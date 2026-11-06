"""Read, map, validate, and chronologically normalize CSV/XLSX uploads."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import SensorDatabaseError


MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024
SUPPORTED_SUFFIXES = (".csv", ".xlsx")
NO_COLUMN = "（未识别）"
MEASUREMENT_COLUMNS = (
    "tilt1_inline_model",
    "tilt1_crossline_model",
    "temperature_celsius",
    "tilt2_inline",
    "tilt2_crossline",
)
FIELD_ALIASES = {
    "timestamp": ("timestamp", "采集时间", "采样时间", "时间", "datetime"),
    "tilt1_inline_model": (
        "tilt1_inline_model",
        "tilt1_inline",
        "1号顺线方向倾斜度（x轴）",
        "1号顺线方向倾斜度",
    ),
    "tilt1_crossline_model": (
        "tilt1_crossline_model",
        "tilt1_crossline",
        "1号横线方向倾斜度（y轴）",
        "1号横线方向倾斜度",
    ),
    "temperature_celsius": (
        "temperature_celsius",
        "temperature",
        "环境温度",
        "温度",
    ),
    "tilt2_inline": (
        "tilt2_inline",
        "2号顺线方向倾斜度（x轴）",
        "2号顺线方向倾斜度",
    ),
    "tilt2_crossline": (
        "tilt2_crossline",
        "2号横线方向倾斜度（y轴）",
        "2号横线方向倾斜度",
    ),
}


@dataclass(frozen=True)
class ColumnMapping:
    """Source columns selected for one import."""

    timestamp: str
    tilt1_inline_model: str
    tilt1_crossline_model: str
    temperature_celsius: str
    tilt2_inline: str
    tilt2_crossline: str

    def selected_columns(self) -> tuple[str, ...]:
        """Return every source column in database field order."""

        return (
            self.timestamp,
            self.tilt1_inline_model,
            self.tilt1_crossline_model,
            self.temperature_celsius,
            self.tilt2_inline,
            self.tilt2_crossline,
        )


@dataclass(frozen=True)
class UploadTable:
    """One uploaded worksheet held in memory before persistence."""

    filename: str
    sheet_name: str | None
    frame: pd.DataFrame


@dataclass(frozen=True)
class NormalizedUpload:
    """Accepted time-series rows, rejected rows, and import audit counts."""

    frame: pd.DataFrame
    rejected_rows: pd.DataFrame
    audit: dict[str, Any]


class UploadTableReader:
    """Read bounded CSV/XLSX uploads without changing the source file."""

    def list_sheets(self, filename: str, content: bytes) -> tuple[str, ...]:
        """Return XLSX sheet names; CSV files have no sheet list."""

        suffix = self._validate(filename, content)
        if suffix == ".csv":
            return ()
        try:
            with pd.ExcelFile(BytesIO(content)) as workbook:
                return tuple(str(name) for name in workbook.sheet_names)
        except (OSError, ValueError, ImportError) as error:
            raise SensorDatabaseError(f"无法读取 XLSX 文件：{error}") from error

    def read(
        self,
        filename: str,
        content: bytes,
        sheet_name: str | None = None,
    ) -> UploadTable:
        """Read the selected table after format and size validation."""

        suffix = self._validate(filename, content)
        try:
            if suffix == ".csv":
                frame = self._read_csv(content)
                selected_sheet = None
            else:
                with pd.ExcelFile(BytesIO(content)) as workbook:
                    selected_sheet = sheet_name or str(workbook.sheet_names[0])
                    if selected_sheet not in workbook.sheet_names:
                        raise SensorDatabaseError(
                            f"上传文件中找不到工作表：{selected_sheet}"
                        )
                    frame = pd.read_excel(workbook, sheet_name=selected_sheet)
        except SensorDatabaseError:
            raise
        except (OSError, ValueError, ImportError) as error:
            raise SensorDatabaseError(f"无法读取上传文件：{error}") from error

        if frame.empty:
            raise SensorDatabaseError("上传文件没有数据行。")
        if len(frame.columns) < 6:
            raise SensorDatabaseError("上传文件至少需要时间、四项倾角和温度六列。")
        return UploadTable(filename, selected_sheet, frame)

    @staticmethod
    def _validate(filename: str, content: bytes) -> str:
        suffix = Path(filename).suffix.casefold()
        if suffix not in SUPPORTED_SUFFIXES:
            raise SensorDatabaseError("仅支持 CSV 和 XLSX 文件。")
        if not content:
            raise SensorDatabaseError("上传文件为空。")
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise SensorDatabaseError("上传文件超过 50 MB。")
        if suffix == ".xlsx" and not content.startswith(b"PK"):
            raise SensorDatabaseError("文件扩展名是 XLSX，但内容不是有效的 XLSX 容器。")
        return suffix

    @staticmethod
    def _read_csv(content: bytes) -> pd.DataFrame:
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return pd.read_csv(
                    BytesIO(content),
                    encoding=encoding,
                    sep=None,
                    engine="python",
                )
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise SensorDatabaseError("无法识别 CSV 的编码或分隔符。")


class UploadNormalizer:
    """Validate required values and sort accepted measurements by timestamp."""

    def normalize(
        self,
        table: UploadTable,
        mapping: ColumnMapping,
    ) -> NormalizedUpload:
        """Reject incomplete rows and combine exact duplicate timestamps."""

        selected = mapping.selected_columns()
        if len(selected) != len(set(selected)):
            raise SensorDatabaseError("不同业务字段不能映射到同一列。")
        missing_columns = [column for column in selected if column not in table.frame]
        if missing_columns:
            raise SensorDatabaseError(f"上传文件缺少映射列：{missing_columns}")

        source = table.frame.copy()
        normalized = pd.DataFrame(index=source.index)
        normalized["source_row"] = np.arange(2, len(source) + 2, dtype=int)
        normalized["timestamp"] = pd.to_datetime(
            source[mapping.timestamp],
            errors="coerce",
        )
        for field in MEASUREMENT_COLUMNS:
            source_column = getattr(mapping, field)
            normalized[field] = pd.to_numeric(source[source_column], errors="coerce")

        required = ["timestamp", *MEASUREMENT_COLUMNS]
        complete_mask = normalized.loc[:, required].notna().all(axis=1)
        finite_mask = pd.Series(False, index=normalized.index)
        numeric_values = normalized.loc[:, list(MEASUREMENT_COLUMNS)].to_numpy(
            dtype=float
        )
        finite_mask.loc[:] = np.isfinite(numeric_values).all(axis=1)
        accepted_mask = complete_mask & finite_mask

        rejected = normalized.loc[~accepted_mask].copy()
        rejected["rejection_reason"] = np.where(
            rejected["timestamp"].isna(),
            "缺少或无法识别时间",
            "倾角或温度缺失、非数值或非有限值",
        )
        accepted = normalized.loc[accepted_mask].copy()
        if accepted.empty:
            raise SensorDatabaseError("没有时间、四项倾角和温度均有效的数据行。")

        grouped = (
            accepted.groupby("timestamp", as_index=False, sort=True)
            .agg(
                source_row_count=("source_row", "size"),
                tilt1_inline_model=("tilt1_inline_model", "median"),
                tilt1_crossline_model=("tilt1_crossline_model", "median"),
                temperature_celsius=("temperature_celsius", "median"),
                tilt2_inline=("tilt2_inline", "median"),
                tilt2_crossline=("tilt2_crossline", "median"),
            )
            .sort_values("timestamp", kind="stable")
            .reset_index(drop=True)
        )
        audit = {
            "source_filename": table.filename,
            "source_sheet": table.sheet_name or "",
            "source_rows": len(source),
            "accepted_rows": int(accepted_mask.sum()),
            "rejected_rows": int((~accepted_mask).sum()),
            "time_series_rows": len(grouped),
            "duplicate_timestamp_rows_combined": int(len(accepted) - len(grouped)),
        }
        return NormalizedUpload(grouped, rejected, audit)


def normalized_column_name(value: object) -> str:
    """Normalize punctuation and whitespace for alias matching."""

    return (
        "".join(str(value).strip().casefold().split())
        .replace("_", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def suggest_mapping(columns: list[object]) -> dict[str, str | None]:
    """Suggest a source column for every required database field."""

    normalized = {normalized_column_name(column): str(column) for column in columns}
    result: dict[str, str | None] = {}
    for field, aliases in FIELD_ALIASES.items():
        result[field] = next(
            (
                normalized[normalized_column_name(alias)]
                for alias in aliases
                if normalized_column_name(alias) in normalized
            ),
            None,
        )
    return result
