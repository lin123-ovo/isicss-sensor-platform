"""Validated CSV/XLSX upload and inference for the active temperature model."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .temperature_pipeline import (
    FEATURE_COLUMNS,
    GATEWAY,
    TARGET_COLUMNS,
    TemperatureModelRepository,
    TemperaturePipelineError,
    absolute_relative_error_percent,
    regression_metrics,
    predict_frame,
    predict_compact36_comparison,
    round_prediction_values,
)


MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024
SUPPORTED_SUFFIXES = (".csv", ".xlsx")
NO_COLUMN = "（不提供）"
FIELD_ALIASES = {
    "timestamp": (
        "timestamp",
        "采集时间",
        "采样时间",
        "时间",
        "datetime",
    ),
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
class UploadColumnMapping:
    """Source columns selected for one upload."""

    timestamp: str
    tilt1Inline: str
    tilt1Crossline: str
    temperature: str
    tilt2Inline: str | None = None
    tilt2Crossline: str | None = None

    def selected_columns(self) -> tuple[str, ...]:
        """Return selected source columns without optional missing targets."""

        return tuple(
            column
            for column in (
                self.timestamp,
                self.tilt1Inline,
                self.tilt1Crossline,
                self.temperature,
                self.tilt2Inline,
                self.tilt2Crossline,
            )
            if column is not None
        )


@dataclass(frozen=True)
class UploadTable:
    """In-memory upload content."""

    filename: str
    sheetName: str | None
    frame: pd.DataFrame


@dataclass(frozen=True)
class NormalizedUpload:
    """Accepted ten-minute rows, rejected source rows, and audit summary."""

    frame: pd.DataFrame
    rejectedRows: pd.DataFrame
    audit: dict[str, Any]
    featureContext: pd.DataFrame | None = None
    inputHistory: pd.DataFrame | None = None


@dataclass(frozen=True)
class PredictionResult:
    """Predictions, optional metrics, and model metadata."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    audit: dict[str, Any]
    modelMetadata: dict[str, Any]
    comparisonMetrics: pd.DataFrame | None = None


class UploadTableReader:
    """Read bounded CSV/XLSX content without persisting the upload."""

    def list_sheets(self, filename: str, content: bytes) -> tuple[str, ...]:
        """Return workbook sheet names; CSV has no sheet names."""

        suffix = self._validate(filename, content)
        if suffix == ".csv":
            return ()
        with pd.ExcelFile(BytesIO(content)) as workbook:
            return tuple(workbook.sheet_names)

    def read(
        self,
        filename: str,
        content: bytes,
        sheetName: str | None = None,
    ) -> UploadTable:
        """Read the selected table and reject empty or underspecified uploads."""

        suffix = self._validate(filename, content)
        if suffix == ".csv":
            frame = self._read_csv(content)
            selectedSheet = None
        else:
            with pd.ExcelFile(BytesIO(content)) as workbook:
                selectedSheet = sheetName or workbook.sheet_names[0]
                if selectedSheet not in workbook.sheet_names:
                    raise TemperaturePipelineError(
                        f"上传文件中找不到工作表：{selectedSheet}"
                    )
                frame = pd.read_excel(workbook, sheet_name=selectedSheet)
        if frame.empty:
            raise TemperaturePipelineError("上传文件没有数据行。")
        if len(frame.columns) < 4:
            raise TemperaturePipelineError(
                "上传文件至少需要时间、1号两方向和温度四列。"
            )
        return UploadTable(filename, selectedSheet, frame)

    @staticmethod
    def _validate(filename: str, content: bytes) -> str:
        suffix = Path(filename).suffix.casefold()
        if suffix not in SUPPORTED_SUFFIXES:
            raise TemperaturePipelineError("仅支持 CSV 和 XLSX 文件。")
        if not content:
            raise TemperaturePipelineError("上传文件为空。")
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise TemperaturePipelineError("上传文件超过50 MB。")
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
        raise TemperaturePipelineError("无法识别 CSV 的编码或分隔符。")


class UploadNormalizer:
    """Normalize prediction uploads according to the active processing profile."""

    def normalize(
        self,
        table: UploadTable,
        mapping: UploadColumnMapping,
    ) -> NormalizedUpload:
        """Validate and normalize rows before the shared cleaning pipeline."""

        from sensor_database.processing import current_processing_version
        if current_processing_version() == "final_v3":
            from sensor_database.import_service import (
                ColumnMapping, UploadNormalizer as DatabaseNormalizer,
                UploadTable as DatabaseTable,
            )
            from sensor_database import SensorDatabaseError

            if mapping.tilt2Inline is None or mapping.tilt2Crossline is None:
                raise TemperaturePipelineError("与数据入库一致，必须选择2号顺线和横线列。")
            try:
                imported = DatabaseNormalizer().normalize(
                    DatabaseTable(table.filename, table.sheetName, table.frame),
                    ColumnMapping(mapping.timestamp, mapping.tilt1Inline, mapping.tilt1Crossline,
                                  mapping.temperature, mapping.tilt2Inline, mapping.tilt2Crossline),
                )
            except SensorDatabaseError as error:
                raise TemperaturePipelineError(str(error)) from error
            return NormalizedUpload(imported.frame, imported.rejected_rows, imported.audit)

        selected = mapping.selected_columns()
        if len(selected) != len(set(selected)):
            raise TemperaturePipelineError("不同业务字段不能映射到同一列。")
        missingColumns = [column for column in selected if column not in table.frame]
        if missingColumns:
            raise TemperaturePipelineError(f"上传文件缺少映射列：{missingColumns}")

        source = table.frame.copy()
        normalized = pd.DataFrame(index=source.index)
        normalized["source_row"] = np.arange(2, len(source) + 2, dtype=int)
        normalized["timestamp"] = pd.to_datetime(
            source[mapping.timestamp], errors="coerce"
        )
        sourceMapping = {
            "tilt1_inline_model": mapping.tilt1Inline,
            "tilt1_crossline_model": mapping.tilt1Crossline,
            "temperature_celsius": mapping.temperature,
        }
        for target, sourceColumn in sourceMapping.items():
            normalized[target] = pd.to_numeric(source[sourceColumn], errors="coerce")
        optionalTargets = {
            "tilt2_inline": mapping.tilt2Inline,
            "tilt2_crossline": mapping.tilt2Crossline,
        }
        for target, sourceColumn in optionalTargets.items():
            normalized[target] = (
                pd.to_numeric(source[sourceColumn], errors="coerce")
                if sourceColumn is not None
                else np.nan
            )

        required = ["timestamp", *FEATURE_COLUMNS]
        completeMask = normalized.loc[:, required].notna().all(axis=1)
        finiteMask = pd.Series(True, index=normalized.index)
        finiteMask.loc[:] = np.isfinite(
            normalized.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        ).all(axis=1)
        acceptedMask = completeMask & finiteMask
        # In personal v3, retain only genuinely empty whole rows as placeholders;
        # arbitrary non-numeric text and partially incomplete rows remain rejected.
        from sensor_database.processing import current_processing_version
        if current_processing_version() == "final_v3":
            mapped = [mapping.tilt1Inline, mapping.tilt1Crossline, mapping.temperature,
                      *[name for name in (mapping.tilt2Inline, mapping.tilt2Crossline) if name is not None]]
            blank = source[mapped].apply(lambda col: col.isna() | col.astype(str).str.strip().isin(("", "缺测"))).all(axis=1)
            acceptedMask |= normalized["timestamp"].notna() & blank
        rejected = normalized.loc[~acceptedMask].copy()
        rejected["rejection_reason"] = np.where(
            rejected["timestamp"].isna(),
            "缺少或无法识别时间",
            "1号倾角或温度缺失、非数值或非有限值",
        )
        accepted = normalized.loc[acceptedMask].copy()
        if accepted.empty:
            raise TemperaturePipelineError(
                "没有时间、1号两方向和温度均有效的预测行。"
            )
        accepted["timestamp"] = accepted["timestamp"].dt.round("10min")
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
            "gateway": GATEWAY,
            "source_filename": table.filename,
            "source_sheet": table.sheetName,
            "source_rows": len(source),
            "accepted_rows": len(accepted),
            "rejected_rows": len(rejected),
            "prediction_rows": len(grouped),
            "temperature_policy": ("required_with_full_short_gap_placeholder"
                                   if current_processing_version() == "final_v3"
                                   else "required_no_imputation_no_fallback"),
        }
        return NormalizedUpload(grouped, rejected, audit)


class InteractivePredictionService:
    """Run a saved three-feature absolute-value model."""

    def __init__(self, projectRoot: Path) -> None:
        self._project_root = projectRoot.expanduser().resolve()
        self._repository = TemperatureModelRepository(projectRoot)

    def predict(
        self,
        upload: NormalizedUpload,
        modelVersion: str | None = None,
        *,
        compareCrosslineBranches: bool = False,
    ) -> PredictionResult:
        """Predict with the active or selected model; targets are never fitted."""

        if not bool(upload.audit.get("cleaning_applied", False)):
            upload = self._clean_transient_upload(upload)

        loaded = (
            self._repository.load_version(modelVersion)
            if modelVersion is not None
            else self._repository.load()
        )
        from . import crossline_compact36 as compact36
        from . import crossline_seven as seven
        from . import crossline_nine_regional as nine
        from . import crossline_tabicl_single as tabicl_single
        from . import crossline_tabm_single as tabm_single
        from sensor_database.processing import current_processing_version

        context = upload.featureContext
        if upload.inputHistory is not None and (
            getattr(loaded.model, 'featureVersion', None) in (compact36.FEATURE_VERSION, seven.FEATURE_VERSION, nine.FEATURE_VERSION, tabicl_single.FEATURE_VERSION, tabm_single.FEATURE_VERSION)
            or current_processing_version() in ('final_v2', 'final_v3')
        ):
            context = upload.inputHistory
        comparisons = None
        if compareCrosslineBranches:
            comparisons = predict_compact36_comparison(loaded.model, upload.frame, context_frame=context)
            prediction = comparisons["ensemble"]
        else:
            prediction = predict_frame(loaded.model, upload.frame, context_frame=context)
        result = upload.frame.loc[
            :,
            ["timestamp", "source_row_count", *FEATURE_COLUMNS, *TARGET_COLUMNS],
        ].copy()
        if "temperature_review_included" in upload.frame:
            result["temperature_review_included"] = upload.frame["temperature_review_included"].astype(bool)
        prediction = round_prediction_values(prediction)
        result["predicted_inline"] = prediction[:, 0]
        result["predicted_crossline"] = prediction[:, 1]
        if comparisons is not None:
            result["predicted_crossline_tabicl"] = comparisons["tabicl"][:, 1]
            result["predicted_crossline_tabm"] = comparisons["tabm"][:, 1]
        result["gateway"] = GATEWAY
        result["model_version"] = str(loaded.metadata["version_id"])
        identity_columns = ["model_version", "gateway"]
        result = result.loc[
            :, identity_columns + [
                column for column in result.columns if column not in identity_columns
            ]
        ]
        result["absolute_error_inline"] = np.where(
            result["tilt2_inline"].notna(),
            np.abs(result["predicted_inline"] - result["tilt2_inline"]),
            np.nan,
        )
        result["absolute_error_crossline"] = np.where(
            result["tilt2_crossline"].notna(),
            np.abs(result["predicted_crossline"] - result["tilt2_crossline"]),
            np.nan,
        )
        result["absolute_relative_error_inline_percent"] = (
            absolute_relative_error_percent(
                result["tilt2_inline"],
                result["predicted_inline"],
            )
        )
        result["absolute_relative_error_crossline_percent"] = (
            absolute_relative_error_percent(
                result["tilt2_crossline"],
                result["predicted_crossline"],
            )
        )
        metricFrames = []
        for direction in ("inline", "crossline"):
            target = f"tilt2_{direction}"
            predicted = f"predicted_{direction}"
            evaluable = result[target].notna() & result[predicted].notna()
            if evaluable.any():
                metricFrames.append(
                    pd.DataFrame(
                        [
                            {
                                "split": "uploaded_evaluation",
                                "model": loaded.model.name,
                                "direction": direction,
                                "sample_count": int(evaluable.sum()),
                                **regression_metrics(
                                    result.loc[evaluable, target].to_numpy(dtype=float),
                                    result.loc[evaluable, predicted].to_numpy(dtype=float),
                                    result.loc[evaluable, "timestamp"],
                                ),
                            }
                        ]
                    )
                )
        metrics = (
            pd.concat(metricFrames, ignore_index=True)
            if metricFrames
            else pd.DataFrame(
                columns=[
                    "split",
                    "model",
                    "direction",
                    "sample_count",
                    "mae",
                    "rmse",
                    "max_abs_error",
                    "max_error_timestamp",
                    "unit",
                    "mean_abs_relative_error_percent",
                    "max_abs_relative_error_percent",
                    "max_relative_error_timestamp",
                    "relative_error_valid_count",
                    "relative_error_excluded_zero_count",
                    "relative_error_unit",
                ]
            )
        )
        comparison_metrics = None
        if comparisons is not None:
            valid = result["tilt2_crossline"].notna()
            rows = []
            if valid.any():
                for method in ("tabicl", "tabm", "ensemble"):
                    rows.append({
                        "method": method,
                        "sample_count": int(valid.sum()),
                        **regression_metrics(
                            result.loc[valid, "tilt2_crossline"].to_numpy(dtype=float),
                            comparisons[method][valid.to_numpy(), 1],
                            result.loc[valid, "timestamp"],
                        ),
                    })
            comparison_metrics = pd.DataFrame(rows)
        audit = dict(upload.audit)
        if comparisons is not None:
            audit["crossline_comparison"] = "archived_components_same_input"
        return PredictionResult(result, metrics, audit, loaded.metadata, comparison_metrics)

    def _clean_transient_upload(self, upload: NormalizedUpload) -> NormalizedUpload:
        """Apply the database cleaning contract without persisting the upload."""

        from sensor_database import SensorDatabaseError, get_sensor_database
        from sensor_database.cleaning import FIELD_SPECS, SensorCleaningPipeline
        from sensor_database.processing import is_final_processing

        try:
            database = get_sensor_database(self._project_root)
            incoming = upload.frame.copy()
            incoming["timestamp"] = pd.to_datetime(incoming["timestamp"], errors="raise")
            requested_timestamps = set(incoming["timestamp"].dt.round("10min"))
            from sensor_database.processing import current_processing_version
            if current_processing_version() == "final_v3":
                cleaning = database.clean_prediction_upload(
                    incoming, temperature_jump_check_enabled=True,
                )
            elif is_final_processing():
                cleaning = database.clean_prediction_upload(incoming)
            else:
                history = database.load_measurements()
                combined = pd.concat([history, incoming], ignore_index=True, sort=False)
                combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="raise")
                combined = (
                    combined.sort_values("timestamp", kind="stable")
                    .drop_duplicates(subset=["timestamp"], keep="last")
                    .reset_index(drop=True)
                )
                cleaning = SensorCleaningPipeline.for_project(self._project_root).clean(combined)
        except (SensorDatabaseError, ValueError) as error:
            raise TemperaturePipelineError(f"上传数据清洗失败：{error}") from error

        selected_times = pd.to_datetime(cleaning.frame["timestamp"])
        select_mask = selected_times.isin(requested_timestamps)
        from sensor_database.processing import current_processing_version
        if is_final_processing() and current_processing_version() == "final_v3" and requested_timestamps:
            # Also expose newly reconstructed missing buckets inside the upload interval.
            inserted = cleaning.frame["is_inserted_gap"].fillna(False)
            select_mask |= inserted & selected_times.between(min(requested_timestamps), max(requested_timestamps))
        selected = cleaning.frame.loc[select_mask].copy()
        selected = selected.sort_values("timestamp", kind="stable").reset_index(drop=True)
        if not requested_timestamps.issubset(set(pd.to_datetime(selected["timestamp"]))):
            raise TemperaturePipelineError("上传数据无法与十分钟清洗时间线完整对应。")
        insufficient = pd.Series(False, index=selected.index)
        for spec in FIELD_SPECS:
            insufficient |= selected[f"{spec.prefix}_hampel_reference_count"].lt(30)
        if current_processing_version() == "final_v3":
            # Match database prediction eligibility; the pipeline already marks
            # unresolved or unavailable inputs, including insufficient references.
            insufficient[:] = False
        unusable = (
            insufficient
            | selected["model_input_unresolved"].astype(bool)
            | ~selected["model_input_available"].astype(bool)
        )
        excluded = selected.get("processed_row_excluded", pd.Series(False, index=selected.index)).astype(bool)
        unusable |= excluded
        counts = {
            "上下文不足": int(insufficient.sum()),
            "未解决异常": int(selected["model_input_unresolved"].sum()),
            "输入缺失": int((~selected["model_input_available"]).sum()),
            "零温度剔除": int(excluded.sum()),
        }
        usable = selected.loc[~unusable].copy()
        if usable.empty:
            raise TemperaturePipelineError(
                "上传数据清洗后没有可预测记录："
                + "，".join(f"{key}{value}条" for key, value in counts.items() if value)
            )
        prediction_columns = [
            "timestamp",
            "source_row_count",
            *FEATURE_COLUMNS,
            *TARGET_COLUMNS,
        ]
        frame = usable.loc[:, prediction_columns].copy()
        for target in TARGET_COLUMNS:
            flag = f"{target}_gap_imputed"
            if flag in usable:
                frame.loc[usable[flag].fillna(False).astype(bool), target] = np.nan
        affected = cleaning.affected_rows.copy()
        if not affected.empty:
            affected = affected.loc[
                pd.to_datetime(affected["timestamp"]).isin(requested_timestamps)
            ].copy()
        audit = {
            **upload.audit,
            "cleaning_policy_sha256": cleaning.policy_sha256,
            "cleaning_field_summary": [
                item.to_dict() for item in cleaning.field_summary
            ],
            "cleaning_affected_rows": affected.to_dict("records"),
            "cleaning_input_rows": len(selected),
            "cleaning_prediction_rows": len(frame),
            "cleaning_skipped_rows": int(unusable.sum()),
            "cleaning_skipped_insufficient_context_rows": counts["上下文不足"],
            "cleaning_skipped_unresolved_rows": counts["未解决异常"],
            "cleaning_skipped_missing_input_rows": counts["输入缺失"],
            "cleaning_skipped_zero_temperature_rows": counts["零温度剔除"],
        }
        if is_final_processing():
            from sensor_database.temperature_cusum import AUDIT_COLUMNS as CUSUM_AUDIT_COLUMNS

            cusum_outlier = selected["temperature_cusum_outlier"].astype(bool)
            cusum_applied = selected["temperature_cusum_candidate_applied"].astype(bool)
            temperature_rows = selected.loc[cusum_outlier | cusum_applied, [
                "timestamp", "temperature_raw_celsius", "temperature_celsius",
                "temperature_review_required", "temperature_review_reason", *CUSUM_AUDIT_COLUMNS,
            ]]
            audit.update({
                "cleaning_temperature_rule_versions": sorted(selected["temperature_rule_version"].unique().tolist()),
                "cleaning_cusum_checked_rows": int(selected["temperature_cusum_checked"].sum()),
                "cleaning_cusum_applied_rows": int(cusum_applied.sum()),
                "cleaning_cusum_review_rows": int((cusum_outlier & selected["temperature_review_required"]).sum()),
                "cleaning_cusum_status_counts": {str(key): int(value) for key, value in
                                               selected["temperature_cusum_status"].value_counts().items()},
                "cleaning_cusum_temperature_rows": temperature_rows.to_dict("records"),
            })
            from sensor_database.temperature_daily_trend import AUDIT_COLUMNS as DAILY_AUDIT_COLUMNS

            daily = selected["temperature_daily_corrected"].astype(bool)
            if selected["temperature_daily_checked"].any() or selected["temperature_rule_version"].str.contains("daily_trend").any():
                audit.update({
                    "cleaning_daily_checked_rows": int(selected["temperature_daily_checked"].sum()),
                    "cleaning_daily_corrected_rows": int(daily.sum()),
                    "cleaning_daily_max_abs_correction": float(selected["temperature_daily_offset_celsius"].abs().max())
                    if selected["temperature_daily_checked"].any() else 0.,
                    "cleaning_daily_temperature_rows": selected.loc[daily, [
                        "timestamp", "temperature_raw_celsius", "temperature_celsius", *DAILY_AUDIT_COLUMNS,
                    ]].to_dict("records"),
                })
        context_mask = pd.Series(True, index=cleaning.frame.index)
        if "model_input_available" in cleaning.frame:
            context_mask &= cleaning.frame["model_input_available"].astype(bool)
        if "model_input_unresolved" in cleaning.frame:
            context_mask &= ~cleaning.frame["model_input_unresolved"].astype(bool)
        if "is_inserted_gap" in cleaning.frame:
            context_mask &= ~cleaning.frame["is_inserted_gap"].astype(bool)
        context = cleaning.frame.loc[context_mask, ["timestamp", *FEATURE_COLUMNS]].copy()
        context = context.sort_values("timestamp", kind="stable").reset_index(drop=True)
        from .crossline_compact36 import exogenous_history

        if "history_barrier" in cleaning.frame:
            context = exogenous_history(cleaning.frame)

        return NormalizedUpload(frame, upload.rejectedRows, audit, context,
                                inputHistory=exogenous_history(cleaning.frame))

def normalized_column_name(value: object) -> str:
    """Normalize common punctuation and whitespace for alias matching."""

    return (
        "".join(str(value).strip().casefold().split())
        .replace("_", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def suggest_mapping(columns: list[object]) -> dict[str, str | None]:
    """Suggest source columns while allowing the UI to override every choice."""

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
