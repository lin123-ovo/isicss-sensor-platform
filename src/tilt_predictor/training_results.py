"""Load and validate the latest ratio-training reports for presentation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .model_registry import ModelRegistryError
from .registry_factory import get_model_registry
from .prediction_reports import (
    build_detailed_prediction_report,
    normalize_prediction_values,
    recalculate_test_metrics,
    remove_retired_metric_columns,
)
from .temperature_pipeline import TemperaturePipelineError


@dataclass(frozen=True)
class TrainingResultArtifacts:
    """Persisted evidence produced by the latest ratio-training run."""

    metadata: dict[str, Any]
    metrics: pd.DataFrame
    candidates: pd.DataFrame
    predictions: pd.DataFrame

    @classmethod
    def load(cls, project_root: Path) -> "TrainingResultArtifacts":
        """Load the canonical active version's report set."""

        root = project_root.expanduser().resolve()
        try:
            registry = get_model_registry(root)
            active = registry.bootstrap_current_version()
            if active is None:
                raise ModelRegistryError("尚无兼容清洗合同的活动模型。")
        except ModelRegistryError as error:
            raise TemperaturePipelineError(str(error)) from error
        return cls.load_from_directory(registry.version_directory(active.version_id))

    @classmethod
    def load_from_directory(cls, directory: Path) -> "TrainingResultArtifacts":
        """Load and validate one version or draft artifact directory."""

        root = directory.expanduser().resolve()
        paths = {
            "训练决策": root / "metadata.json",
            "评估指标": root / "metrics.csv",
            "候选参数": root / "parameter_candidates.csv",
            "测试集预测": root / "test_predictions.csv",
        }
        missing = [label for label, path in paths.items() if not path.is_file()]
        if missing:
            raise TemperaturePipelineError(
                f"最近一次训练结果不完整，缺少：{', '.join(missing)}。"
            )

        try:
            metadata = json.loads(
                paths["训练决策"].read_text(encoding="utf-8-sig")
            )
            metrics = pd.read_csv(paths["评估指标"])
            candidates = pd.read_csv(paths["候选参数"])
            predictions = pd.read_csv(
                paths["测试集预测"],
                parse_dates=["timestamp"],
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise TemperaturePipelineError(f"无法读取最近一次训练结果：{error}") from error

        cls._validate(metadata, metrics, candidates, predictions)
        predictions = normalize_prediction_values(predictions)
        metrics = recalculate_test_metrics(predictions, metadata, metrics)
        return cls(metadata, metrics, candidates, predictions)

    @staticmethod
    def _validate(
        metadata: dict[str, Any],
        metrics: pd.DataFrame,
        candidates: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> None:
        """Reject partial or stale-shaped reports before rendering them."""

        required_metadata = {
            "model",
            "parameters",
            "raw_input_equations",
            "status",
            "split",
            "sample_counts",
        }
        missing_metadata = sorted(required_metadata - set(metadata))
        if missing_metadata:
            raise TemperaturePipelineError(
                f"训练决策缺少字段：{missing_metadata}。"
            )

        required_columns = {
            "评估指标": (
                metrics,
                {
                    "split",
                    "model",
                    "direction",
                    "mae",
                    "rmse",
                    "max_abs_error",
                    "mean_abs_relative_error_percent",
                    "max_abs_relative_error_percent",
                    "relative_error_valid_count",
                    "relative_error_excluded_zero_count",
                },
            ),
            "候选参数": (
                candidates,
                {"rank", "model", "parameters_json", "mean_mae"},
            ),
            "测试集预测": (
                predictions,
                {
                    "timestamp",
                    "tilt2_inline",
                    "tilt2_crossline",
                    "predicted_inline",
                    "predicted_crossline",
                    "absolute_error_inline",
                    "absolute_error_crossline",
                    "absolute_relative_error_inline_percent",
                    "absolute_relative_error_crossline_percent",
                },
            ),
        }
        for label, (frame, expected) in required_columns.items():
            missing_columns = sorted(expected - set(frame.columns))
            if missing_columns:
                raise TemperaturePipelineError(
                    f"{label}缺少字段：{missing_columns}。"
                )

    def selected_test_metrics(self) -> pd.DataFrame:
        """Return only the selected model's two final-test metric rows."""

        selected_model = str(self.metadata["model"])
        selected = self.metrics.loc[
            (self.metrics["split"] == "test")
            & (self.metrics["model"] == selected_model)
        ].copy()
        if set(selected["direction"]) != {"inline", "crossline"}:
            raise TemperaturePipelineError("最终测试指标缺少顺线或横线结果。")
        selected["direction"] = pd.Categorical(
            selected["direction"],
            categories=["inline", "crossline"],
            ordered=True,
        )
        selected = selected.sort_values("direction", kind="stable").reset_index(
            drop=True
        )
        return remove_retired_metric_columns(selected)

    def detailed_predictions(
        self,
        source_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return the complete row-level report, upgrading legacy artifacts if needed."""

        return build_detailed_prediction_report(
            self.predictions,
            self.metadata,
            self.metrics,
            source_frame=source_frame,
        )

    def equation_rows(self) -> pd.DataFrame:
        """Return learned raw-input intercepts and coefficients by target."""

        labels = {
            "tilt2_inline": "2号顺线",
            "tilt2_crossline": "2号横线",
            "tilt2_crossline_delta": "2号横线变化量",
            "intercept": "截距",
            "tilt1_inline_model": "1号顺线系数",
            "tilt1_crossline_model": "1号横线系数",
            "temperature_celsius": "温度系数",
            "tilt1_inline_delta": "1号顺线变化量系数",
            "tilt1_crossline_delta": "1号横线变化量系数",
            "temperature_delta": "温度变化量系数",
            "temperature_change_10m": "10分钟温差系数",
            "temperature_change_1h": "1小时温差系数",
            "temperature_rolling_mean_1h_delta": "1小时平均温度变化系数",
        }
        from .model_explanation import FEATURE_LABELS

        for name, label in FEATURE_LABELS.items():
            if "^" in name or " " in name:
                labels[name] = f"{label}系数"
        rows = []
        for target, equation in self.metadata["raw_input_equations"].items():
            row = {"预测目标": labels.get(target, target)}
            row.update(
                {labels.get(name, name): value for name, value in equation.items()}
            )
            rows.append(row)
        return pd.DataFrame(rows)
