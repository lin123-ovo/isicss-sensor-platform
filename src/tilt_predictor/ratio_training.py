"""Chronological ratio-based training from the local CSV sensor database."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .registry_factory import get_model_registry
from .prediction_reports import build_detailed_prediction_report
from . import crossline_tabpfn_v2 as tabpfn_v2
from . import crossline_probability as probability
from . import crossline_compact36 as compact36
from . import crossline_nine_regional as nine
from . import regional_weight_search as weight_search
from . import crossline_tabicl_single as tabicl_single
from . import crossline_tabicl_16 as tabicl_16
from .tabicl_options import normalize_options
from .temperature_pipeline import (
    COMPACT_RECIPES,
    compact_recipe_metadata,
    DIRECTION_TARGETS,
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    FIXED_MODEL_NAME,
    GATEWAY,
    TARGET_COLUMNS,
    FittedDualModel,
    TemperaturePipelineError,
    WorkflowConfig,
    baseline_metrics,
    absolute_relative_error_percent,
    extract_linear_equations,
    extract_feature_importances,
    extract_derived_feature_columns,
    fit_direction_model,
    fit_model,
    metrics_table,
    normalize_model_parameters,
    normalize_direction_models,
    direction_models_from_parameters,
    predict_frame,
    regression_metrics,
    round_prediction_values,
)


CSV_DATABASE_DATASET_VERSION = "csg124_cleaned_database_v2"
OPTIMIZATION_OBJECTIVE = (
    "mean_mae_degrees_across_3_time_folds_per_direction"
)
OPTIMIZATION_TARGETS = {
    "inline": "tilt2_inline_absolute",
    "crossline": "tilt2_crossline_absolute",
}
MIN_OPTUNA_TRIALS = 10
MAX_OPTUNA_TRIALS = 200
OPTUNA_TRIAL_STEP = 10
MIN_AUTO_RATIO_COUNT = 2
MAX_AUTO_RATIO_COUNT = 20
MIN_TEST_PERCENTAGE = 10
MAX_TEST_PERCENTAGE = 90
DEFAULT_TEST_PERCENTAGE = 30


def validate_optimization_trials(value: int) -> int:
    """Validate the UI-controlled Optuna trial count per direction."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("Optuna 试验次数必须是整数。")
    trials = int(value)
    if not MIN_OPTUNA_TRIALS <= trials <= MAX_OPTUNA_TRIALS:
        raise ValueError(
            f"Optuna 每方向试验次数必须在 {MIN_OPTUNA_TRIALS} 到 "
            f"{MAX_OPTUNA_TRIALS} 次之间。"
        )
    if trials % OPTUNA_TRIAL_STEP != 0:
        raise ValueError(f"Optuna 试验次数必须按 {OPTUNA_TRIAL_STEP} 次递增。")
    return trials


def automatic_split_ratios(
    count: int, *, maximum_train_weight: float = 8.0
) -> tuple["SplitRatio", ...]:
    """Return evenly spaced train/validation ratios with one shared test holdout.

    Every generated candidate reserves the newest 10% for the final test.
    This keeps the test period identical while different train/validation
    allocations are compared using validation evidence only. The default
    upper training weight remains 8.0 for customer callers; the full UI
    requests 8.9. Bounds or counts that cannot yield unique one-decimal
    ratios raise TemperaturePipelineError.
    """

    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TemperaturePipelineError("自动比较的比例数量必须是整数。")
    normalized_count = int(count)
    if not MIN_AUTO_RATIO_COUNT <= normalized_count <= MAX_AUTO_RATIO_COUNT:
        raise TemperaturePipelineError(
            f"自动比较的比例数量必须在 {MIN_AUTO_RATIO_COUNT} 到 "
            f"{MAX_AUTO_RATIO_COUNT} 之间。"
        )
    if (
        isinstance(maximum_train_weight, bool)
        or not isinstance(maximum_train_weight, Real)
        or not math.isfinite(maximum_train_weight)
        or not 6.1 <= maximum_train_weight <= 8.9
    ):
        raise TemperaturePipelineError("自动比较的训练权重上限必须在6.1到8.9之间。")
    if not math.isclose(
        maximum_train_weight, round(maximum_train_weight, 1), rel_tol=0.0, abs_tol=1e-9
    ):
        raise TemperaturePipelineError("自动比较的训练权重上限最多支持一位小数。")
    train_weights = np.linspace(6.0, maximum_train_weight, normalized_count)
    ratios = tuple(
        SplitRatio(round(float(train), 1), round(9.0 - float(train), 1), 1.0)
        for train in train_weights
    )
    for ratio in ratios:
        ratio.validate()
    if len({ratio.label for ratio in ratios}) != normalized_count:
        raise TemperaturePipelineError("当前训练权重范围无法生成指定数量的不同比例。")
    return ratios


def multi_ratio_ranking_row(
    metadata: dict[str, Any],
) -> dict[str, float | str | int]:
    """Extract crossline-relative ranking evidence from one candidate."""

    def optional_float(value: Any) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    try:
        requested_ratio = metadata["split"]["requested_ratio"]
        ratio = str(requested_ratio["label"])
        directions = metadata["evaluation"]["validation"]["directions"]
        inline_mae = float(directions["inline"]["mae"])
        crossline_mae = float(directions["crossline"]["mae"])
        validation_crossline = directions["crossline"]
        test_directions = metadata["evaluation"]["test"]["directions"]
        test_inline_mae = float(test_directions["inline"]["mae"])
        test_crossline_mae = float(test_directions["crossline"]["mae"])
        test_crossline = test_directions["crossline"]
    except (KeyError, TypeError, ValueError) as error:
        raise TemperaturePipelineError("自动比例比较缺少完整的评估指标。") from error
    return {
        "ratio": ratio,
        "validation_crossline_mean_abs_relative_error_percent": optional_float(
            validation_crossline.get("mean_abs_relative_error_percent")
        ),
        "validation_crossline_relative_error_valid_count": int(
            validation_crossline.get("relative_error_valid_count", 0)
        ),
        "validation_crossline_relative_error_excluded_zero_count": int(
            validation_crossline.get("relative_error_excluded_zero_count", 0)
        ),
        "validation_sample_count": int(
            metadata.get("sample_counts", {}).get("validation", 0)
        ),
        "validation_inline_mae": inline_mae,
        "validation_crossline_mae": crossline_mae,
        "test_crossline_mean_abs_relative_error_percent": optional_float(
            test_crossline.get("mean_abs_relative_error_percent")
        ),
        "test_crossline_relative_error_valid_count": int(
            test_crossline.get("relative_error_valid_count", 0)
        ),
        "test_crossline_relative_error_excluded_zero_count": int(
            test_crossline.get("relative_error_excluded_zero_count", 0)
        ),
        "test_inline_mae": test_inline_mae,
        "test_crossline_mae": test_crossline_mae,
    }


@dataclass(frozen=True)
class SplitRatio:
    """Positive train, validation, and test weights totaling ten."""

    train: float
    validation: float
    test: float

    def validate(self) -> None:
        """Require positive, one-decimal weights whose sum is exactly ten."""

        self._normalized_weights()

    def _normalized_weights(self) -> tuple[float, float, float]:
        """Return stable one-decimal weights after validating the contract."""

        values = (self.train, self.validation, self.test)
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise TemperaturePipelineError("训练、验证和测试比例必须是数字。")
        numeric_values = tuple(float(value) for value in values)
        if any(not math.isfinite(value) for value in numeric_values):
            raise TemperaturePipelineError("训练、验证和测试比例必须是有限数字。")
        normalized = tuple(round(value, 1) for value in numeric_values)
        if any(
            not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9)
            for value, rounded in zip(numeric_values, normalized, strict=True)
        ):
            raise TemperaturePipelineError("训练、验证和测试比例最多支持一位小数。")
        if any(value <= 0.0 for value in normalized):
            raise TemperaturePipelineError("训练、验证和测试比例都必须大于0。")
        if not math.isclose(sum(normalized), 10.0, rel_tol=0.0, abs_tol=1e-9):
            raise TemperaturePipelineError("训练、验证和测试比例之和必须等于10。")
        return normalized

    @property
    def label(self) -> str:
        """Return a compact user-facing ratio label."""

        values = self._normalized_weights()
        return ":".join(f"{value:.1f}".rstrip("0").rstrip(".") for value in values)

    def fractions(self) -> tuple[float, float, float]:
        """Return normalized fractions that sum to one."""

        train, validation, test = self._normalized_weights()
        return train / 10.0, validation / 10.0, test / 10.0


@dataclass(frozen=True)
class RatioPartitions:
    """Three non-overlapping chronological data partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    ratio: SplitRatio


@dataclass(frozen=True)
class TrainTestPercentage:
    """Chronological train/test split expressed as an integer test percentage."""

    test: int

    def validate(self) -> None:
        """Require the bounded percentage exposed by the training UI."""

        if isinstance(self.test, bool) or not isinstance(self.test, (int, np.integer)):
            raise TemperaturePipelineError("测试集比例必须是整数百分比。")
        if not MIN_TEST_PERCENTAGE <= int(self.test) <= MAX_TEST_PERCENTAGE:
            raise TemperaturePipelineError(
                f"测试集比例必须在 {MIN_TEST_PERCENTAGE}% 到 "
                f"{MAX_TEST_PERCENTAGE}% 之间。"
            )

    @property
    def test_percent(self) -> int:
        """Return the validated test percentage."""

        self.validate()
        return int(self.test)

    @property
    def train_percent(self) -> int:
        """Return the complementary training percentage."""

        return 100 - self.test_percent

    @property
    def label(self) -> str:
        """Return the persisted user-facing percentage label."""

        return f"{self.train_percent}%:{self.test_percent}%"


@dataclass(frozen=True)
class TrainTestPartitions:
    """Two non-overlapping chronological data partitions."""

    train: pd.DataFrame
    test: pd.DataFrame
    percentage: TrainTestPercentage


@dataclass(frozen=True)
class HyperparameterSearchResult:
    """Complete search evidence and selected direction-specific linear parameters."""

    search: pd.DataFrame
    summary: pd.DataFrame
    selected_parameters: dict[str, Any]
    metadata: dict[str, Any]


def chronological_ratio_split(
    frame: pd.DataFrame,
    ratio: SplitRatio,
    *,
    shared_test_holdout: bool = False,
) -> RatioPartitions:
    """Split chronologically, optionally reserving ceil(10%) before allocation.

    Automatic comparison uses the shared holdout so rounding train/validation
    counts cannot change the final test period. Other callers retain the
    three-way largest-remainder allocation.
    """

    ratio.validate()
    required = ["timestamp", *FEATURE_COLUMNS, *TARGET_COLUMNS]
    missing = [column for column in required if column not in frame]
    if missing:
        raise TemperaturePipelineError(f"比例切分数据缺少字段：{missing}")

    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="raise")
    ordered = ordered.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if ordered["timestamp"].duplicated().any():
        raise TemperaturePipelineError("比例切分前发现重复时间主键。")
    matrix = ordered.loc[:, [*FEATURE_COLUMNS, *TARGET_COLUMNS]].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise TemperaturePipelineError("比例切分数据包含缺失或非有限值。")

    fractions = np.asarray(ratio.fractions(), dtype=float)
    remaining_rows = len(ordered)
    if shared_test_holdout:
        if ratio.test != 1.0:
            raise TemperaturePipelineError("自动比例比较必须保留10%的测试集。")
        test_count = (len(ordered) + 9) // 10
        remaining_rows -= test_count
        raw_counts = fractions[:2] / fractions[:2].sum() * remaining_rows
    else:
        raw_counts = fractions * len(ordered)
    counts = np.floor(raw_counts).astype(int)
    remainder = remaining_rows - int(counts.sum())
    allocation_order = np.argsort(-(raw_counts - counts), kind="stable")
    for index in allocation_order[:remainder]:
        counts[index] += 1
    if shared_test_holdout:
        counts = np.append(counts, test_count)
    train_count, validation_count, test_count = (int(value) for value in counts)
    if train_count < 20 or validation_count < 5 or test_count < 5:
        raise TemperaturePipelineError(
            "按当前比例切分后样本太少："
            f"train={train_count}, validation={validation_count}, test={test_count}。"
        )

    validation_end = train_count + validation_count
    train = ordered.iloc[:train_count].copy()
    validation = ordered.iloc[train_count:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()
    if not train["timestamp"].max() < validation["timestamp"].min():
        raise TemperaturePipelineError("训练集与验证集发生时间泄漏。")
    if not validation["timestamp"].max() < test["timestamp"].min():
        raise TemperaturePipelineError("验证集与测试集发生时间泄漏。")
    return RatioPartitions(train, validation, test, ratio)


def chronological_train_test_split(
    frame: pd.DataFrame,
    percentage: TrainTestPercentage,
) -> TrainTestPartitions:
    """Split a timestamp-sorted frame into an older train and newer test period."""

    percentage.validate()
    required = ["timestamp", *FEATURE_COLUMNS, *TARGET_COLUMNS]
    missing = [column for column in required if column not in frame]
    if missing:
        raise TemperaturePipelineError(f"训练测试切分数据缺少字段：{missing}")

    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="raise")
    ordered = ordered.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if ordered["timestamp"].duplicated().any():
        raise TemperaturePipelineError("训练测试切分前发现重复时间主键。")
    matrix = ordered.loc[:, [*FEATURE_COLUMNS, *TARGET_COLUMNS]].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise TemperaturePipelineError("训练测试切分数据包含缺失或非有限值。")

    test_count = int(math.ceil(len(ordered) * percentage.test_percent / 100.0))
    train_count = len(ordered) - test_count
    if train_count < 20 or test_count < 5:
        raise TemperaturePipelineError(
            "按当前测试比例切分后样本太少："
            f"train={train_count}, test={test_count}。"
        )

    train = ordered.iloc[:train_count].copy()
    test = ordered.iloc[train_count:].copy()
    if not train["timestamp"].max() < test["timestamp"].min():
        raise TemperaturePipelineError("训练集与测试集发生时间泄漏。")
    return TrainTestPartitions(train, test, percentage)


class RatioTrainingWorkflow:
    """Search and evaluate a model with a user-selected chronological ratio."""

    def __init__(self, project_root: Path, *, registry: Any | None = None) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.config = WorkflowConfig.load(self.project_root)
        self.registry = registry or get_model_registry(self.project_root)

    def preview(
        self,
        frame: pd.DataFrame,
        ratio: SplitRatio,
        *,
        shared_test_holdout: bool = False,
    ) -> dict[str, Any]:
        """Return counts and date boundaries without fitting or writing artifacts."""

        return self._split_metadata(
            chronological_ratio_split(frame, ratio, shared_test_holdout=shared_test_holdout)
        )

    def preview_train_test(
        self,
        frame: pd.DataFrame,
        percentage: TrainTestPercentage,
    ) -> dict[str, Any]:
        """Return two-way counts and date boundaries without fitting or writing."""

        return self._train_test_metadata(
            chronological_train_test_split(frame, percentage)
        )

    def train(
        self,
        frame: pd.DataFrame,
        ratio: SplitRatio,
        database_sha256: str,
        database_rows: int,
        excluded_missing_targets: int,
        fixed_parameters: dict[str, Any] | None = None,
        optimization_trials: int | None = None,
        *,
        shared_test_holdout: bool = False,
        direction_models: dict[str, str] | None = None,
        feature_context: pd.DataFrame | None = None,
        fusion_search: dict[str, Any] | None = None,
        tabicl_options: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        """Train one persistent draft while leaving the active model unchanged."""

        if optimization_trials is None:
            optimization_trials = self.config.optimizationTrials
        optimization_trials = validate_optimization_trials(optimization_trials)
        partitions = chronological_ratio_split(
            frame, ratio, shared_test_holdout=shared_test_holdout
        )
        models = normalize_direction_models(direction_models)
        searched_parameters = self._tabicl_parameters(models, fixed_parameters, tabicl_options)
        if models['crossline'] == nine.MODEL_NAME:
            if fixed_parameters is not None:
                raise TemperaturePipelineError('九档自动比例请先搜索当前数据，不使用历史参数。')
            if feature_context is None or not isinstance(fusion_search, dict):
                raise TemperaturePipelineError('请先搜索当前数据的九档融合比例。')
            strategy = 'train_test' if isinstance(partitions, TrainTestPartitions) else 'ratio'
            label = percentage.label if strategy == 'train_test' else ratio.label
            expected = weight_search.fingerprint(frame, feature_context, database_sha256,
                                                self._cleaning_policy_sha256(), strategy, label)
            if fusion_search.get('fingerprint') != expected:
                raise TemperaturePipelineError('数据、清洗配置或划分已变化，请重新搜索融合比例。')
            weights = fusion_search.get('weights')
            if not isinstance(weights, list) or len(weights) != 3:
                raise TemperaturePipelineError('搜索比例无效，请重新搜索。')
            searched_parameters = nine.normalize_parameters({**nine.FROZEN_PARAMETERS,
                'negative_weight': weights[0], 'middle_weight': weights[1], 'near_zero_weight': weights[2]})
        folds = self._rolling_folds(partitions.train)
        if fixed_parameters is None:
            search_result = self._optimize_model_parameters(
                folds, optimization_trials=optimization_trials, direction_models=models,
                searched_crossline=searched_parameters,
                **({'feature_context': feature_context} if models['crossline'] in COMPACT_RECIPES or (models['crossline'] == probability.MODEL_NAME and feature_context is not None) else {}),
            )
            parameter_mode = "automatic_search"
        else:
            parameters = self._validate_fixed_parameters(fixed_parameters)
            if direction_models_from_parameters(parameters) != models:
                raise TemperaturePipelineError("历史参数的算法组合与本次选择不一致。")
            search_result = self._evaluate_fixed_parameters(folds, parameters,
                **({'feature_context': feature_context} if models['crossline'] in COMPACT_RECIPES or (models['crossline'] == probability.MODEL_NAME and feature_context is not None) else {}))
            parameter_mode = "saved_version_parameters"
        search = search_result.search
        summary = search_result.summary
        selected_parameters = search_result.selected_parameters

        context_kwargs = {'context_frame': frame} if models['crossline'] == probability.MODEL_NAME else {}
        if models['crossline'] in (probability.MODEL_NAME, *COMPACT_RECIPES):
            context_kwargs = {'context_frame': feature_context if feature_context is not None else frame}
        validation_model = fit_model(partitions.train, selected_parameters, **context_kwargs)
        validation_prediction = predict_frame(
            validation_model,
            partitions.validation,
            **context_kwargs,
        )
        validation_metrics = metrics_table(
            partitions.validation,
            validation_prediction,
            split="validation",
            modelName=selected_parameters["model"],
        )
        validation_baselines = baseline_metrics(
            partitions.train,
            partitions.validation,
            split="validation",
        )
        train_and_validation = pd.concat(
            [partitions.train, partitions.validation],
            ignore_index=True,
        ).sort_values("timestamp", kind="stable")
        final_model = fit_model(train_and_validation, selected_parameters, **context_kwargs)
        test_prediction = predict_frame(
            final_model,
            partitions.test,
            **context_kwargs,
        )
        test_metrics = metrics_table(
            partitions.test,
            test_prediction,
            split="test",
            modelName=selected_parameters["model"],
        )
        test_baselines = baseline_metrics(
            train_and_validation,
            partitions.test,
            split="test",
        )
        split_metadata = self._split_metadata(partitions)
        metadata = {
            "schema_version": 7,
            "gateway": GATEWAY,
            "dataset_version": CSV_DATABASE_DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "feature_columns": list(FEATURE_COLUMNS),
            "direction_feature_columns": {
                "inline": list(FEATURE_COLUMNS),
                "crossline": list(FEATURE_COLUMNS),
            },
            "direction_contracts": {
                direction: f"absolute_value_{model}" for direction, model in models.items()
            },
            "target_columns": list(TARGET_COLUMNS),
            "training_source": "csv_database",
            "source_sha256": database_sha256,
            "cleaning_policy_sha256": self._cleaning_policy_sha256(),
            "model": selected_parameters["model"],
            "parameters": selected_parameters,
            "parameter_mode": parameter_mode,
            **({"fusion_weight_search": fusion_search} if searched_parameters is not None else {}),
            "hyperparameter_optimization": search_result.metadata,
            "raw_input_equations": extract_linear_equations(final_model),
            "feature_importances": extract_feature_importances(final_model),
            "derived_feature_columns": extract_derived_feature_columns(final_model),
            "status": "TRAINED_CANDIDATE",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": split_metadata,
            "sample_counts": {
                "database_rows": database_rows,
                "eligible_rows": len(frame),
                "excluded_missing_targets": excluded_missing_targets,
                "train": len(partitions.train),
                "validation": len(partitions.validation),
                "test": len(partitions.test),
                "final_fit": len(train_and_validation),
                "validation_crossline_evaluable": len(partitions.validation),
                "test_crossline_evaluable": len(partitions.test),
            },
            "evaluation": {
                "validation": self._metric_snapshot(validation_metrics),
                "test": self._metric_snapshot(test_metrics),
            },
            "limitation": (
                "同一传感器数据库内的时间留出测试。"
                "如果反复根据测试结果更换比例，测试集将不再是一次性盲测。"
            ),
        }
        run_id = self._save_draft(
            final_model,
            metadata,
            search,
            summary,
            test_metrics,
            test_baselines,
            partitions.test,
            test_prediction,
            validation_metrics=validation_metrics,
            validation_baselines=validation_baselines,
        )
        return {**metadata, "run_id": run_id}

    def train_train_test(
        self,
        frame: pd.DataFrame,
        percentage: TrainTestPercentage,
        database_sha256: str,
        database_rows: int,
        excluded_missing_targets: int,
        fixed_parameters: dict[str, Any] | None = None,
        optimization_trials: int | None = None,
        *,
        direction_models: dict[str, str] | None = None,
        feature_context: pd.DataFrame | None = None,
        fusion_search: dict[str, Any] | None = None,
        tabicl_options: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        """Train on the older period and evaluate once on the newest holdout."""

        if optimization_trials is None:
            optimization_trials = self.config.optimizationTrials
        optimization_trials = validate_optimization_trials(optimization_trials)
        partitions = chronological_train_test_split(frame, percentage)
        models = normalize_direction_models(direction_models)
        searched_parameters = self._tabicl_parameters(models, fixed_parameters, tabicl_options)
        if models['crossline'] == nine.MODEL_NAME:
            if fixed_parameters is not None:
                raise TemperaturePipelineError('九档自动比例请先搜索当前数据，不使用历史参数。')
            if feature_context is None or not isinstance(fusion_search, dict):
                raise TemperaturePipelineError('请先搜索当前数据的九档融合比例。')
            strategy = 'train_test' if isinstance(partitions, TrainTestPartitions) else 'ratio'
            label = percentage.label if strategy == 'train_test' else ratio.label
            expected = weight_search.fingerprint(frame, feature_context, database_sha256,
                                                self._cleaning_policy_sha256(), strategy, label)
            if fusion_search.get('fingerprint') != expected:
                raise TemperaturePipelineError('数据、清洗配置或划分已变化，请重新搜索融合比例。')
            weights = fusion_search.get('weights')
            if not isinstance(weights, list) or len(weights) != 3:
                raise TemperaturePipelineError('搜索比例无效，请重新搜索。')
            searched_parameters = nine.normalize_parameters({**nine.FROZEN_PARAMETERS,
                'negative_weight': weights[0], 'middle_weight': weights[1], 'near_zero_weight': weights[2]})
        folds = self._rolling_folds(partitions.train)
        if fixed_parameters is None:
            search_result = self._optimize_model_parameters(
                folds,
                optimization_trials=optimization_trials,
                direction_models=models,
                searched_crossline=searched_parameters,
                **({'feature_context': feature_context} if models['crossline'] in COMPACT_RECIPES or (models['crossline'] == probability.MODEL_NAME and feature_context is not None) else {}),
            )
            parameter_mode = "automatic_search"
        else:
            parameters = self._validate_fixed_parameters(fixed_parameters)
            if direction_models_from_parameters(parameters) != models:
                raise TemperaturePipelineError("历史参数的算法组合与本次选择不一致。")
            search_result = self._evaluate_fixed_parameters(folds, parameters,
                **({'feature_context': feature_context} if models['crossline'] in COMPACT_RECIPES or (models['crossline'] == probability.MODEL_NAME and feature_context is not None) else {}))
            parameter_mode = "saved_version_parameters"

        selected_parameters = search_result.selected_parameters
        context_kwargs = {'context_frame': frame} if models['crossline'] == probability.MODEL_NAME else {}
        if models['crossline'] in (probability.MODEL_NAME, *COMPACT_RECIPES):
            context_kwargs = {'context_frame': feature_context if feature_context is not None else frame}
        final_model = fit_model(partitions.train, selected_parameters, **context_kwargs)
        test_prediction = predict_frame(final_model, partitions.test, **context_kwargs)
        test_metrics = metrics_table(
            partitions.test,
            test_prediction,
            split="test",
            modelName=selected_parameters["model"],
        )
        test_baselines = baseline_metrics(
            partitions.train,
            partitions.test,
            split="test",
        )
        split_metadata = self._train_test_metadata(partitions)
        metadata = {
            "schema_version": 8,
            "gateway": GATEWAY,
            "dataset_version": CSV_DATABASE_DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "feature_columns": list(FEATURE_COLUMNS),
            "direction_feature_columns": {
                "inline": list(FEATURE_COLUMNS),
                "crossline": list(FEATURE_COLUMNS),
            },
            "direction_contracts": {
                direction: f"absolute_value_{model}" for direction, model in models.items()
            },
            "target_columns": list(TARGET_COLUMNS),
            "training_source": "csv_database",
            "source_sha256": database_sha256,
            "cleaning_policy_sha256": self._cleaning_policy_sha256(),
            "model": selected_parameters["model"],
            "parameters": selected_parameters,
            "parameter_mode": parameter_mode,
            **({"fusion_weight_search": fusion_search} if searched_parameters is not None else {}),
            "hyperparameter_optimization": search_result.metadata,
            "raw_input_equations": extract_linear_equations(final_model),
            "feature_importances": extract_feature_importances(final_model),
            "derived_feature_columns": extract_derived_feature_columns(final_model),
            "status": "TRAINED_CANDIDATE",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": split_metadata,
            "sample_counts": {
                "database_rows": database_rows,
                "eligible_rows": len(frame),
                "excluded_missing_targets": excluded_missing_targets,
                "train": len(partitions.train),
                "test": len(partitions.test),
                "final_fit": len(partitions.train),
                "test_crossline_evaluable": len(partitions.test),
            },
            "evaluation": {
                "test": self._metric_snapshot(test_metrics),
            },
            "limitation": (
                "同一传感器数据库内的时间留出测试。"
                "超参数只在训练段内部进行三折前向验证；"
                "如果反复根据测试结果更换比例，测试集将不再是一次性盲测。"
            ),
        }
        run_id = self._save_draft(
            final_model,
            metadata,
            search_result.search,
            search_result.summary,
            test_metrics,
            test_baselines,
            partitions.test,
            test_prediction,
        )
        return {**metadata, "run_id": run_id}

    def train_multiple_ratios(
        self,
        frame: pd.DataFrame,
        *,
        ratio_count: int,
        database_sha256: str,
        database_rows: int,
        excluded_missing_targets: int,
        fixed_parameters: dict[str, Any] | None = None,
        optimization_trials: int | None = None,
        maximum_train_weight: float = 8.0,
        direction_models: dict[str, str] | None = None,
        feature_context: pd.DataFrame | None = None,
        tabicl_options: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        """Train several ratio candidates and retain only the validation winner.

        The final 10% test holdout is identical for every candidate. Selection
        uses validation crossline mean absolute relative error only, so test
        rows never choose a training ratio.
        """

        self._tabicl_parameters(normalize_direction_models(direction_models), fixed_parameters, tabicl_options)
        ratios = automatic_split_ratios(
            ratio_count, maximum_train_weight=maximum_train_weight
        )
        completed: list[dict[str, Any]] = []
        generated_run_ids: list[str] = []
        try:
            for ratio in ratios:
                metadata = self.train(
                    frame,
                    ratio,
                    database_sha256,
                    database_rows,
                    excluded_missing_targets,
                    fixed_parameters=fixed_parameters,
                    **({"tabicl_options": tabicl_options} if tabicl_options is not None else {}),
                    optimization_trials=optimization_trials,
                    shared_test_holdout=True,
                    **({'feature_context': feature_context} if feature_context is not None else {}),
                    **({"direction_models": direction_models} if direction_models is not None else {}),
                )
                generated_run_ids.append(str(metadata["run_id"]))
                completed.append({"metadata": metadata})
                completed[-1]["ranking"] = multi_ratio_ranking_row(metadata)

            scorable = [
                candidate
                for candidate in completed
                if self._has_crossline_relative_score(candidate["ranking"])
            ]
            if not scorable:
                raise TemperaturePipelineError(
                    "所有比例的验证集横线真值均为0，"
                    "无法计算平均绝对相对误差并选择候选模型。"
                )
        except Exception:
            for run_id in generated_run_ids:
                try:
                    self.registry.delete_draft(run_id)
                except Exception:
                    pass
            raise

        ordered = sorted(
            completed,
            key=lambda item: (
                not self._has_crossline_relative_score(item["ranking"]),
                self._crossline_relative_score(item["ranking"]),
                (0. if normalize_direction_models(direction_models)["crossline"] in COMPACT_RECIPES
                 else float(item["ranking"]["validation_crossline_mae"])),
                -int(item["ranking"]["validation_sample_count"]),
                str(item["ranking"]["ratio"]),
            ),
        )
        winner = ordered[0]
        winner_run_id = str(winner["metadata"]["run_id"])
        rankings: list[dict[str, float | str | int | bool]] = []
        for rank, candidate in enumerate(ordered, start=1):
            candidate_run_id = str(candidate["metadata"]["run_id"])
            rankings.append(
                {
                    "rank": rank,
                    **candidate["ranking"],
                    "selected": candidate_run_id == winner_run_id,
                }
            )
            if candidate_run_id != winner_run_id:
                self.registry.delete_draft(candidate_run_id)
        return {
            "winner": winner["metadata"],
            "rankings": rankings,
            "ratios": tuple(ratio.label for ratio in ratios),
            "selection_rule": (
                "validation_crossline_mean_abs_relative_error_then_"
                "crossline_mae_then_validation_sample_count"
            ),
        }

    @staticmethod
    def _has_crossline_relative_score(ranking: dict[str, Any]) -> bool:
        """Return whether a candidate has a usable validation ranking score."""

        score = float(
            ranking["validation_crossline_mean_abs_relative_error_percent"]
        )
        return (
            int(ranking["validation_crossline_relative_error_valid_count"]) > 0
            and math.isfinite(score)
        )

    @staticmethod
    def _crossline_relative_score(ranking: dict[str, Any]) -> float:
        """Return a sortable score, placing unscorable candidates last."""

        if not RatioTrainingWorkflow._has_crossline_relative_score(ranking):
            return float("inf")
        return float(
            ranking["validation_crossline_mean_abs_relative_error_percent"]
        )

    def search_regional_weights(
        self, frame: pd.DataFrame, feature_context: pd.DataFrame, database_sha256: str,
        strategy: str, split: SplitRatio | TrainTestPercentage,
        *, progress: Any = None, cancelled: Any = None,
    ) -> dict[str, Any]:
        """Fit purged fold branches and select nine-level weights without test labels."""
        if strategy == 'ratio' and isinstance(split, SplitRatio):
            partitions = chronological_ratio_split(frame, split)
        elif strategy == 'train_test' and isinstance(split, TrainTestPercentage):
            partitions = chronological_train_test_split(frame, split)
        else:
            raise TemperaturePipelineError('九档比例搜索仅支持单一划分或按测试比例训练。')
        binding = weight_search.fingerprint(frame, feature_context, database_sha256,
                                            self._cleaning_policy_sha256(), strategy, split.label)
        folds = self._rolling_folds(partitions.train)
        collected = []
        for index, (_, train, validation) in enumerate(folds):
            if cancelled and cancelled():
                raise weight_search.SearchCancelled('已取消比例搜索。')
            if progress: progress('训练验证窗口', index, 3)
            cutoff = pd.to_datetime(validation.timestamp).min()-pd.Timedelta(hours=24)
            fit = train.loc[pd.to_datetime(train.timestamp) < cutoff].copy()
            if len(fit) < 400:
                raise TemperaturePipelineError('九档融合在24小时净化后至少需要400条训练记录。')
            estimator = fit_direction_model(fit, 'crossline', nine.FROZEN_PARAMETERS,
                                            context_frame=feature_context)
            features = nine.features_for(validation, feature_context)
            a, b = estimator.named_steps['regressor']._branch_probabilities(features)
            collected.append((a, b, validation['tilt2_crossline'].to_numpy(dtype=float)))
        if progress: progress('搜索融合比例', 0, 101**3)
        result = weight_search.search_probabilities(collected,
            progress=(lambda done,total: progress('搜索融合比例',done,total)) if progress else None,
            cancelled=cancelled)
        result.update(fingerprint=binding, source_sha256=database_sha256,
                      cleaning_policy_sha256=self._cleaning_policy_sha256(), strategy=strategy,
                      split_label=split.label, recipe=nine.recipe_metadata())
        return result

    def _rolling_folds(
        self,
        train_frame: pd.DataFrame,
    ) -> tuple[tuple[str, pd.DataFrame, pd.DataFrame], ...]:
        splitter = TimeSeriesSplit(n_splits=3)
        folds = []
        for fold_index, (train_index, validation_index) in enumerate(
            splitter.split(train_frame),
            start=1,
        ):
            train = train_frame.iloc[train_index].copy()
            validation = train_frame.iloc[validation_index].copy()
            if not train["timestamp"].max() < validation["timestamp"].min():
                raise TemperaturePipelineError(f"滚动折 {fold_index} 存在时间泄漏。")
            folds.append((f"ratio_forward_fold_{fold_index}", train, validation))
        return tuple(folds)

    @staticmethod
    def _tabicl_parameters(
        models: Mapping[str, str], fixed_parameters: Mapping[str, Any] | None,
        settings: Mapping[str, object] | None,
    ) -> dict[str, Any] | None:
        """Validate a single manually configured candidate without overriding saved parameters."""
        if settings is None:
            return None
        if fixed_parameters is not None:
            raise TemperaturePipelineError('已保存版本参数不能与手动TabICLv2设置同时使用。')
        recipe = {tabicl_single.MODEL_NAME: tabicl_single, tabicl_16.MODEL_NAME: tabicl_16}.get(models['crossline'])
        if recipe is None:
            raise TemperaturePipelineError('上下文比例和NLL校准仅支持独立TabICLv2模型。')
        try:
            return recipe.normalize_parameters({**recipe.FROZEN_PARAMETERS, 'recipe_version': 2,
                                                **normalize_options(settings)})
        except ValueError as error:
            raise TemperaturePipelineError(str(error)) from error

    def _optimize_model_parameters(
        self,
        folds: tuple[tuple[str, pd.DataFrame, pd.DataFrame], ...],
        optimization_trials: int,
        direction_models: dict[str, str] | None = None,
        feature_context: pd.DataFrame | None = None,
        searched_crossline: dict[str, Any] | None = None,
    ) -> HyperparameterSearchResult:
        """Run one deterministic MAE study for each target direction."""

        models = normalize_direction_models(direction_models)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        search_frames: list[pd.DataFrame] = []
        summary_frames: list[pd.DataFrame] = []
        selected_directions: dict[str, dict[str, float | int | str]] = {}
        direction_metadata: dict[str, dict[str, Any]] = {}
        for direction_index, direction in enumerate(DIRECTION_TARGETS):
            seed = self.config.optimizationSeed + direction_index
            if models[direction] == tabpfn_v2.MODEL_NAME:
                parameters = dict(tabpfn_v2.FROZEN_PARAMETERS)
                result = self._evaluate_direction_candidate(
                    folds, direction=direction, candidate_id=1, trial_number=0, parameters=parameters,
                )
                objective = float(np.mean([row["mae"] for row in result]))
                for row in result:
                    row.update(trial_state="COMPLETE", objective_value=objective, failure_reason="")
                search = pd.DataFrame(result)
                summary = self._summarize_search(search)
                metadata = {"model": tabpfn_v2.MODEL_NAME, "engine": "fixed_tabpfn_v2",
                            "trials_requested": 1, "trials_completed": 1, "trials_failed": 0,
                            "seed": 42, "best_trial_number": 0, "best_objective_value": objective,
                            "objective": "mean_mae_degrees"}
            elif models[direction] in (probability.MODEL_NAME, *COMPACT_RECIPES):
                recipe = COMPACT_RECIPES.get(models[direction], probability)
                parameters = (searched_crossline if models[direction] == nine.MODEL_NAME or
                              (models[direction] in (tabicl_single.MODEL_NAME, tabicl_16.MODEL_NAME) and searched_crossline is not None)
                              else dict(recipe.FROZEN_PARAMETERS))
                if parameters is None:
                    raise TemperaturePipelineError("请先搜索九档融合比例。")
                result = self._evaluate_direction_candidate(folds, direction=direction, candidate_id=1,
                    trial_number=0, parameters=parameters,
                    **({'feature_context': feature_context} if models[direction] in COMPACT_RECIPES or (models[direction] == probability.MODEL_NAME and feature_context is not None) else {}))
                if models[direction] == nine.MODEL_NAME:
                    eligible = [r for r in result if r['relative_error_valid_count'] > 0]
                    if not eligible:
                        raise TemperaturePipelineError('验证数据没有可计算相对误差的真值。')
                    objective = float(np.average([r['mean_abs_relative_error_percent'] for r in eligible],
                                                 weights=[r['relative_error_valid_count'] for r in eligible]))
                else:
                    objective = float(np.average([r['mean_abs_relative_error_percent'] for r in result],
                        weights=[r['validation_count'] for r in result] if models[direction] in COMPACT_RECIPES else None))
                for r in result:
                    r.update(trial_state='COMPLETE', objective_value=objective, failure_reason='')
                search = pd.DataFrame(result); summary = self._summarize_search(search)
                metadata = {'model': models[direction], 'engine': ('configured_tabicl_recipe' if parameters.get('recipe_version') == 2 and models[direction] in (tabicl_single.MODEL_NAME, tabicl_16.MODEL_NAME) else 'frozen_probability_recipe'),
                            'trials_requested': 1, 'trials_completed': 1, 'trials_failed': 0,
                            'seed': 20261022, 'best_trial_number': 0, 'best_objective_value': objective,
                            'objective': 'mean_absolute_relative_error_percent', 'purge_hours': 24}
                audits = self._calibration_fold_audits(result)
                if audits:
                    metadata['calibration_folds'] = audits
            else:
                search, summary, parameters, metadata = self._optimize_direction(
                    folds, direction, seed, optimization_trials, model_name=models[direction],
                )
            search_frames.append(search)
            summary_frames.append(summary)
            selected_directions[direction] = parameters
            direction_metadata[direction] = metadata

        search = pd.concat(search_frames, ignore_index=True)
        summary = pd.concat(summary_frames, ignore_index=True)
        family = models["inline"] if len(set(models.values())) == 1 else "dual_linear"
        selected_parameters = normalize_model_parameters({
            "model": family,
            "directions": selected_directions,
        })
        metadata = {
            "engine": "optuna",
            "sampler": "TPESampler",
            "direction": "minimize",
            "objective": OPTIMIZATION_OBJECTIVE,
            "objective_targets": OPTIMIZATION_TARGETS,
            "jobs": self.config.optimizationJobs,
            "by_direction": direction_metadata,
            "search_space": {
                "epsilon": {
                    "minimum": self.config.huberEpsilonMinimum,
                    "maximum": self.config.huberEpsilonMaximum,
                    "distribution": "uniform",
                },
                "alpha": {
                    "allows_zero": self.config.huberAlphaAllowsZero,
                    "minimum_positive": self.config.huberAlphaMinimumPositive,
                    "maximum": self.config.huberAlphaMaximum,
                    "positive_distribution": "log",
                },
                "max_iter": self.config.huberMaxIter,
            },
        }
        metadata["search_space_by_direction"] = {
            direction: self._search_space(model) for direction, model in models.items()
        }
        if family != FIXED_MODEL_NAME:
            metadata.pop("search_space")
        return HyperparameterSearchResult(
            search,
            summary,
            selected_parameters,
            metadata,
        )

    def _search_space(self, model: str) -> dict[str, Any]:
        """Describe the exact search domain used by one direction."""

        if model == tabpfn_v2.MODEL_NAME:
            return {"policy": "fixed", **tabpfn_v2.FROZEN_PARAMETERS}
        if model == nine.MODEL_NAME:
            return {'policy': 'regional_mare_grid', 'minimum': 0, 'maximum': 100, 'step_percent': 1,
                    'regions': ['negative', 'middle', 'near_zero']}
        if model in COMPACT_RECIPES:
            return {'policy': 'fixed_compact36_recipe', **COMPACT_RECIPES[model].FROZEN_PARAMETERS}
        if model == probability.MODEL_NAME:
            return {'policy': 'fixed_original_recipe', **probability.FROZEN_PARAMETERS}
        if model == "catboost":
            return {
                "iterations": {"minimum": 100, "maximum": 600, "step": 100},
                "depth": {"minimum": 3, "maximum": 8},
                "learning_rate": {"minimum": .01, "maximum": .2, "distribution": "log"},
                "l2_leaf_reg": {"minimum": .1, "maximum": 30., "distribution": "log"},
                "random_seed": self.config.optimizationSeed,
            }
        if model in ("ridge", "polynomial_ridge"):
            space = {"alpha": {"minimum": 1e-6,
                               "maximum": 1e4 if model == "polynomial_ridge" else 1e3,
                               "distribution": "log"}}
            if model == "polynomial_ridge":
                space.update(degree=2, include_bias=False, interaction_only=False)
            return space
        if model == "elasticnet":
            return {
                "alpha": {"minimum": 1e-6, "maximum": 10.0, "distribution": "log"},
                "l1_ratio": {"minimum": 0.05, "maximum": 1.0, "distribution": "uniform"},
                "max_iter": 10000,
            }
        return {
            "epsilon": {"minimum": self.config.huberEpsilonMinimum,
                        "maximum": self.config.huberEpsilonMaximum},
            "alpha": {"minimum_positive": self.config.huberAlphaMinimumPositive,
                      "maximum": self.config.huberAlphaMaximum, "allows_zero": True},
            "max_iter": self.config.huberMaxIter,
        }

    def _optimize_direction(
        self,
        folds: tuple[tuple[str, pd.DataFrame, pd.DataFrame], ...],
        direction: str,
        seed: int,
        optimization_trials: int,
        *,
        model_name: str = FIXED_MODEL_NAME,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str], dict[str, Any]]:
        """Optimize one target direction and preserve every trial result."""

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        search_rows: list[dict[str, Any]] = []

        def objective(trial: optuna.Trial) -> float:
            if model_name == FIXED_MODEL_NAME:
                alpha_is_zero = trial.suggest_categorical("alpha_is_zero", [True, False])
                parameters = {
                    "epsilon": trial.suggest_float(
                        "epsilon", self.config.huberEpsilonMinimum, self.config.huberEpsilonMaximum,
                    ),
                    "alpha": (0.0 if alpha_is_zero else trial.suggest_float(
                        "alpha_positive", self.config.huberAlphaMinimumPositive,
                        self.config.huberAlphaMaximum, log=True,
                    )),
                    "max_iter": self.config.huberMaxIter,
                }
            elif model_name == "ridge":
                parameters = {"alpha": trial.suggest_float("alpha", 1e-6, 1e3, log=True)}
            elif model_name == "polynomial_ridge":
                parameters = {"alpha": trial.suggest_float("alpha", 1e-6, 1e4, log=True),
                              "degree": 2, "include_bias": False, "interaction_only": False}
            elif model_name == "catboost":
                parameters = {
                    "iterations": trial.suggest_int("iterations", 100, 600, step=100),
                    "depth": trial.suggest_int("depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", .01, .2, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", .1, 30., log=True),
                    "random_seed": self.config.optimizationSeed,
                }
            else:
                parameters = {
                    "alpha": trial.suggest_float("alpha", 1e-6, 10.0, log=True),
                    "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 1.0),
                    "max_iter": 10000,
                }
            parameters = {"model": model_name, **parameters}
            trial.set_user_attr("normalized_parameters", parameters)
            try:
                trial_rows = self._evaluate_direction_candidate(
                    folds,
                    direction=direction,
                    candidate_id=trial.number + 1,
                    trial_number=trial.number,
                    parameters=parameters,
                )
            except (TemperaturePipelineError, ValueError, FloatingPointError) as error:
                trial.set_user_attr("failure_reason", str(error))
                raise
            objective_value = float(
                np.mean([float(row["mae"]) for row in trial_rows])
            )
            for row in trial_rows:
                row["trial_state"] = "COMPLETE"
                row["objective_value"] = objective_value
                row["failure_reason"] = ""
            search_rows.extend(trial_rows)
            return objective_value

        study.optimize(
            objective,
            n_trials=optimization_trials,
            n_jobs=self.config.optimizationJobs,
            catch=(TemperaturePipelineError, ValueError, FloatingPointError),
            show_progress_bar=False,
        )
        completed = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        failed = [
            trial for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL
        ]
        for trial in failed:
            search_rows.append(
                self._failed_trial_row(
                    trial,
                    direction,
                    str(trial.user_attrs.get("failure_reason", "试验执行失败")),
                )
            )
        if not completed:
            raise TemperaturePipelineError(
                f"{direction} 方向的 Optuna {optimization_trials} 次 {model_name} "
                "试验全部失败，未生成训练候选。"
            )
        if len({trial.number for trial in completed}) != len(completed):
            raise TemperaturePipelineError(f"{direction} 方向存在重复试验编号。")

        search = pd.DataFrame(search_rows)
        summary = self._summarize_search(search)
        best_trial = study.best_trial
        parameters = dict(best_trial.user_attrs["normalized_parameters"])
        metadata = {
            "model": model_name,
            "trials_requested": optimization_trials,
            "trials_completed": len(completed),
            "trials_failed": len(failed),
            "seed": seed,
            "best_trial_number": best_trial.number,
            "best_objective_value": float(best_trial.value),
        }
        return search, summary, parameters, metadata

    def _evaluate_fixed_parameters(
        self,
        folds: tuple[tuple[str, pd.DataFrame, pd.DataFrame], ...],
        parameters: dict[str, Any],
        feature_context: pd.DataFrame | None = None,
    ) -> HyperparameterSearchResult:
        """Evaluate one saved parameter set without creating an Optuna study."""

        rows: list[dict[str, Any]] = []
        by_direction: dict[str, dict[str, Any]] = {}
        for direction in DIRECTION_TARGETS:
            direction_parameters = {
                **parameters["directions"][direction],
                "model": direction_models_from_parameters(parameters)[direction],
            }
            direction_rows = self._evaluate_direction_candidate(
                folds,
                direction=direction,
                candidate_id=1,
                trial_number=0,
                parameters=direction_parameters,
                **({'feature_context': feature_context} if direction_parameters['model'] in COMPACT_RECIPES or (direction_parameters['model'] == probability.MODEL_NAME and feature_context is not None) else {}),
            )
            objective_value = float(
                np.mean([float(row['mean_abs_relative_error_percent'] if direction_parameters['model'] in (probability.MODEL_NAME, *COMPACT_RECIPES) else row["mae"]) for row in direction_rows])
            )
            if direction_parameters['model'] in COMPACT_RECIPES:
                objective_value = float(np.average([r['mean_abs_relative_error_percent'] for r in direction_rows],
                    weights=[r['validation_count'] for r in direction_rows]))
            for row in direction_rows:
                row["trial_state"] = "COMPLETE"
                row["objective_value"] = objective_value
                row["failure_reason"] = ""
            rows.extend(direction_rows)
            by_direction[direction] = {
                "trials_requested": 1,
                "trials_completed": 1,
                "trials_failed": 0,
                "seed": None,
                "best_trial_number": 0,
                "best_objective_value": objective_value,
            }
        for direction, entry in by_direction.items():
            audits = self._calibration_fold_audits([r for r in rows if r['direction'] == direction])
            if audits:
                entry['calibration_folds'] = audits
        search = pd.DataFrame(rows)
        summary = self._summarize_search(search)
        metadata = {
            "engine": "saved_version_parameters",
            "sampler": None,
            "direction": "minimize",
            "objective": OPTIMIZATION_OBJECTIVE,
            "objective_targets": OPTIMIZATION_TARGETS,
            "jobs": 1,
            "by_direction": by_direction,
        }
        return HyperparameterSearchResult(search, summary, parameters, metadata)

    @staticmethod
    def _calibration_fold_audits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Persist calibration coverage for every time fold, separate from error scoring."""
        return [{'fold': row['fold'], **json.loads(row['calibration_audit_json'])}
                for row in rows if row.get('calibration_audit_json')]

    @staticmethod
    def _evaluate_direction_candidate(
        folds: tuple[tuple[str, pd.DataFrame, pd.DataFrame], ...],
        *,
        direction: str,
        candidate_id: int,
        trial_number: int,
        parameters: dict[str, float | int | str],
        feature_context: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate final absolute predictions on every forward time fold."""

        rows: list[dict[str, Any]] = []
        parameters_json = json.dumps(
            {"model": FIXED_MODEL_NAME, **parameters},
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        target = DIRECTION_TARGETS[direction]
        for fold_name, train, validation in folds:
            probability_model = parameters.get('model') in (probability.MODEL_NAME, *COMPACT_RECIPES)
            recipe = COMPACT_RECIPES.get(parameters.get('model'), probability)
            context = pd.concat([train, validation], ignore_index=True) if probability_model else None
            if feature_context is not None:
                context = feature_context
            if probability_model:
                cutoff = pd.to_datetime(validation.timestamp).min()-pd.Timedelta(hours=24)
                train = train.loc[pd.to_datetime(train.timestamp) < cutoff].copy()
                if len(train) < 300:
                    raise TemperaturePipelineError('横线概率融合在24小时净化后至少需要300条训练记录，请增加数据或训练比例。')
            estimator = fit_direction_model(train, direction, parameters,
                                            **({'context_frame': context} if probability_model else {}))
            evaluation = validation
            features = evaluation.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
            prediction = estimator.predict(recipe.features_for(evaluation, context) if probability_model else features)
            scored_target = OPTIMIZATION_TARGETS[direction]
            metrics = regression_metrics(
                evaluation[target].to_numpy(dtype=float),
                prediction,
                evaluation["timestamp"],
            )
            calibration_columns = {}
            if probability_model:
                calibration = getattr(estimator.named_steps['regressor'], 'training_info_', {}).get('calibration', {})
                if calibration.get('enabled'):
                    calibration_columns['calibration_audit_json'] = json.dumps(calibration, ensure_ascii=False, allow_nan=False)
            rows.append(
                {
                    **calibration_columns,
                    "candidate_id": candidate_id,
                    "trial_number": trial_number,
                    "fold": fold_name,
                    "direction": direction,
                    "model": parameters.get("model", FIXED_MODEL_NAME),
                    "epsilon": parameters.get("epsilon", np.nan),
                    "alpha": parameters.get("alpha", np.nan),
                    "l1_ratio": parameters.get("l1_ratio", np.nan),
                    "max_iter": parameters.get("max_iter", np.nan),
                    **{name: parameters.get(name, np.nan) for name in (
                        "iterations", "depth", "learning_rate", "l2_leaf_reg", "random_seed",
                        "degree", "include_bias", "interaction_only",
                    )},
                    "parameters_json": parameters_json,
                    "train_count": len(train),
                    "validation_count": len(evaluation),
                    "train_end": train["timestamp"].max(),
                    "validation_start": validation["timestamp"].min(),
                    "scored_target": scored_target,
                    **metrics,
                }
            )
        return rows

    @staticmethod
    def _failed_trial_row(
        trial: optuna.trial.FrozenTrial,
        direction: str,
        failure_reason: str,
    ) -> dict[str, Any]:
        """Create one auditable row for a failed Optuna trial."""

        parameters = trial.user_attrs.get("normalized_parameters", {})
        parameters_json = (
            json.dumps(
                parameters,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            if parameters
            else ""
        )
        return {
            "candidate_id": trial.number + 1,
            "trial_number": trial.number,
            "trial_state": "FAIL",
            "fold": "",
            "direction": direction,
            "model": parameters.get("model", FIXED_MODEL_NAME),
            "l1_ratio": parameters.get("l1_ratio", np.nan),
            "epsilon": parameters.get("epsilon", np.nan),
            "alpha": parameters.get("alpha", np.nan),
            "max_iter": parameters.get("max_iter", np.nan),
            **{name: parameters.get(name, np.nan) for name in (
                "iterations", "depth", "learning_rate", "l2_leaf_reg", "random_seed",
                "degree", "include_bias", "interaction_only",
            )},
            "parameters_json": parameters_json,
            "scored_target": OPTIMIZATION_TARGETS.get(direction, ""),
            "objective_value": np.nan,
            "failure_reason": failure_reason[:500],
        }

    @staticmethod
    def _validate_fixed_parameters(
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize saved linear parameters without searching alternatives."""

        try:
            return normalize_model_parameters(parameters)
        except TemperaturePipelineError as error:
            raise TemperaturePipelineError("历史版本的模型参数不完整或无效。") from error

    @staticmethod
    def _summarize_search(search: pd.DataFrame) -> pd.DataFrame:
        complete = search.loc[search["trial_state"] == "COMPLETE"].copy()
        if "l1_ratio" not in complete:
            complete["l1_ratio"] = np.nan
        summary = (
            complete.groupby(
                [
                    "direction",
                    "candidate_id",
                    "trial_number",
                    "model",
                    "epsilon",
                    "alpha",
                    "l1_ratio",
                    "max_iter",
                    "parameters_json",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                objective_value=("objective_value", "first"),
                mean_mae=("mae", "mean"),
                worst_mae=("mae", "max"),
                mean_rmse=("rmse", "mean"),
            )
            .sort_values(
                [
                    "direction",
                    "mean_mae",
                    "worst_mae",
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        summary["rank"] = summary.groupby("direction", sort=False).cumcount() + 1
        return summary

    @staticmethod
    def _metric_snapshot(metrics: pd.DataFrame) -> dict[str, Any]:
        """Store compact validation/test metrics for candidate comparison."""

        directions: dict[str, dict[str, float | int | None]] = {}
        for direction in DIRECTION_TARGETS:
            rows = metrics.loc[metrics["direction"] == direction]
            if len(rows) != 1:
                raise TemperaturePipelineError(
                    f"{direction} 方向缺少唯一的评估指标。"
                )
            row = rows.iloc[0]
            mean_relative_error = float(row["mean_abs_relative_error_percent"])
            directions[direction] = {
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "max_abs_error": float(row["max_abs_error"]),
                "mean_abs_relative_error_percent": (
                    mean_relative_error
                    if math.isfinite(mean_relative_error)
                    else None
                ),
                "relative_error_valid_count": int(
                    row["relative_error_valid_count"]
                ),
                "relative_error_excluded_zero_count": int(
                    row["relative_error_excluded_zero_count"]
                ),
            }
        return {"directions": directions}

    @staticmethod
    def _split_metadata(partitions: RatioPartitions) -> dict[str, Any]:
        frames = {
            "train": partitions.train,
            "validation": partitions.validation,
            "test": partitions.test,
        }
        train_weight, validation_weight, test_weight = (
            partitions.ratio._normalized_weights()
        )
        train_fraction, validation_fraction, test_fraction = partitions.ratio.fractions()
        fractions = {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        }
        return {
            "method": "chronological_ratio",
            "requested_ratio": {
                "label": partitions.ratio.label,
                "train": train_weight,
                "validation": validation_weight,
                "test": test_weight,
            },
            "requested_fractions": fractions,
            "partitions": {
                name: {
                    "rows": len(partition),
                    "timestamp_min": partition["timestamp"].min().isoformat(),
                    "timestamp_max": partition["timestamp"].max().isoformat(),
                }
                for name, partition in frames.items()
            },
        }

    @staticmethod
    def _train_test_metadata(partitions: TrainTestPartitions) -> dict[str, Any]:
        """Describe one percentage-based chronological train/test split."""

        frames = {"train": partitions.train, "test": partitions.test}
        percentage = partitions.percentage
        return {
            "method": "chronological_train_test_percentage",
            "requested_ratio": {
                "label": percentage.label,
                "train_percent": percentage.train_percent,
                "test_percent": percentage.test_percent,
            },
            "requested_fractions": {
                "train": percentage.train_percent / 100.0,
                "test": percentage.test_percent / 100.0,
            },
            "partitions": {
                name: {
                    "rows": len(partition),
                    "timestamp_min": partition["timestamp"].min().isoformat(),
                    "timestamp_max": partition["timestamp"].max().isoformat(),
                }
                for name, partition in frames.items()
            },
        }

    def _cleaning_policy_sha256(self) -> str:
        """Bind every candidate to the exact reviewed cleaning contract."""

        from sensor_database.cleaning import SensorCleaningPipeline

        return SensorCleaningPipeline.for_project(self.project_root).processing_sha256

    def _save_draft(
        self,
        model: FittedDualModel,
        metadata: dict[str, Any],
        search: pd.DataFrame,
        summary: pd.DataFrame,
        test_metrics: pd.DataFrame,
        test_baselines: pd.DataFrame,
        test_frame: pd.DataFrame,
        test_prediction: np.ndarray,
        validation_metrics: pd.DataFrame | None = None,
        validation_baselines: pd.DataFrame | None = None,
    ) -> str:
        """Persist a complete candidate run and return its stable run ID."""

        from sensor_database.cleaning import SensorCleaningPipeline

        pipeline = SensorCleaningPipeline.for_project(self.project_root)
        if direction_models_from_parameters(model.parameters)['crossline'] == probability.MODEL_NAME:
            metadata.update({'schema_version': 9, 'feature_version': probability.FEATURE_VERSION,
                        'direction_feature_columns': {'inline': list(FEATURE_COLUMNS), 'crossline': list(probability.FEATURE_COLUMNS)},
                        'crossline_probability': probability.recipe_metadata()})
            optimization = dict(metadata['hyperparameter_optimization'])
            optimization['objective_by_direction'] = {'inline': OPTIMIZATION_OBJECTIVE,
                'crossline': 'fixed_original_recipe_3_purged_time_folds_MARE'}
            optimization['objective'] = 'inline_MAE_and_fixed_crossline_probability_recipe_MARE'
            metadata['hyperparameter_optimization'] = optimization
            # Keep the caller's result consistent with the actual persisted draft.
            model.featureVersion = probability.FEATURE_VERSION
            model.directionFeatureColumns = {'inline': FEATURE_COLUMNS, 'crossline': probability.FEATURE_COLUMNS}
        if direction_models_from_parameters(model.parameters)['crossline'] in COMPACT_RECIPES:
            recipe = COMPACT_RECIPES[direction_models_from_parameters(model.parameters)['crossline']]
            cross = model.estimators[1].named_steps['regressor']
            metadata.update({'schema_version': 11 if recipe is not compact36 else 10, 'feature_version': recipe.FEATURE_VERSION,
                'direction_feature_columns': {'inline': list(FEATURE_COLUMNS), 'crossline': list(compact36.BASE_FEATURE_COLUMNS)},
                'derived_feature_columns': {'crossline': list(compact36.FEATURE_COLUMNS)},
                'crossline_compact36': compact_recipe_metadata(recipe, model.parameters['directions']['crossline']),
                'compact36_training_info': cross.training_info_, 'library_versions': cross.versions_})
            optimization = dict(metadata['hyperparameter_optimization'])
            optimization['objective_by_direction'] = {'inline': OPTIMIZATION_OBJECTIVE,
                'crossline': ('searched_regional_weights_pooled_MARE' if recipe is nine else 'fixed_compact36_3_purged_time_folds_pooled_MARE')}
            optimization['objective'] = ('inline_MAE_and_searched_regional_crossline_MARE' if recipe is nine
                                         else 'inline_MAE_and_fixed_compact36_crossline_MARE')
            metadata['hyperparameter_optimization'] = optimization
        if pipeline.processing_version in ("final_v1", "final_v2", "final_v3"):
            metadata["processing_version"] = pipeline.processing_version
            metadata["base_cleaning_policy_sha256"] = pipeline.policy.sha256
        metric_frames = []
        if validation_metrics is not None:
            metric_frames.append(validation_metrics)
        if validation_baselines is not None:
            metric_frames.append(validation_baselines)
        metric_frames.extend([test_metrics, test_baselines])
        combined_metrics = pd.concat(metric_frames, ignore_index=True)
        prediction_report = test_frame.loc[
            :, ["timestamp", *FEATURE_COLUMNS, *TARGET_COLUMNS]
        ].copy()
        prediction_report["predicted_inline"] = test_prediction[:, 0]
        prediction_report["predicted_crossline"] = test_prediction[:, 1]
        prediction_report["absolute_error_inline"] = np.abs(
            prediction_report["predicted_inline"] - prediction_report["tilt2_inline"]
        )
        prediction_report["absolute_error_crossline"] = np.abs(
            prediction_report["predicted_crossline"]
            - prediction_report["tilt2_crossline"]
        )
        prediction_report["absolute_relative_error_inline_percent"] = (
            absolute_relative_error_percent(
                prediction_report["tilt2_inline"],
                prediction_report["predicted_inline"],
            )
        )
        prediction_report["absolute_relative_error_crossline_percent"] = (
            absolute_relative_error_percent(
                prediction_report["tilt2_crossline"],
                prediction_report["predicted_crossline"],
            )
        )
        prediction_report = build_detailed_prediction_report(
            prediction_report,
            metadata,
            combined_metrics,
        )
        draft = self.registry.create_training_draft(
            model_payload={"model": model, "metadata": metadata},
            metadata=metadata,
            metrics=combined_metrics,
            parameter_candidates=summary,
            parameter_search=search,
            test_predictions=prediction_report,
        )
        return draft.run_id
