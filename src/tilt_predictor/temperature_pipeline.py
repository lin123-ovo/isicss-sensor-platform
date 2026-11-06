"""Temperature-aware CSG124 cleaning, model selection, and holdout evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import joblib
from catboost import CatBoostError, CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .crossline_residual import (
    CrosslineResidualConfig,
    RESIDUAL_FEATURE_COLUMNS,
    build_residual_features,
)
from . import crossline_tabpfn_v2 as tabpfn_v2
from . import crossline_probability as probability
from . import crossline_compact36 as compact36
from . import crossline_seven as seven
from . import crossline_nine_regional as nine
from . import crossline_tabicl_single as tabicl_single
from . import crossline_tabicl_16 as tabicl_16
from . import crossline_tabm_single as tabm_single
from .crossline_single import SingleBranchRegressor


GATEWAY = "CSG124"
DATASET_VERSION = "isicss_local_v1"
FEATURE_VERSION = "tilt_temperature_v1"
HYBRID_FEATURE_VERSION = "tilt_temperature_residual_v1"
SUPPORTED_FEATURE_VERSIONS = (FEATURE_VERSION, HYBRID_FEATURE_VERSION, probability.FEATURE_VERSION,
                              compact36.FEATURE_VERSION, seven.FEATURE_VERSION, nine.FEATURE_VERSION, tabicl_single.FEATURE_VERSION, tabm_single.FEATURE_VERSION)
COMPACT_RECIPES = {compact36.MODEL_NAME: compact36, seven.MODEL_NAME: seven, nine.MODEL_NAME: nine, tabicl_single.MODEL_NAME: tabicl_single, tabicl_16.MODEL_NAME: tabicl_16, tabm_single.MODEL_NAME: tabm_single}
FIXED_MODEL_NAME = "huber"
LINEAR_MODEL_LABELS = {"huber": "Huber", "ridge": "Ridge", "elasticnet": "ElasticNet"}
MODEL_LABELS = {**LINEAR_MODEL_LABELS, "catboost": "CatBoost", "polynomial_ridge": "二次项 Ridge",
                tabpfn_v2.MODEL_NAME: "TabPFN v2（独立回归）",
                probability.MODEL_NAME: "TabICLv2＋CatBoost（概率选档）",
                compact36.MODEL_NAME: "TabICLv2＋TabM（36维组合）",
                seven.MODEL_NAME: "TabICLv2＋TabM（七档分段融合）",
                nine.MODEL_NAME: "TabICLv2＋TabM（九档自动比例）",
                tabicl_single.MODEL_NAME: "TabICLv2（独立单模型）",
                tabicl_16.MODEL_NAME: "TabICLv2（独立单模型，16成员）",
                tabm_single.MODEL_NAME: "TabM（独立单模型）"}
DUAL_LINEAR_MODEL_NAME = "dual_linear"
FEATURE_COLUMNS = (
    "tilt1_inline_model",
    "tilt1_crossline_model",
    "temperature_celsius",
)
TARGET_COLUMNS = ("tilt2_inline", "tilt2_crossline")
POLYNOMIAL_FEATURE_COLUMNS = (
    *FEATURE_COLUMNS,
    "tilt1_inline_model^2", "tilt1_inline_model tilt1_crossline_model",
    "tilt1_inline_model temperature_celsius", "tilt1_crossline_model^2",
    "tilt1_crossline_model temperature_celsius", "temperature_celsius^2",
)
DIRECTION_TARGETS = {
    "inline": "tilt2_inline",
    "crossline": "tilt2_crossline",
}
EXPECTED_BUCKET_MINUTES = 10
RELATIVE_ERROR_ZERO_TOLERANCE = 1e-12
PREDICTION_DECIMAL_PLACES = 2
PREDICTION_QUANTUM = Decimal("0.01")
PREDICTION_VALUE_COLUMNS = (
    "predicted_inline",
    "predicted_crossline",
)


class TemperaturePipelineError(RuntimeError):
    """Raised when the active data or model violates the approved contract."""


def round_prediction_values(values: Any) -> np.ndarray:
    """Round finite prediction values to the public two-decimal contract.

    Decimal ``ROUND_HALF_UP`` implements ordinary decimal rounding for both
    positive and negative sensor values. Missing values remain missing.
    """

    prediction = np.asarray(values, dtype=float)
    rounded = prediction.copy()
    infinite_mask = np.isinf(prediction)
    if infinite_mask.any():
        raise TemperaturePipelineError("预测结果包含无穷大。")
    flat_prediction = prediction.reshape(-1)
    flat_rounded = rounded.reshape(-1)
    for index in np.flatnonzero(np.isfinite(flat_prediction)):
        flat_rounded[index] = float(
            Decimal(str(float(flat_prediction[index]))).quantize(
                PREDICTION_QUANTUM,
                rounding=ROUND_HALF_UP,
            )
        )
    return rounded


@dataclass(frozen=True)
class WorkflowConfig:
    """Validated data, split, and search configuration."""

    projectRoot: Path
    sourcePath: Path
    sourceSheet: str
    sourceSha256: str
    columns: dict[str, str]
    expectedCounts: dict[str, int]
    selectionStart: pd.Timestamp
    selectionEnd: pd.Timestamp
    validationStart: pd.Timestamp
    validationEnd: pd.Timestamp
    testStart: pd.Timestamp
    testEnd: pd.Timestamp
    optimizer: str
    optimizationTrials: int
    optimizationSeed: int
    optimizationJobs: int
    huberEpsilonMinimum: float
    huberEpsilonMaximum: float
    huberAlphaAllowsZero: bool
    huberAlphaMinimumPositive: float
    huberAlphaMaximum: float
    huberEpsilons: tuple[float, ...]
    huberAlphas: tuple[float, ...]
    huberMaxIter: int

    @classmethod
    def load(cls, projectRoot: Path) -> "WorkflowConfig":
        """Load and validate the two editable JSON configuration files."""

        root = projectRoot.expanduser().resolve()
        dataset = _read_json(root / "configs" / "active_dataset.json")
        search = _read_json(root / "configs" / "hyperparameter_search.json")
        if dataset.get("gateway") != GATEWAY:
            raise TemperaturePipelineError("活动数据必须且只能属于 CSG124。")
        if dataset.get("dataset_version") != DATASET_VERSION:
            raise TemperaturePipelineError("活动数据版本与代码合同不一致。")
        if dataset.get("feature_version") != FEATURE_VERSION:
            raise TemperaturePipelineError("活动特征版本与代码合同不一致。")

        source = _require_mapping(dataset, "source")
        columns = _require_mapping(dataset, "columns")
        expectedCounts = _require_mapping(dataset, "expected_counts")
        split = _require_mapping(dataset, "split")
        optimization = _require_mapping(search, "optimization")
        huber = _require_mapping(search, "huber")
        requiredColumnKeys = {
            "timestamp",
            *FEATURE_COLUMNS,
            *TARGET_COLUMNS,
        }
        if set(columns) != requiredColumnKeys:
            raise TemperaturePipelineError(
                f"columns 必须且只能包含 {sorted(requiredColumnKeys)}。"
            )
        configuredSourcePath = Path(str(source["path"])).expanduser()
        sourcePath = (
            configuredSourcePath
            if configuredSourcePath.is_absolute()
            else root / configuredSourcePath
        ).resolve()
        optimizer = str(optimization.get("engine", ""))
        if optimizer != "optuna_tpe":
            raise TemperaturePipelineError("optimization.engine 必须是 optuna_tpe。")
        jobs = _positive_int(optimization, "jobs")
        if jobs != 1:
            raise TemperaturePipelineError("Optuna 当前只允许单进程 jobs=1。")
        epsilonMinimum = _finite_float(huber, "epsilon_min")
        epsilonMaximum = _finite_float(huber, "epsilon_max")
        alphaMinimum = _finite_float(huber, "alpha_min_positive")
        alphaMaximum = _finite_float(huber, "alpha_max")
        if epsilonMinimum <= 1.0 or epsilonMaximum <= epsilonMinimum:
            raise TemperaturePipelineError("Huber epsilon 范围必须满足 1 < min < max。")
        if alphaMinimum <= 0.0 or alphaMaximum <= alphaMinimum:
            raise TemperaturePipelineError("Huber alpha 范围必须满足 0 < min < max。")
        alphaAllowsZero = huber.get("alpha_allow_zero")
        if not isinstance(alphaAllowsZero, bool):
            raise TemperaturePipelineError("huber.alpha_allow_zero 必须是布尔值。")
        return cls(
            projectRoot=root,
            sourcePath=sourcePath,
            sourceSheet=str(source["sheet"]),
            sourceSha256=str(source["sha256"]),
            columns={str(key): str(value) for key, value in columns.items()},
            expectedCounts={str(key): int(value) for key, value in expectedCounts.items()},
            selectionStart=pd.Timestamp(split["selection_start"]),
            selectionEnd=pd.Timestamp(split["selection_end_exclusive"]),
            validationStart=pd.Timestamp(split["validation_start"]),
            validationEnd=pd.Timestamp(split["validation_end_exclusive"]),
            testStart=pd.Timestamp(split["test_start"]),
            testEnd=pd.Timestamp(split["test_end_exclusive"]),
            optimizer=optimizer,
            optimizationTrials=_positive_int(optimization, "trials"),
            optimizationSeed=_positive_int(optimization, "seed"),
            optimizationJobs=jobs,
            huberEpsilonMinimum=epsilonMinimum,
            huberEpsilonMaximum=epsilonMaximum,
            huberAlphaAllowsZero=alphaAllowsZero,
            huberAlphaMinimumPositive=alphaMinimum,
            huberAlphaMaximum=alphaMaximum,
            huberEpsilons=tuple(np.linspace(epsilonMinimum, epsilonMaximum, 10)),
            huberAlphas=(
                0.0,
                *(float(value) for value in np.geomspace(alphaMinimum, alphaMaximum, 4)),
            ),
            huberMaxIter=_positive_int(huber, "max_iter"),
        )


@dataclass(frozen=True)
class CleanedDataset:
    """Cleaned ten-minute data and its complete audit evidence."""

    frame: pd.DataFrame
    rejectedRows: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class TimePartitions:
    """Chronological model-selection, validation, and test partitions."""

    selection: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass
class FittedDualModel:
    """Two direction regressors plus an optional legacy residual model."""

    name: str
    parameters: dict[str, Any]
    estimators: Any
    featureVersion: str = FEATURE_VERSION
    directionFeatureColumns: Any = None
    crosslinePredictsDelta: bool = False
    crosslineResidualEstimator: Any = None
    crosslineResidualConfig: Any = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return both direction predictions from the three-feature matrix."""

        if direction_models_from_parameters(self.parameters)['crossline'] in (probability.MODEL_NAME, *COMPACT_RECIPES):
            if not isinstance(features, pd.DataFrame):
                raise TemperaturePipelineError("横线概率模型需要时间与历史，请使用predict_frame或带时间的DataFrame。")
            return predict_frame(self, features)
        if self.name in (*MODEL_LABELS, DUAL_LINEAR_MODEL_NAME):
            try:
                result = np.column_stack(
                    [estimator.predict(features) for estimator in self.estimators]
                )
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                if direction_models_from_parameters(self.parameters)["crossline"] != tabpfn_v2.MODEL_NAME:
                    raise
                raise TemperaturePipelineError(f"横线 TabPFN v2 预测失败：{error}") from error
        else:
            raise TemperaturePipelineError(f"不支持的模型类型：{self.name}")
        prediction = np.asarray(result, dtype=float)
        if prediction.ndim != 2 or prediction.shape[1] != len(TARGET_COLUMNS):
            raise TemperaturePipelineError(f"预测结果形状错误：{prediction.shape}")
        if not np.isfinite(prediction).all():
            raise TemperaturePipelineError("预测结果包含 NaN 或无穷大。")
        return round_prediction_values(prediction)

    def features_for(self, direction: str) -> tuple[str, ...]:
        """Return the archived feature contract for one direction."""

        contracts = getattr(self, "directionFeatureColumns", None)
        if isinstance(contracts, dict) and direction in contracts:
            return tuple(contracts[direction])
        return FEATURE_COLUMNS

    @property
    def has_crossline_residual(self) -> bool:
        """Return whether this artifact contains a complete residual stage."""

        return (
            getattr(self, "crosslineResidualEstimator", None) is not None
            and getattr(self, "crosslineResidualConfig", None) is not None
        )


@dataclass(frozen=True)
class LoadedModel:
    """Validated archived model and user-facing metadata."""

    model: FittedDualModel
    metadata: dict[str, Any]
    artifactPath: Path


class TemperatureDatasetProcessor:
    """Read the approved workbook without modifying it and build model rows."""

    def __init__(self, config: WorkflowConfig) -> None:
        self._config = config

    def clean(self) -> CleanedDataset:
        """Drop incomplete rows, round to ten minutes, and aggregate duplicates."""

        self._validate_source()
        try:
            raw = pd.read_excel(
                self._config.sourcePath,
                sheet_name=self._config.sourceSheet,
            )
        except (ValueError, OSError) as error:
            raise TemperaturePipelineError(f"无法读取活动 Excel：{error}") from error
        missingColumns = [
            sourceName
            for sourceName in self._config.columns.values()
            if sourceName not in raw.columns
        ]
        if missingColumns:
            raise TemperaturePipelineError(f"活动 Excel 缺少列：{missingColumns}")

        normalized = pd.DataFrame(index=raw.index)
        timestampSource = self._config.columns["timestamp"]
        normalized["timestamp"] = pd.to_datetime(
            raw[timestampSource], errors="coerce"
        )
        for targetName in (*FEATURE_COLUMNS, *TARGET_COLUMNS):
            normalized[targetName] = pd.to_numeric(
                raw[self._config.columns[targetName]], errors="coerce"
            )
        normalized["source_row"] = np.arange(2, len(raw) + 2, dtype=int)

        required = ["timestamp", *FEATURE_COLUMNS, *TARGET_COLUMNS]
        completeMask = normalized.loc[:, required].notna().all(axis=1)
        finiteMask = pd.Series(True, index=normalized.index)
        numericColumns = [*FEATURE_COLUMNS, *TARGET_COLUMNS]
        finiteMask.loc[:] = np.isfinite(
            normalized.loc[:, numericColumns].to_numpy(dtype=float)
        ).all(axis=1)
        acceptedMask = completeMask & finiteMask
        rejected = normalized.loc[~acceptedMask].copy()
        rejected["rejection_reason"] = np.where(
            normalized.loc[~acceptedMask, "timestamp"].isna(),
            "缺少或无法识别采集时间",
            "温度或倾角缺失、非数值或非有限值",
        )

        accepted = normalized.loc[acceptedMask].copy()
        if accepted.empty:
            raise TemperaturePipelineError("没有六个必需字段均有效的数据行。")
        accepted["timestamp"] = accepted["timestamp"].dt.round(
            f"{EXPECTED_BUCKET_MINUTES}min"
        )
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
        gaps = grouped["timestamp"].diff().gt(
            pd.Timedelta(minutes=EXPECTED_BUCKET_MINUTES * 2)
        )
        grouped["segment_id"] = gaps.cumsum().add(1).astype(int)
        grouped["model_sample_usable"] = True
        self._validate_expected_counts(raw, accepted, rejected, grouped)

        ranges = {
            column: {
                "min": float(grouped[column].min()),
                "max": float(grouped[column].max()),
            }
            for column in (*FEATURE_COLUMNS, *TARGET_COLUMNS)
        }
        audit = {
            "schema_version": 1,
            "gateway": GATEWAY,
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "source": {
                "path": str(self._config.sourcePath),
                "sheet": self._config.sourceSheet,
                "sha256": self._config.sourceSha256,
                "source_was_modified": False,
            },
            "counts": {
                "source_rows": int(len(raw)),
                "rejected_rows": int(len(rejected)),
                "accepted_rows": int(len(accepted)),
                "ten_minute_buckets": int(len(grouped)),
                "duplicate_buckets": int(grouped["source_row_count"].gt(1).sum()),
            },
            "timestamp_min": grouped["timestamp"].min().isoformat(),
            "timestamp_max": grouped["timestamp"].max().isoformat(),
            "column_ranges": ranges,
            "temperature_policy": "required_no_imputation_no_zero_fill",
        }
        return CleanedDataset(grouped, rejected, audit)

    def _validate_source(self) -> None:
        if not self._config.sourcePath.is_file():
            raise TemperaturePipelineError(
                f"活动数据文件不存在：{self._config.sourcePath}"
            )
        actualHash = sha256_file(self._config.sourcePath)
        if actualHash != self._config.sourceSha256:
            raise TemperaturePipelineError(
                "活动数据文件 SHA-256 已变化，禁止使用未经重新审核的数据。"
            )

    def _validate_expected_counts(
        self,
        raw: pd.DataFrame,
        accepted: pd.DataFrame,
        rejected: pd.DataFrame,
        grouped: pd.DataFrame,
    ) -> None:
        actual = {
            "source_rows": len(raw),
            "rejected_rows": len(rejected),
            "accepted_rows": len(accepted),
            "ten_minute_buckets": len(grouped),
        }
        if actual != self._config.expectedCounts:
            raise TemperaturePipelineError(
                f"活动数据计数与审核配置不一致：expected={self._config.expectedCounts}, "
                f"actual={actual}"
            )


class TemperatureModelWorkflow:
    """Coordinate cleaning, forward-only search, evaluation, and model archiving."""

    def __init__(self, projectRoot: Path) -> None:
        self.config = WorkflowConfig.load(projectRoot)
        self.processor = TemperatureDatasetProcessor(self.config)

    def clean_and_save(self) -> CleanedDataset:
        """Clean the source workbook and write reproducible CSV/JSON artifacts."""

        cleaned = self.processor.clean()
        interim = self.config.projectRoot / "data" / "interim"
        reports = self.config.projectRoot / "reports" / "metrics"
        interim.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)
        cleaned.frame.to_csv(
            interim / "CSG124_temperature_cleaned.csv",
            index=False,
            encoding="utf-8-sig",
        )
        cleaned.rejectedRows.to_csv(
            interim / "CSG124_temperature_rejected_rows.csv",
            index=False,
            encoding="utf-8-sig",
        )
        write_json(reports / "temperature_cleaning_audit.json", cleaned.audit)
        return cleaned

    def split(self, frame: pd.DataFrame) -> TimePartitions:
        """Create non-overlapping chronological selection, validation, and test sets."""

        timestamp = pd.to_datetime(frame["timestamp"], errors="raise")
        selection = frame.loc[
            timestamp.between(
                self.config.selectionStart,
                self.config.selectionEnd,
                inclusive="left",
            )
        ].copy()
        validation = frame.loc[
            timestamp.between(
                self.config.validationStart,
                self.config.validationEnd,
                inclusive="left",
            )
        ].copy()
        test = frame.loc[
            timestamp.between(
                self.config.testStart,
                self.config.testEnd,
                inclusive="left",
            )
        ].copy()
        if (len(selection), len(validation), len(test)) != (575, 144, 144):
            raise TemperaturePipelineError(
                "时间切分样本数异常，应为 selection=575, validation=144, test=144，"
                f"实际为 {(len(selection), len(validation), len(test))}。"
            )
        if not selection["timestamp"].max() < validation["timestamp"].min():
            raise TemperaturePipelineError("选参集与独立验证集发生时间重叠。")
        if not validation["timestamp"].max() < test["timestamp"].min():
            raise TemperaturePipelineError("独立验证集与最终测试集发生时间重叠。")
        return TimePartitions(selection, validation, test)

    def rolling_folds(
        self, selection: pd.DataFrame
    ) -> tuple[tuple[str, pd.DataFrame, pd.DataFrame], ...]:
        """Build three expanding, forward-only folds from September 14-17."""

        timestamp = pd.to_datetime(selection["timestamp"], errors="raise")
        folds: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
        for day in (15, 16, 17):
            validationStart = pd.Timestamp(year=2026, month=9, day=day)
            validationEnd = validationStart + pd.Timedelta(days=1)
            train = selection.loc[timestamp < validationStart].copy()
            validation = selection.loc[
                timestamp.between(validationStart, validationEnd, inclusive="left")
            ].copy()
            if train.empty or validation.empty:
                raise TemperaturePipelineError(f"滚动折 09-{day} 没有足够数据。")
            if not train["timestamp"].max() < validation["timestamp"].min():
                raise TemperaturePipelineError(f"滚动折 09-{day} 存在时间泄漏。")
            folds.append((f"validate_2026_09_{day:02d}", train, validation))
        return tuple(folds)

    def train(self) -> dict[str, Any]:
        """Run selection, validation, one-time holdout evaluation, and archiving."""

        cleaned = self.clean_and_save()
        partitions = self.split(cleaned.frame)
        candidates = tuple(self._candidate_parameters())
        searchRows: list[dict[str, Any]] = []
        for foldName, train, validation in self.rolling_folds(partitions.selection):
            for candidateIndex, parameters in enumerate(candidates, start=1):
                model = fit_model(train, parameters)
                prediction = predict_frame(model, validation)
                for directionIndex, (direction, target) in enumerate(
                    DIRECTION_TARGETS.items()
                ):
                    metrics = regression_metrics(
                        validation[target].to_numpy(dtype=float),
                        prediction[:, directionIndex],
                        validation["timestamp"],
                    )
                    searchRows.append(
                        {
                            "candidate_id": candidateIndex,
                            "fold": foldName,
                            "direction": direction,
                            "model": parameters["model"],
                            "parameters_json": json.dumps(
                                parameters,
                                ensure_ascii=False,
                                sort_keys=True,
                                allow_nan=False,
                            ),
                            "train_count": len(train),
                            "validation_count": len(validation),
                            "train_end": train["timestamp"].max(),
                            "validation_start": validation["timestamp"].min(),
                            **metrics,
                        }
                    )
        search = pd.DataFrame(searchRows)
        summary = (
            search.groupby(
                ["candidate_id", "model", "parameters_json"], as_index=False
            )
            .agg(
                mean_mae=("mae", "mean"),
                mean_rmse=("rmse", "mean"),
                worst_mae=("mae", "max"),
            )
            .sort_values(
                ["mean_mae", "worst_mae"],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        summary["rank"] = np.arange(1, len(summary) + 1, dtype=int)
        selectedId = int(summary.iloc[0]["candidate_id"])
        selectedParameters = dict(candidates[selectedId - 1])

        validationModel = fit_model(partitions.selection, selectedParameters)
        validationPrediction = predict_frame(validationModel, partitions.validation)
        validationMetrics = metrics_table(
            partitions.validation,
            validationPrediction,
            split="validation",
            modelName=str(selectedParameters["model"]),
        )
        validationBaselines = baseline_metrics(
            partitions.selection,
            partitions.validation,
            split="validation",
        )
        trainAndValidation = pd.concat(
            [partitions.selection, partitions.validation], ignore_index=True
        ).sort_values("timestamp", kind="stable")
        finalModel = fit_model(trainAndValidation, selectedParameters)
        testPrediction = predict_frame(finalModel, partitions.test)
        testMetrics = metrics_table(
            partitions.test,
            testPrediction,
            split="test",
            modelName=str(selectedParameters["model"]),
        )
        testBaselines = baseline_metrics(
            trainAndValidation,
            partitions.test,
            split="test",
        )
        equations = extract_linear_equations(finalModel)
        createdAt = datetime.now(timezone.utc).isoformat()
        metadata = {
            "schema_version": 1,
            "gateway": GATEWAY,
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "feature_columns": list(FEATURE_COLUMNS),
            "target_columns": list(TARGET_COLUMNS),
            "model": selectedParameters["model"],
            "parameters": selectedParameters,
            "raw_input_equations": equations,
            "status": "TRAINED_CANDIDATE",
            "created_at_utc": createdAt,
            "source_sha256": self.config.sourceSha256,
            "sample_counts": {
                "selection": len(partitions.selection),
                "validation": len(partitions.validation),
                "test": len(partitions.test),
                "final_fit": len(trainAndValidation),
            },
            "limitation": (
                "同一文件内约6天的时间留出测试，不代表跨月或跨季节验证。"
            ),
        }
        self._save_training_artifacts(
            finalModel,
            metadata,
            search,
            summary,
            validationMetrics,
            validationBaselines,
            testMetrics,
            testBaselines,
            partitions.test,
            testPrediction,
        )
        return metadata

    def _candidate_parameters(self) -> Iterable[dict[str, float | int | str]]:
        for epsilon in self.config.huberEpsilons:
            for alpha in self.config.huberAlphas:
                yield {
                    "model": FIXED_MODEL_NAME,
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "max_iter": self.config.huberMaxIter,
                }

    def _save_training_artifacts(
        self,
        model: FittedDualModel,
        metadata: dict[str, Any],
        search: pd.DataFrame,
        summary: pd.DataFrame,
        validationMetrics: pd.DataFrame,
        validationBaselines: pd.DataFrame,
        testMetrics: pd.DataFrame,
        testBaselines: pd.DataFrame,
        testFrame: pd.DataFrame,
        testPrediction: np.ndarray,
    ) -> None:
        metricsDir = self.config.projectRoot / "reports" / "metrics"
        predictionsDir = self.config.projectRoot / "reports" / "predictions"
        modelDir = self.config.projectRoot / "models" / GATEWAY
        for directory in (metricsDir, predictionsDir, modelDir):
            directory.mkdir(parents=True, exist_ok=True)
        search.to_csv(
            metricsDir / "temperature_hyperparameter_search.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary.to_csv(
            metricsDir / "temperature_hyperparameter_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        combinedMetrics = pd.concat(
            [validationMetrics, validationBaselines, testMetrics, testBaselines],
            ignore_index=True,
        )
        combinedMetrics.to_csv(
            metricsDir / "temperature_model_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        predictionReport = testFrame.loc[:, ["timestamp", *TARGET_COLUMNS]].copy()
        predictionReport["predicted_inline"] = testPrediction[:, 0]
        predictionReport["predicted_crossline"] = testPrediction[:, 1]
        predictionReport["absolute_error_inline"] = np.abs(
            predictionReport["predicted_inline"] - predictionReport["tilt2_inline"]
        )
        predictionReport["absolute_error_crossline"] = np.abs(
            predictionReport["predicted_crossline"]
            - predictionReport["tilt2_crossline"]
        )
        predictionReport["absolute_relative_error_inline_percent"] = (
            absolute_relative_error_percent(
                predictionReport["tilt2_inline"],
                predictionReport["predicted_inline"],
            )
        )
        predictionReport["absolute_relative_error_crossline_percent"] = (
            absolute_relative_error_percent(
                predictionReport["tilt2_crossline"],
                predictionReport["predicted_crossline"],
            )
        )
        predictionReport.to_csv(
            predictionsDir / "CSG124_temperature_holdout_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        write_json(metricsDir / "temperature_model_decision.json", metadata)
        write_json(
            self.config.projectRoot
            / "configs"
            / "selected_model_parameters.json",
            {
                "schema_version": 1,
                "gateway": GATEWAY,
                "dataset_version": DATASET_VERSION,
                "feature_version": FEATURE_VERSION,
                "model": metadata["model"],
                "parameters": metadata["parameters"],
                "status": metadata["status"],
                "source_sha256": metadata["source_sha256"],
            },
        )
        joblib.dump(
            {"model": model, "metadata": metadata},
            modelDir / "tilt_temperature_v1.joblib",
        )


class TemperatureModelRepository:
    """Load validated model artifacts from the local model registry."""

    def __init__(self, projectRoot: Path) -> None:
        self._projectRoot = projectRoot.expanduser().resolve()

    def load(self) -> LoadedModel:
        """Load the active model without changing registry state."""

        from .model_registry import ModelRegistryError
        from .registry_factory import get_model_registry

        try:
            registry = get_model_registry(self._projectRoot)
            active = registry.bootstrap_current_version()
            if active is None:
                raise ModelRegistryError("尚无兼容清洗合同的活动模型。")
        except (ModelRegistryError, OSError, ValueError) as error:
            raise TemperaturePipelineError(f"活动模型无法加载：{error}") from error
        return self.load_version(active.version_id)

    def load_version(self, versionId: str) -> LoadedModel:
        """Load one saved version for a single prediction without activating it."""

        from .model_registry import ModelRegistryError
        from .registry_factory import get_model_registry

        try:
            registry = get_model_registry(self._projectRoot)
            registry.bootstrap_current_version()
            version = registry.get_version(versionId)
            artifact = registry.version_directory(version.version_id) / "model.joblib"
            metadata_path = registry.version_directory(version.version_id) / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            metadata = {**metadata, "version_id": version.version_id}
            payload = joblib.load(artifact)
        except (ModelRegistryError, OSError, ValueError) as error:
            raise TemperaturePipelineError(
                f"模型版本 {versionId} 无法加载：{error}"
            ) from error
        if not isinstance(payload, dict):
            raise TemperaturePipelineError("模型归档格式无效。")
        model = payload.get("model")
        if not isinstance(metadata, dict) or not isinstance(model, FittedDualModel):
            raise TemperaturePipelineError("模型归档缺少模型或元数据。")
        feature_version = metadata.get("feature_version")
        if feature_version not in SUPPORTED_FEATURE_VERSIONS:
            raise TemperaturePipelineError(
                f"模型特征版本 {feature_version} 已停用，请重新训练。"
            )
        if tuple(metadata.get("feature_columns", ())) != FEATURE_COLUMNS:
            raise TemperaturePipelineError("模型特征列与当前三特征合同不一致。")
        normalized = normalize_model_parameters(metadata.get("parameters", {}))
        if model.name != normalized["model"] or metadata.get("model") != model.name:
            raise TemperaturePipelineError("模型归档与元数据算法不一致。")
        if normalize_model_parameters(model.parameters) != normalized:
            raise TemperaturePipelineError("模型归档与元数据参数不一致。")
        expected_models = direction_models_from_parameters(normalized)
        estimator_types = {"huber": HuberRegressor, "ridge": Ridge, "polynomial_ridge": Ridge,
                           "elasticnet": ElasticNet, "catboost": CatBoostRegressor,
                           tabpfn_v2.MODEL_NAME: tabpfn_v2.TabPFNV2Regressor,
                           probability.MODEL_NAME: probability.CrosslineProbabilityRegressor,
                           compact36.MODEL_NAME: compact36.Compact36Regressor,
                           seven.MODEL_NAME: seven.SevenRegionalRegressor,
                           nine.MODEL_NAME: nine.NineRegionalRegressor,
                           tabicl_single.MODEL_NAME: SingleBranchRegressor,
                           tabicl_16.MODEL_NAME: SingleBranchRegressor,
                           tabm_single.MODEL_NAME: SingleBranchRegressor}
        if len(model.estimators) != 2 or any(
            not isinstance(estimator, Pipeline)
            or not isinstance(estimator.named_steps.get("regressor"), estimator_types[expected_models[direction]])
            for direction, estimator in zip(DIRECTION_TARGETS, model.estimators, strict=True)
        ):
            raise TemperaturePipelineError("模型归档中的实际估计器与算法声明不一致。")
        for direction, estimator in zip(DIRECTION_TARGETS, model.estimators, strict=True):
            if expected_models[direction] == probability.MODEL_NAME:
                validate_probability_artifact(model, metadata)
            elif expected_models[direction] in COMPACT_RECIPES:
                validate_compact36_artifact(model, metadata)
            elif expected_models[direction] == "polynomial_ridge":
                _validate_polynomial_estimator(estimator, normalized["directions"][direction])
                derived = metadata.get("derived_feature_columns")
                if not isinstance(derived, dict) or derived.get(direction) != list(POLYNOMIAL_FEATURE_COLUMNS):
                    raise TemperaturePipelineError("二次项 Ridge 的派生特征元数据不一致。")
            elif "polynomial" in estimator.named_steps:
                raise TemperaturePipelineError("普通模型归档包含未声明的二次项 Ridge 变换。")
        if feature_version == HYBRID_FEATURE_VERSION:
            if not model.has_crossline_residual:
                raise TemperaturePipelineError("横线两阶段模型缺少残差估计器。")
            try:
                residual_config = CrosslineResidualConfig.from_mapping(
                    model.crosslineResidualConfig
                )
                residual_metadata = metadata.get("crossline_residual")
                if not isinstance(residual_metadata, dict):
                    raise ValueError("横线两阶段模型缺少残差元数据。")
                metadata_config = residual_metadata.get("parameters")
                if not isinstance(metadata_config, dict):
                    raise ValueError("横线两阶段模型缺少残差元数据。")
                archived_config = CrosslineResidualConfig.from_mapping(
                    metadata_config
                )
            except ValueError as error:
                raise TemperaturePipelineError(f"横线残差参数无效：{error}") from error
            if residual_config.to_dict() != archived_config.to_dict():
                raise TemperaturePipelineError("横线残差模型与元数据参数不一致。")
        trainingSource = metadata.get("training_source", "approved_excel")
        if feature_version == probability.FEATURE_VERSION and expected_models['crossline'] != probability.MODEL_NAME:
            raise TemperaturePipelineError("概率特征版本与横线算法不一致。")
        if feature_version == compact36.FEATURE_VERSION and expected_models['crossline'] != compact36.MODEL_NAME:
            raise TemperaturePipelineError("36维特征版本与横线算法不一致。")
        if feature_version == nine.FEATURE_VERSION and expected_models['crossline'] != nine.MODEL_NAME:
            raise TemperaturePipelineError('九档分段特征版本与横线算法不一致。')
        if feature_version == seven.FEATURE_VERSION and expected_models['crossline'] != seven.MODEL_NAME:
            raise TemperaturePipelineError("七档分段特征版本与横线算法不一致。")
        sourceSha256 = str(metadata.get("source_sha256", ""))
        if trainingSource == "approved_excel":
            if sourceSha256 != WorkflowConfig.load(self._projectRoot).sourceSha256:
                raise TemperaturePipelineError("模型来源不是当前唯一活动数据。")
        elif trainingSource == "csv_database":
            if re.fullmatch(r"[0-9a-f]{64}", sourceSha256) is None:
                raise TemperaturePipelineError("传感器数据库模型缺少有效的来源指纹。")
        else:
            raise TemperaturePipelineError("模型训练数据来源未受支持。")
        return LoadedModel(model, metadata, artifact)


def fit_model(
    train: pd.DataFrame,
    parameters: Mapping[str, Any],
    *,
    context_frame: pd.DataFrame | None = None,
) -> FittedDualModel:
    """Fit independently selected direction regressors from the same inputs."""

    features = train.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    targets = train.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise TemperaturePipelineError("训练矩阵包含缺失值或非有限值。")
    normalizedParameters = normalize_model_parameters(parameters)
    modelName = str(normalizedParameters["model"])
    if modelName in (*MODEL_LABELS, DUAL_LINEAR_MODEL_NAME):
        models = direction_models_from_parameters(normalizedParameters)
        estimators = [
            fit_direction_model(train, direction, {
                **normalizedParameters["directions"][direction], "model": models[direction]
            }, context_frame=context_frame)
            for direction in DIRECTION_TARGETS
        ]
        version = probability.FEATURE_VERSION if models['crossline'] == probability.MODEL_NAME else FEATURE_VERSION
        contracts = {'inline': FEATURE_COLUMNS, 'crossline': probability.FEATURE_COLUMNS} if version == probability.FEATURE_VERSION else None
        if models['crossline'] in COMPACT_RECIPES:
            version = COMPACT_RECIPES[models['crossline']].FEATURE_VERSION
            contracts = {'inline': FEATURE_COLUMNS, 'crossline': compact36.BASE_FEATURE_COLUMNS}
        return FittedDualModel(modelName, normalizedParameters, tuple(estimators),
                               featureVersion=version, directionFeatureColumns=contracts)
    raise TemperaturePipelineError(
        f"不支持的模型类型：{modelName}"
    )


def available_direction_models(direction: str) -> tuple[str, ...]:
    """Return the algorithms allowed for one target; polynomial Ridge is crossline-only."""

    if direction not in DIRECTION_TARGETS:
        raise TemperaturePipelineError(f"不支持的预测方向：{direction}")
    return tuple(model for model in MODEL_LABELS if direction == "crossline" or model not in ("polynomial_ridge", tabpfn_v2.MODEL_NAME, probability.MODEL_NAME, *COMPACT_RECIPES))


def normalize_direction_models(models: Mapping[str, str] | None = None) -> dict[str, str]:
    """Validate both algorithm choices; omitted choices retain the Huber default."""

    if models is None:
        return dict.fromkeys(DIRECTION_TARGETS, FIXED_MODEL_NAME)
    if not isinstance(models, Mapping) or set(models) != set(DIRECTION_TARGETS):
        raise TemperaturePipelineError("训练模型必须同时包含顺线和横线方向。")
    normalized = {direction: str(models[direction]).lower() for direction in DIRECTION_TARGETS}
    if any(model not in MODEL_LABELS for model in normalized.values()):
        raise TemperaturePipelineError(
            "训练模型选择无效，请从支持的顺线和横线模型中选择。"
        )
    if normalized["inline"] in ("polynomial_ridge", tabpfn_v2.MODEL_NAME, probability.MODEL_NAME, *COMPACT_RECIPES):
        raise TemperaturePipelineError("该模型暂时仅支持横线预测。")
    return normalized


def direction_models_from_parameters(parameters: Mapping[str, Any]) -> dict[str, str]:
    """Read direction algorithms from new or historical Huber documents."""

    if not isinstance(parameters, Mapping):
        raise TemperaturePipelineError("模型参数必须是对象。")
    family = str(parameters.get("model", "")).lower()
    directions = parameters.get("directions", {})
    if not isinstance(directions, Mapping):
        raise TemperaturePipelineError("模型 directions 必须是对象。")
    models = {}
    for direction in DIRECTION_TARGETS:
        document = directions.get(direction, {})
        if not isinstance(document, Mapping):
            raise TemperaturePipelineError(f"{direction} 模型参数必须是对象。")
        models[direction] = str(document.get("model", family)).lower()
    return normalize_direction_models(models)


def model_combination_label(parameters: Mapping[str, Any]) -> str:
    """Return a user-facing label for the actual direction algorithms."""

    models = direction_models_from_parameters(parameters)
    return f"顺线 {MODEL_LABELS[models['inline']]}｜横线 {MODEL_LABELS[models['crossline']]}"


def normalize_model_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Validate direction model parameters while preserving historical documents."""

    models = direction_models_from_parameters(parameters)
    family = str(parameters.get("model", "")).lower()
    expected_family = models["inline"] if len(set(models.values())) == 1 else DUAL_LINEAR_MODEL_NAME
    if family != expected_family:
        raise TemperaturePipelineError("整体模型类型与两个方向的算法不一致。")
    if expected_family == FIXED_MODEL_NAME:
        return normalize_huber_parameters(parameters)
    directions = parameters.get("directions")
    if not isinstance(directions, Mapping) or set(directions) != set(DIRECTION_TARGETS):
        raise TemperaturePipelineError("模型参数必须同时包含顺线和横线方向。")
    normalized = {
        direction: normalize_direction_parameters(direction, {**directions[direction], "model": model})
        for direction, model in models.items()
    }
    return {"model": expected_family, "directions": normalized}


def normalize_direction_parameters(direction: str, document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a single direction without pretending both directions use its algorithm."""

    if not isinstance(document, Mapping):
        raise TemperaturePipelineError("方向模型参数必须是对象。")
    model = str(document.get("model", FIXED_MODEL_NAME)).lower()
    if model not in available_direction_models(direction):
        raise TemperaturePipelineError("方向算法无效，或该模型仅支持横线预测。")
    if model == FIXED_MODEL_NAME:
        values = normalize_huber_parameters({**document, "model": model})["directions"][direction]
    elif model == tabpfn_v2.MODEL_NAME:
        if set(document) != set(tabpfn_v2.FROZEN_PARAMETERS) or type(document.get("max_context_rows")) is not int or document["max_context_rows"] != tabpfn_v2.MAX_CONTEXT_ROWS:
            raise TemperaturePipelineError("TabPFN v2 固定参数无效。")
        return dict(tabpfn_v2.FROZEN_PARAMETERS)
    elif model == "catboost":
        values = normalize_catboost_parameters(document)
    elif model in (probability.MODEL_NAME, *COMPACT_RECIPES):
        try:
            return COMPACT_RECIPES.get(model, probability).normalize_parameters(document)
        except ValueError as error:
            raise TemperaturePipelineError(str(error)) from error
    else:
        try:
            alpha = float(document["alpha"])
            if isinstance(document["alpha"], bool) or not np.isfinite(alpha) or alpha <= 0:
                raise ValueError("alpha 必须是有限正数")
            values = {"alpha": alpha}
            if model == "elasticnet":
                l1_ratio = float(document["l1_ratio"])
                max_iter = document["max_iter"]
                if (isinstance(document["l1_ratio"], bool) or not np.isfinite(l1_ratio)
                        or not 0 < l1_ratio <= 1 or isinstance(max_iter, bool)
                        or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0):
                    raise ValueError("l1_ratio 或 max_iter 无效")
                values.update(l1_ratio=l1_ratio, max_iter=int(max_iter))
            elif model == "polynomial_ridge":
                degree = document.get("degree", 2)
                if (isinstance(degree, bool) or not isinstance(degree, (int, np.integer))
                        or degree != 2 or document.get("include_bias", False) is not False
                        or document.get("interaction_only", False) is not False):
                    raise ValueError("二次项 Ridge 固定二阶、无常数列且包含平方项与交互项")
                values.update(degree=2, include_bias=False, interaction_only=False)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise TemperaturePipelineError(f"{direction} {model} 参数无效：{error}") from error
    return {"model": model, **values}


def _validate_polynomial_estimator(estimator: Pipeline, parameters: Mapping[str, Any]) -> None:
    """Reject falsely declared or incompatible polynomial artifacts before prediction."""

    steps = estimator.named_steps
    polynomial = steps.get("polynomial")
    scaler = steps.get("scaler")
    regressor = steps.get("regressor")
    if (tuple(steps) != ("polynomial", "scaler", "regressor")
            or not isinstance(polynomial, PolynomialFeatures)
            or polynomial.degree != 2 or polynomial.include_bias is not False
            or polynomial.interaction_only is not False
            or getattr(polynomial, "n_features_in_", None) != 3
            or getattr(polynomial, "n_output_features_", None) != 9
            or not isinstance(scaler, StandardScaler)
            or not scaler.with_mean or not scaler.with_std
            or getattr(scaler, "n_features_in_", None) != 9
            or not isinstance(regressor, Ridge)
            or getattr(regressor, "n_features_in_", None) != 9
            or regressor.alpha != parameters["alpha"] or regressor.solver != "svd"
            or not regressor.fit_intercept):
        raise TemperaturePipelineError("二次项 Ridge 归档的变换、标准化器或参数不一致。")
    if tuple(polynomial.get_feature_names_out(FEATURE_COLUMNS)) != POLYNOMIAL_FEATURE_COLUMNS:
        raise TemperaturePipelineError("二次项 Ridge 归档的派生特征顺序不一致。")


def normalize_catboost_parameters(document: Mapping[str, Any]) -> dict[str, float | int]:
    """Validate bounded CPU tree parameters before training or loading artifacts."""

    try:
        values: dict[str, float | int] = {}
        for name, minimum, maximum in (
            ("iterations", 1, 2000), ("depth", 1, 10), ("random_seed", 0, 2**31 - 1)
        ):
            value = document[name]
            if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                    or not minimum <= value <= maximum):
                raise ValueError(f"{name} 必须为 {minimum} 至 {maximum} 的整数")
            values[name] = int(value)
        for name, maximum in (("learning_rate", 1.0), ("l2_leaf_reg", 1000.0)):
            value = document[name]
            if isinstance(value, bool) or not np.isfinite(float(value)):
                raise ValueError(f"{name} 必须是有限数值")
            number = float(value)
            if not (0 < number <= maximum if name == "learning_rate" else 0 <= number <= maximum):
                raise ValueError(f"{name} 超出允许范围")
            values[name] = number
        return values
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise TemperaturePipelineError(f"CatBoost 参数无效：{error}") from error


def normalize_huber_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy shared and direction-specific Huber parameter documents."""

    if str(parameters.get("model", "")).lower() != FIXED_MODEL_NAME:
        raise TemperaturePipelineError("当前策略仅允许 Huber 模型。")
    rawDirections = parameters.get("directions")
    if rawDirections is None:
        directionDocuments: Mapping[str, Any] = {
            direction: parameters for direction in DIRECTION_TARGETS
        }
    elif isinstance(rawDirections, Mapping):
        if set(rawDirections) != set(DIRECTION_TARGETS):
            raise TemperaturePipelineError("Huber 参数必须同时包含顺线和横线方向。")
        directionDocuments = rawDirections
    else:
        raise TemperaturePipelineError("Huber directions 必须是对象。")

    normalizedDirections: dict[str, dict[str, float | int]] = {}
    for direction in DIRECTION_TARGETS:
        document = directionDocuments[direction]
        if not isinstance(document, Mapping):
            raise TemperaturePipelineError(f"{direction} Huber 参数必须是对象。")
        try:
            epsilon = float(document["epsilon"])
            alpha = float(document["alpha"])
            maxIter = int(document["max_iter"])
        except (KeyError, TypeError, ValueError) as error:
            raise TemperaturePipelineError(
                f"{direction} Huber 参数缺失或类型无效。"
            ) from error
        if not np.isfinite(epsilon) or not np.isfinite(alpha):
            raise TemperaturePipelineError(f"{direction} Huber 参数必须为有限值。")
        if epsilon <= 1.0 or alpha < 0.0 or maxIter <= 0:
            raise TemperaturePipelineError(
                f"{direction} Huber 参数必须满足 epsilon>1、alpha>=0、max_iter>0。"
            )
        normalizedDirections[direction] = {
            "epsilon": epsilon,
            "alpha": alpha,
            "max_iter": maxIter,
        }
    return {"model": FIXED_MODEL_NAME, "directions": normalizedDirections}


def fit_direction_model(
    train: pd.DataFrame,
    direction: str,
    parameters: Mapping[str, Any],
    *,
    context_frame: pd.DataFrame | None = None,
) -> Pipeline:
    """Fit one regressor; standardize only when the selected algorithm is linear."""

    if direction not in DIRECTION_TARGETS:
        raise TemperaturePipelineError(f"不支持的预测方向：{direction}")
    model = str(parameters.get("model", FIXED_MODEL_NAME)).lower()
    normalized = normalize_direction_parameters(direction, parameters)
    features = train.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    target = train[DIRECTION_TARGETS[direction]].to_numpy(dtype=float)
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise TemperaturePipelineError("训练矩阵包含缺失值或非有限值。")
    if model in COMPACT_RECIPES:
        try:
            train = train.sort_values('timestamp', kind='stable')
            target = train[DIRECTION_TARGETS[direction]].to_numpy(dtype=float)
            derived = compact36.features_for(train, context_frame)
            regressor = (SingleBranchRegressor('tabicl' if model in (tabicl_single.MODEL_NAME, tabicl_16.MODEL_NAME) else 'tabm',
                                                n_estimators=16 if model == tabicl_16.MODEL_NAME else 4,
                                                **({'context_fraction': normalized['context_fraction'],
                                                    'temperature_calibration': normalized['temperature_calibration'],
                                                    'parameter_version': 2} if normalized.get('recipe_version') == 2
                                                    and model in (tabicl_single.MODEL_NAME, tabicl_16.MODEL_NAME) else {}))
                         if model in (tabicl_single.MODEL_NAME, tabicl_16.MODEL_NAME, tabm_single.MODEL_NAME) else nine.NineRegionalRegressor(normalized['negative_weight'], normalized['middle_weight'], normalized['near_zero_weight'])
                         if model == nine.MODEL_NAME else seven.SevenRegionalRegressor() if model == seven.MODEL_NAME
                         else compact36.Compact36Regressor())
            estimator = Pipeline([('scaler', 'passthrough'), ('regressor', regressor)])
            estimator.fit(derived, target, regressor__timestamps=pd.to_datetime(train.timestamp))
            source = compact36.history_for(train, context_frame)
            estimator.named_steps['regressor'].history_frame_ = probability.input_frame(source)
            estimator.named_steps['regressor'].training_info_['history_rows'] = len(source)
            return estimator
        except ValueError as error:
            raise TemperaturePipelineError(f"36维组合训练失败：{error}") from error
    if model == probability.MODEL_NAME:
        try:
            derived = probability.features_for(train, context_frame)
            estimator = Pipeline([('scaler', 'passthrough'),
                                  ('regressor', probability.CrosslineProbabilityRegressor())])
            estimator.fit(derived, target)
            source = pd.concat([probability.input_frame(p) for p in (context_frame, train) if p is not None])
            source = source.drop_duplicates('timestamp', keep='last')
            end = pd.to_datetime(train.timestamp).max()
            history = source[pd.to_datetime(source.timestamp).between(end-pd.Timedelta(minutes=210), end)].copy()
            estimator.named_steps['regressor'].history_frame_ = probability.input_frame(history)
            return estimator
        except ValueError as error:
            raise TemperaturePipelineError(f"横线概率融合训练失败：{error}") from error
    if model == tabpfn_v2.MODEL_NAME:
        order = pd.to_datetime(train["timestamp"], errors="raise").argsort(kind="stable")
        features, target = features[order], target[order]
        regressor = tabpfn_v2.TabPFNV2Regressor()
    elif model == FIXED_MODEL_NAME:
        regressor = HuberRegressor(
            epsilon=float(normalized["epsilon"]), alpha=float(normalized["alpha"]),
            max_iter=int(normalized["max_iter"]),
        )
    elif model in ("ridge", "polynomial_ridge"):
        regressor = Ridge(alpha=float(normalized["alpha"]), solver="svd")
    elif model == "catboost":
        regressor = CatBoostRegressor(
            **{name: value for name, value in normalized.items() if name != "model"},
            loss_function="RMSE", task_type="CPU", thread_count=1,
            verbose=False, allow_writing_files=False, allow_const_label=True,
        )
    else:
        regressor = ElasticNet(
            alpha=float(normalized["alpha"]), l1_ratio=float(normalized["l1_ratio"]),
            max_iter=int(normalized["max_iter"]), selection="cyclic", tol=1e-4,
        )
    steps = []
    if model == "polynomial_ridge":
        steps.append(("polynomial", PolynomialFeatures(
            degree=2, include_bias=False, interaction_only=False,
        )))
    estimator = Pipeline(
        [*steps,
            ("scaler", "passthrough" if model in ("catboost", tabpfn_v2.MODEL_NAME) else StandardScaler()),
            ("regressor", regressor),
        ]
    )
    try:
        estimator.fit(features, target)
    except CatBoostError as error:
        raise TemperaturePipelineError(f"{direction} CatBoost 训练失败：{error}") from error
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        if model != tabpfn_v2.MODEL_NAME:
            raise
        raise TemperaturePipelineError(f"横线 TabPFN v2 初始化或训练失败：{error}") from error
    return estimator


def predict_compact36_comparison(
    model: FittedDualModel,
    frame: pd.DataFrame,
    *,
    context_frame: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """Predict three archived Compact36 branches using identical input history.

    No model state is changed. The individual branches are components of the
    saved ensemble, rather than newly trained standalone model versions.
    """
    if getattr(model, "featureVersion", None) != compact36.FEATURE_VERSION:
        raise TemperaturePipelineError("分支对照仅支持36维固定比例融合模型。")
    try:
        estimator = model.estimators[1].named_steps["regressor"]
        if type(estimator) is not compact36.Compact36Regressor:
            raise TemperaturePipelineError("模型横线组件不支持分支对照。")
        features = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(features).all():
            raise TemperaturePipelineError("温度或倾角输入缺失，禁止预测。")
        derived = compact36.features_for(frame, context_frame, estimator.history_frame_)
        inline = model.estimators[0].predict(features)
        icl, tabm = estimator._branch_probabilities(derived)
        probabilities = {"tabicl": icl, "tabm": tabm, "ensemble": .75 * icl + .25 * tabm}
        return {
            name: round_prediction_values(np.column_stack([inline, probability.select_level(values)]))
            for name, values in probabilities.items()
        }
    except (KeyError, ValueError) as error:
        raise TemperaturePipelineError(f"横线分支对照预测失败：{error}") from error


def predict_frame(
    model: FittedDualModel,
    frame: pd.DataFrame,
    *,
    context_frame: pd.DataFrame | None = None,
) -> np.ndarray:
    """Predict a frame, applying the archived crossline residual stage when present."""

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise TemperaturePipelineError(f"预测数据缺少三特征列：{missing}")
    features = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise TemperaturePipelineError("温度或倾角输入缺失，禁止预测。")
    if getattr(model, 'featureVersion', None) in (probability.FEATURE_VERSION, compact36.FEATURE_VERSION, seven.FEATURE_VERSION, nine.FEATURE_VERSION, tabicl_single.FEATURE_VERSION, tabm_single.FEATURE_VERSION):
        feature_module = compact36 if model.featureVersion in (compact36.FEATURE_VERSION, seven.FEATURE_VERSION, nine.FEATURE_VERSION, tabicl_single.FEATURE_VERSION, tabm_single.FEATURE_VERSION) else probability
        try:
            cross = model.estimators[1].named_steps['regressor']
            derived = feature_module.features_for(frame, context_frame, cross.history_frame_)
            inline = model.estimators[0].predict(features)
            return round_prediction_values(np.column_stack([inline, model.estimators[1].predict(derived)]))
        except ValueError as error:
            raise TemperaturePipelineError(f"横线概率融合预测失败：{error}") from error
    if not getattr(model, "has_crossline_residual", False):
        return model.predict(features)

    context = (context_frame if context_frame is not None else frame).copy()
    context_missing = [
        column for column in ("timestamp", *FEATURE_COLUMNS) if column not in context
    ]
    if context_missing:
        raise TemperaturePipelineError(f"横线残差上下文缺少字段：{context_missing}")
    context["timestamp"] = pd.to_datetime(context["timestamp"], errors="raise")
    context = context.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if context["timestamp"].duplicated().any():
        raise TemperaturePipelineError("横线残差上下文存在重复时间戳。")
    if "history_barrier" in context:
        # Carry the boundary after excluded observations have been removed.
        context["history_segment"] = context["history_barrier"].fillna(False).astype(bool).cumsum()
        finite = np.isfinite(context.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
        context = context.loc[finite & ~context["history_barrier"].fillna(False).astype(bool)].reset_index(drop=True)
    context_features = context.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(context_features).all():
        raise TemperaturePipelineError("横线残差上下文包含缺失或非有限输入。")

    base_prediction = model.predict(context_features)
    try:
        residual_features = build_residual_features(
            context,
            base_prediction[:, 1],
        )
        if tuple(residual_features.columns) != RESIDUAL_FEATURE_COLUMNS:
            raise ValueError("横线残差特征合同不一致。")
        config = CrosslineResidualConfig.from_mapping(model.crosslineResidualConfig)
        residual_prediction = np.asarray(
            model.crosslineResidualEstimator.predict(residual_features),
            dtype=float,
        )
    except ValueError as error:
        raise TemperaturePipelineError(f"横线残差预测失败：{error}") from error
    if residual_prediction.shape != (len(context),) or not np.isfinite(
        residual_prediction
    ).all():
        raise TemperaturePipelineError("横线残差预测形状或数值无效。")
    base_prediction[:, 1] = (
        base_prediction[:, 1] + config.correction_scale * residual_prediction
    )
    combined = pd.DataFrame(
        round_prediction_values(base_prediction),
        index=context["timestamp"],
        columns=TARGET_COLUMNS,
    )
    requested_timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if requested_timestamps.duplicated().any():
        raise TemperaturePipelineError("预测数据存在重复时间戳。")
    missing_timestamps = requested_timestamps.loc[
        ~requested_timestamps.isin(combined.index)
    ]
    if not missing_timestamps.empty:
        raise TemperaturePipelineError("横线残差上下文未覆盖全部预测时间。")
    return combined.reindex(requested_timestamps).to_numpy(dtype=float)


def extract_linear_equations(
    model: FittedDualModel,
) -> dict[str, dict[str, float]]:
    """Convert standardized linear coefficients back to raw input units."""

    equations: dict[str, dict[str, float]] = {}
    for targetIndex, targetName in enumerate(TARGET_COLUMNS):
        estimator = model.estimators[targetIndex]
        regressor = estimator.named_steps["regressor"]
        if isinstance(regressor, (CatBoostRegressor, tabpfn_v2.TabPFNV2Regressor, probability.CrosslineProbabilityRegressor, compact36.Compact36Regressor, SingleBranchRegressor)):
            continue
        scaler = estimator.named_steps["scaler"]
        coefficients = np.asarray(regressor.coef_, dtype=float)
        intercept = np.asarray(regressor.intercept_, dtype=float).reshape(-1)
        scaledCoefficients = coefficients.reshape(-1)
        rawCoefficients = scaledCoefficients / np.asarray(
            scaler.scale_, dtype=float
        )
        scaledIntercept = float(intercept[0])
        rawIntercept = scaledIntercept - float(
            np.dot(rawCoefficients, np.asarray(scaler.mean_, dtype=float))
        )
        equations[targetName] = {
            "intercept": rawIntercept,
            **{
                feature: float(coefficient)
                for feature, coefficient in zip(
                    (POLYNOMIAL_FEATURE_COLUMNS if "polynomial" in estimator.named_steps else FEATURE_COLUMNS),
                    rawCoefficients,
                    strict=True,
                )
            },
        }
    return equations


def extract_derived_feature_columns(model: FittedDualModel) -> dict[str, list[str]]:
    """Record internal expanded features while preserving the external three-input contract."""

    result = {
        direction: list(POLYNOMIAL_FEATURE_COLUMNS)
        for direction, estimator in zip(DIRECTION_TARGETS, model.estimators, strict=True)
        if "polynomial" in estimator.named_steps
    }
    if direction_models_from_parameters(model.parameters)['crossline'] == probability.MODEL_NAME:
        result['crossline'] = list(probability.FEATURE_COLUMNS)
    if direction_models_from_parameters(model.parameters)['crossline'] in COMPACT_RECIPES:
        result['crossline'] = list(compact36.FEATURE_COLUMNS)
    return result


def compact_recipe_metadata(recipe, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Keep old recipe metadata exact while describing configurable TabICL archives."""
    return (recipe.recipe_metadata(parameters) if recipe in (tabicl_single, tabicl_16)
            else recipe.recipe_metadata())


def validate_compact36_artifact(model: FittedDualModel, metadata: Mapping[str, Any]) -> None:
    """Validate external34/internal36 contracts and lazy portable CPU components."""
    cross = model.estimators[1].named_steps['regressor']
    recipe = COMPACT_RECIPES[direction_models_from_parameters(model.parameters)['crossline']]
    expected_type = (SingleBranchRegressor if recipe in (tabicl_single, tabicl_16, tabm_single) else nine.NineRegionalRegressor if recipe is nine else seven.SevenRegionalRegressor if recipe is seven else compact36.Compact36Regressor)
    if (metadata.get('feature_version') != recipe.FEATURE_VERSION
            or getattr(model, 'featureVersion', None) != recipe.FEATURE_VERSION
            or metadata.get('crossline_compact36') != compact_recipe_metadata(recipe, model.parameters['directions']['crossline'])
            or metadata.get('direction_feature_columns', {}).get('crossline') != list(compact36.BASE_FEATURE_COLUMNS)
            or metadata.get('derived_feature_columns', {}).get('crossline') != list(compact36.FEATURE_COLUMNS)
            or model.features_for('crossline') != compact36.BASE_FEATURE_COLUMNS
            or model.features_for('inline') != FEATURE_COLUMNS
            or metadata.get('direction_feature_columns', {}).get('inline') != list(FEATURE_COLUMNS)
            or type(cross) is not expected_type
            or tuple(model.estimators[1].named_steps) != ('scaler', 'regressor')
            or model.estimators[1].named_steps['scaler'] != 'passthrough'):
        raise TemperaturePipelineError('36维组合归档的配置或两层特征合同不一致。')
    if recipe in (tabicl_single, tabicl_16, tabm_single) and cross.branch != ('tabicl' if recipe in (tabicl_single, tabicl_16) else 'tabm'):
        raise TemperaturePipelineError('单模型归档分支与算法不一致。')
    if recipe in (tabicl_single, tabicl_16) and getattr(cross, 'n_estimators', 4) != (16 if recipe is tabicl_16 else 4):
        raise TemperaturePipelineError('TabICLv2模型成员数与算法声明不一致。')
    if recipe in (tabicl_single, tabicl_16):
        params = model.parameters['directions']['crossline']
        modern = params.get('recipe_version') == 2
        if (getattr(cross, 'parameter_version', 1) != (2 if modern else 1)
                or getattr(cross, 'context_fraction', 1.) != params.get('context_fraction', 1.)
                or getattr(cross, 'temperature_calibration', False) is not params.get('temperature_calibration', False)
                or (modern and metadata.get('compact36_training_info') != cross.training_info_)):
            raise TemperaturePipelineError('TabICLv2设置或校准记录与保存模型不一致。')
    if recipe is nine:
        params = model.parameters['directions']['crossline']
        if any(getattr(cross, key) != params[key] for key in ('negative_weight', 'middle_weight', 'near_zero_weight')):
            raise TemperaturePipelineError('九档模型归档比例与参数不一致。')
    try:
        cross.validate_state()
    except (ValueError, AttributeError, KeyError, TypeError, RuntimeError) as error:
        raise TemperaturePipelineError(f'36维组合组件状态无效：{error}') from error


def validate_probability_artifact(model: FittedDualModel, metadata: Mapping[str, Any]) -> None:
    """Check both probability components and causal history against archived declarations."""
    from tabicl import TabICLClassifier

    cross = model.estimators[1].named_steps['regressor']
    tab = getattr(cross, 'tabicl_', None); cat = getattr(cross, 'catboost_', None)
    if (metadata.get('feature_version') != probability.FEATURE_VERSION
            or getattr(model, 'featureVersion', None) != probability.FEATURE_VERSION
            or metadata.get('crossline_probability') != probability.recipe_metadata()
            or metadata.get('direction_feature_columns', {}).get('crossline') != list(probability.FEATURE_COLUMNS)
            or model.features_for('crossline') != probability.FEATURE_COLUMNS
            or not isinstance(cross, probability.CrosslineProbabilityRegressor)
            or not isinstance(tab, TabICLClassifier) or not isinstance(cat, probability.CatBoostClassifier)
            or tab.n_estimators != 4 or tab.norm_methods != ['none', 'power']
            or tab.average_logits is not True or tab.softmax_temperature != 1.
            or tab.random_state != 20261022 or getattr(cross, 'n_features_in_', None) != 25):
        raise TemperaturePipelineError('横线概率融合归档、特征或固定参数声明不一致。')
    expected = {'iterations': 300, 'depth': 5, 'learning_rate': .03, 'l2_leaf_reg': 3., 'random_seed': 20261022}
    if any(cat.get_params().get(k) != v for k,v in expected.items()):
        raise TemperaturePipelineError('横线概率融合的CatBoost组件参数不一致。')
    try:
        probability.input_frame(cross.history_frame_)
    except (AttributeError, ValueError) as error:
        raise TemperaturePipelineError('横线概率融合缺少有效的历史输入上下文。') from error


def extract_feature_importances(model: FittedDualModel) -> dict[str, dict[str, float]]:
    """Archive tree feature importances separately from linear coefficients."""

    return {
        target: dict(zip(FEATURE_COLUMNS, map(float, regressor.feature_importances_), strict=True))
        for target, estimator in zip(TARGET_COLUMNS, model.estimators, strict=True)
        if isinstance(regressor := estimator.named_steps["regressor"], CatBoostRegressor)
    }


def metrics_table(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    split: str,
    modelName: str,
) -> pd.DataFrame:
    """Return direction-specific regression metrics with units and timestamps."""

    rows = []
    for directionIndex, (direction, target) in enumerate(DIRECTION_TARGETS.items()):
        rows.append(
            {
                "split": split,
                "model": modelName,
                "direction": direction,
                "sample_count": len(frame),
                **regression_metrics(
                    frame[target].to_numpy(dtype=float),
                    prediction[:, directionIndex],
                    frame["timestamp"],
                ),
            }
        )
    return pd.DataFrame(rows)


def baseline_metrics(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    """Evaluate constant and per-direction fixed-bias baselines."""

    trainTargets = train.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=float)
    evaluationTargets = evaluation.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=float)
    constantPrediction = np.tile(trainTargets.mean(axis=0), (len(evaluation), 1))
    matchingFeatures = ("tilt1_inline_model", "tilt1_crossline_model")
    trainInput = train.loc[:, list(matchingFeatures)].to_numpy(dtype=float)
    evaluationInput = evaluation.loc[:, list(matchingFeatures)].to_numpy(dtype=float)
    bias = (trainTargets - trainInput).mean(axis=0)
    fixedBiasPrediction = evaluationInput + bias
    rows: list[dict[str, Any]] = []
    for modelName, prediction in (
        ("constant_baseline", constantPrediction),
        ("fixed_bias_baseline", fixedBiasPrediction),
    ):
        for directionIndex, (direction, target) in enumerate(
            DIRECTION_TARGETS.items()
        ):
            rows.append(
                {
                    "split": split,
                    "model": modelName,
                    "direction": direction,
                    "sample_count": len(evaluation),
                    **regression_metrics(
                        evaluationTargets[:, directionIndex],
                        prediction[:, directionIndex],
                        evaluation["timestamp"],
                    ),
                }
            )
    return pd.DataFrame(rows)


def regression_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    timestamps: pd.Series,
) -> dict[str, Any]:
    """Calculate absolute and absolute-relative regression error evidence."""

    actualArray = np.asarray(actual, dtype=float)
    predictionArray = round_prediction_values(prediction)
    if actualArray.shape != predictionArray.shape or actualArray.size == 0:
        raise TemperaturePipelineError("指标输入形状不一致或为空。")
    absoluteError = np.abs(predictionArray - actualArray)
    maximumIndex = int(np.argmax(absoluteError))
    timestampValues = pd.to_datetime(timestamps, errors="raise").reset_index(drop=True)
    relativeError = absolute_relative_error_percent(actualArray, predictionArray)
    relativeValidMask = np.isfinite(relativeError)
    relativeValidCount = int(relativeValidMask.sum())
    excludedZeroCount = int(actualArray.size - relativeValidCount)
    if relativeValidCount:
        validRelativeError = relativeError[relativeValidMask]
        validIndices = np.flatnonzero(relativeValidMask)
        relativeMaximumIndex = int(validIndices[np.argmax(validRelativeError)])
        meanRelativeError = float(np.mean(validRelativeError))
        maximumRelativeError = float(relativeError[relativeMaximumIndex])
        maximumRelativeTimestamp = timestampValues.iloc[relativeMaximumIndex]
    else:
        meanRelativeError = float("nan")
        maximumRelativeError = float("nan")
        maximumRelativeTimestamp = pd.NaT
    return {
        "mae": float(np.mean(absoluteError)),
        "rmse": float(np.sqrt(np.mean(np.square(predictionArray - actualArray)))),
        "max_abs_error": float(absoluteError[maximumIndex]),
        "max_error_timestamp": timestampValues.iloc[maximumIndex],
        "unit": "degree",
        "mean_abs_relative_error_percent": meanRelativeError,
        "max_abs_relative_error_percent": maximumRelativeError,
        "max_relative_error_timestamp": maximumRelativeTimestamp,
        "relative_error_valid_count": relativeValidCount,
        "relative_error_excluded_zero_count": excludedZeroCount,
        "relative_error_unit": "percent",
    }


def absolute_relative_error_percent(
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
) -> np.ndarray:
    """Return row-level absolute relative error percentages; zero truth is NaN."""

    actualArray = np.asarray(actual, dtype=float)
    predictionArray = np.asarray(prediction, dtype=float)
    if actualArray.shape != predictionArray.shape:
        raise TemperaturePipelineError("相对误差输入形状不一致。")
    relativeError = np.full(actualArray.shape, np.nan, dtype=float)
    validMask = np.abs(actualArray) > RELATIVE_ERROR_ZERO_TOLERANCE
    relativeError[validMask] = (
        np.abs(predictionArray[validMask] - actualArray[validMask])
        / np.abs(actualArray[validMask])
        * 100.0
    )
    return relativeError


def load_active_cleaned(projectRoot: Path) -> pd.DataFrame:
    """Load the saved active data and enforce the three-feature contract."""

    path = (
        projectRoot.expanduser().resolve()
        / "data"
        / "interim"
        / "CSG124_temperature_cleaned.csv"
    )
    if not path.is_file():
        raise TemperaturePipelineError("找不到9月清洗数据，请先执行清洗。")
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", *FEATURE_COLUMNS, *TARGET_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TemperaturePipelineError(f"9月清洗数据缺少字段：{missing}")
    return frame


def sha256_file(path: Path, chunkSize: int = 1 << 20) -> str:
    """Return a streaming SHA-256 digest without modifying the file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunkSize), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict UTF-8 JSON after converting pandas and NumPy values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TemperaturePipelineError(f"找不到配置文件：{path}") from error
    except json.JSONDecodeError as error:
        raise TemperaturePipelineError(f"配置文件不是有效 JSON：{path}") from error
    if not isinstance(payload, dict):
        raise TemperaturePipelineError(f"配置文件必须是 JSON 对象：{path}")
    return payload


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TemperaturePipelineError(f"配置项 {key} 必须是 JSON 对象。")
    return value


def _float_tuple(payload: dict[str, Any], key: str) -> tuple[float, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise TemperaturePipelineError(f"配置项 {key} 必须是非空数组。")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise TemperaturePipelineError(f"配置项 {key} 必须只包含数字。") from error
    if not all(np.isfinite(result)):
        raise TemperaturePipelineError(f"配置项 {key} 不能包含非有限值。")
    return result


def _positive_float_tuple(payload: dict[str, Any], key: str) -> tuple[float, ...]:
    result = _float_tuple(payload, key)
    if any(item <= 0.0 for item in result):
        raise TemperaturePipelineError(f"配置项 {key} 必须全部大于0。")
    return result


def _nonnegative_float_tuple(
    payload: dict[str, Any], key: str
) -> tuple[float, ...]:
    result = _float_tuple(payload, key)
    if any(item < 0.0 for item in result):
        raise TemperaturePipelineError(f"配置项 {key} 不能包含负数。")
    return result


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TemperaturePipelineError(f"配置项 {key} 必须是正整数。")
    return value


def _finite_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        raise TemperaturePipelineError(f"配置项 {key} 必须是有限数字。")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TemperaturePipelineError(f"配置项 {key} 必须是有限数字。") from error
    if not np.isfinite(result):
        raise TemperaturePipelineError(f"配置项 {key} 必须是有限数字。")
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
