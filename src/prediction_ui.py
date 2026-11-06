"""Streamlit interface for the active CSG124 temperature model."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from sensor_app.runtime_paths import resolve_runtime_root  # noqa: E402
from sensor_app.table_display import display_dataframe, display_data_editor, two_decimal_text  # noqa: E402
from sensor_database.cleaning_options import PERSONAL_FIXED_TEMPERATURE_RULE_VERSION
from sensor_database.processing import current_processing_version, is_final_processing  # noqa: E402
from sensor_app.upload_options import native_upload_file_types  # noqa: E402
from sensor_app.ui_components import (  # noqa: E402
    render_page_header,
)

PROJECT_ROOT = resolve_runtime_root(PROJECT_ROOT)

RELATIVE_METRIC_COLUMNS = {
    "mean_abs_relative_error_percent",
    "max_abs_relative_error_percent",
    "max_relative_error_timestamp",
    "relative_error_valid_count",
    "relative_error_excluded_zero_count",
}
RELATIVE_PREDICTION_COLUMNS = {
    "absolute_relative_error_inline_percent",
    "absolute_relative_error_crossline_percent",
}
DIRECT_PREDICTION_MODE = "直接预测（不保存到数据库）"
DATABASE_PREDICTION_MODE = "保存到数据库后按比例预测"
FULL_AUTO_MAX_TRAIN_WEIGHT = 8.9


def _prediction_upload_file_types(
    *,
    is_frozen: bool | None = None,
    platform_name: str | None = None,
) -> tuple[str, str] | None:
    """Return chooser filters consistent with the data-import entry point."""

    return native_upload_file_types(
        is_frozen=is_frozen,
        platform_name=platform_name,
    )

PREDICTION_DISPLAY_NAMES = {
    "model_version": "模型版本号",
    "gateway": "设备号",
    "timestamp": "数据时间",
    "tilt1_crossline_model": "1号横线(°)",
    "predicted_crossline_tabicl": "横线预测·仅TabICLv2(°)",
    "predicted_crossline_tabm": "横线预测·仅TabM(°)",
    "tilt1_inline_model": "1号顺线(°)",
    "temperature_celsius": "温度(°C)",
    "temperature_review_included": "温度待判断参与预测",
    "tilt2_crossline": "2号横线真值(°)",
    "tilt2_inline": "2号顺线真值(°)",
    "predicted_crossline": "横线预测值(°)",
    "predicted_inline": "顺线预测值(°)",
    "error_crossline": "横线有符号误差(°)",
    "absolute_error_crossline": "横线绝对误差(°)",
    "squared_error_crossline": "横线平方误差(°²)",
    "absolute_relative_error_crossline_percent": "横线绝对相对误差(%)",
    "error_inline": "顺线有符号误差(°)",
    "absolute_error_inline": "顺线绝对误差(°)",
    "squared_error_inline": "顺线平方误差(°²)",
    "absolute_relative_error_inline_percent": "顺线绝对相对误差(%)",
}

PREDICTION_COMPARISON_ORDER = (
    "model_version",
    "gateway",
    "timestamp",
    "source_row_count",
    "tilt1_inline_model",
    "tilt1_crossline_model",
    "temperature_celsius",
    "tilt2_inline",
    "predicted_inline",
    "tilt2_crossline",
    "predicted_crossline",
    "error_inline",
    "absolute_error_inline",
    "squared_error_inline",
    "absolute_relative_error_inline_percent",
    "error_crossline",
    "absolute_error_crossline",
    "squared_error_crossline",
    "absolute_relative_error_crossline_percent",
)

TEST_PREDICTION_HIDDEN_DISPLAY_COLUMNS = frozenset(
    {
        "gateway",
    }
)
TEST_PREDICTION_LEADING_DISPLAY_COLUMNS = (
    "model_version",
    "timestamp",
)
UPLOAD_PREDICTION_HIDDEN_DISPLAY_COLUMNS = frozenset(
    {
        "gateway",
        "source_row_count",
    }
)

from tilt_predictor.interactive_prediction import (  # noqa: E402
    NO_COLUMN,
    InteractivePredictionService,
    UploadColumnMapping,
    UploadNormalizer,
    UploadTableReader,
    suggest_mapping,
)
from tilt_predictor.database_prediction import (  # noqa: E402
    build_database_prediction_upload,
)
from sensor_database import (  # noqa: E402
    SensorDatabaseError,
    get_sensor_database,
)
from tilt_predictor.model_registry import (  # noqa: E402
    DraftRun,
    ModelRegistryError,
    ModelVersion,
)
from tilt_predictor.registry_factory import get_model_registry  # noqa: E402
from tilt_predictor.ratio_training import (  # noqa: E402
    DEFAULT_TEST_PERCENTAGE,
    MAX_AUTO_RATIO_COUNT,
    MAX_TEST_PERCENTAGE,
    MAX_OPTUNA_TRIALS,
    MIN_AUTO_RATIO_COUNT,
    MIN_TEST_PERCENTAGE,
    MIN_OPTUNA_TRIALS,
    OPTUNA_TRIAL_STEP,
    RatioTrainingWorkflow,
    SplitRatio,
    TrainTestPercentage,
    automatic_split_ratios,
)
from tilt_predictor.temperature_pipeline import (  # noqa: E402
    FEATURE_VERSION,
    GATEWAY,
    MODEL_LABELS,
    available_direction_models,
    normalize_model_parameters,
    direction_models_from_parameters,
    model_combination_label,
    HYBRID_FEATURE_VERSION,
    PREDICTION_VALUE_COLUMNS,
    SUPPORTED_FEATURE_VERSIONS,
    TemperatureModelRepository,
    TemperaturePipelineError,
    round_prediction_values,
)

from tilt_predictor.training_results import TrainingResultArtifacts  # noqa: E402
from tilt_predictor import crossline_probability as probability
from tilt_predictor import crossline_compact36 as compact36
from tilt_predictor.model_explanation import tabicl_settings_summary, tabicl_calibration_rows
from tilt_predictor import crossline_tabicl_single as tabicl_single
from tilt_predictor import crossline_tabicl_16 as tabicl_16
from tilt_predictor import crossline_tabm_single as tabm_single
from tilt_predictor import crossline_seven as seven
from tilt_predictor import crossline_nine_regional as nine
from tilt_predictor import regional_weight_search as regional_search
from tilt_predictor.regional_search_job import RegionalSearchJob
from tilt_predictor.prediction_reports import (  # noqa: E402
    remove_retired_metric_columns,
)
from tilt_predictor.model_explanation import (  # noqa: E402
    build_model_explanation,
    format_coefficient,
)
from tilt_predictor.data_information import (  # noqa: E402
    CrosslineInformationReport,
    analyze_crossline_information,
)


def order_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Place both directions' ground-truth/prediction pairs before error columns."""

    frame = remove_retired_metric_columns(frame)
    preferred = [
        column for column in PREDICTION_COMPARISON_ORDER if column in frame.columns
    ]
    remaining = [column for column in frame.columns if column not in preferred]
    return frame.loc[:, [*preferred, *remaining]].copy()


def build_test_prediction_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the compact row-level prediction table used by the training page.

    The downloadable CSV intentionally keeps the complete audit contract. The
    page table hides only redundant identity columns while keeping the model
    version and timestamp first.
    """

    frame = remove_retired_metric_columns(frame)
    leading = [
        column
        for column in TEST_PREDICTION_LEADING_DISPLAY_COLUMNS
        if column in frame.columns
    ]
    remaining = [
        column
        for column in frame.columns
        if column not in TEST_PREDICTION_HIDDEN_DISPLAY_COLUMNS
        and column not in TEST_PREDICTION_LEADING_DISPLAY_COLUMNS
    ]
    return frame.loc[:, [*leading, *remaining]].copy()


def draft_version_label(run_id: str, drafts: tuple[DraftRun, ...]) -> str:
    """Return the compact public label for one persistent training candidate.

    Draft directories keep their ``run-...`` identifiers for audit tracing.  The
    training page instead exposes chronological ``xN`` labels so they cannot be
    confused with immutable, published ``vN`` model versions.
    """

    ordered = sorted(
        drafts,
        key=lambda draft: (draft.created_at_utc, draft.run_id),
    )
    for number, draft in enumerate(ordered, start=1):
        if draft.run_id == run_id:
            return f"x{number}"
    return "x?"


def _is_current_huber_candidate(metadata: dict[str, object]) -> bool:
    """Return whether a draft matches the current dual-Huber training contract."""

    contracts = metadata.get("direction_contracts")
    return (
        metadata.get("feature_version") == FEATURE_VERSION
        and isinstance(contracts, dict)
        and contracts.get("inline") == "absolute_value_huber"
        and contracts.get("crossline") == "absolute_value_huber"
        and "crossline_residual" not in metadata
    )


def _is_current_linear_candidate(metadata: dict[str, object]) -> bool:
    """Accept current absolute linear candidates, including mixed algorithms."""

    if metadata.get("feature_version") not in (FEATURE_VERSION, probability.FEATURE_VERSION, compact36.FEATURE_VERSION, seven.FEATURE_VERSION, nine.FEATURE_VERSION, tabicl_single.FEATURE_VERSION, tabm_single.FEATURE_VERSION) or "crossline_residual" in metadata:
        return False
    try:
        parameters = normalize_model_parameters(metadata.get("parameters", {}))
        models = direction_models_from_parameters(parameters)
    except TemperaturePipelineError:
        return False
    return metadata.get("model") == parameters["model"] and metadata.get("direction_contracts") == {
        direction: f"absolute_value_{model}" for direction, model in models.items()
    }


def _parameters_match_models(metadata: dict, models: dict[str, str]) -> bool:
    """Only reuse a saved version with the selected direction algorithms."""

    if metadata.get("feature_version") not in SUPPORTED_FEATURE_VERSIONS:
        return False
    try:
        parameters = normalize_model_parameters(metadata.get("parameters", {}))
        return direction_models_from_parameters(parameters) == models
    except TemperaturePipelineError:
        return False


def _latest_current_huber_draft(
    registry: object,
    drafts: tuple[DraftRun, ...],
    *,
    include_linear: bool = False,
) -> DraftRun | None:
    """Find the newest readable draft for the current training contract.

    Older hybrid drafts remain stored for audit compatibility, but must not be
    surfaced as the candidate that a user can save after the rollback.
    """

    for draft in drafts:
        try:
            directory = registry.draft_directory(draft.run_id)
            metadata = json.loads(
                (directory / "metadata.json").read_text(encoding="utf-8-sig")
            )
        except (AttributeError, OSError, ValueError, json.JSONDecodeError):
            continue
        predicate = _is_current_linear_candidate if include_linear else _is_current_huber_candidate
        if isinstance(metadata, dict) and predicate(metadata):
            return draft
    return None


def with_display_model_version(frame: pd.DataFrame, version_label: str) -> pd.DataFrame:
    """Return a copy whose public model-version column uses ``version_label``."""

    display = frame.copy()
    if "model_version" in display.columns:
        display["model_version"] = version_label
    return display


def build_upload_prediction_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return upload predictions without redundant identity and audit columns.

    The underlying prediction frame and downloadable CSV retain these fields so
    import provenance remains available outside the on-screen comparison table.
    """

    frame = remove_retired_metric_columns(frame)
    visible_columns = [
        column
        for column in frame.columns
        if column not in UPLOAD_PREDICTION_HIDDEN_DISPLAY_COLUMNS
    ]
    return frame.loc[:, visible_columns].copy()


def _has_relative_metric_columns(frame: pd.DataFrame) -> bool:
    """Return whether a report contains the complete relative-error contract."""

    return RELATIVE_METRIC_COLUMNS.issubset(frame.columns)


def _render_prediction_audit(
    audit: dict[str, object],
    *,
    visible: bool,
) -> None:
    """Render diagnostic audit JSON only on editions that explicitly expose it."""

    if visible:
        st.json(audit)


def _select_prediction_mode(*, allow_database_prediction: bool) -> str:
    """Select a prediction mode without exposing persistence in customer builds."""

    if not allow_database_prediction:
        st.caption("只在内存中执行本次上传预测，不写入传感器数据库。")
        return DIRECT_PREDICTION_MODE
    return st.radio(
        "预测方式",
        (DIRECT_PREDICTION_MODE, DATABASE_PREDICTION_MODE),
        horizontal=True,
        key="upload_prediction_mode",
    )


def format_metric_table_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only copy with every metric rounded to two decimals.

    Persisted reports and CSV downloads keep their original precision. Only
    numeric columns whose names identify evaluation metrics are formatted.
    """

    formatted = remove_retired_metric_columns(frame)
    metric_tokens = (
        "mae",
        "rmse",
        "relative_error",
        "objective_value",
        "误差",
        "改善率",
    )
    for column in formatted.columns:
        column_name = str(column).lower()
        if column_name.endswith(("_count", "_timestamp")):
            continue
        if not any(token in column_name for token in metric_tokens):
            continue
        numeric = pd.to_numeric(formatted[column], errors="coerce")
        numeric_mask = numeric.notna()
        if not numeric_mask.any():
            continue
        formatted[column] = formatted[column].astype(object)
        formatted.loc[numeric_mask, column] = numeric.loc[numeric_mask].map(
            lambda value: f"{value:.2f}"
        )
    return formatted


def prediction_column_config(frame: pd.DataFrame) -> dict[str, object]:
    """Format prediction values and keep compact identity columns in Streamlit."""

    config: dict[str, object] = {}
    for candidate in (
        "model_version",
        PREDICTION_DISPLAY_NAMES["model_version"],
    ):
        if candidate in frame.columns:
            config[candidate] = st.column_config.TextColumn(width=120)
    for column in PREDICTION_VALUE_COLUMNS:
        for candidate in (column, PREDICTION_DISPLAY_NAMES.get(column, column)):
            if candidate in frame.columns:
                config[candidate] = st.column_config.NumberColumn(format="%.2f")
    return config


def prediction_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize public prediction columns without retired legacy metrics."""

    export = remove_retired_metric_columns(frame)
    for column in PREDICTION_VALUE_COLUMNS:
        if column not in export:
            continue
        numeric = pd.to_numeric(export[column], errors="coerce")
        rounded = pd.Series(
            round_prediction_values(numeric.to_numpy(dtype=float)),
            index=export.index,
        )
        export[column] = rounded.map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}"
        )
    return export.to_csv(index=False).encode("utf-8-sig")


def format_coefficient_table_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Format coefficient cells compactly without changing persisted values."""

    formatted = frame.copy()
    for column in formatted.columns:
        if column == "预测目标":
            continue
        formatted[column] = formatted[column].map(
            lambda value: None
            if pd.isna(value)
            else format_coefficient(float(value))
            if not is_final_processing()
            else two_decimal_text(value)
        )
    return formatted


def _visible_relative_metric_cards(
    row: pd.Series,
    *,
    has_relative_metrics: bool,
) -> list[tuple[str, str]]:
    """Return the compact relative-error cards shown in one summary row."""

    cards: list[tuple[str, str]] = []
    if has_relative_metrics:
        cards.append(
            (
                "平均绝对相对误差",
                f"{row['mean_abs_relative_error_percent']:.2f}%",
            )
        )
    return cards


def show_prediction_metric_cards(
    metrics: pd.DataFrame,
    *,
    heading: str,
    caveat: str | None = None,
) -> None:
    """Render one consistent metric-card summary for both prediction directions."""

    st.subheader(heading)
    if caveat:
        st.caption(caveat)

    direction_labels = {"inline": "顺线", "crossline": "横线"}
    has_relative_metrics = _has_relative_metric_columns(metrics)
    if not has_relative_metrics:
        st.warning(
            "当前结果不含完整相对误差字段；"
            "MAE、RMSE和最大绝对误差仍可查看。"
        )

    for _, row in metrics.iterrows():
        direction_key = str(row["direction"])
        direction = direction_labels.get(direction_key, direction_key)
        st.markdown(f"**{direction}（{int(row['sample_count'])} 条）**")
        mae_column, rmse_column, maximum_column = st.columns(3)
        mae_column.metric("MAE", f"{row['mae']:.2f}°")
        rmse_column.metric("RMSE", f"{row['rmse']:.2f}°")
        maximum_column.metric("最大绝对误差", f"{row['max_abs_error']:.2f}°")
        st.caption(f"最大误差发生时间：{row['max_error_timestamp']}")

        relative_cards = _visible_relative_metric_cards(
            row,
            has_relative_metrics=has_relative_metrics,
        )
        if relative_cards:
            relative_columns = st.columns(len(relative_cards))
            for column, (label, value) in zip(
                relative_columns,
                relative_cards,
                strict=True,
            ):
                column.metric(label, value)
        if has_relative_metrics:
            st.caption(
                f"相对误差有效 {int(row['relative_error_valid_count'])} 条；"
                f"真值为0排除 "
                f"{int(row['relative_error_excluded_zero_count'])} 条。"
            )


# Keep the established private name available for callers and tests that have
# not migrated to the public shared renderer yet.
_show_prediction_metric_cards = show_prediction_metric_cards


def _model_contract_summary(metadata: dict[str, object]) -> tuple[str, str | None]:
    """Return a concise model label plus direction-specific contract detail."""

    model_name = str(metadata.get("model", "UNKNOWN")).upper()
    if metadata.get("feature_version") == HYBRID_FEATURE_VERSION:
        return (
            f"顺线 {model_name}｜横线两阶段残差",
            "顺线：三特征绝对值 Huber｜横线：Huber基础预测 + 输入侧时序残差模型",
        )
    if metadata.get("parameters"):
        return model_combination_label(metadata["parameters"]), "顺线、横线均直接预测2号绝对倾角"
    return f"三特征绝对值 {model_name}", "顺线、横线均直接预测2号绝对倾角"


def _show_crossline_data_information(measurements: pd.DataFrame) -> None:
    """Show whether the current data can describe crossline changes."""

    st.subheader("横线数据有效信息诊断")
    st.caption(
        "只读检查当前传感器数据库的数值分辨率、10分钟时间覆盖和±60分钟变化滞后；"
        "不会清洗原始证据、训练模型或切换活动版本。"
    )
    try:
        report: CrosslineInformationReport = analyze_crossline_information(
            measurements
        )
    except TemperaturePipelineError as error:
        st.warning(f"暂时无法完成横线数据诊断：{error}")
        return

    summary = report.summary
    source_column, bucket_column, resolution_column, lag_column = st.columns(4)
    source_column.metric("数据库记录", int(summary["source_rows"]))
    bucket_column.metric(
        "有效10分钟桶",
        int(summary["effective_bucket_rows"]),
        delta=f"合并 {int(summary['collapsed_rows'])} 条同桶记录",
        delta_color="off",
    )
    resolution_column.metric(
        "2号横线原始步进",
        f"{summary['target_source_resolution_degrees']:.2f}°",
    )
    lag_column.metric(
        "1号横线最佳变化相关",
        f"{summary['best_crossline_lag_correlation']:.2f}",
        delta=f"{int(summary['best_crossline_lag_minutes']):+d} 分钟",
        delta_color="off",
    )
    st.caption(
        f"10分钟时间覆盖率 {summary['bucket_coverage_percent']:.2f}%｜"
        f"缺少 {int(summary['missing_bucket_rows'])} 个时间桶｜"
        f"2号横线原始取值 {int(summary['target_source_unique_values'])} 种。"
        "正滞后表示1号变化领先2号，负滞后表示2号领先1号。"
    )
    best_lags = report.best_lags.rename(
        columns={
            "feature_label": "输入信号",
            "lag_minutes": "最佳滞后(分钟)",
            "relationship": "时间关系",
            "pair_count": "有效变化对数",
            "pearson_correlation": "Pearson相关",
            "spearman_correlation": "Spearman相关",
        }
    ).loc[
        :,
        [
            "输入信号",
            "最佳滞后(分钟)",
            "时间关系",
            "有效变化对数",
            "Pearson相关",
            "Spearman相关",
        ],
    ]
    st.markdown("#### 各输入与2号横线变化的最佳时间关系")
    display_dataframe(
        best_lags,
        hide_index=True,
        width="stretch",
        column_config={
            "Pearson相关": st.column_config.NumberColumn(format="%.2f"),
            "Spearman相关": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    with st.expander("查看字段精度和全部滞后扫描"):
        field_quality = report.field_quality.rename(
            columns={
                "label": "字段",
                "source_unique_values": "原始不同取值数",
                "source_resolution": "原始最小步进",
                "unchanged_fraction": "相邻10分钟不变比例",
                "minimum": "最小值",
                "maximum": "最大值",
            }
        ).loc[
            :,
            [
                "字段",
                "原始不同取值数",
                "原始最小步进",
                "相邻10分钟不变比例",
                "最小值",
                "最大值",
            ],
        ]
        display_dataframe(
            field_quality,
            hide_index=True,
            width="stretch",
            column_config={
                "原始最小步进": st.column_config.NumberColumn(format="%.2f"),
                "相邻10分钟不变比例": st.column_config.NumberColumn(format="%.2%%"),
                "最小值": st.column_config.NumberColumn(format="%.2f"),
                "最大值": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        lag_chart = report.lag_analysis.pivot(
            index="lag_minutes",
            columns="feature_label",
            values="pearson_correlation",
        )
        st.line_chart(lag_chart)
        st.caption(
            "相关系数只用于发现可能的时间偏移，不代表因果关系；"
            "接近0表示当前数据中没有稳定的线性变化关系。"
        )


def _select_column(
    label: str,
    options: list[str],
    suggested: str | None,
    *,
    key: str,
    optional: bool = False,
) -> str | None:
    """Render one explicit column selector with an alias-based default."""

    displayed = [NO_COLUMN, *options] if optional else options
    if optional and suggested is None:
        index = 0
    elif suggested in displayed:
        index = displayed.index(suggested)
    else:
        index = 1 if optional else 0
    selected = st.selectbox(label, displayed, index=index, key=key)
    return None if selected == NO_COLUMN else selected


def _show_model_status() -> None:
    """Show active data/model identity without treating a missing report as pass."""

    st.sidebar.subheader("当前模型")
    st.sidebar.write(f"网关：{GATEWAY}")
    st.sidebar.write("模型族：三特征回归")
    try:
        database_status = get_sensor_database(PROJECT_ROOT).status()
    except (TemperaturePipelineError, SensorDatabaseError) as error:
        st.sidebar.warning(f"传感器数据库不可用：{error}")
    else:
        st.sidebar.write(f"传感器数据库：{database_status.measurement_rows} 条")
        st.sidebar.write(f"导入次数：{database_status.import_count}")
    try:
        loaded = TemperatureModelRepository(PROJECT_ROOT).load()
    except TemperaturePipelineError as error:
        st.sidebar.warning(str(error))
        return
    status = str(loaded.metadata.get("status", "UNKNOWN"))
    feature_version = str(loaded.metadata.get("feature_version", FEATURE_VERSION))
    st.sidebar.write(f"特征版本：{feature_version}")
    st.sidebar.write(
        f"活动版本：{loaded.metadata.get('version_id', '迁移中的兼容版本')}"
    )
    st.sidebar.write(
        f"模型数据版本：{loaded.metadata.get('dataset_version', 'UNKNOWN')}"
    )
    split_ratio = loaded.metadata.get("split", {}).get("requested_ratio", {})
    if split_ratio:
        split_method = loaded.metadata.get("split", {}).get("method")
        split_label = (
            "训练/测试"
            if split_method == "chronological_train_test_percentage"
            else "训练/验证/测试"
        )
        st.sidebar.write(f"{split_label}：{split_ratio.get('label', 'UNKNOWN')}")
    st.sidebar.write(f"模型：{loaded.model.name}")
    st.sidebar.write(f"参数：{loaded.model.parameters}")
    with st.sidebar.expander("当前活动模型公式"):
        _show_model_explanation(loaded.metadata, showHeading=False)
    models = direction_models_from_parameters(loaded.model.parameters)
    st.sidebar.write(f"顺线：三特征绝对值 {MODEL_LABELS[models['inline']]}")
    if feature_version == HYBRID_FEATURE_VERSION:
        st.sidebar.write("横线：Huber基础预测 + 输入侧时序残差模型")
    else:
        st.sidebar.write(f"横线：{MODEL_LABELS[models['crossline']]}" if models['crossline'] in (probability.MODEL_NAME, compact36.MODEL_NAME, seven.MODEL_NAME, nine.MODEL_NAME, tabicl_single.MODEL_NAME, tabicl_16.MODEL_NAME, tabm_single.MODEL_NAME)
                         else f"横线：三特征绝对值 {MODEL_LABELS[models['crossline']]}")
def _show_latest_training_results(
    selected_ratio_label: str,
    artifact_directory: Path | None = None,
    candidate_version_label: str | None = None,
) -> None:
    """Render the latest model, metrics, and final-test predictions."""

    st.divider()
    st.subheader("当前训练候选" if artifact_directory else "当前活动模型结果")
    try:
        result = (
            TrainingResultArtifacts.load_from_directory(artifact_directory)
            if artifact_directory is not None
            else TrainingResultArtifacts.load(PROJECT_ROOT)
        )
    except TemperaturePipelineError as error:
        st.info(f"尚无可展示的完整训练结果：{error}")
        return

    test_metrics = result.selected_test_metrics()

    metadata = result.metadata
    trained_ratio = metadata["split"]["requested_ratio"]["label"]
    created_at = pd.Timestamp(metadata["created_at_utc"])
    if created_at.tzinfo is not None:
        created_at = created_at.tz_convert("Asia/Shanghai")
    created_label = created_at.strftime("%Y-%m-%d %H:%M:%S")
    st.caption(
        f"结果生成时间：{created_label}（北京时间）｜"
        f"训练比例：{trained_ratio}｜数据版本：{metadata['dataset_version']}"
    )
    if trained_ratio != selected_ratio_label:
        st.info(
            f"当前控件选择的是 {selected_ratio_label}，下方仍是最近一次已完成的 "
            f"{trained_ratio} 训练结果。点击上方训练按钮后才会更新。"
        )
    model_column, sample_column, ratio_column = st.columns(3)
    model_label, model_contract_detail = _model_contract_summary(metadata)
    model_column.metric("选中模型", model_label)
    sample_column.metric("最终拟合样本", metadata["sample_counts"]["final_fit"])
    ratio_column.metric("测试样本", metadata["sample_counts"]["test"])
    if model_contract_detail:
        st.caption(model_contract_detail)
    settings_summary = tabicl_settings_summary(metadata)
    if settings_summary:
        st.caption(settings_summary)
    optimization = metadata.get("hyperparameter_optimization")
    if isinstance(optimization, dict):
        st.markdown("#### 超参数搜索结果")
        by_direction = optimization.get("by_direction")
        if isinstance(by_direction, dict):
            optimization_rows = []
            for direction, values in by_direction.items():
                optimization_rows.append(
                    {
                        "方向": "顺线" if direction == "inline" else "横线",
                        "成功/请求": (
                            f"{values.get('trials_completed', 0)} / "
                            f"{values.get('trials_requested', 0)}"
                        ),
                        "失败": values.get("trials_failed", 0),
                        "最优Trial": values.get("best_trial_number", "—"),
                        ("最优目标值" if metadata.get('feature_version') in (compact36.FEATURE_VERSION, seven.FEATURE_VERSION, nine.FEATURE_VERSION, tabicl_single.FEATURE_VERSION, tabm_single.FEATURE_VERSION)
                         else "最优MAE(°)"): values.get("best_objective_value", "—"),
                        **({'评价指标': 'MARE(%)' if direction == 'crossline' else 'MAE(°)'}
                           if metadata.get('feature_version') in (compact36.FEATURE_VERSION, seven.FEATURE_VERSION, nine.FEATURE_VERSION, tabicl_single.FEATURE_VERSION, tabm_single.FEATURE_VERSION) else {}),
                        "随机种子": values.get("seed", "—"),
                    }
                )
            display_dataframe(
                format_metric_table_for_display(pd.DataFrame(optimization_rows)),
                hide_index=True,
                width="stretch",
            )
            failed_count = sum(
                int(values.get("trials_failed", 0))
                for values in by_direction.values()
            )
        else:
            trial_column, best_column, failed_column = st.columns(3)
            trial_column.metric(
                "成功试验",
                f"{optimization.get('trials_completed', 0)} / "
                f"{optimization.get('trials_requested', 0)}",
            )
            best_column.metric("最优 Trial", optimization.get("best_trial_number", "—"))
            failed_count = int(optimization.get("trials_failed", 0))
            failed_column.metric("失败试验", failed_count)
        st.caption(
            f"搜索引擎：{optimization.get('engine', 'UNKNOWN')}｜"
            f"采样器：{optimization.get('sampler') or '无'}｜"
            f"目标：{optimization.get('objective', 'UNKNOWN')}"
        )
        if failed_count:
            st.warning(
                f"有 {failed_count} 次超参数试验失败；失败原因已写入完整试验报告，"
                "其余成功试验仍参与排名。"
            )

    st.markdown("#### 最终模型参数")
    parameter_document = metadata["parameters"]
    directions = parameter_document.get("directions")
    if isinstance(directions, dict):
        parameter_rows = [
            {
                "方向": "顺线" if direction == "inline" else "横线",
                "算法": MODEL_LABELS[direction_models_from_parameters(parameter_document)[direction]],
                "epsilon": _parameter_text(values.get("epsilon")),
                "alpha": _parameter_text(values.get("alpha")),
                "l1_ratio": _parameter_text(values.get("l1_ratio")),
                "max_iter": _parameter_text(values.get("max_iter")),
                "degree": _parameter_text(values.get("degree")),
                **{name: _parameter_text(values.get(name)) for name in (
                    "iterations", "depth", "learning_rate", "l2_leaf_reg", "random_seed", "n_estimators", "tabicl_weight", "context_fraction", "temperature_calibration"
                )},
            }
            for direction, values in directions.items()
        ]
    else:
        parameter_rows = [
            {"参数": name, "数值": str(value)}
            for name, value in parameter_document.items()
        ]
    parameter_column, equation_column = st.columns([1, 2])
    with parameter_column:
        st.caption("超参数（训练前配置或选参得到）")
        display_dataframe(pd.DataFrame(parameter_rows), hide_index=True, width="stretch")
    with equation_column:
        if metadata.get("feature_importances") or metadata.get("derived_feature_columns"):
            _show_model_explanation(metadata, showHeading=False)
        else:
            st.caption("训练后学到的原始输入线性系数与截距")
            display_dataframe(
                format_coefficient_table_for_display(result.equation_rows()),
                hide_index=True,
                width="stretch",
            )
    if not metadata.get("feature_importances") and not metadata.get("derived_feature_columns"):
        st.caption(
            "顺线和横线预测方程形式：预测值 = 截距 + "
            "1号顺线系数×1号顺线 + 1号横线系数×1号横线 + 温度系数×温度。"
        )
    residual = metadata.get("crossline_residual")
    if isinstance(residual, dict):
        residual_parameters = residual.get("parameters")
        if isinstance(residual_parameters, dict):
            st.caption(
                "横线最终结果 = Huber基础预测 + 修正强度 × 残差预测；"
                "残差模型只使用当前及过去的1号顺线、1号横线、温度和时间。"
            )
            display_dataframe(
                pd.DataFrame(
                    [
                        {
                            "损失函数": residual_parameters.get("loss", "—"),
                            "学习率": residual_parameters.get("learning_rate", "—"),
                            "叶节点数": residual_parameters.get(
                                "max_leaf_nodes", "—"
                            ),
                            "最小叶样本": residual_parameters.get(
                                "min_samples_leaf", "—"
                            ),
                            "L2": residual_parameters.get(
                                "l2_regularization", "—"
                            ),
                            "修正强度": residual_parameters.get(
                                "correction_scale", "—"
                            ),
                        }
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

    show_prediction_metric_cards(test_metrics, heading="最终测试集指标")

    split_method = metadata.get("split", {}).get("method")
    metrics_expander_label = (
        "查看测试集与基线的全部指标"
        if split_method == "chronological_train_test_percentage"
        else "查看验证集、测试集与基线的全部指标"
    )
    with st.expander(metrics_expander_label):
        display_dataframe(
            format_metric_table_for_display(result.metrics),
            hide_index=True,
            width="stretch",
        )

    st.markdown("#### 测试集逐条预测结果")
    try:
        source_frame = get_sensor_database(PROJECT_ROOT).training_snapshot().frame
        detailed_predictions = result.detailed_predictions(source_frame)
    except (TemperaturePipelineError, SensorDatabaseError) as error:
        st.error(f"逐条预测结果无法补齐：{error}")
        return
    if candidate_version_label is not None:
        detailed_predictions = with_display_model_version(
            detailed_predictions,
            candidate_version_label,
        )
    prediction_chart = detailed_predictions.set_index("timestamp")
    has_relative_predictions = RELATIVE_PREDICTION_COLUMNS.issubset(
        detailed_predictions.columns
    )
    tab_labels = ["顺线：真值与预测", "横线：真值与预测", "绝对误差"]
    if has_relative_predictions:
        tab_labels.append("相对误差")
    chart_tabs = st.tabs(tab_labels)
    inline_tab, crossline_tab, error_tab = chart_tabs[:3]
    with inline_tab:
        st.line_chart(
            prediction_chart.loc[:, ["tilt2_inline", "predicted_inline"]].rename(
                columns={"tilt2_inline": "顺线真值", "predicted_inline": "顺线预测"}
            )
        )
    with crossline_tab:
        st.line_chart(
            prediction_chart.loc[
                :, ["tilt2_crossline", "predicted_crossline"]
            ].rename(
                columns={
                    "tilt2_crossline": "横线真值",
                    "predicted_crossline": "横线预测",
                }
            )
        )
    with error_tab:
        st.line_chart(
            prediction_chart.loc[
                :, ["absolute_error_inline", "absolute_error_crossline"]
            ].rename(
                columns={
                    "absolute_error_inline": "顺线绝对误差",
                    "absolute_error_crossline": "横线绝对误差",
                }
            )
        )
    if has_relative_predictions:
        with chart_tabs[3]:
            st.line_chart(
                prediction_chart.loc[
                    :,
                    [
                        "absolute_relative_error_inline_percent",
                        "absolute_relative_error_crossline_percent",
                    ],
                ].rename(
                    columns={
                        "absolute_relative_error_inline_percent": (
                            "顺线绝对相对误差(%)"
                        ),
                        "absolute_relative_error_crossline_percent": (
                            "横线绝对相对误差(%)"
                        ),
                    }
                )
            )
    aggregate_display_names = {}
    metric_labels = {
        "mae": "MAE(°)",
        "rmse": "RMSE(°)",
        "max_abs_error": "最大绝对误差(°)",
        "mean_abs_relative_error_percent": "平均绝对相对误差(%)",
        "max_abs_relative_error_percent": "最大绝对相对误差(%)",
    }
    for direction, label in (("crossline", "横线"), ("inline", "顺线")):
        for metric, metric_label in metric_labels.items():
            aggregate_display_names[f"test_{metric}_{direction}"] = (
                f"测试集{label}{metric_label}"
            )
    table_predictions = build_test_prediction_display_frame(detailed_predictions)
    display_predictions = table_predictions.rename(
        columns={**PREDICTION_DISPLAY_NAMES, **aggregate_display_names}
    )
    st.caption(
        "单行误差按该时刻计算；以“测试集”开头的指标是整段测试集汇总值，"
        "按要求附在每一行，因此同一方向的这些值会重复。"
    )
    display_dataframe(
        format_metric_table_for_display(display_predictions),
        column_config=prediction_column_config(display_predictions),
        hide_index=True,
        width="stretch",
        height=360,
    )
    st.download_button(
        "下载测试集预测结果 CSV",
        prediction_csv_bytes(detailed_predictions),
        file_name="CSG124_final_test_predictions.csv",
        mime="text/csv",
    )

    with st.expander("查看 Optuna 试验排名与模型参数"):
        display_dataframe(
            format_metric_table_for_display(result.candidates),
            hide_index=True,
            width="stretch",
        )
    with st.expander("查看完整训练元数据"):
        st.json(metadata)


def _parameter_text(value: float | int | None) -> str:
    """Keep inapplicable parameters distinct from numeric zero in tables."""

    if value is None:
        return "—"
    return two_decimal_text(value) if is_final_processing() else f"{value:g}"


def _version_label(version: ModelVersion, metadata: dict | None = None) -> str:
    """Return a version label with the actual algorithms instead of Huber-only parameters."""

    note = f"｜{version.note}" if version.note else ""
    if metadata is not None:
        model_label = model_combination_label(metadata["parameters"])
    elif version.model in MODEL_LABELS:
        name = MODEL_LABELS[version.model]
        model_label = f"顺线 {name}｜横线 {name}"
    else:
        model_label = version.model
    return f"{version.version_id}｜{model_label}{note}"


def _prediction_model_label(
    version: ModelVersion,
    active_version_id: str,
    *,
    compact_model_labels: bool,
    metadata: dict | None = None,
) -> str:
    """Return one upload-prediction model name for the selected edition."""

    if compact_model_labels:
        return version.version_id
    active_suffix = "｜当前活动" if version.version_id == active_version_id else ""
    return f"{_version_label(version, metadata)}{active_suffix}"


def _show_model_explanation(
    metadata: dict[str, object],
    *,
    showHeading: bool = True,
) -> None:
    """Render formulas and coefficients derived from exactly one model version."""

    explanation = build_model_explanation(metadata)
    if showHeading:
        st.markdown("#### 模型公式与特征重要性" if explanation.featureImportanceRows
                    else "#### 模型公式与回归系数")
    st.caption(
        "横线为36维概率模型与相对误差选档；下表系数只描述顺线，温度分解参数随横线模型归档。"
        if metadata.get('feature_version') in (compact36.FEATURE_VERSION, seven.FEATURE_VERSION, nine.FEATURE_VERSION, tabicl_single.FEATURE_VERSION, tabm_single.FEATURE_VERSION) else
        "二次项 Ridge 包含原始项、平方项及两两交互项；下表展示原始量纲下的系数与截距。"
        if metadata.get("derived_feature_columns") else
        "线性模型展示训练后的截距与系数；CatBoost 展示树模型的特征重要性。"
        if explanation.featureImportanceRows else
        "epsilon、alpha、l1_ratio、max_iter 是各算法适用的训练超参数；"
        "截距和各特征系数是训练后学到的模型参数。"
    )
    st.write(f"模型结构：{explanation.structureLabel}")
    if not explanation.formulas:
        st.warning("该历史版本未保存可展示公式。")
        return
    st.code("\n".join(explanation.formulas), language=None)
    calibration_rows = tabicl_calibration_rows(metadata)
    if calibration_rows:
        st.caption('NLL校准覆盖情况：NLL仅统计模型已覆盖档位的有效记录。')
        if any(row['未覆盖'] for row in calibration_rows):
            st.warning('部分内部校准记录的档位尚未出现在拟合上下文中，已单独计数；'
                       '这些记录不参与温度拟合，验证/测试集不会因此删行。')
        display_dataframe(pd.DataFrame(calibration_rows), hide_index=True, width='stretch')
    if explanation.coefficientRows:
        display_dataframe(
            pd.DataFrame(explanation.coefficientRows), hide_index=True, width="stretch",
        )
    if explanation.featureImportanceRows:
        display_dataframe(
            pd.DataFrame(explanation.featureImportanceRows), hide_index=True, width="stretch",
        )
        st.caption("特征重要性反映该模型对输入的使用程度，不表示因果关系。")


def _model_library_metrics_for_display(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a display-only summary without changing saved metric precision."""

    column_names = {
        "direction": "预测方向",
        "mae": "平均绝对误差(°)",
        "rmse": "均方根误差(°)",
        "max_abs_error": "最大绝对误差(°)",
        "mean_abs_relative_error_percent": "平均绝对相对误差(%)",
    }
    displayed = metrics.reindex(columns=list(column_names)).copy()
    displayed["direction"] = displayed["direction"].astype(object).replace(
        {"inline": "顺线", "crossline": "横线"}
    )
    for column in list(column_names)[1:]:
        numeric = pd.to_numeric(displayed[column], errors="coerce")
        numeric = numeric.replace([float("inf"), float("-inf")], float("nan"))
        displayed[column] = numeric.map(
            lambda value: f"{value:.2f}" if pd.notna(value) else "—"
        )
    return displayed.rename(columns=column_names)


def _show_model_library(database_sha256: str) -> tuple[ModelRegistry, tuple[ModelVersion, ...]]:
    """Render saved model versions, parameter reuse, and explicit activation."""

    st.subheader("模型版本库")
    registry = get_model_registry(PROJECT_ROOT)
    try:
        active = registry.bootstrap_current_version()
        versions = registry.list_versions()
    except ModelRegistryError as error:
        st.error(f"模型版本库不可用：{error}")
        return registry, ()
    if not versions or active is None:
        st.info("尚无兼容清洗合同的模型。请先运行训练，候选模型不会自动发布。")
        return registry, ()

    flash = st.session_state.pop("model_registry_message", None)
    if flash:
        st.success(flash)
    metadata_by_version = {
        version.version_id: registry.get_metadata(version.version_id)
        for version in versions
    }
    feature_versions = {
        version_id: str(metadata.get("feature_version", FEATURE_VERSION))
        for version_id, metadata in metadata_by_version.items()
    }
    table = pd.DataFrame(
        [
            {
                "版本号": version.version_id,
                "备注": version.note,
                "训练模型": model_combination_label(metadata_by_version[version.version_id]["parameters"]),
                "顺线 epsilon": _parameter_text(version.inline_epsilon),
                "顺线 alpha": _parameter_text(version.inline_alpha),
                "横线 epsilon": _parameter_text(version.crossline_epsilon),
                "横线 alpha": _parameter_text(version.crossline_alpha),
                "max_iter": _parameter_text(version.max_iter),
                "训练比例": version.split_ratio,
                "TabICLv2设置": tabicl_settings_summary(metadata_by_version[version.version_id]),
                "活动版本": "是" if version.version_id == active.version_id else "否",
            }
            for version in versions
        ]
    )
    edited_table = display_data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=[column for column in table.columns if column != "备注"],
        column_config={
            "备注": st.column_config.TextColumn(
                "备注",
                help="可直接修改，最多保存 200 个字符。",
                max_chars=200,
            )
        },
        key="model_version_note_editor",
    )
    st.caption("备注编辑完成后自动保存；刷新页面或重新启动后仍会保留。")
    notes_by_version = {
        str(row["版本号"]): str(row["备注"])
        for row in edited_table.to_dict(orient="records")
    }
    changed_notes = {
        version.version_id: notes_by_version[version.version_id]
        for version in versions
        if notes_by_version[version.version_id] != version.note
    }
    if changed_notes:
        try:
            for version_id, note in changed_notes.items():
                registry.update_note(version_id, note)
        except ModelRegistryError as error:
            st.error(f"备注保存失败：{error}")
        else:
            st.session_state["model_registry_message"] = (
                f"已自动保存 {len(changed_notes)} 个模型版本的备注。"
            )
            st.rerun()
    by_id = {version.version_id: version for version in versions}
    if st.session_state.get("model_library_version") not in by_id:
        st.session_state["model_library_version"] = active.version_id
    selected_id = st.selectbox(
        "选择历史模型版本",
        list(by_id),
        format_func=lambda value: _version_label(by_id[value], metadata_by_version[value]),
        key="model_library_version",
    )
    selected = by_id[selected_id]
    selected_metadata = metadata_by_version[selected_id]
    st.markdown("#### 主要信息")
    try:
        result = TrainingResultArtifacts.load_from_directory(
            registry.version_directory(selected_id)
        )
        display_dataframe(
            _model_library_metrics_for_display(result.selected_test_metrics()),
            hide_index=True,
            width="stretch",
        )
    except TemperaturePipelineError as error:
        st.warning(f"该版本的指标无法展示：{error}")
    _show_model_explanation(selected_metadata)
    if selected.source_sha256 != database_sha256:
        st.warning(
            "该历史模型的数据 SHA-256 与当前传感器数据库不同。"
            "可以复用参数重新训练，但直接激活前请确认数据版本差异。"
        )
    reuse_column, activate_column = st.columns(2)
    with reuse_column:
        compatible_parameters = (
            feature_versions[selected_id] in SUPPORTED_FEATURE_VERSIONS
        )
        if not compatible_parameters:
            st.caption("该版本不兼容当前模型参数合同，不能复用参数。")
        if st.button(
            "使用此版本参数重新训练",
            width="stretch",
            disabled=not compatible_parameters,
        ):
            selected_models = direction_models_from_parameters(registry.get_parameters(selected_id))
            for direction, model in selected_models.items():
                st.session_state[f"training_model_{direction}"] = model
            st.session_state["training_parameter_mode"] = "使用已保存版本参数"
            st.session_state["training_parameter_version"] = selected_id
            st.rerun()
    with activate_column:
        confirmed = st.checkbox(
            f"确认将 {selected_id} 设为活动预测模型",
            key=f"confirm_activate_{selected_id}",
        )
        if st.button(
            "设为活动预测模型",
            disabled=not confirmed or selected_id == active.version_id,
            width="stretch",
        ):
            try:
                registry.activate_version(selected_id)
            except ModelRegistryError as error:
                st.error(f"切换失败，原活动模型保持不变：{error}")
            else:
                st.session_state["model_registry_message"] = (
                    f"已将 {selected_id} 设为活动预测模型。"
                )
                st.rerun()
    return registry, versions


def _tabicl_parameter_controls(saved: dict | None = None) -> dict | None:
    """Manual settings for new training; saved-version settings remain read-only."""
    st.markdown('#### TabICLv2 参数')
    if saved is not None:
        st.number_input('上下文保留比例（%）', min_value=1, max_value=100,
                        value=round(saved.get('context_fraction', 1.)*100), step=1, disabled=True)
        st.checkbox('启用 NLL 温度校准', value=saved.get('temperature_calibration', False), disabled=True)
        st.caption('以上设置来自已保存版本；重新训练时会重新估计温度，已保存模型预测始终使用自己的归档温度。')
        return None
    percentage = st.number_input('上下文保留比例（%）', min_value=1, max_value=100, value=50, step=1,
        key='tabicl_context_percent', help='按训练温度分布选取上下文；每次拟合选样后至少需要300条，不足时请提高比例或增加数据。')
    enabled = st.checkbox('启用 NLL 温度校准', value=False, key='tabicl_temperature_calibration',
        help='训练内部按80%/20%时间划分并净化24小时，自动估计温度；未覆盖档位单独计数，有效校准至少50条且两个档位。保存后预测自动使用该温度。')
    st.caption('这两项由你手动设置，横线不进行Optuna搜索；校准会增加训练耗时。上下文选样不使用2号真值。')
    return {'context_fraction': int(percentage)/100, 'temperature_calibration': bool(enabled)}


def _show_training() -> None:
    """Train drafts, reuse saved parameters, and publish only on confirmation."""

    render_page_header(
        "MODEL LAB",
        "模型训练与状态",
        "检查数据健康、运行训练、评估候选并管理活动模型版本。",
        status="模型工作流",
    )
    database = get_sensor_database(PROJECT_ROOT)
    workflow = RatioTrainingWorkflow(PROJECT_ROOT)
    try:
        snapshot = database.training_snapshot()
        measurements = database.load_cleaned_measurements()
    except (TemperaturePipelineError, SensorDatabaseError) as error:
        st.error(str(error))
        return

    registry, versions = _show_model_library(snapshot.database_sha256)
    st.divider()
    _show_crossline_data_information(measurements)
    st.divider()
    st.subheader("按比例划分并重新训练")
    st.write(
        "顺线和横线使用1号顺线、1号横线和温度训练所选模型，"
        "直接预测2号绝对倾角；2号数据只作为训练标签和评分真值。"
        "训练完成后先成为候选，只有点击保存并激活才会用于预测。"
    )
    model_columns = st.columns(2)
    direction_models = {}
    for column, direction, label in zip(
        model_columns, ("inline", "crossline"), ("顺线训练模型", "横线训练模型"), strict=True
    ):
        direction_models[direction] = column.selectbox(
            label, list(available_direction_models(direction)), format_func=MODEL_LABELS.__getitem__,
            key=f"training_model_{direction}",
        )
    st.caption(model_combination_label({"model": "dual_linear", "directions": {
        direction: {"model": model} for direction, model in direction_models.items()
    }}))
    if "catboost" in direction_models.values():
        st.caption("CatBoost 使用 CPU 决策树，自动搜索树数、深度、学习率和叶节点正则化；"
                   "搜索次数越多，训练耗时越长。")
    if direction_models["crossline"] == "tabpfn_v2":
        st.info("TabPFN v2 使用三个当前输入特征与固定参数，CPU 上最多使用训练段最近 1000 条记录作为上下文；首次使用需要获取 v2 权重。训练后仍需手动保存并激活。")
    if direction_models["crossline"] == "polynomial_ridge":
        st.caption("横线二次项 Ridge 将当前三个输入扩展为9项（原始项、平方项及两两交互项），"
                   "再标准化训练；alpha在 0.000001–10000 范围内自动搜索。")
    uses_probability = direction_models['crossline'] in (probability.MODEL_NAME, compact36.MODEL_NAME, seven.MODEL_NAME, nine.MODEL_NAME, tabicl_single.MODEL_NAME, tabicl_16.MODEL_NAME, tabm_single.MODEL_NAME)
    uses_compact36 = direction_models['crossline'] in (compact36.MODEL_NAME, seven.MODEL_NAME, nine.MODEL_NAME, tabicl_single.MODEL_NAME, tabicl_16.MODEL_NAME, tabm_single.MODEL_NAME)
    uses_single = direction_models['crossline'] in (tabicl_single.MODEL_NAME, tabicl_16.MODEL_NAME, tabm_single.MODEL_NAME)
    if direction_models['crossline'] == nine.MODEL_NAME:
        st.info('九档自动比例：保留两端，先搜索三个区域的固定比例，再点击训练生成候选。比例不是动态门控。')
    elif direction_models['crossline'] == seven.MODEL_NAME:
        st.info('横线七档分段融合：负向端TabICL/TabM为52.5%/47.5%，中间82.5%/17.5%，靠零端90%/10%。'
                '每路先去掉−0.10°和−0.02°概率并归一化，最终只输出−0.09°至−0.03°七档。'
                'TabICL使用训练池75%工况覆盖上下文，TabM使用完整训练池；保留原固定弱风险校正。'
                '比例固定，不自动学习；真实端点仍参与训练和评估，三折验证净化24小时。'
                '按当前数据重新训练，效果须重新评估；选择本选项不会启动训练或切换模型。')
    elif direction_models['crossline'] == tabicl_16.MODEL_NAME:
        st.info('横线 TabICLv2 独立单模型，使用16个集成成员；其余特征和九档选值规则与4成员单模型相同。耗时及内存可能增加；训练后需手动保存并激活。')
    elif direction_models['crossline'] in (tabicl_single.MODEL_NAME, tabm_single.MODEL_NAME):
        st.info('横线独立单模型：只训练所选模型，使用36维因果输入和九档相对误差选值；与融合模型独立归档。训练后须手动保存发布。')
    elif uses_compact36:
        st.info('横线36维组合：TabICLv2概率75%＋TabM概率25%，两个分支均使用36维，按预计相对误差从九档选值。'
                '温度状态30/360分钟；串行加载模型，保留完整输入历史，不使用预测期2号真值。'
                '固定参数，三折验证净化24小时；TabM内部80/20时间划分另净化24小时并按NLL早停。'
                '内部训练段至少200条，数据不足会报错；当前App重新训练的误差须独立评估。')
    elif uses_probability:
        st.info("横线使用原方案：TabICLv2与CatBoost概率各50%，按最小预计相对误差从−0.10至−0.02九档选值。"
                "使用当前及过去3小时输入，不使用预测期2号真值。横线参数固定，三折验证净化24小时，"
                "净化后每折至少300条。按当前数据库重新训练，误差需重新评估。")
    mode_options = ["自动选择模型参数", "使用已保存版本参数"]
    if st.session_state.get("training_parameter_mode") == "自动选择 Huber 参数":
        st.session_state["training_parameter_mode"] = mode_options[0]
    mode = st.radio(
        "模型参数来源",
        mode_options,
        horizontal=True,
        key="training_parameter_mode",
    )
    fixed_parameters = None
    optimization_trials = workflow.config.optimizationTrials
    if mode == "使用已保存版本参数":
        compatible_versions = tuple(
            version
            for version in versions
            if _parameters_match_models(registry.get_metadata(version.version_id), direction_models)
        )
        if not compatible_versions:
            st.error("没有与当前算法组合匹配的历史版本，请改用“自动选择模型参数”。")
            return
        by_id = {version.version_id: version for version in compatible_versions}
        active_id = registry.get_active_version().version_id
        if st.session_state.get("training_parameter_version") not in by_id:
            st.session_state["training_parameter_version"] = (
                active_id if active_id in by_id else next(iter(by_id))
            )
        selected_parameter_id = st.selectbox(
            "参数来源版本",
            list(by_id),
            format_func=lambda value: _version_label(by_id[value], registry.get_metadata(value)),
            key="training_parameter_version",
        )
        fixed_parameters = registry.get_parameters(selected_parameter_id)
        st.info(
            "本次将严格使用所选版本中两个方向的算法参数；"
            "使用当前时间比例重新训练，不搜索其他参数。"
        )
        st.json(fixed_parameters)
    else:
        st.info(("顺线搜索模型参数；横线独立单模型采用本次设置并报告净化时间验证的相对误差。训练完成后仍需手动保存并激活。"
                 if uses_single else "顺线搜索模型参数；横线使用本次单独搜索的九档融合比例。训练完成后仍需手动保存并激活。"
                 if direction_models["crossline"] == nine.MODEL_NAME else "顺线搜索模型参数；横线采用固定概率融合方案并报告净化时间验证的相对误差。训练完成后仍需手动保存并激活。")
                if uses_probability else "顺线和横线分别对所选算法最小化三折时间序列的平均 MAE；训练完成后仍需手动保存并激活。")
        optimization_trials = int(
            st.number_input(
                "Optuna每方向试验次数",
                min_value=MIN_OPTUNA_TRIALS,
                max_value=MAX_OPTUNA_TRIALS,
                value=100,
                step=OPTUNA_TRIAL_STEP,
                help=(
                    ("仅顺线运行该次数；横线固定方案只验证1组。" if uses_probability else "顺线和横线各运行该次数；")
                    + f"允许范围 {MIN_OPTUNA_TRIALS}-"
                    f"{MAX_OPTUNA_TRIALS} 次，按 {OPTUNA_TRIAL_STEP} 次递增。"
                ),
                key="optimization_trials_per_direction_v2",
            )
        )
        st.caption((f"当前顺线搜索 {optimization_trials} 次；横线使用搜索所得融合比例。" if direction_models["crossline"] == nine.MODEL_NAME else f"当前顺线搜索 {optimization_trials} 次；横线按本次设置验证1组参数，不重复进行Optuna搜索。")
                   if uses_probability else f"当前将运行顺线 {optimization_trials} 次 + 横线 {optimization_trials} 次，共 {optimization_trials * 2} 次。")

    tabicl_options = None
    if direction_models['crossline'] in (tabicl_single.MODEL_NAME, tabicl_16.MODEL_NAME):
        tabicl_options = _tabicl_parameter_controls(
            fixed_parameters['directions']['crossline'] if fixed_parameters is not None else None)

    training_strategy = st.radio(
        "比例训练方式",
        ["单一比例训练", "按测试比例训练", "自动比较多个比例"],
        horizontal=True,
        key="training_ratio_strategy",
    )
    automatic_ratio_count: int | None = None
    train_test_percentage: TrainTestPercentage | None = None
    auto_ratios: tuple[SplitRatio, ...] = ()
    labels = {"train": "训练集", "validation": "验证集", "test": "测试集"}
    if training_strategy == "自动比较多个比例":
        automatic_ratio_count = int(
            st.number_input(
                "自动比较的比例数量",
                min_value=MIN_AUTO_RATIO_COUNT,
                max_value=MAX_AUTO_RATIO_COUNT,
                value=20,
                step=1,
                help=(
                    "系统固定最后 10% 为同一测试集（行数向上取整），并在 60% 到 89% 的"
                    "训练集占比间生成均匀方案，比例权重保留一位小数。"
                    "默认比较 20 组，可在 2–20 组之间调整。"
                ),
                key="automatic_ratio_count_v2",
            )
        )
        try:
            auto_ratios = automatic_split_ratios(
                automatic_ratio_count,
                maximum_train_weight=FULL_AUTO_MAX_TRAIN_WEIGHT,
            )
            preview_rows = []
            for candidate_ratio in auto_ratios:
                preview = workflow.preview(
                    snapshot.frame, candidate_ratio, shared_test_holdout=True
                )
                partitions = preview["partitions"]
                preview_rows.append(
                    {
                        "比例": candidate_ratio.label,
                        "训练样本": partitions["train"]["rows"],
                        "验证样本": partitions["validation"]["rows"],
                        "测试样本": partitions["test"]["rows"],
                        "测试开始时间": partitions["test"]["timestamp_min"],
                    }
                )
        except TemperaturePipelineError as error:
            st.error(str(error))
            return
        ratio = auto_ratios[-1]
        display_dataframe(pd.DataFrame(preview_rows), hide_index=True, width="stretch")
        st.info(
            "所有方案共享最后 10% 的测试集。系统只以验证集横线平均绝对"
            "相对误差选择比例；测试集指标仅用于最终报告，不参与选择。"
        )
        st.caption(
            "最高比例为 8.9:0.1:1（89%训练、1%验证、10%测试）。"
            "验证集较小，评估更容易波动；增加搜索量不保证误差下降。"
            "最终模型仍合并训练集和验证集拟合，约使用90%的数据。"
        )
        selected_ratio_label = ratio.label
    elif training_strategy == "按测试比例训练":
        percentage_column, _ = st.columns([3, 1])
        with percentage_column:
            test_percentage = int(
                st.slider(
                    "测试集比例",
                    min_value=MIN_TEST_PERCENTAGE,
                    max_value=MAX_TEST_PERCENTAGE,
                    value=DEFAULT_TEST_PERCENTAGE,
                    step=1,
                    format="%d%%",
                    key="training_test_percentage",
                    help=(
                        "数据按 timestamp 升序排列；前段用于训练，"
                        "最后这一比例只用于最终测试。"
                    ),
                )
            )
        train_test_percentage = TrainTestPercentage(test_percentage)
        selected_ratio_label = train_test_percentage.label
        try:
            preview = workflow.preview_train_test(
                snapshot.frame,
                train_test_percentage,
            )
        except TemperaturePipelineError as error:
            st.error(str(error))
            return
        preview_rows = [
            {
                "数据集": labels[name],
                "样本数": partition["rows"],
                "开始时间": partition["timestamp_min"],
                "结束时间": partition["timestamp_max"],
            }
            for name, partition in preview["partitions"].items()
        ]
        display_dataframe(pd.DataFrame(preview_rows), hide_index=True, width="stretch")
        st.info(
            f"将按时间使用前 {train_test_percentage.train_percent}% 数据训练，"
            f"后 {train_test_percentage.test_percent}% 数据测试。"
            "自动选参只在训练段内部进行三折时间序列验证，测试段不参与选参或拟合。"
        )
    else:
        preset_options = ["8:1:1", "6:2:2", "7:2:1", "自定义"]
        default_ratio = "8:1:1"
        preset = st.selectbox(
            "训练集 : 验证集 : 测试集",
            preset_options,
            index=preset_options.index(default_ratio),
            help=(
                "比例支持一位小数，三项之和必须等于10；"
                "例如 6.5:2:1.5 等于 65% / 20% / 15%。"
            ),
            key="training_split_ratio_v3",
        )
        if preset == "自定义":
            train_column, validation_column, test_column = st.columns(3)
            train_weight = float(
                train_column.number_input(
                    "训练集权重",
                    min_value=0.1,
                    max_value=9.8,
                    value=6.0,
                    step=0.1,
                    format="%.1f",
                )
            )
            validation_weight = float(
                validation_column.number_input(
                    "验证集权重",
                    min_value=0.1,
                    max_value=9.8,
                    value=2.0,
                    step=0.1,
                    format="%.1f",
                )
            )
            test_weight = float(
                test_column.number_input(
                    "测试集权重",
                    min_value=0.1,
                    max_value=9.8,
                    value=2.0,
                    step=0.1,
                    format="%.1f",
                )
            )
            st.caption(
                "当前比例合计："
                f"{train_weight + validation_weight + test_weight:.1f} / 10.0"
            )
            ratio = SplitRatio(train_weight, validation_weight, test_weight)
        else:
            weights = (int(value) for value in preset.split(":"))
            ratio = SplitRatio(*weights)

        try:
            preview = workflow.preview(snapshot.frame, ratio)
        except TemperaturePipelineError as error:
            st.error(str(error))
            return
        preview_rows = [
            {
                "数据集": labels[name],
                "样本数": partition["rows"],
                "开始时间": partition["timestamp_min"],
                "结束时间": partition["timestamp_max"],
            }
            for name, partition in preview["partitions"].items()
        ]
        display_dataframe(pd.DataFrame(preview_rows), hide_index=True, width="stretch")
        selected_ratio_label = ratio.label
    st.caption(
        f"当前可训练 {len(snapshot.frame)} 条；"
        f"因缺少2号真值排除 {snapshot.excluded_missing_targets} 条。"
    )
    st.warning(
        "测试集只应用于最终评估。"
        "如果根据测试误差反复切换比例，它就不再是一次性盲测。"
    )
    fusion_search = None
    if direction_models['crossline'] == nine.MODEL_NAME:
        st.markdown('#### 九档融合比例搜索')
        if automatic_ratio_count is not None:
            st.warning('九档比例搜索暂不支持自动比较多个划分，请切换为单一比例或按测试比例。')
        elif fixed_parameters is not None:
            st.warning('九档自动比例请切换为自动选择模型参数并搜索当前数据。')
        else:
            selected_split = train_test_percentage if train_test_percentage is not None else ratio
            strategy = 'train_test' if train_test_percentage is not None else 'ratio'
            binding = regional_search.fingerprint(snapshot.frame, measurements, snapshot.database_sha256,
                                                   workflow._cleaning_policy_sha256(), strategy, selected_split.label)
            job = st.session_state.get('nine_regional_search_job')
            state = job.snapshot() if job is not None else None
            running = state is not None and state['status'] == 'running'
            if st.button('搜索融合比例', disabled=running, key='nine_search_start'):
                st.session_state.pop('nine_regional_search_result', None)
                job = RegionalSearchJob(workflow, snapshot.frame.copy(deep=True), measurements.copy(deep=True),
                                        snapshot.database_sha256, strategy, selected_split, binding)
                st.session_state['nine_regional_search_job'] = job
                st.rerun()
            if state is not None:
                if running:
                    st.progress(min(1.0, state['done']/max(1,state['total'])),
                                text=f"{state['phase']}：{state['done']} / {state['total']}")
                    a,b = st.columns(2)
                    if a.button('刷新搜索进度', key='nine_search_refresh'): st.rerun()
                    if b.button('取消搜索', key='nine_search_cancel'):
                        job.cancel()
                        st.info('已请求取消；当前模型训练阶段结束后停止。')
                elif state['status'] == 'complete':
                    if job.fingerprint == binding:
                        st.session_state['nine_regional_search_result'] = state['result']
                    else:
                        st.warning('搜索完成，但数据或划分已变化，请重新搜索。')
                elif state['status'] in ('failed','cancelled'):
                    st.warning(state['message'])
            result = st.session_state.get('nine_regional_search_result')
            if result is not None and result.get('fingerprint') == binding and not running:
                fusion_search = result
                st.caption('以下为训练期选参成绩，测试集未参与搜索；不是独立测试成绩。')
                st.dataframe(pd.DataFrame([{'区域': label, 'TabICL(%)': w*100, 'TabM(%)': (1-w)*100}
                    for label,w in zip(('负向段','中间段','靠零段'), result['weights'], strict=True)]),
                    hide_index=True, width='stretch')
                report=[]
                for entry in result['folds']:
                    for group in ('baseline','selected'):
                        m=entry[group];report.append({'窗口': entry['fold'], '方案': '基线' if group=='baseline' else '搜索最优',
                            'MARE(%)': m['mare_percent'], '有效数': m['relative_valid'],
                            '零值排除数': m['zero_excluded'], '跨区间错误': m['cross_region_errors']})
                for group in ('baseline','selected'):
                    m=result[group];report.append({'窗口':'合并','方案':'基线' if group=='baseline' else '搜索最优',
                        'MARE(%)':m['mare_percent'],'有效数':m['relative_valid'],
                        '零值排除数':m['zero_excluded'],'跨区间错误':m['cross_region_errors']})
                st.dataframe(pd.DataFrame(report), hide_index=True, width='stretch')
            elif result is not None and result.get('fingerprint') != binding:
                st.warning('数据、清洗配置或划分已改变，请重新搜索。')

    if automatic_ratio_count is not None:
        button_label = (
            f"自动比较 {automatic_ratio_count} 组比例并保留最佳候选"
        )
        if fixed_parameters is None:
            st.caption(
                (f"本次比较 {automatic_ratio_count} 组比例，每组顺线搜索 {optimization_trials} 次，横线验证固定概率方案。"
                if uses_probability else f"本次将运行 {automatic_ratio_count} 组比例，每组顺线和横线各 "
                f"{optimization_trials} 次 Optuna 搜索，共 "
                f"{automatic_ratio_count * optimization_trials * 2} 次。")
            )
    elif train_test_percentage is not None:
        button_label = (
            f"按前 {train_test_percentage.train_percent}% 训练、"
            f"后 {train_test_percentage.test_percent}% 测试"
        )
    else:
        button_label = (
            f"运行双方向 Optuna 各 {optimization_trials} 次搜索并训练"
            if fixed_parameters is None
            else "按当前比例使用已保存参数训练"
        )
        if uses_probability and fixed_parameters is None:
            button_label = (f"运行顺线 {optimization_trials} 次搜索＋横线单模型训练" if uses_single else f"运行顺线 {optimization_trials} 次搜索＋九档融合训练" if direction_models["crossline"] == nine.MODEL_NAME else f"运行顺线 {optimization_trials} 次搜索＋横线固定概率融合训练")
    if st.button(button_label, type="primary", disabled=(direction_models["crossline"] == nine.MODEL_NAME and fusion_search is None)):
        try:
            if automatic_ratio_count is not None:
                spinner_message = (
                    f"正在比较 {automatic_ratio_count} 组时间比例；"
                    "只会保留验证集最优的训练候选……"
                )
            else:
                spinner_message = (
                    "正在运行顺线、横线各 "
                    f"{optimization_trials} 次搜索和时间留出评估……"
                    if fixed_parameters is None
                    else "正在使用历史参数训练并运行时间留出评估……"
                )
                if uses_probability and fixed_parameters is None:
                    spinner_message = ('正在搜索顺线参数、使用搜索比例训练九档模型并进行时间留出评估……' if direction_models['crossline'] == nine.MODEL_NAME else '正在搜索顺线参数、验证横线固定概率方案并进行时间留出评估……')
            with st.spinner(spinner_message):
                if automatic_ratio_count is not None:
                    comparison = workflow.train_multiple_ratios(
                        snapshot.frame,
                        ratio_count=automatic_ratio_count,
                        maximum_train_weight=FULL_AUTO_MAX_TRAIN_WEIGHT,
                        database_sha256=snapshot.database_sha256,
                        database_rows=snapshot.database_rows,
                        excluded_missing_targets=snapshot.excluded_missing_targets,
                        direction_models=direction_models,
                        **({'feature_context': measurements} if uses_compact36 or current_processing_version() in ('final_v2', 'final_v3') else {}),
                        fixed_parameters=fixed_parameters,
                        **({"tabicl_options": tabicl_options} if tabicl_options is not None else {}),
                        optimization_trials=(
                            optimization_trials if fixed_parameters is None else None
                        ),
                    )
                    metadata = comparison["winner"]
                    st.session_state["latest_ratio_comparison"] = comparison[
                        "rankings"
                    ]
                elif train_test_percentage is not None:
                    metadata = workflow.train_train_test(
                        snapshot.frame,
                        train_test_percentage,
                        snapshot.database_sha256,
                        snapshot.database_rows,
                        snapshot.excluded_missing_targets,
                        direction_models=direction_models,
                        **({'feature_context': measurements} if uses_compact36 or current_processing_version() in ('final_v2', 'final_v3') else {}),
                        fixed_parameters=fixed_parameters,
                        **({"tabicl_options": tabicl_options} if tabicl_options is not None else {}),
                        **({"fusion_search": fusion_search} if direction_models["crossline"] == nine.MODEL_NAME else {}),
                        optimization_trials=(
                            optimization_trials if fixed_parameters is None else None
                        ),
                    )
                    st.session_state.pop("latest_ratio_comparison", None)
                else:
                    metadata = workflow.train(
                        snapshot.frame,
                        ratio,
                        snapshot.database_sha256,
                        snapshot.database_rows,
                        snapshot.excluded_missing_targets,
                        direction_models=direction_models,
                        **({'feature_context': measurements} if uses_compact36 or current_processing_version() in ('final_v2', 'final_v3') else {}),
                        fixed_parameters=fixed_parameters,
                        **({"tabicl_options": tabicl_options} if tabicl_options is not None else {}),
                        **({"fusion_search": fusion_search} if direction_models["crossline"] == nine.MODEL_NAME else {}),
                        optimization_trials=(
                            optimization_trials if fixed_parameters is None else None
                        ),
                    )
                    st.session_state.pop("latest_ratio_comparison", None)
        except TemperaturePipelineError as error:
            st.error(str(error))
        except ModelRegistryError as error:
            st.error(f"自动比较的候选清理失败：{error}")
        else:
            st.session_state["latest_training_run_id"] = metadata["run_id"]
            st.session_state["latest_training_ratio_label"] = metadata["split"][
                "requested_ratio"
            ]["label"]
            st.session_state.pop("latest_training_ratio", None)
            st.success(
                f"{model_combination_label(metadata['parameters'])} 候选训练完成，比例 "
                f"{metadata['split']['requested_ratio']['label']}，"
                f"状态：{metadata['status']}。"
                "当前预测模型尚未改变。"
            )

    comparison_rows = st.session_state.get("latest_ratio_comparison")
    if comparison_rows:
        st.markdown("#### 自动比例比较排名")
        st.caption(
            "排名仅按验证集横线平均绝对相对误差，相同时依次按横线 MAE "
            "和验证样本数决定；测试集指标仅供最终报告。"
            "未选中的临时候选已删除。"
        )
        comparison_frame = format_metric_table_for_display(
            pd.DataFrame(comparison_rows)
        ).rename(
            columns={
                "rank": "排名",
                "ratio": "比例",
                "validation_crossline_mean_abs_relative_error_percent": (
                    "验证横线平均绝对相对误差(%)"
                ),
                "validation_crossline_relative_error_valid_count": (
                    "验证横线相对误差有效数"
                ),
                "validation_crossline_relative_error_excluded_zero_count": (
                    "验证横线零真值排除数"
                ),
                "validation_sample_count": "验证样本数",
                "validation_inline_mae": "验证顺线MAE(°)",
                "validation_crossline_mae": "验证横线MAE(°)",
                "test_crossline_mean_abs_relative_error_percent": (
                    "测试横线平均绝对相对误差(%)-仅报告"
                ),
                "test_crossline_relative_error_valid_count": (
                    "测试横线相对误差有效数-仅报告"
                ),
                "test_crossline_relative_error_excluded_zero_count": (
                    "测试横线零真值排除数-仅报告"
                ),
                "test_inline_mae": "测试顺线MAE(°)",
                "test_crossline_mae": "测试横线MAE(°)",
                "selected": "已保留",
            }
        )
        display_dataframe(
            comparison_frame,
            hide_index=True,
            width="stretch",
        )

    run_id = st.session_state.get("latest_training_run_id")
    candidate_label: str | None = None
    if not run_id:
        persistent_drafts = registry.list_drafts()
        compatible_draft = _latest_current_huber_draft(registry, persistent_drafts, include_linear=True)
        if compatible_draft is not None:
            run_id = compatible_draft.run_id
            candidate_label = draft_version_label(str(run_id), persistent_drafts)
            st.info(f"已加载最近持久化训练候选：{candidate_label}")
    if run_id:
        display_ratio_label = selected_ratio_label
        stored_ratio_label = st.session_state.get("latest_training_ratio_label")
        if isinstance(stored_ratio_label, str) and stored_ratio_label:
            display_ratio_label = stored_ratio_label
        if candidate_label is None:
            candidate_label = draft_version_label(
                str(run_id),
                registry.list_drafts(),
            )
        try:
            draft_directory = registry.draft_directory(str(run_id))
            draft_result = TrainingResultArtifacts.load_from_directory(draft_directory)
        except (ModelRegistryError, TemperaturePipelineError) as error:
            st.warning(f"训练候选无法读取：{error}")
        else:
            if not _is_current_linear_candidate(draft_result.metadata):
                st.warning(
                    "该候选属于旧版模型合同，已保留审计记录，"
                    "但不能在当前三特征训练页保存或激活。"
                )
                return
            _show_latest_training_results(
                display_ratio_label,
                draft_directory,
                candidate_label,
            )
            st.markdown("#### 保存训练候选")
            note = st.text_input(
                "版本备注（可选；留空将自动保存训练比例）",
                max_chars=200,
                key=f"draft_note_{run_id}",
            )
            if st.button("保存并设为活动模型", type="primary"):
                try:
                    saved = registry.save_draft(str(run_id), note=note, activate=True)
                except ModelRegistryError as error:
                    st.error(f"保存失败，原活动模型保持不变：{error}")
                else:
                    st.session_state.pop("latest_training_run_id", None)
                    st.session_state.pop("latest_ratio_comparison", None)
                    st.session_state.pop("latest_training_ratio", None)
                    st.session_state.pop("latest_training_ratio_label", None)
                    st.session_state["model_registry_message"] = (
                        f"候选已保存为 {saved.version_id}，并设为活动预测模型。"
                    )
                    st.rerun()
    else:
        _show_latest_training_results(selected_ratio_label)


def _show_prediction(
    *,
    compact_model_labels: bool = False,
    show_audit_details: bool = True,
    allow_database_prediction: bool = True,
) -> None:
    """Predict an upload directly or persist it before bounded prediction."""

    render_page_header(
        "PREDICTION",
        "上传新数据预测",
        "按上传、字段映射、参数、执行和结果评估的顺序完成一次预测。",
        status="等待上传",
    )
    try:
        registry = get_model_registry(PROJECT_ROOT)
        active_version = registry.bootstrap_current_version()
        versions = registry.list_versions()
    except ModelRegistryError as error:
        st.error(f"模型版本库不可用：{error}")
        return
    if not versions or active_version is None:
        st.info("尚无兼容清洗合同的模型，请先训练候选模型并手动发布。")
        return

    versions_by_id = {version.version_id: version for version in versions}
    if st.session_state.get("upload_prediction_model_version") not in versions_by_id:
        st.session_state["upload_prediction_model_version"] = active_version.version_id
    selected_version = st.selectbox(
        "预测模型版本",
        list(versions_by_id),
        format_func=lambda value: _prediction_model_label(
            versions_by_id[value],
            active_version.version_id,
            compact_model_labels=compact_model_labels,
            metadata=None if compact_model_labels else registry.get_metadata(value),
        ),
        key="upload_prediction_model_version",
        help="只影响本次预测，不会切换当前活动模型。",
    )
    st.caption(f"本次使用 {selected_version}；不会切换当前活动模型。")
    try:
        selected_metadata = registry.get_metadata(selected_version)
    except ModelRegistryError as error:
        st.error(f"模型版本 {selected_version} 无法读取：{error}")
        return
    if selected_metadata.get("feature_version") not in SUPPORTED_FEATURE_VERSIONS:
        st.error("所选版本不兼容当前预测合同，请重新训练并发布。")
        return
    compare_crossline_branches = False
    if selected_metadata.get("feature_version") == compact36.FEATURE_VERSION:
        compare_crossline_branches = st.checkbox(
            "对照横线：仅 TabICLv2、仅 TabM 与融合模型",
            key="upload_compare_compact36_branches",
            help="使用所选模型版本中已保存的两个分支，同一批数据计算三种结果；不重新训练。",
        )
        if compare_crossline_branches:
            st.caption("三个结果使用相同清洗数据、历史上下文和选档规则；这是模型分支对照，不是新训练的独立模型。")
    st.caption(
        "必须提供时间、四项倾角和温度，与数据入库的校验规则一致。"
        "2号两方向仅用于事后误差评估，不参与预测输入。" if current_processing_version() == "final_v3" else
        "必须提供时间、1号顺线、1号横线和温度。2号两方向可选，仅用于事后误差评估。"
    )
    if is_final_processing() and current_processing_version() != "final_v3":
        st.caption(
            "直接上传预测按当天走势修复明显尖峰或凸起，用两侧正常温度平滑衔接；正常起伏保留。"
            "预测表和下载中的温度是实际送入模型的处理值；原值及修正记录另列。"
        )
    upload = st.file_uploader(
        "选择 CSV 或 XLSX",
        type=_prediction_upload_file_types(),
        help=(
            "请选择 CSV 或 XLSX。为兼容 macOS 桌面版，文件选择框可能显示所有文件；"
            "提交后系统仍会校验文件格式和内容。"
        ),
    )
    if upload is None:
        return
    content = upload.getvalue()
    reader = UploadTableReader()
    try:
        sheets = reader.list_sheets(upload.name, content)
        sheet = st.selectbox("工作表", sheets) if sheets else None
        table = reader.read(upload.name, content, sheet)
    except TemperaturePipelineError as error:
        st.error(str(error))
        return

    columns = [str(column) for column in table.frame.columns]
    suggestions = suggest_mapping(columns)
    timestamp = _select_column(
        "时间列", columns, suggestions["timestamp"], key="timestamp"
    )
    tilt1Inline = _select_column(
        "1号顺线列",
        columns,
        suggestions["tilt1_inline_model"],
        key="tilt1_inline",
    )
    tilt1Crossline = _select_column(
        "1号横线列",
        columns,
        suggestions["tilt1_crossline_model"],
        key="tilt1_crossline",
    )
    temperature = _select_column(
        "环境温度列",
        columns,
        suggestions["temperature_celsius"],
        key="temperature",
    )
    tilt2Inline = _select_column(
        "2号顺线真值列" if current_processing_version() == "final_v3" else "2号顺线真值列（可选）",
        columns,
        suggestions["tilt2_inline"],
        key="tilt2_inline",
        optional=current_processing_version() != "final_v3",
    )
    tilt2Crossline = _select_column(
        "2号横线真值列" if current_processing_version() == "final_v3" else "2号横线真值列（可选）",
        columns,
        suggestions["tilt2_crossline"],
        key="tilt2_crossline",
        optional=current_processing_version() != "final_v3",
    )
    prediction_mode = _select_prediction_mode(
        allow_database_prediction=allow_database_prediction,
    )
    prediction_percentage = 100
    if prediction_mode == DATABASE_PREDICTION_MODE:
        prediction_percentage = st.slider(
            "入库后预测比例",
            min_value=1,
            max_value=100,
            value=100,
            step=1,
            format="%d%%",
            help="完成入库后，按时间选择当前传感器数据库末尾 N% 的记录执行预测。",
            key="upload_database_prediction_percentage",
        )
        st.caption(
            "数据会先写入传感器数据库，再按时间选择数据库末尾"
            f" {prediction_percentage}% 的记录预测。"
        )
        button_label = "保存到传感器数据库并按比例预测"
    else:
        if allow_database_prediction:
            st.caption("只在内存中执行本次上传预测，不写入传感器数据库。")
        button_label = "直接执行预测"

    prediction_temperature_check = False
    if current_processing_version() == "final_v3":
        prediction_temperature_check = True
        st.caption("温度自动执行异常检测与修复；待判断记录保留原值，不参与预测。")

    if not st.button(button_label, type="primary"):
        return
    try:
        mapping = UploadColumnMapping(
            timestamp=str(timestamp),
            tilt1Inline=str(tilt1Inline),
            tilt1Crossline=str(tilt1Crossline),
            temperature=str(temperature),
            tilt2Inline=tilt2Inline,
            tilt2Crossline=tilt2Crossline,
        )
        normalized = UploadNormalizer().normalize(table, mapping)
    except TemperaturePipelineError as error:
        st.error(str(error))
        return

    if current_processing_version() == "final_v3":
        normalized.audit["temperature_jump_check_enabled"] = prediction_temperature_check
    prediction_upload = normalized
    import_result = None
    if prediction_mode == DATABASE_PREDICTION_MODE:
        database = get_sensor_database(PROJECT_ROOT)
        try:
            import_result = database.import_upload(
                normalized, content,
                **({"temperature_jump_check_enabled": prediction_temperature_check,
                    "temperature_rule_version": PERSONAL_FIXED_TEMPERATURE_RULE_VERSION}
                   if current_processing_version() == "final_v3" else {}),
            )
            measurements = database.load_cleaned_measurements()
            prediction_upload = build_database_prediction_upload(
                measurements,
                prediction_percentage,
            )
        except (TemperaturePipelineError, SensorDatabaseError) as error:
            st.error(f"入库或构建预测范围失败：{error}")
            return
        if import_result.duplicate:
            st.info("文件内容已入库，本次不重复新增导入批次。")
        else:
            st.success(
                "传感器数据库已更新："
                f"新增 {import_result.inserted_rows} 条，"
                f"更新 {import_result.updated_rows} 条，"
                f"未变 {import_result.unchanged_rows} 条，"
                f"当前共 {import_result.database_rows} 条。"
            )
    try:
        result = InteractivePredictionService(PROJECT_ROOT).predict(
            prediction_upload,
            modelVersion=str(selected_version),
            compareCrosslineBranches=compare_crossline_branches,
        )
    except TemperaturePipelineError as error:
        if import_result is None:
            st.error(f"直接预测失败：{error}")
        else:
            st.warning(f"数据已入库，但预测失败：{error}")
        return

    if import_result is None:
        st.success(
            f"已使用 {selected_version} 生成 {len(result.predictions)} 个10分钟时间桶预测；"
            "本次数据未写入传感器数据库。"
        )
        audit = {**result.audit, "persistence_mode": "in_memory_only"}
    else:
        st.success(
            f"已使用 {selected_version} 预测数据库末尾 {prediction_percentage}%："
            f"共 {len(result.predictions)} 条。"
        )
        audit = {
            **result.audit,
            "persistence_mode": "saved_then_database_percentage",
            "csv_database": import_result.to_dict(),
        }
    skipped_rows = int(result.audit.get("cleaning_skipped_rows", 0))
    if skipped_rows:
        st.warning(
            "上传数据已按统一清洗规则处理；"
            f"跳过 {skipped_rows} 个无法安全用于预测的10分钟桶，"
            f"已对其余 {len(result.predictions)} 个桶完成预测。"
        )
    _render_prediction_audit(audit, visible=show_audit_details)
    _show_prediction_temperature_summary(result.audit)
    if result.comparisonMetrics is not None:
        st.subheader("横线模型分支对照")
        st.caption("同一批记录的事后评估；如果包含模型训练期数据，不能视为独立测试。")
        if result.comparisonMetrics.empty:
            st.info("缺少可评估的2号横线真值，已展示三种预测值，无法计算对照误差。")
        else:
            comparison = result.comparisonMetrics.rename(columns={
                "method": "方法", "sample_count": "评估条数", "mae": "MAE (°)",
                "rmse": "RMSE (°)", "max_abs_error": "最大绝对误差 (°)",
                "mean_abs_relative_error_percent": "平均绝对相对误差 (%)",
            }).copy()
            comparison["方法"] = comparison["方法"].map({
                "tabicl": "仅 TabICLv2", "tabm": "仅 TabM", "ensemble": "融合 75% / 25%",
            })
            columns = ["方法", "评估条数", "MAE (°)", "RMSE (°)", "最大绝对误差 (°)", "平均绝对相对误差 (%)"]
            display_dataframe(comparison[columns], width="stretch", hide_index=True)
    ordered_predictions = build_upload_prediction_display_frame(
        order_prediction_columns(result.predictions)
    )
    st.caption("先展示顺线与横线的“真值 → 预测值”对照，再展示各方向误差，便于逐行比对。")
    display_dataframe(
        ordered_predictions.rename(columns=PREDICTION_DISPLAY_NAMES),
        column_config=prediction_column_config(
            ordered_predictions.rename(columns=PREDICTION_DISPLAY_NAMES)
        ),
        width="stretch",
        hide_index=True,
    )
    st.line_chart(
        result.predictions.loc[
            :, ["timestamp", "predicted_inline", "predicted_crossline"]
        ],
        x="timestamp",
    )
    if not result.metrics.empty:
        _show_prediction_metric_cards(
            result.metrics,
            heading="本次预测指标",
            caveat=(
                "只使用本次上传中同时具有2号传感器真值且模型成功输出的行计算；"
                "这是本次文件的事后评估，不代表独立测试集成绩。"
            ),
        )
        with st.expander("查看指标明细表"):
            display_dataframe(
                format_metric_table_for_display(result.metrics),
                width="stretch",
                hide_index=True,
            )
        st.line_chart(
            result.predictions.loc[
                :,
                [
                    "timestamp",
                    "absolute_error_inline",
                    "absolute_error_crossline",
                ],
            ],
            x="timestamp",
        )
        st.line_chart(
            result.predictions.loc[
                :,
                [
                    "timestamp",
                    "absolute_relative_error_inline_percent",
                    "absolute_relative_error_crossline_percent",
                ],
            ],
            x="timestamp",
        )
    st.download_button(
        "下载预测 CSV",
        prediction_csv_bytes(ordered_predictions),
        file_name="CSG124_temperature_predictions.csv",
        mime="text/csv",
    )
    if not normalized.rejectedRows.empty:
        st.warning(f"已拒绝 {len(normalized.rejectedRows)} 行缺温度或缺倾角的输入。")
        display_dataframe(normalized.rejectedRows, width="stretch")


def _show_prediction_temperature_summary(audit: dict[str, object]) -> None:
    """展示上传预测实际执行的CUSUM及模型使用的处理后温度。"""
    if "cleaning_daily_checked_rows" in audit:
        st.markdown("#### 本次预测的温度处理")
        checked, corrected, maximum = st.columns(3)
        checked.metric("当天趋势已检查", int(audit["cleaning_daily_checked_rows"]))
        corrected.metric("温度异常段修复", int(audit["cleaning_daily_corrected_rows"]))
        maximum.metric("最大修正幅度", f"{float(audit['cleaning_daily_max_abs_correction']):.2f}℃")
        st.caption("仅修复有可靠进入和恢复边界的明显异常段，两侧正常温度平滑衔接；正常波动保留，修复值属于估计值。")
        records = audit.get("cleaning_daily_temperature_rows", [])
        if records:
            table = pd.DataFrame(records).loc[:, [
                "timestamp", "temperature_raw_celsius", "temperature_celsius",
                "temperature_daily_trend_celsius", "temperature_daily_offset_celsius",
            ]].rename(columns={
                "timestamp": "时间", "temperature_raw_celsius": "原温度℃", "temperature_celsius": "处理后温度℃",
                "temperature_daily_trend_celsius": "当天趋势℃", "temperature_daily_offset_celsius": "修正扣除℃",
            })
            with st.expander("查看当天温度趋势修正与原值"):
                display_dataframe(table, width="stretch", hide_index=True)
        elif not int(audit["cleaning_daily_checked_rows"]):
            st.info("当天连续观测不足，保留原温度；缺失、物理越界和零温度仍按既有规则处理。")
        return
    if "cleaning_cusum_checked_rows" not in audit:
        return
    st.markdown("#### 本次预测的温度处理")
    checked, applied, pending = st.columns(3)
    checked.metric("CUSUM已检查", int(audit["cleaning_cusum_checked_rows"]))
    applied.metric("CUSUM温度已修正", int(audit["cleaning_cusum_applied_rows"]))
    pending.metric("CUSUM待判断", int(audit["cleaning_cusum_review_rows"]))
    st.caption("温度修复使用前后文估计；待判断的桶不进入本次预测，原始文件和已入库数据保持不变。")
    if not int(audit["cleaning_cusum_checked_rows"]):
        st.info("本次没有可执行CUSUM的温度桶：可能不在12:00—18:00检查窗，或历史/上午校准数据不足。原跳变处理仍执行。")
    records = audit.get("cleaning_cusum_temperature_rows", [])
    if not records:
        return
    table = pd.DataFrame(records).loc[:, [
        "timestamp", "temperature_raw_celsius", "temperature_celsius",
        "temperature_cusum_reference_celsius", "temperature_cusum_offset_celsius",
        "temperature_cusum_candidate_applied", "temperature_review_required",
    ]].rename(columns={
        "timestamp": "时间", "temperature_raw_celsius": "原温度℃",
        "temperature_celsius": "处理后温度℃", "temperature_cusum_reference_celsius": "参考温度℃",
        "temperature_cusum_offset_celsius": "候选扣除℃",
        "temperature_cusum_candidate_applied": "候选已采用", "temperature_review_required": "待判断",
    })
    with st.expander("查看本次预测的温度修正与原值"):
        display_dataframe(table, width="stretch", hide_index=True)


def _show_database() -> None:
    """Show the CSV-backed measurement table and append-only import log."""

    st.subheader("传感器数据库")
    database = get_sensor_database(PROJECT_ROOT)
    try:
        status = database.status()
        measurements = database.load_cleaned_measurements()
        import_log = database.load_import_log()
    except TemperaturePipelineError as error:
        st.error(str(error))
        return

    first, second, third = st.columns(3)
    first.metric("测量记录", status.measurement_rows)
    second.metric("导入次数", status.import_count)
    third.metric("最新时间", status.timestamp_max or "暂无")
    st.caption("主键为 gateway + timestamp；重复数据不重复累加，新值会更新原时间桶。")

    if measurements.empty:
        st.info("尚无入库数据。")
    else:
        display_dataframe(measurements.tail(100), width="stretch")
        st.download_button(
            "下载测量数据库",
            measurements.to_csv(index=False).encode("utf-8-sig"),
            file_name="CSG124_measurements.csv",
            mime="text/csv",
        )

    st.subheader("导入日志")
    if import_log.empty:
        st.info("尚无导入日志。")
    else:
        display_dataframe(import_log.tail(100), width="stretch")
        st.download_button(
            "下载导入日志",
            import_log.to_csv(index=False).encode("utf-8-sig"),
            file_name="import_log.csv",
            mime="text/csv",
        )


def main() -> None:
    """Render the complete single-gateway application."""

    st.set_page_config(page_title="CSG124温度倾角预测", layout="wide")
    st.title("CSG124 倾角与温度预测")
    st.info(
        "当前只开放 CSG124。CSG421 没有新的有效数据，未训练温度模型。"
    )
    _show_model_status()
    trainingTab, predictionTab, databaseTab = st.tabs(
        ["模型训练与状态", "上传预测", "传感器数据库"]
    )
    with trainingTab:
        _show_training()
    with predictionTab:
        _show_prediction()
    with databaseTab:
        _show_database()


def show_model_status() -> None:
    """Render the migrated model identity in the shared sidebar."""

    _show_model_status()


def show_training() -> None:
    """Render chronological split controls and the latest training evidence."""

    _show_training()


def show_prediction(
    *,
    compact_model_labels: bool = False,
    show_audit_details: bool = True,
    allow_database_prediction: bool = True,
) -> None:
    """Render the previous upload, persistence, prediction, and error workflow."""

    _show_prediction(
        compact_model_labels=compact_model_labels,
        show_audit_details=show_audit_details,
        allow_database_prediction=allow_database_prediction,
    )


if __name__ == "__main__":
    main()
