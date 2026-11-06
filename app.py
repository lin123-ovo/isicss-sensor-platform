"""Streamlit UI for importing, viewing, and deleting sensor CSV batches."""

from __future__ import annotations

from collections.abc import Callable
from html import escape
import hashlib
import importlib
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from sensor_app.runtime_paths import resolve_runtime_root  # noqa: E402
from sensor_app.table_display import display_dataframe  # noqa: E402
from sensor_app.temperature_details import (  # noqa: E402
    temperature_category_masks,
    temperature_detail_table,
)
from sensor_database.processing import (  # noqa: E402
    ProcessingVersion, current_processing_version, use_processing_version,
    is_final_processing, retained_processed_rows,
)
from sensor_app.upload_options import native_upload_file_types  # noqa: E402
from sensor_database.cleaning_options import PERSONAL_FIXED_TEMPERATURE_RULE_VERSION  # noqa: E402
from sensor_database.temperature_cusum import RULE_VERSIONS as CUSUM_VERSIONS  # noqa: E402
from sensor_database.temperature_daily_trend import RULE_VERSION as DAILY_TEMPERATURE_VERSION  # noqa: E402
MODERN_TEMPERATURE_VERSIONS = (*CUSUM_VERSIONS, DAILY_TEMPERATURE_VERSION)
from sensor_app.dashboard import (  # noqa: E402
    DashboardSnapshot,
    load_dashboard_snapshot,
    load_workspace_snapshot,
)
from sensor_app.ui_components import (  # noqa: E402
    apply_nexrun_theme,
    render_alert,
    render_empty_state,
    render_metric_card,
    render_page_header,
    render_result_table,
)

SOURCE_ROOT = PROJECT_ROOT
PROJECT_ROOT = resolve_runtime_root(SOURCE_ROOT)


def _discard_stale_local_module(
    module_name: str,
    expected_path: Path,
    required_attributes: tuple[str, ...],
) -> None:
    """Discard an incomplete or foreign module left by a hot-reload process."""

    cached = sys.modules.get(module_name)
    if cached is None:
        return

    cached_file = getattr(cached, "__file__", None)
    cached_path = Path(cached_file).resolve() if cached_file else None
    expected_path = expected_path.resolve()
    is_complete = all(hasattr(cached, name) for name in required_attributes)
    if cached_path == expected_path and is_complete:
        return

    sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


NAVIGATION_ICONS = {
    "系统概览": "dashboard",
    "模型训练": "model_training",
    "上传预测": "upload_file",
    "数据入库": "database",
    "传感器数据库": "table_chart",
    "导入记录": "history",
}


def _format_navigation_label(page: str) -> str:
    """Attach a bundled Material icon to each persistent navigation label."""

    return f":material/{NAVIGATION_ICONS[page]}: {page}"


def _import_upload_file_types(
    *,
    is_frozen: bool | None = None,
    platform_name: str | None = None,
) -> tuple[str, str] | None:
    """Return native chooser filters without excluding spreadsheets on macOS.

    The macOS pywebview runtime passes browser MIME filters to an incompatible
    native chooser branch in both source and packaged launches, which greys out
    CSV/XLSX files. The import reader still validates extensions and content
    before any database write, so removing only this presentation-layer filter
    is safe.
    """

    return native_upload_file_types(
        is_frozen=is_frozen,
        platform_name=platform_name,
    )


def _apply_ui_theme() -> None:
    """Apply the current user-selected Nexrun theme."""

    apply_nexrun_theme(str(st.session_state.get("ui_theme", "跟随系统")))


def _enable_navigation_tooltips() -> None:
    """Add sidebar navigation labels and forward sidebar scrolling to main content."""

    components.html(
        """
        <script>
        (() => {
          const parentWindow = window.parent;
          const doc = parentWindow.document;
          const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
          if (!sidebar) return;

          const getMainScrollTarget = () => {
            const candidates = [
              doc.querySelector('[data-testid="stMain"]'),
              doc.querySelector('[data-testid="stAppViewContainer"]'),
              doc.body,
              doc.scrollingElement
            ];
            return candidates.find(
              (element) => {
                if (!element || element.scrollHeight <= element.clientHeight) {
                  return false;
                }
                const overflowY = parentWindow.getComputedStyle(element).overflowY;
                return overflowY === 'auto' || overflowY === 'scroll';
              }
            ) || doc.scrollingElement;
          };

          const pageNames = [
            '系统概览',
            '模型训练',
            '上传预测',
            '数据入库',
            '传感器数据库',
            '导入记录'
          ];
          const options = sidebar.querySelectorAll(
            '[data-testid="stRadio"] label[data-testid="stRadioOption"]'
          );
          options.forEach((option, index) => {
            const name = pageNames[index];
            if (!name) return;
            option.title = name;
            option.setAttribute('aria-label', name);
          });

          if (sidebar.dataset.nexrunScrollForwarding === 'enabled') return;
          sidebar.dataset.nexrunScrollForwarding = 'enabled';
          sidebar.addEventListener('wheel', (event) => {
            if (!event.deltaY) return;
            const scrollTarget = getMainScrollTarget();
            if (!scrollTarget) return;
            event.preventDefault();
            scrollTarget.scrollBy({ top: event.deltaY, behavior: 'auto' });
          }, { passive: false });
        })();
        </script>
        """,
        height=0,
        width=0,
    )

from sensor_database import (  # noqa: E402
    ColumnMapping,
    SensorDatabaseError,
    UploadNormalizer,
    UploadTableReader,
    suggest_mapping,
    get_sensor_database,
)
_discard_stale_local_module(
    "prediction_ui",
    SOURCE_ROOT / "src" / "prediction_ui.py",
    (
        "PREDICTION_DISPLAY_NAMES",
        "format_metric_table_for_display",
        "order_prediction_columns",
        "prediction_column_config",
        "prediction_csv_bytes",
        "show_prediction",
        "show_training",
    ),
)
from prediction_ui import (  # noqa: E402
    PREDICTION_DISPLAY_NAMES,
    format_metric_table_for_display,
    order_prediction_columns,
    prediction_column_config,
    prediction_csv_bytes,
    show_prediction,
    show_training,
)
from tilt_predictor.database_prediction import (  # noqa: E402
    build_database_prediction_upload,
)
from tilt_predictor.interactive_prediction import InteractivePredictionService  # noqa: E402
from tilt_predictor.model_registry import (  # noqa: E402
    ModelRegistryError,
    ModelVersion,
)
from tilt_predictor.mysql_model_registry import MySQLModelRegistry  # noqa: E402
from tilt_predictor.registry_factory import get_model_registry  # noqa: E402
from tilt_predictor.temperature_pipeline import (  # noqa: E402
    TemperaturePipelineError,
)


def _select_column(
    label: str,
    columns: list[str],
    suggested: str | None,
    key: str,
) -> str:
    """Render one required source-column selector."""

    index = columns.index(suggested) if suggested in columns else 0
    return str(st.selectbox(label, columns, index=index, key=key))


def _show_import(*, temperature_rule_version: str = "temperature_jump_v1", fixed_temperature: bool = False) -> None:
    """Read an upload, let the user verify mappings, and commit one batch."""

    render_page_header(
        "DATA IMPORT",
        "导入传感器数据",
        "上传、映射、校验、十分钟聚合并写入 MySQL 可审计数据库。",
        status="等待上传",
    )
    st.write(
        "上传 CSV 或 XLSX 后，系统会拒绝缺少时间、倾角或温度的行。" +
        ("倾角沿用现有清洗规则；温度自动执行异常检测与修复。" if fixed_temperature else
         "倾角沿用现有清洗规则；温度开启下方开关后使用新版异常检测与修复流程。")
        if temperature_rule_version in ("temperature_jump_v3", *MODERN_TEMPERATURE_VERSIONS) else
        "上传 CSV 或 XLSX 后，系统会拒绝缺少时间、倾角或温度的行，"
        "再对1号顺线、1号横线和温度执行冻结的 Hampel 清洗规则。"
    )
    upload = st.file_uploader("选择数据文件", type=_import_upload_file_types())
    content = upload.getvalue() if upload is not None else b""
    _reset_temperature_import_choice(upload.name if upload is not None else None, content)
    repair_temperature = temperature_rule_version in ("temperature_jump_v2", "temperature_jump_v3", *MODERN_TEMPERATURE_VERSIONS)
    review_temperature = temperature_rule_version in ("temperature_jump_v3", *MODERN_TEMPERATURE_VERSIONS)
    check_temperature = True if fixed_temperature else st.toggle(
        "启用温度异常检测与修复" if repair_temperature else "启用温度突变检查（超过2℃/10分钟）",
        key="import_temperature_jump_check",
    )
    st.caption(
        ("按当天走势识别明显尖峰或凸起片段，用两侧正常温度平滑衔接；正常升降和小幅波动保留原值。"
         "不按前几天同刻温度强行对齐，也不因一般趋势偏差剔除整段数据。原温度与修正记录保留。"
         if temperature_rule_version == DAILY_TEMPERATURE_VERSION else
         "短跳点插值，连续偏移段按前后趋势整体修正；事件最长6小时且不跨日。"
         "无法可靠修复时保留原温度、标记待判断，不参与训练；数据库预测默认排除，可在预测区手动包含。修复值为估计值。"
         if review_temperature else (
             "短跳点插值，连续偏移段按前后趋势整体修正；无法可靠修复的段落隔离。修复值为估计值，原始数据保留。"
             if repair_temperature else "短尖峰符合条件时插值；持续异常段隔离，原始数据保留。"
         ))
        + ("温度处理固定启用。" if fixed_temperature else "关闭后仅使用现有清洗规则。")
        + "同一10分钟桶含多个批次记录时，整个桶采用最近入库批次的设置。"
    )
    if review_temperature:
        st.caption("新修复规则只对后续导入生效；历史已入库数据需重新导入后应用，原始文件无需修改。")
    if temperature_rule_version in CUSUM_VERSIONS:
        st.caption(
            "已加入 CUSUM：以前7天同刻温度为参考，经当天10:00—11:50校准，检查12:00—18:00的持续偏差。"
            "正向修复候选通过物理范围及边界检查后直接采用，保留原值与修正量；"
            "负向报警或候选未通过检查时保留原值、标记待判断。历史不足时跳过CUSUM，仍执行原跳变处理。"
            "CUSUM修复使用前后文，属于离线估计。"
        )
    if upload is None:
        render_empty_state("等待数据文件", "请选择一个 CSV 或 XLSX 文件开始导入工作流。")
        return

    reader = UploadTableReader()
    try:
        sheets = reader.list_sheets(upload.name, content)
        selected_sheet = st.selectbox("工作表", sheets) if sheets else None
        table = reader.read(upload.name, content, selected_sheet)
    except SensorDatabaseError as error:
        st.error(str(error))
        return

    st.caption(f"源数据：{len(table.frame)} 行，{len(table.frame.columns)} 列")
    display_dataframe(table.frame.head(20), width="stretch", hide_index=True)

    columns = [str(column) for column in table.frame.columns]
    suggestions = suggest_mapping(columns)
    st.markdown("#### 确认字段映射")
    first, second = st.columns(2)
    with first:
        timestamp = _select_column(
            "时间列",
            columns,
            suggestions["timestamp"],
            "mapping_timestamp",
        )
        tilt1_inline = _select_column(
            "1号顺线列",
            columns,
            suggestions["tilt1_inline_model"],
            "mapping_tilt1_inline",
        )
        tilt1_crossline = _select_column(
            "1号横线列",
            columns,
            suggestions["tilt1_crossline_model"],
            "mapping_tilt1_crossline",
        )
    with second:
        temperature = _select_column(
            "温度列",
            columns,
            suggestions["temperature_celsius"],
            "mapping_temperature",
        )
        tilt2_inline = _select_column(
            "2号顺线列",
            columns,
            suggestions["tilt2_inline"],
            "mapping_tilt2_inline",
        )
        tilt2_crossline = _select_column(
            "2号横线列",
            columns,
            suggestions["tilt2_crossline"],
            "mapping_tilt2_crossline",
        )

    if not st.button("校验并加入数据库", type="primary"):
        return
    try:
        mapping = ColumnMapping(
            timestamp=timestamp,
            tilt1_inline_model=tilt1_inline,
            tilt1_crossline_model=tilt1_crossline,
            temperature_celsius=temperature,
            tilt2_inline=tilt2_inline,
            tilt2_crossline=tilt2_crossline,
        )
        normalized = UploadNormalizer().normalize(table, mapping)
        result = get_sensor_database(PROJECT_ROOT).import_upload(
            normalized, content, temperature_jump_check_enabled=check_temperature,
            temperature_rule_version=temperature_rule_version,
        )
    except SensorDatabaseError as error:
        st.error(f"入库失败：{error}")
        return

    if result.duplicate:
        st.info(
            f"文件内容及温度检查设置与最近等价批次相同，未重复入库。"
            f"已有批次 ID：{result.import_id}。"
        )
    else:
        st.success(
            f"导入成功。批次 ID：{result.import_id}。"
            f"新增 {result.inserted_rows} 条，更新 {result.updated_rows} 条，"
            f"未变化 {result.unchanged_rows} 条，数据库现有 {result.database_rows} 条。"
        )
    if result.rejected_rows:
        st.warning(f"有 {result.rejected_rows} 行因字段不完整或无效而未入库。")
        display_dataframe(normalized.rejected_rows, width="stretch", hide_index=True)
    check_label = "温度异常检测与修复" if repair_temperature else "温度突变检查"
    st.caption(f"本次{check_label}：{'开启' if result.temperature_jump_check_enabled else '关闭'}")
    suspect, repaired, isolated = st.columns(3)
    suspect.metric("本批次温度可疑桶", result.temperature_suspicious_rows)
    repaired.metric("本批次温度修复桶", result.temperature_repaired_rows)
    isolated.metric("本批次温度隔离桶", result.temperature_isolated_rows)
    if repair_temperature:
        interpolation, offset = st.columns(2)
        interpolation.metric("本批次温度插值修复", result.temperature_interpolated_rows)
        offset.metric("本批次温度偏移修正", result.temperature_offset_corrected_rows)
    if review_temperature:
        review, hard = st.columns(2)
        review.metric("本批次温度待判断", result.temperature_review_rows)
        hard.metric("本批次温度硬异常", result.temperature_hard_unresolved_rows)
        st.caption("待判断保留原值，不属于已修复且不参与训练；数据库预测默认排除，可在预测区手动包含。")
    st.caption("以上统计为本批次涉及的时间桶；隔离是保守筛查，不代表已确认设备故障。")
    st.markdown("#### 入库后的数据库清洗概况")
    bucket_col, usable_col, imputed_col, unresolved_col = st.columns(4)
    bucket_col.metric("10分钟有效桶", result.bucket_rows)
    usable_col.metric("模型可用", result.usable_rows)
    imputed_col.metric("短异常插值", result.imputed_rows)
    unresolved_col.metric("未解决异常", result.unresolved_rows)
    if result.field_summary:
        summary = pd.DataFrame(result.field_summary).rename(
            columns={
                "label": "字段",
                "physical_outliers": "物理越界",
                "bucket_conflicts": "桶内冲突",
                "hampel_outliers": "Hampel异常",
                "outliers": "异常合计",
                "imputed": "已插值",
                "unresolved": "未解决",
            }
        )
        display_dataframe(
            summary.drop(columns=["field"], errors="ignore"),
            width="stretch",
            hide_index=True,
        )
    if result.affected_rows:
        with st.expander("查看受影响的逐行原值、清洗值和原因"):
            display_dataframe(
                pd.DataFrame(result.affected_rows),
                width="stretch",
                hide_index=True,
            )
    if result.cleaning_audit_path:
        st.caption(f"清洗审计：{result.cleaning_audit_path}")


def _reset_temperature_import_choice(filename: str | None, content: bytes) -> None:
    """新文件恢复默认开启，工作表/映射重绘时保留当前选择。"""

    identity = (filename, hashlib.sha256(content).hexdigest())
    if st.session_state.get("import_temperature_file_identity") != identity:
        st.session_state["import_temperature_file_identity"] = identity
        st.session_state["import_temperature_jump_check"] = True


def _filter_measurements_for_display(
    measurements: pd.DataFrame,
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    """Filter a copy for display without changing prediction inputs or storage."""

    if measurements.empty or "timestamp" not in measurements.columns:
        return measurements.copy()
    timestamps = pd.to_datetime(measurements["timestamp"], errors="coerce")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
    mask = timestamps.ge(start) & timestamps.lt(end)
    return measurements.loc[mask].copy()


def _simplified_cleaned_measurements(measurements: pd.DataFrame) -> pd.DataFrame:
    """返回六列中文清洗数据副本；缺少必需列时抛出 ValueError。"""

    column_names = {
        "timestamp": "时间",
        "tilt1_inline_model": "1号顺线",
        "tilt1_crossline_model": "1号横线",
        "temperature_celsius": "温度",
        "tilt2_inline": "2号顺线",
        "tilt2_crossline": "2号横线",
    }
    missing = [column for column in column_names if column not in measurements.columns]
    if missing:
        raise ValueError(f"无法生成简化清洗数据，缺少字段：{'、'.join(missing)}")
    return retained_processed_rows(measurements).loc[:, list(column_names)].rename(columns=column_names).copy()


def _cleaned_measurements_csv_bytes(measurements: pd.DataFrame) -> bytes:
    """导出六列中文 CSV，保留精度和空值；缺少必需列时抛出 ValueError。"""

    version = measurements.get("processing_version", pd.Series(dtype=str))
    final_v3 = current_processing_version() == "final_v3" or (not version.empty and version.eq("final_v3").all())
    return _simplified_cleaned_measurements(measurements).to_csv(
        index=False, float_format="%.2f" if final_v3 else None
    ).encode("utf-8-sig")


def _final_measurements_for_display(measurements: pd.DataFrame) -> pd.DataFrame:
    """Display final numbers and readable missing cells; exports remain numeric."""

    displayed = _simplified_cleaned_measurements(measurements)
    for column in displayed.columns[1:]:
        displayed[column] = displayed[column].map(
            lambda value: "缺测" if pd.isna(value) else f"{float(value):.2f}"
        )
    return displayed


def _show_database(
    *, compact_model_labels: bool = False, simplify_database_view: bool = False
) -> None:
    """Show current measurements, batch deletion, and CSV downloads."""

    render_page_header(
        "DATABASE",
        "当前传感器数据库",
        "浏览已入库测量值，并在不改动数据库的情况下执行预测。",
    )
    database = get_sensor_database(PROJECT_ROOT)
    try:
        status = database.status()
        raw_measurements = database.load_measurements()
        measurements = database.load_cleaned_measurements()
    except SensorDatabaseError as error:
        st.error(str(error))
        return

    final_processing = is_final_processing()
    audit_measurements = measurements
    measurements = retained_processed_rows(measurements)
    row_metric, usable_metric, unresolved_metric, import_metric, time_metric = st.columns(5)
    row_metric.metric("10分钟清洗记录", len(measurements))
    usable_metric.metric("可训练样本" if final_processing else "模型可用", status.usable_rows)
    unresolved_metric.metric("未解决输入" if final_processing else "未解决异常", status.unresolved_rows)
    import_metric.metric("有效导入批次", status.active_imports)
    time_metric.metric("最新原始时间", _timestamp_label(status.timestamp_max))
    st.caption(
        "默认展示最终处理数据；整行连续缺测1～2条可用前后实测插值并保留两位小数。"
        "2号插值仅为估计值，不作为训练或评分真值。"
        if current_processing_version() == "final_v3" else
        "默认展示完成异常处理、温度修正及短缺测插值后的最终数据；2号插值仅为估计值，不作为训练或评分真值。"
        if final_processing else
        "默认展示十分钟清洗模型表；精确时间原值保存在独立只读页签中。"
    )

    try:
        import_log = database.load_import_log()
    except SensorDatabaseError as error:
        st.error(f"无法读取导入记录：{error}")
        return
    _show_batch_delete(database, import_log)
    _show_model_delete(compact_model_labels=compact_model_labels)

    if "temperature_zero_invalid" in audit_measurements:
        repaired, excluded = st.columns(2)
        repaired.metric("零温度插值", int(audit_measurements["temperature_zero_interpolated"].sum()))
        excluded.metric("零温度剔除", int(audit_measurements["processed_row_excluded"].sum()))
        st.caption("连续1～2个零温度点仅在两端可靠时插值；超过2个或无法插值则剔除整行，原始记录保留。")
        zero_rows = audit_measurements.loc[audit_measurements["temperature_zero_invalid"]]
        if not zero_rows.empty:
            with st.expander("零温度处理记录（只读）"):
                display_dataframe(zero_rows.loc[:, [
                    "timestamp", "temperature_raw_celsius", "temperature_celsius",
                    "temperature_zero_run_length", "temperature_zero_interpolated",
                    "processed_row_excluded", "processed_exclusion_reason",
                ]], width="stretch", hide_index=True)
    if measurements.empty and audit_measurements.empty:
        render_empty_state("数据库为空", "请先从数据入库页面导入一批传感器数据。")
        return

    _show_database_prediction(
        audit_measurements,
        compact_model_labels=compact_model_labels,
    )
    st.markdown("#### 清洗质量汇总")
    quality_summary = []
    for prefix, label in (
        ("tilt1_inline", "1号顺线"),
        ("tilt1_crossline", "1号横线"),
        ("temperature", "温度"),
    ):
        quality_summary.append(
            {
                "字段": label,
                "物理越界": int(measurements[f"{prefix}_physical_outlier"].sum()),
                "桶内冲突": int(measurements[f"{prefix}_bucket_conflict"].sum()),
                "Hampel异常": int(measurements[f"{prefix}_hampel_outlier"].sum()),
                "已插值": int(measurements[f"{prefix}_imputed"].sum()),
                "未解决": int(measurements[f"{prefix}_unresolved"].sum()),
            }
        )
    display_dataframe(pd.DataFrame(quality_summary), width="stretch", hide_index=True)
    if simplify_database_view and "temperature_offset_corrected" in measurements:
        category_masks = temperature_category_masks(measurements)
        for column, (category, mask) in zip(st.columns(5), category_masks.items(), strict=True):
            column.metric(category, int(mask.sum()))
        st.caption("待判断温度保留原值且不参与训练；数据库预测默认排除，可在本页预测区手动包含。六列下载不包含质量标记。")
        with st.expander("温度处理明细（只读）", expanded=True):
            category = st.selectbox(
                "查看温度明细类别",
                list(category_masks),
                key="database_temperature_detail_category",
                format_func=lambda name: f"{name}（{int(category_masks[name].sum())} 条）",
            )
            st.caption(
                "明细范围与上方统计一致，不受下方时间、质量状态筛选影响。"
                "同一记录可能属于多个类别，各类别数量不能直接相加。"
                "原始温度为10分钟桶内原值中位数；处理后为空表示未得到可用温度。"
            )
            details = temperature_detail_table(measurements, category)
            st.caption(f"{category}：共 {len(details)} 条，按时间升序显示全部记录。")
            if details.empty:
                st.info("该类别暂无记录。")
            display_dataframe(
                details, width="stretch", hide_index=True, height=360,
                column_config={
                    name: st.column_config.NumberColumn(format="%.2f")
                    for name in ("原始温度（℃）", "处理后温度（℃）", "偏移修正量（℃）")
                    if name in details
                },
            )
    st.markdown("#### 只读时间筛选")
    timestamp_values = pd.to_datetime(measurements["timestamp"], errors="coerce").dropna()
    filtered = measurements.copy()
    if not timestamp_values.empty:
        minimum_date = timestamp_values.min().date()
        maximum_date = timestamp_values.max().date()
        selected_range = st.date_input(
            "显示时间范围",
            value=(minimum_date, maximum_date),
            min_value=minimum_date,
            max_value=maximum_date,
            key="database_display_date_range",
            help="只筛选下方页面明细，不改变数据库内容，也不改变上方预测输入。",
        )
        if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
            filtered = _filter_measurements_for_display(
                measurements,
                selected_range[0],
                selected_range[1],
            )
    status_options = ["全部", "clean", "imputed", "unresolved", "inserted_gap"]
    if simplify_database_view:
        status_options.insert(3, "corrected")
        status_options.insert(4, "review_required")
    selected_status = st.selectbox(
        "质量状态",
        status_options,
        key="database_quality_status",
        help="仅筛选页面明细，不改变数据库、训练或预测输入。",
    )
    if selected_status != "全部":
        filtered = filtered.loc[
            filtered["quality_status"].astype(str).eq(selected_status)
        ].copy()
    affected_only = st.checkbox(
        "只显示发生过异常、插值或未解决的观测",
        key="database_affected_only",
    )
    if affected_only:
        filtered = filtered.loc[
            filtered["input_outlier"].astype(bool)
            | filtered["model_input_imputed"].astype(bool)
            | filtered["model_input_unresolved"].astype(bool)
            | filtered["is_inserted_gap"].astype(bool)
            | filtered.get("tilt2_inline_gap_imputed", pd.Series(False, index=filtered.index)).astype(bool)
            | filtered.get("tilt2_crossline_gap_imputed", pd.Series(False, index=filtered.index)).astype(bool)
        ].copy()
    total_column, visible_column = st.columns(2)
    total_column.metric("数据库总量", len(measurements))
    visible_column.metric("当前显示", len(filtered))
    st.caption("该筛选仅影响下方表格；数据库预测始终使用未筛选的完整数据库。")
    cleaned_tab, raw_tab = st.tabs(["清洗模型数据", "原始精确时间数据"])
    with cleaned_tab:
        title = "最终处理后数据" if final_processing else (
            "简化清洗数据" if simplify_database_view else "原值、清洗值与质量状态"
        )
        st.markdown(f"#### {title}")
        try:
            displayed = (
                _final_measurements_for_display(filtered) if final_processing else
                _simplified_cleaned_measurements(filtered) if simplify_database_view else filtered
            )
            display_dataframe(displayed, width="stretch", hide_index=True, height=520)
            cleaned_csv = _cleaned_measurements_csv_bytes(measurements)
        except ValueError as error:
            st.error(str(error))
        else:
            st.download_button(
                "下载清洗数据库 CSV",
                cleaned_csv,
                file_name="CSG124_cleaned_measurements.csv",
                mime="text/csv",
            )
        if final_processing:
            if "tilt2_inline_gap_imputed" in filtered:
                st.caption("2号短缺测可按前后实测值插值；插值标记和原值见下方详情，六列 CSV 不包含标记。")
            with st.expander("处理标记与原值（只读）"):
                display_dataframe(filtered, width="stretch", hide_index=True, height=360)
        if simplify_database_view:
            _show_temperature_cusum_audit(filtered)
            _show_temperature_daily_audit(filtered)
    with raw_tab:
        st.caption("该表保留入库后的精确时间原值和来源追溯信息，不做就地清洗。")
        display_dataframe(raw_measurements, width="stretch", hide_index=True, height=520)
        st.download_button(
            "下载原始有效数据库 CSV",
            raw_measurements.to_csv(index=False).encode("utf-8-sig"),
            file_name="CSG124_measurements.csv",
            mime="text/csv",
        )


def _show_temperature_daily_audit(measurements: pd.DataFrame) -> None:
    """独立展示当天趋势修正，避免与旧CUSUM审计字段混淆。"""
    if "temperature_daily_checked" not in measurements:
        return
    current = measurements.loc[measurements["temperature_rule_version"].eq(DAILY_TEMPERATURE_VERSION)]
    if current.empty:
        return
    checked, corrected = st.columns(2)
    checked.metric("当天趋势已检查", int(current["temperature_daily_checked"].sum()))
    corrected.metric("温度异常段修复", int(current["temperature_daily_corrected"].sum()))
    st.caption("仅修复明显异常的局部尖峰或凸起，用同日两侧正常点平滑衔接；正常起伏保留，未闭合片段保留原值。")
    with st.expander("当天温度趋势处理记录（只读）"):
        table = current.loc[:, ["timestamp", "temperature_raw_celsius", "temperature_celsius",
                                "temperature_daily_trend_celsius", "temperature_daily_offset_celsius",
                                "temperature_daily_status"]].rename(columns={
            "timestamp": "时间", "temperature_raw_celsius": "原温度℃", "temperature_celsius": "处理后温度℃",
            "temperature_daily_trend_celsius": "当天趋势℃", "temperature_daily_offset_celsius": "修正扣除℃",
            "temperature_daily_status": "执行状态",
        })
        table["执行状态"] = table["执行状态"].replace({
            "checked": "正常保留", "event_bridged": "异常段平滑修复",
            "unclosed_event_kept_raw": "无可靠恢复点，保留原值",
            "insufficient_same_day_context": "当天参考不足，保留原值", "not_observed": "无实际观测"})
        display_dataframe(table, width="stretch", hide_index=True, height=360)
        st.download_button("下载当天温度趋势审计CSV", table.to_csv(index=False).encode("utf-8-sig"),
                           file_name="CSG124_temperature_daily_audit.csv", mime="text/csv")


def _show_temperature_cusum_audit(measurements: pd.DataFrame) -> None:
    """展示筛选范围内的CUSUM执行状态、原值及已采用/未采用的修复候选。"""
    if "temperature_cusum_status" not in measurements:
        return
    current = measurements.loc[measurements["temperature_rule_version"].isin(CUSUM_VERSIONS)]
    if current.empty:
        return
    checked, alarm, applied, review = st.columns(4)
    checked.metric("CUSUM已检查", int(current["temperature_cusum_checked"].sum()))
    alarm.metric("CUSUM报警桶", int((current["temperature_cusum_alarm_positive"] | current["temperature_cusum_alarm_negative"]).sum()))
    applied.metric("CUSUM候选已采用", int(current["temperature_cusum_candidate_applied"].sum()))
    review.metric("CUSUM待判断", int((current["temperature_cusum_outlier"] & current["temperature_review_required"]).sum()))
    st.caption("以上为当前筛选范围。CUSUM报警表示偏离历史参考；已采用的修复仍是估计值。")
    with st.expander("CUSUM温度处理记录（只读）"):
        status_labels = {
            "outside_scan_window": "不在午后检查窗", "insufficient_morning_reference": "上午参考不足",
            "insufficient_calibration": "历史校准不足", "checked": "已检查",
            "blocked_or_missing_reference": "已有异常或参考缺失",
            "candidate_boundary_rejected": "候选边界检查未通过", "not_observed": "无实际观测",
        }
        table = current.loc[:, [
            "timestamp", "temperature_raw_celsius", "temperature_celsius",
            "temperature_cusum_reference_celsius", "temperature_cusum_candidate_celsius",
            "temperature_cusum_offset_celsius", "temperature_cusum_positive", "temperature_cusum_negative",
            "temperature_cusum_candidate_applied", "temperature_review_required", "temperature_cusum_status",
        ]].copy()
        table["temperature_cusum_status"] = table["temperature_cusum_status"].replace(status_labels)
        table = table.rename(columns={
            "timestamp": "时间", "temperature_raw_celsius": "原温度℃", "temperature_celsius": "处理后温度℃",
            "temperature_cusum_reference_celsius": "历史参考℃", "temperature_cusum_candidate_celsius": "修复候选℃",
            "temperature_cusum_offset_celsius": "候选扣除℃", "temperature_cusum_positive": "正向CUSUM",
            "temperature_cusum_negative": "负向CUSUM", "temperature_cusum_candidate_applied": "候选已采用",
            "temperature_review_required": "待判断", "temperature_cusum_status": "执行状态",
        })
        display_dataframe(table, width="stretch", hide_index=True, height=360)
        st.download_button("下载CUSUM温度审计CSV", table.to_csv(index=False).encode("utf-8-sig"),
                           file_name="CSG124_temperature_cusum_audit.csv", mime="text/csv")


def _database_prediction_model_label(
    version: ModelVersion,
    active_version_id: str,
    *,
    compact_model_labels: bool,
) -> str:
    """Return the model name used by the database-prediction selector."""

    version_id = str(version.version_id)
    if compact_model_labels:
        return version_id
    note = " ".join(version.note.split()) or "无备注"
    return f"{version_id}｜{note}"


def _show_database_prediction(
    measurements: pd.DataFrame,
    *,
    compact_model_labels: bool = False,
) -> None:
    """Predict the latest selected share with one immutable saved model."""

    st.markdown("#### 使用已训练模型预测")
    st.caption(
        "选择模型版本后，按时间取数据库末尾 N% 的记录执行预测。"
        "该操作不会重新训练模型、不会切换活动模型，也不会修改数据库。"
    )
    registry = get_model_registry(PROJECT_ROOT)
    try:
        active = registry.bootstrap_current_version()
        versions = registry.list_versions()
    except ModelRegistryError as error:
        st.error(f"模型版本库不可用：{error}")
        return
    if not versions or active is None:
        st.info("尚无兼容清洗合同的模型，请先训练候选模型并手动发布。")
        return

    by_id = {version.version_id: version for version in versions}
    version_ids = list(by_id)
    default_index = version_ids.index(active.version_id)
    model_column, percentage_column = st.columns([1.5, 1], gap="medium")
    with model_column:
        selected_version = st.selectbox(
            "预测模型",
            version_ids,
            index=default_index,
            format_func=lambda version_id: _database_prediction_model_label(
                by_id[version_id],
                active.version_id,
                compact_model_labels=compact_model_labels,
            ),
            key="database_prediction_model",
        )
    with percentage_column:
        percentage = st.slider(
            "预测比例",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
            format="%d%%",
            key="database_prediction_percentage",
            help="按 timestamp 升序排列后，选择末尾这一比例的数据。",
        )
    selected_count = max(1, (len(measurements) * int(percentage) + 99) // 100)
    prediction_total = len(measurements)
    preview_upload = None
    include_temperature_review = False
    if current_processing_version() in ("final_v2", "final_v3"):
        include_temperature_review = st.checkbox(
            "包含温度待判断数据",
            value=False,
            key="database_prediction_include_temperature_review",
            help="仅本次数据库预测使用待判断记录的温度原值；仍排除缺失、硬异常及其他不可用输入，不改变训练或数据库。",
        )
        try:
            preview_upload = build_database_prediction_upload(
                measurements, int(percentage), include_temperature_review=include_temperature_review,
            )
        except TemperaturePipelineError as error:
            st.info(str(error))
            return
        selected_count = len(preview_upload.frame)
        prediction_total = preview_upload.audit["database_rows"]
    st.caption(
        f"将使用 {selected_version} 预测数据库最后 {percentage}%："
        f"共 {selected_count} / {prediction_total} 条。"
    )
    if include_temperature_review and preview_upload is not None:
        st.caption(
            f"本次所选记录含温度待判断 {preview_upload.audit['temperature_review_prediction_rows']} 条，"
            "使用保留原值预测，结果会标明这些记录。比例按包含待判断后的可预测记录计算。"
        )
    if st.button(
        "执行数据库预测",
        type="primary",
        key="run_database_prediction",
    ):
        try:
            upload = (preview_upload if preview_upload is not None
                      else build_database_prediction_upload(
                          measurements, int(percentage), include_temperature_review=include_temperature_review,
                      ))
            result = InteractivePredictionService(PROJECT_ROOT).predict(
                upload,
                modelVersion=str(selected_version),
            )
        except (TemperaturePipelineError, ModelRegistryError) as error:
            st.error(f"数据库预测失败：{error}")
        else:
            st.session_state["database_prediction_result"] = result
            st.session_state["database_prediction_signature"] = (
                str(selected_version),
                int(percentage),
                include_temperature_review,
            )

    signature = st.session_state.get("database_prediction_signature")
    result = st.session_state.get("database_prediction_result")
    if result is None or signature != (
        str(selected_version),
        int(percentage),
        include_temperature_review,
    ):
        return

    st.success(
        f"预测完成：模型 {selected_version}，共 {len(result.predictions)} 条记录。"
    )
    ordered_predictions = order_prediction_columns(result.predictions)
    prediction_columns = [
        column
        for column in ordered_predictions.columns
        if column in PREDICTION_DISPLAY_NAMES
    ]
    display = ordered_predictions.loc[:, prediction_columns].rename(
        columns=PREDICTION_DISPLAY_NAMES
    )
    st.caption("先展示顺线与横线的“真值 → 预测值”对照，再展示各方向误差，便于逐行比对。")
    display_dataframe(
        display,
        column_config=prediction_column_config(display),
        width="stretch",
        hide_index=True,
        height=420,
    )
    chart = result.predictions.loc[
        :, ["timestamp", "predicted_inline", "predicted_crossline"]
    ]
    st.line_chart(chart, x="timestamp")
    if not result.metrics.empty:
        _show_database_prediction_metrics(result.metrics)
    st.download_button(
        "下载数据库预测 CSV",
        prediction_csv_bytes(ordered_predictions),
        file_name=(
            f"CSG124_database_predictions_{selected_version}_{percentage}pct.csv"
        ),
        mime="text/csv",
        key="download_database_prediction",
    )


def _show_database_prediction_metrics(metrics: pd.DataFrame) -> None:
    """Show database prediction metrics with the same cards as final testing."""

    st.markdown("##### 本次数据库预测指标")
    st.caption(
        "只使用所选记录中已有2号传感器真值且模型成功输出的行计算；"
        "这些记录可能参与过模型训练，因此不代表独立测试集成绩。"
    )
    direction_labels = {"inline": "顺线", "crossline": "横线"}
    for _, row in metrics.iterrows():
        direction = direction_labels.get(str(row["direction"]), str(row["direction"]))
        st.markdown(f"**{direction}（{int(row['sample_count'])} 条）**")
        mae_column, rmse_column, maximum_column = st.columns(3)
        mae_column.metric("MAE", f"{row['mae'] * 1000:.2f} m°")
        rmse_column.metric("RMSE", f"{row['rmse'] * 1000:.2f} m°")
        maximum_column.metric("MaxAE", f"{row['max_abs_error'] * 1000:.2f} m°")
        st.caption(f"最大误差发生时间：{row['max_error_timestamp']}")

        relative_column = st.columns(1)[0]
        relative_column.metric(
            "MAPE",
            f"{row['mean_abs_relative_error_percent']:.2f}%",
        )
        st.caption(
            f"相对误差有效 {int(row['relative_error_valid_count'])} 条；"
            f"真值为0排除 {int(row['relative_error_excluded_zero_count'])} 条。"
        )

    with st.expander("查看指标明细表"):
        display_metrics = metrics.copy()
        for column in ("mae", "rmse", "max_abs_error"):
            if column in display_metrics:
                display_metrics[column] = pd.to_numeric(display_metrics[column], errors="coerce") * 1000
        if "unit" in display_metrics:
            display_metrics["unit"] = "m°"
        metric_table = format_metric_table_for_display(display_metrics)
        for column in ("mae", "rmse", "max_abs_error"):
            if column in metric_table:
                metric_table[column] = display_metrics[column].map(
                    lambda value: f"{value:.2f}" if pd.notna(value) else "—"
                )
        metric_table = metric_table.rename(columns={
            "mae": "MAE (m°)", "rmse": "RMSE (m°)", "max_abs_error": "MaxAE (m°)",
            "mean_abs_relative_error_percent": "MAPE (%)",
        })
        display_dataframe(
            metric_table,
            width="stretch",
            hide_index=True,
        )


def _batch_label(row: pd.Series) -> str:
    """Build a readable label while keeping the exact batch ID visible."""

    imported = pd.Timestamp(row["imported_at_utc"])
    if imported.tzinfo is not None:
        imported = imported.tz_convert("Asia/Shanghai")
    return (
        f"{imported.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"{row['source_filename']} | {row['time_series_rows']} 条 | "
        f"{row['import_id']}"
    )


def _show_batch_delete(database: object, import_log: pd.DataFrame) -> None:
    """Delete one active import beside the database it currently affects."""

    flash_message = st.session_state.pop("delete_message", None)
    if flash_message:
        st.success(flash_message)

    active = import_log.loc[import_log["status"] == "active"].copy()
    st.markdown("#### 删除一次导入")
    render_alert(
        "删除是可审计的危险操作",
        "删除会移除所选批次对当前数据库的影响，并从剩余批次重新构建主表；"
        "执行前系统会自动备份。",
        variant="warning",
    )
    with st.container(border=True):
        if active.empty:
            st.info("没有可删除的有效导入批次。")
            return
        active = active.sort_values("imported_at_utc", ascending=False)
        options = active.to_dict("records")
        selected = st.selectbox(
            "选择导入批次",
            options,
            format_func=lambda row: _batch_label(pd.Series(row)),
            key="database_delete_import_batch",
        )
        confirmed = st.checkbox(
            "我确认删除这个批次的数据，并理解当前数据库会被重新构建。",
            key="database_delete_import_confirmed",
        )
        if not st.button(
            "删除所选导入批次",
            disabled=not confirmed,
            key="database_delete_import_button",
        ):
            return
        try:
            result = database.delete_import(str(selected["import_id"]))
        except SensorDatabaseError as error:
            st.error(f"删除失败：{error}")
            return
        st.session_state["delete_message"] = (
            f"已删除批次 {result.import_id}：移除当前记录 "
            f"{result.removed_current_rows} 条，恢复较早版本 {result.restored_previous_rows} 条，"
            f"数据库现有 {result.database_rows} 条。备份："
            f"{result.backup_path.relative_to(PROJECT_ROOT)}"
        )
        st.rerun()


def _database_delete_model_label(
    version: ModelVersion,
    active_version_id: str | None,
    *,
    compact_model_labels: bool,
) -> str:
    """Return the model name used by the destructive version selector."""

    version_id = str(version.version_id)
    if compact_model_labels:
        return version_id
    active_suffix = " | 当前活动" if version_id == active_version_id else ""
    return (
        f"{version_id} | {str(version.model).upper()} | "
        f"{version.split_ratio}{active_suffix}"
    )


def _show_model_delete(*, compact_model_labels: bool = False) -> None:
    """Delete one saved model beside the database it is used to predict."""

    flash_message = st.session_state.pop("model_delete_message", None)
    if flash_message:
        st.success(flash_message)

    st.markdown("#### 删除模型版本")
    render_alert(
        "删除模型不可恢复",
        "删除会永久移除该模型及其文件；若删除当前活动模型，系统会自动切换到最新的剩余模型。",
        variant="warning",
    )
    try:
        registry = get_model_registry(PROJECT_ROOT)
        versions = registry.list_versions()
    except ModelRegistryError as error:
        st.error(f"无法读取模型版本：{error}")
        return
    active = None
    if versions:
        try:
            active = registry.get_active_version()
        except ModelRegistryError:
            st.warning("当前没有活动模型；删除不会切换活动模型。")

    with st.container(border=True):
        if not versions:
            st.info("没有可删除的模型版本。")
            return
        by_id = {version.version_id: version for version in versions}
        selected_id = st.selectbox(
            "选择模型版本",
            list(by_id),
            format_func=lambda version_id: _database_delete_model_label(
                by_id[version_id],
                active.version_id if active else None,
                compact_model_labels=compact_model_labels,
            ),
            key="database_delete_model_version",
        )
        remaining_versions = [
            version for version in versions if version.version_id != selected_id
        ]
        if active and selected_id == active.version_id:
            if remaining_versions:
                replacement_id = remaining_versions[0].version_id
                st.caption(f"删除后会自动切换活动模型至 {replacement_id}。")
            else:
                st.caption("删除后将没有活动模型，上传预测暂不可用。")
        else:
            st.caption("删除不可恢复，不影响当前活动模型。")
        confirmed = st.checkbox(
            f"我确认永久删除 {selected_id} 及其模型文件。",
            key="database_delete_model_confirmed",
        )
        if not st.button(
            "删除所选模型版本",
            disabled=not confirmed,
            key="database_delete_model_button",
        ):
            return
        if not isinstance(registry, MySQLModelRegistry):
            st.error("当前运行环境不支持删除模型版本。")
            return
        try:
            replacement_id = registry.delete_version(selected_id)
        except ModelRegistryError as error:
            st.error(f"删除失败，模型库未更改：{error}")
            return
        if replacement_id:
            message = (
                f"已删除 {selected_id}，活动预测模型已自动切换至 "
                f"{replacement_id}。"
            )
        elif active and selected_id == active.version_id:
            message = f"已删除 {selected_id}，当前没有活动预测模型。"
        else:
            message = f"已删除 {selected_id}。"
        st.session_state["model_delete_message"] = message
        st.rerun()


def _show_import_history(*, temperature_rule_version: str = "temperature_jump_v1") -> None:
    """Show read-only import audit history."""

    render_page_header(
        "AUDIT TRAIL",
        "导入记录",
        "按批次追踪来源、哈希、行数和删除记录。删除入口已移至传感器数据库页面。",
    )
    database = get_sensor_database(PROJECT_ROOT)
    try:
        import_log = database.load_import_log()
    except SensorDatabaseError as error:
        st.error(str(error))
        return

    if import_log.empty:
        render_empty_state("暂无导入记录", "完成一次数据入库后，这里会显示完整审计时间线。")
        return
    st.markdown("#### 最近审计活动")
    _render_import_timeline(
        import_log.sort_values("imported_at_utc", ascending=False).head(8)
    )
    st.markdown("#### 完整审计记录")
    from sensor_database.cleaning_options import parse_cleaning_options

    import_log = import_log.copy()
    check_label = (
        "温度异常检测与修复"
        if temperature_rule_version in ("temperature_jump_v2", "temperature_jump_v3", *MODERN_TEMPERATURE_VERSIONS) else "温度突变检查"
    )
    import_log[check_label] = import_log["cleaning_options_json"].map(
        lambda value: (
            (f"开启（修复{options.rule_version.rsplit('_', 1)[-1]}）"
             if options.rule_version in ("temperature_jump_v2", "temperature_jump_v3", *MODERN_TEMPERATURE_VERSIONS) else "开启（旧版检查）")
            if (options := parse_cleaning_options(value)) and options.enabled else "关闭"
        ) if temperature_rule_version in ("temperature_jump_v2", "temperature_jump_v3", *MODERN_TEMPERATURE_VERSIONS) else (
            "开启" if (options := parse_cleaning_options(value)) and options.enabled else "关闭"
        )
    )
    display_columns = [
        "imported_at_utc",
        "source_filename",
        check_label,
        "time_series_rows",
        "inserted_rows",
        "updated_rows",
        "unchanged_rows",
        "cleaning_bucket_rows",
        "cleaning_usable_rows",
        "cleaning_imputed_rows",
        "cleaning_unresolved_rows",
        "status",
        "deleted_at_utc",
        "import_id",
    ]
    display_dataframe(
        import_log.loc[:, display_columns].sort_values(
            "imported_at_utc",
            ascending=False,
        ),
        width="stretch",
        hide_index=True,
    )


def _display_value(value: object, *, suffix: str = "") -> str:
    """Return a readable dashboard value without leaking Python nulls."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "不可用"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}{suffix}"
    return f"{value}{suffix}"


def _timestamp_label(value: str | None) -> str:
    """Format persisted timestamps for narrow dashboard surfaces."""

    if not value:
        return "不可用"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%Y-%m-%d %H:%M")


def _go_to_page(page: str) -> None:
    """Move the persistent sidebar navigation to another page."""

    st.session_state["main_navigation"] = page


def _render_import_timeline(imports: pd.DataFrame) -> None:
    """Render the latest import activity using the reference timeline language."""

    if imports.empty:
        render_empty_state("暂无活动", "完成一次数据入库后，这里会显示最近的审计事件。")
        return
    items: list[str] = []
    for _, row in imports.iterrows():
        imported = pd.to_datetime(row.get("imported_at_utc"), errors="coerce", utc=True)
        if pd.isna(imported):
            imported_label = "时间未知"
        else:
            imported_label = imported.tz_convert("Asia/Shanghai").strftime("%m-%d %H:%M")
        status = "已删除" if str(row.get("status", "")) == "deleted" else "有效批次"
        source = escape(str(row.get("source_filename", "未知文件")))
        rows = _display_value(row.get("time_series_rows"), suffix=" 条")
        items.append(
            '<div class="nexrun-timeline-item">'
            f'<div class="nexrun-timeline-title">{source}</div>'
            f'<div class="nexrun-timeline-meta">{imported_label} · {status} · {rows}</div>'
            '</div>'
        )
    st.markdown(
        f'<div class="nexrun-timeline">{"".join(items)}</div>',
        unsafe_allow_html=True,
    )


def _overview_chart_data(measurements: pd.DataFrame) -> pd.DataFrame:
    """Prepare a bounded chronological frame for read-only overview charts."""

    required = [
        "timestamp",
        "tilt1_inline_model",
        "tilt1_crossline_model",
        "tilt2_inline",
        "tilt2_crossline",
        "temperature_celsius",
    ]
    if measurements.empty or any(column not in measurements for column in required):
        return pd.DataFrame()
    chart = measurements.loc[:, required].copy()
    chart["timestamp"] = pd.to_datetime(chart["timestamp"], errors="coerce")
    chart = chart.dropna(subset=["timestamp"]).sort_values("timestamp").tail(240)
    for column in required[1:]:
        chart[column] = pd.to_numeric(chart[column], errors="coerce")
    return chart


def _show_overview(
    snapshot: DashboardSnapshot,
    *,
    show_coverage_and_quick_access: bool = True,
) -> None:
    """Render the new read-only Nexrun-style system overview."""

    render_page_header(
        "SYSTEM OVERVIEW",
        "系统概览",
        "集中查看数据覆盖、导入审计、活动模型和传感器趋势。",
    )
    metric_columns = st.columns(3, gap="medium")
    metric_columns[0].metric(
        "10分钟清洗记录",
        _display_value(snapshot.cleaned_rows),
    )
    metric_columns[1].metric("有效导入批次", _display_value(snapshot.active_imports))
    metric_columns[2].metric("活动模型", _display_value(snapshot.active_model_version))

    for warning in snapshot.warnings:
        st.warning(warning)

    trend_column, activity_column = st.columns([2.1, 1], gap="large")
    with trend_column:
        st.subheader("传感器趋势")
        chart = _overview_chart_data(snapshot.measurements)
        if chart.empty:
            render_empty_state("暂无趋势", "数据库中还没有可用于绘图的完整测量记录。")
        else:
            tilt_tab, temperature_tab = st.tabs(["倾角趋势", "温度趋势"])
            with tilt_tab:
                st.line_chart(
                    chart.rename(
                        columns={
                            "tilt1_inline_model": "1号顺线",
                            "tilt1_crossline_model": "1号横线",
                            "tilt2_inline": "2号顺线",
                            "tilt2_crossline": "2号横线",
                        }
                    ),
                    x="timestamp",
                    y=["1号顺线", "1号横线", "2号顺线", "2号横线"],
                    color=["#3547D3", "#10C5E8", "#711AC8", "#E526A8"],
                    height=330,
                )
            with temperature_tab:
                st.line_chart(
                    chart.rename(columns={"temperature_celsius": "环境温度"}),
                    x="timestamp",
                    y="环境温度",
                    color="#10C5E8",
                    height=330,
                )
            st.caption("为保证概览流畅，仅展示数据库最近 240 条记录；原始数据未被修改。")
    with activity_column:
        st.subheader("最近导入活动")
        _render_import_timeline(snapshot.recent_imports)

    if show_coverage_and_quick_access:
        st.subheader("数据覆盖与快捷入口")
        coverage_column, quick_column = st.columns([1.2, 1.8], gap="large")
        with coverage_column:
            st.markdown(
                '<div class="nexrun-card">'
                '<div class="nexrun-card-kicker">DATA COVERAGE</div>'
                '<div class="nexrun-card-title">数据库时间范围</div>'
                f'<div class="nexrun-card-copy">开始：{escape(_timestamp_label(snapshot.timestamp_min))}<br>'
                f'结束：{escape(_timestamp_label(snapshot.timestamp_max))}<br>'
                f'最近导入：{escape(_display_value(snapshot.latest_import_time))}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with quick_column:
            action_columns = st.columns(4)
            action_columns[0].button(
                "进入模型训练",
                type="primary",
                use_container_width=True,
                on_click=_go_to_page,
                args=("模型训练",),
            )
            action_columns[1].button(
                "上传预测文件",
                use_container_width=True,
                on_click=_go_to_page,
                args=("上传预测",),
            )
            action_columns[2].button(
                "导入新数据",
                use_container_width=True,
                on_click=_go_to_page,
                args=("数据入库",),
            )
            action_columns[3].button(
                "查看清洗数据库",
                use_container_width=True,
                on_click=_go_to_page,
                args=("传感器数据库",),
            )
            st.caption("快捷入口只负责页面跳转，不会自动训练、预测或写入数据库。")


def main(
    *,
    page_title: str = "工业传感器智能校准与软测量系统",
    workspace_title: str = "工业传感器智能校准与软测量系统",
    sidebar_title: str = "ISICSS",
    training_renderer=show_training,
    dashboard_loader=load_dashboard_snapshot,
    navigation_change_handler: Callable[[str], str | None] | None = None,
    show_overview_coverage_and_quick_access: bool = True,
    compact_model_labels: bool = False,
    show_prediction_audit: bool = True,
    allow_database_prediction: bool = True,
    simplify_database_view: bool = False,
    temperature_rule_version: str = "temperature_jump_v1",
    processing_version: ProcessingVersion = "legacy",
) -> None:
    """Render the local sensor database application under an explicit profile."""

    with use_processing_version(processing_version):
        st.set_page_config(
            page_title=page_title,
            layout="wide",
            initial_sidebar_state="auto",
        )
        _apply_ui_theme()
        _enable_navigation_tooltips()
        st.sidebar.markdown(f"## {sidebar_title}")
        st.sidebar.caption("本地传感器数据与模型工作台")
        st.sidebar.markdown("#### 工作区")
        if st.session_state.get("main_navigation") == "导入记录与删除":
            st.session_state["main_navigation"] = "导入记录"
        navigation = st.sidebar.radio(
            "功能导航",
            list(NAVIGATION_ICONS),
            format_func=_format_navigation_label,
            key="main_navigation",
        )
        st.sidebar.markdown("#### 本地运行")
        st.sidebar.caption("业务数据与模型来自 MySQL；本机仅保留运行配置和审计文件。")
        st.sidebar.divider()
        # Custom/customer loaders keep their existing behavior. Only the personal
        # edition skips the full measurement projection outside its overview.
        loader = (
            load_workspace_snapshot
            if is_final_processing() and navigation != "系统概览"
            and dashboard_loader is load_dashboard_snapshot
            else dashboard_loader
        )
        snapshot = loader(PROJECT_ROOT)
        st.sidebar.caption(f"当前网关 · {snapshot.gateway}")

        title_column, status_column = st.columns([3.3, 1.6], gap="large")
        with title_column:
            st.markdown('<div class="workspace-eyebrow">LOCAL SENSOR WORKSPACE</div>', unsafe_allow_html=True)
            st.title(workspace_title)
            st.caption("数据入库 · 模型训练 · 预测评估 · 版本审计")
        with status_column:
            st.markdown(
                '<div class="workspace-status">'
                f'<span>网关 {escape(snapshot.gateway)}</span>'
                f'<span>模型 {escape(_display_value(snapshot.active_model_version))}</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        navigation_error = (
            navigation_change_handler(navigation)
            if navigation_change_handler is not None
            else None
        )
        if navigation_error:
            st.error(navigation_error)

        if navigation == "系统概览":
            _show_overview(
                snapshot,
                show_coverage_and_quick_access=show_overview_coverage_and_quick_access,
            )
        elif navigation == "模型训练":
            training_renderer()
        elif navigation == "上传预测":
            show_prediction(
                compact_model_labels=compact_model_labels,
                show_audit_details=show_prediction_audit,
                allow_database_prediction=allow_database_prediction,
            )
        elif navigation == "数据入库":
            _show_import(temperature_rule_version=temperature_rule_version,
                         fixed_temperature=processing_version == "final_v3")
        elif navigation == "传感器数据库":
            _show_database(
                compact_model_labels=compact_model_labels,
                simplify_database_view=simplify_database_view,
            )
        else:
            _show_import_history(temperature_rule_version=temperature_rule_version)


if __name__ == "__main__":
    main(
        simplify_database_view=True, temperature_rule_version=PERSONAL_FIXED_TEMPERATURE_RULE_VERSION,
        processing_version="final_v3",
    )
