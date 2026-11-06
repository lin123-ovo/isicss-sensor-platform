"""Version-specific linear equations and tree feature importances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def tabicl_settings_summary(metadata: Mapping[str, Any]) -> str:
    """Describe the saved model's context and learned temperature, including legacy defaults."""
    params = metadata.get('parameters', {}).get('directions', {}).get('crossline', {})
    if params.get('model') not in ('tabicl_v2_single36', 'tabicl_v2_single36_16'):
        return ''
    info = metadata.get('compact36_training_info', {})
    total = info.get('training_n', metadata.get('sample_counts', {}).get('final_fit', '—'))
    count = info.get('context_n', total)
    fraction = params.get('context_fraction', 1.)
    enabled = params.get('temperature_calibration', False)
    temperature = info.get('calibration', {}).get('temperature', 1.)
    result = (f"上下文保留 {fraction:.0%}（{count}/{total}条）｜"
              f"NLL温度校准：{'已启用' if enabled else '关闭'}｜温度 T={temperature:.6g}")
    calibration = info.get('calibration', {})
    if enabled and 'calibration_used_n' in calibration:
        result += (f"｜有效校准 {calibration['calibration_used_n']}/{calibration['calibration_n']}条"
                   f"（覆盖率{calibration['calibration_coverage']:.1%}），未覆盖{calibration['calibration_uncovered_n']}条")
        if calibration['uncovered_class_counts']:
            detail = '、'.join(f"{-0.10+.01*item['class']:.2f}° {item['count']}条"
                              for item in calibration['uncovered_class_counts'])
            result += f"（{detail}）；NLL仅统计已覆盖记录"
    return result


def tabicl_calibration_rows(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expose final and per-fold conditional NLL coverage without hiding unsupported bins."""
    records = [{'stage': entry['fold'], **entry} for entry in
               metadata.get('hyperparameter_optimization', {}).get('by_direction', {}).get('crossline', {}).get('calibration_folds', [])]
    final = metadata.get('compact36_training_info', {}).get('calibration', {})
    if final.get('enabled'):
        records.append({'stage': '最终保存模型', **final})
    return [{'阶段': entry['stage'], '温度': entry['temperature'], '校准总数': entry['calibration_n'],
             '有效校准': entry.get('calibration_used_n', entry['calibration_n']),
             '未覆盖': entry.get('calibration_uncovered_n', 0),
             '覆盖率(%)': 100*entry.get('calibration_coverage', 1.),
             '未覆盖档位': '、'.join(f"{-0.10+.01*item['class']:.2f}° {item['count']}条"
                                     for item in entry.get('uncovered_class_counts', [])) or '无',
             'NLL校准前（有效记录）': entry['nll_before'], 'NLL校准后（有效记录）': entry['nll_after']}
            for entry in records]


MODEL_STRUCTURE_LABELS = {
    "tilt_temperature_tabicl_single36_v1": "横线 TabICLv2（独立单模型）",
    "tilt_temperature_tabm_single36_v1": "横线 TabM（独立单模型）",
    "tilt_temperature_nine_search_v1": "横线 TabICLv2＋TabM（九档自动比例）",
    "tilt_temperature_seven_regional_v1": "横线 TabICLv2＋TabM（七档分段融合）",
    "tilt_temperature_compact36_v1": "横线 TabICLv2＋TabM（36维组合）",
    "tilt_temperature_v1": "三特征绝对值 Huber",
    "tilt_temperature_residual_v1": (
        "顺线三特征 Huber｜横线 Huber + 输入侧时序残差模型"
    ),
    "tilt_hybrid_12h_v2": "顺线绝对值＋横线12小时局部基准变化量 Huber",
}

FEATURE_LABELS = {
    "intercept": "截距",
    "tilt1_inline_model": "1号顺线",
    "tilt1_crossline_model": "1号横线",
    "temperature_celsius": "温度",
    "tilt1_inline_delta": "1号顺线变化量",
    "tilt1_crossline_delta": "1号横线变化量",
    "temperature_delta": "温度变化量",
    "temperature_change_10m": "10分钟温差",
    "temperature_change_1h": "1小时温差",
    "temperature_rolling_mean_1h_delta": "1小时平均温度变化量",
    "tilt1_inline_model^2": "1号顺线²",
    "tilt1_inline_model tilt1_crossline_model": "1号顺线×1号横线",
    "tilt1_inline_model temperature_celsius": "1号顺线×温度",
    "tilt1_crossline_model^2": "1号横线²",
    "tilt1_crossline_model temperature_celsius": "1号横线×温度",
    "temperature_celsius^2": "温度²",
}

FEATURE_UNITS = {
    "intercept": "—",
    "tilt1_inline_model": "°",
    "tilt1_crossline_model": "°",
    "temperature_celsius": "°C",
    "tilt1_inline_delta": "°",
    "tilt1_crossline_delta": "°",
    "temperature_delta": "°C",
    "temperature_change_10m": "°C",
    "temperature_change_1h": "°C",
    "temperature_rolling_mean_1h_delta": "°C",
    "tilt1_inline_model^2": "°²",
    "tilt1_inline_model tilt1_crossline_model": "°²",
    "tilt1_inline_model temperature_celsius": "°·°C",
    "tilt1_crossline_model^2": "°²",
    "tilt1_crossline_model temperature_celsius": "°·°C",
    "temperature_celsius^2": "°C²",
}

TARGET_LABELS = {
    "tilt2_inline": "2号顺线",
    "tilt2_crossline": "2号横线",
    "tilt2_crossline_delta": "2号横线变化量",
}

TARGET_OUTPUTS = {
    "tilt2_inline": "2号顺线绝对倾角",
    "tilt2_crossline": "2号横线绝对倾角",
    "tilt2_crossline_delta": "2号横线相对局部基准的变化量",
}


@dataclass(frozen=True)
class ModelExplanation:
    """Display-ready explanation derived only from one version's metadata."""

    structureLabel: str
    formulas: tuple[str, ...]
    coefficientRows: tuple[dict[str, str], ...]
    calibrationSummary: str | None
    usesLocalBaseline: bool
    featureImportanceRows: tuple[dict[str, str | float], ...] = ()


def model_structure_label(
    featureVersion: str,
    calibrationPolicy: Mapping[str, Any] | None = None,
) -> str:
    """Return a readable structure label without rejecting unknown versions."""

    if featureVersion == "tilt_hybrid_12h_v2" and calibrationPolicy:
        period = calibrationPolicy.get("period_hours", 12)
        return f"顺线绝对值＋横线{period}小时局部基准变化量 Huber"
    return MODEL_STRUCTURE_LABELS.get(featureVersion, featureVersion)


def format_coefficient(value: float) -> str:
    """Format six significant digits while preserving tiny non-zero values."""

    return f"{float(value):.6g}"


def build_model_explanation(metadata: Mapping[str, Any]) -> ModelExplanation:
    """Build formulas and long-form coefficient rows for one saved version."""

    featureVersion = str(metadata.get("feature_version", "UNKNOWN"))
    structure_label = model_structure_label(featureVersion, _calibration_policy(metadata))
    if featureVersion in ("tilt_temperature_v1", "tilt_temperature_probability_v1", "tilt_temperature_compact36_v1", "tilt_temperature_seven_regional_v1", "tilt_temperature_nine_search_v1", "tilt_temperature_tabicl_single36_v1", "tilt_temperature_tabm_single36_v1") and metadata.get("parameters"):
        from .temperature_pipeline import model_combination_label

        structure_label = model_combination_label(metadata["parameters"])
    rawEquations = metadata.get("raw_input_equations")
    if not isinstance(rawEquations, Mapping):
        rawEquations = {}

    directionFeatures = metadata.get("direction_feature_columns", {})
    derivedFeatures = metadata.get("derived_feature_columns", {})
    if isinstance(directionFeatures, Mapping) and isinstance(derivedFeatures, Mapping):
        directionFeatures = {**directionFeatures, **derivedFeatures}
    formulas: list[str] = []
    rows: list[dict[str, str]] = []
    importance_rows: list[dict[str, str | float]] = []
    usesLocalBaseline = _uses_local_baseline(metadata)
    for target, rawEquation in rawEquations.items():
        if not isinstance(rawEquation, Mapping):
            continue
        equation = {
            str(name): float(value)
            for name, value in rawEquation.items()
        }
        orderedFeatures = _ordered_features(
            str(target), equation, directionFeatures
        )
        formulas.append(
            _formula_text(
                str(target),
                equation,
                orderedFeatures,
                crosslineIsBase=(
                    featureVersion == "tilt_temperature_residual_v1"
                    and str(target) == "tilt2_crossline"
                ),
            )
        )
        for name in ("intercept", *orderedFeatures):
            if name not in equation:
                continue
            rows.append(
                {
                    "预测目标": TARGET_LABELS.get(str(target), str(target)),
                    "特征含义": FEATURE_LABELS.get(name, name),
                    "内部字段名": name,
                    "数值": format_coefficient(equation[name]),
                    "输入单位": FEATURE_UNITS.get(name, "未声明"),
                    "输出含义": TARGET_OUTPUTS.get(str(target), str(target)),
                }
            )

    importances = metadata.get("feature_importances", {})
    if isinstance(importances, Mapping):
        for target, values in importances.items():
            if not isinstance(values, Mapping):
                continue
            formulas.insert(0 if target == "tilt2_inline" else len(formulas),
                f"{TARGET_LABELS.get(target, target)}预测值 = "
                "CatBoost(1号顺线, 1号横线, 温度)"
            )
            importance_rows.extend({
                "预测目标": TARGET_LABELS.get(target, target),
                "特征": FEATURE_LABELS.get(feature, feature),
                "特征重要性(%)": float(value),
            } for feature, value in values.items())

    if metadata.get("parameters", {}).get("directions", {}).get("crossline", {}).get("model") == "tabpfn_v2":
        formulas.append("2号横线预测值 = TabPFN v2(1号顺线, 1号横线, 温度；最近最多1000条训练样本)")

    settings_summary = tabicl_settings_summary(metadata)
    if settings_summary:
        formulas.append(settings_summary)
    if featureVersion in ('tilt_temperature_tabicl_single36_v1', 'tilt_temperature_tabm_single36_v1'):
        formulas.append('2号横线预测值 = 单模型九档概率中预计绝对相对误差最小的档位')
    if featureVersion == 'tilt_temperature_nine_search_v1':
        cross = metadata.get('parameters', {}).get('directions', {}).get('crossline', {})
        values = [float(cross[k]) for k in ('negative_weight', 'middle_weight', 'near_zero_weight')]
        ratios = '，'.join(f'{label}{100*w:.1f}%/{100*(1-w):.1f}%'
                          for label,w in zip(('负向','中间','靠零'), values, strict=True))
        formulas.extend(['横线九档融合：保留−0.10°和−0.02°端点，融合后逐行归一化',
                         f'TabICLv2/TabM比例：{ratios}；比例由该版本训练期搜索确定',
                         '2号横线预测值 = 九档中弱校正预计相对风险最小的档位'])
    if featureVersion == 'tilt_temperature_seven_regional_v1':
        formulas.extend([
            '横线七档融合：每路先去掉−0.10°和−0.02°概率，再归一化',
            'TabICLv2/TabM比例：负向52.5%/47.5%，中间82.5%/17.5%，靠零90%/10%；融合后逐行归一化',
            'TabICLv2使用75%工况覆盖上下文，TabM使用完整训练池；三段比例与原实验弱风险校正固定',
            '2号横线预测值 = 七档中原弱校正预计相对风险最小的档位（−0.09至−0.03，步长0.01）',
        ])
    if featureVersion == 'tilt_temperature_compact36_v1':
        formulas.append('横线九档概率 = 75% × TabICLv2概率 + 25% × TabM概率（两个分支均使用36维）')
        formulas.append('36维 = 25项温度历史 + 30/360分钟温度状态5项 + 两轴60/180分钟均值偏移4项 + 温度残差2项')
        formulas.append('温度残差 = 当前1号倾角 − Ridge(标准化当前温度、快慢状态；alpha=10，仅拟合训练段)')
        formulas.append('2号横线预测值 = 九档中预计绝对相对误差最小的档位（−0.10至−0.02，步长0.01）')
    if featureVersion == 'tilt_temperature_probability_v1':
        formulas.append('横线九档概率 = 50% × TabICLv2概率 + 50% × CatBoost概率（当前及过去3小时的25项输入特征）')
        formulas.append('2号横线预测值 = 九档中预计绝对相对误差最小的档位（−0.10至−0.02，步长0.01）')
    if usesLocalBaseline:
        formulas.append(
            "横线绝对预测值 = 2号横线局部基准 + 横线变化量预测值"
        )
    if featureVersion == "tilt_temperature_residual_v1":
        scale = (
            metadata.get("crossline_residual", {})
            .get("parameters", {})
            .get("correction_scale", "未声明")
        )
        formulas.append(
            f"2号横线最终预测值 = Huber基础预测值 + {scale} × 残差模型预测值"
        )
    return ModelExplanation(
        structureLabel=structure_label,
        formulas=tuple(formulas),
        coefficientRows=tuple(rows),
        calibrationSummary=_calibration_summary(metadata),
        usesLocalBaseline=usesLocalBaseline,
        featureImportanceRows=tuple(importance_rows),
    )


def _ordered_features(
    target: str,
    equation: Mapping[str, float],
    directionFeatures: Any,
) -> tuple[str, ...]:
    direction = "crossline" if "crossline" in target else "inline"
    configured = ()
    if isinstance(directionFeatures, Mapping):
        rawConfigured = directionFeatures.get(direction, ())
        if isinstance(rawConfigured, (list, tuple)):
            configured = tuple(str(name) for name in rawConfigured)
    remaining = tuple(
        name
        for name in equation
        if name != "intercept" and name not in configured
    )
    return tuple(name for name in configured if name in equation) + remaining


def _formula_text(
    target: str,
    equation: Mapping[str, float],
    orderedFeatures: tuple[str, ...],
    *,
    crosslineIsBase: bool = False,
) -> str:
    if target == "tilt2_inline":
        output = "2号顺线预测值"
    elif target == "tilt2_crossline":
        output = "2号横线Huber基础预测值" if crosslineIsBase else "2号横线预测值"
    elif target == "tilt2_crossline_delta":
        output = "横线变化量预测值"
    else:
        output = target
    terms = [format_coefficient(equation.get("intercept", 0.0))]
    terms.extend(
        f"({format_coefficient(equation[name])} × {FEATURE_LABELS.get(name, name)})"
        for name in orderedFeatures
    )
    return f"{output} = " + " + ".join(terms)


def _uses_local_baseline(metadata: Mapping[str, Any]) -> bool:
    contracts = metadata.get("direction_contracts", {})
    return isinstance(contracts, Mapping) and (
        contracts.get("crossline") == "local_baseline_delta_huber"
    )


def _calibration_summary(metadata: Mapping[str, Any]) -> str | None:
    policy = _calibration_policy(metadata)
    if not isinstance(policy, Mapping) or not policy:
        return None
    return (
        f"横线校准：每 {policy.get('period_hours', '未声明')} 小时一个周期，"
        f"前 {policy.get('window_minutes', '未声明')} 分钟建立局部基准，"
        f"最少 {policy.get('minimum_samples', '未声明')} 条样本。"
    )


def _calibration_policy(
    metadata: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return a mapping-shaped calibration policy when metadata provides one."""

    policy = metadata.get("calibration_policy")
    return policy if isinstance(policy, Mapping) else None
