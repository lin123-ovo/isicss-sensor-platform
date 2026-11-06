"""Display-only two-decimal formatting for the self-use edition's tables."""

from copy import deepcopy
from decimal import Decimal, InvalidOperation
from numbers import Real
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
import streamlit as st

from sensor_database.processing import is_final_processing


def two_decimal_text(value: object) -> str:
    """Format a numeric display cell, including already formatted coefficients."""

    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return str(value)
    if not number.is_finite() or number.adjusted() > 308:
        return str(value)
    rendered = f"{number:.2f}"
    return "0.00" if rendered == "-0.00" else rendered


def _text_numeric_column(name: object) -> bool:
    label = str(name).lower()
    return any(token in label for token in (
        "epsilon", "alpha", "l1_ratio", "max_iter", "coefficient", "intercept",
        "系数", "截距", "数值", "误差", "mae", "rmse", "objective",
        "1号顺线", "1号横线", "2号顺线", "2号横线", "温度",
    ))


def prepare_table_display(
    data: pd.DataFrame, column_config: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build independent display data/config; leave numbers and caller config intact."""

    displayed = data.copy(deep=True)
    config = deepcopy(column_config) if column_config is not None else {}
    for name in displayed.columns:
        column = displayed[name]
        if is_bool_dtype(column.dtype):
            continue
        if is_numeric_dtype(column.dtype):
            if name in config and config[name] is None:
                continue
            existing = config.get(name)
            if isinstance(existing, dict):
                type_config = existing.get("type_config", {})
                existing_format = type_config.get("format") or ""
                if existing_format.endswith("%%") or existing_format == "percent":
                    existing["type_config"]["format"] = "%.2%%"
                    continue
                if type_config.get("type", "number") != "number":
                    continue
                existing.setdefault("type_config", {})["format"] = "%.2f"
                existing["type_config"]["type"] = "number"
            else:
                config[name] = st.column_config.NumberColumn(
                    label=existing if isinstance(existing, str) else None, format="%.2f"
                )
        else:
            text_numbers = _text_numeric_column(name)
            displayed[name] = column.map(
                lambda value: two_decimal_text(value)
                if (isinstance(value, (Real, Decimal))
                    or (text_numbers and isinstance(value, str)))
                and not isinstance(value, bool) and pd.notna(value)
                else value
            )
    return displayed, config


def display_dataframe(data: Any, **kwargs: Any) -> Any:
    """Render all full-edition numeric columns with two decimals."""

    if is_final_processing() and isinstance(data, pd.DataFrame):
        data, kwargs["column_config"] = prepare_table_display(data, kwargs.get("column_config"))
    return st.dataframe(data, **kwargs)


def display_data_editor(data: Any, **kwargs: Any) -> Any:
    """Apply the same formatting to disabled numeric cells while preserving edits."""

    if is_final_processing() and isinstance(data, pd.DataFrame):
        data, kwargs["column_config"] = prepare_table_display(data, kwargs.get("column_config"))
    return st.data_editor(data, **kwargs)
