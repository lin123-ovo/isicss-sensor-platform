"""Edition-scoped derived processing, independent of signed import policies."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal

import pandas as pd

from .errors import SensorDatabaseError


ProcessingVersion = Literal["legacy", "final_v1", "final_v2", "final_v3"]
MINIMUM_TILT_TOLERANCE_DEGREES = 0.02
COMPARISON_ATOL = 1e-12
MAX_MISSING_BUCKETS = 2
MAX_ENDPOINT_MINUTES = 30
FINAL_VALUE_DECIMAL_PLACES = 2
FINAL_VALUE_ROUNDING = ROUND_HALF_UP
_FINAL_VALUE_QUANTUM = Decimal("0.01")
_PROCESSING_VERSION: ContextVar[ProcessingVersion] = ContextVar(
    "sensor_processing_version", default="legacy"
)


def current_processing_version() -> ProcessingVersion:
    """Return the current caller's profile; unrelated sessions remain isolated."""

    return _PROCESSING_VERSION.get()


def is_final_processing(version: str | None = None) -> bool:
    """Whether this workflow uses the personal edition's derived processing."""

    return (version or current_processing_version()) in ("final_v1", "final_v2", "final_v3")


def retained_processed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy public processed rows, retaining excluded observations only in audit."""

    if "processed_row_excluded" not in frame:
        return frame.copy()
    return frame.loc[~frame["processed_row_excluded"].fillna(False).astype(bool)].copy()


def round_final_measurement(value: float) -> float:
    """Round a processed measurement to cents using decimal half-up rounding.

    Missing values remain missing. Call only after anomaly detection and repairs
    so quantization cannot change their decisions or interpolation endpoints.
    """

    return float(Decimal(str(value)).quantize(
        _FINAL_VALUE_QUANTUM, rounding=FINAL_VALUE_ROUNDING
    ))


@contextmanager
def use_processing_version(version: ProcessingVersion) -> Iterator[None]:
    """Select derived processing for one workflow, restoring it even on rerun."""

    if version not in ("legacy", "final_v1", "final_v2", "final_v3"):
        raise SensorDatabaseError(f"不支持的数据处理版本：{version}")
    token = _PROCESSING_VERSION.set(version)
    try:
        yield
    finally:
        _PROCESSING_VERSION.reset(token)


def final_processing_settings(version: ProcessingVersion = "final_v1") -> dict[str, object]:
    """Return the complete versioned settings included in the derived hash."""

    settings = {
        "version": version,
        "minimum_tilt_tolerance_degrees": MINIMUM_TILT_TOLERANCE_DEGREES,
        "comparison_atol": COMPARISON_ATOL,
        "max_missing_buckets": MAX_MISSING_BUCKETS,
        "max_endpoint_minutes": MAX_ENDPOINT_MINUTES,
        "same_day_only": True,
        "observed_non_imputed_endpoints_only": True,
        "value_decimal_places": FINAL_VALUE_DECIMAL_PLACES,
        "value_rounding": FINAL_VALUE_ROUNDING,
        "rounded_fields": [
            "tilt1_inline_model", "tilt1_crossline_model", "temperature_celsius",
            "tilt2_inline", "tilt2_crossline",
        ],
    }
    if version in ("final_v2", "final_v3"):
        settings["target_short_gap_interpolation"] = {
            "max_missing_buckets": MAX_MISSING_BUCKETS,
            "same_day_only": True,
            "exclude_from_training_and_scoring": True,
        }
        settings["zero_temperature"] = {
            "sentinel": 0.0, "stage": "unrounded_bucket_median",
            "max_interpolated_buckets": MAX_MISSING_BUCKETS,
            "unrepairable": "exclude_whole_row_keep_audit",
            "history_barrier": True,
        }
    if version == "final_v3":
        settings["whole_row_short_gap"] = {
            "max_missing_buckets": 2, "allow_cross_day": True,
            "reliable_observed_endpoints": True, "rounding": "decimal_half_up_2",
            "imputed_targets_excluded_from_training_and_scoring": True,
        }
    return settings
