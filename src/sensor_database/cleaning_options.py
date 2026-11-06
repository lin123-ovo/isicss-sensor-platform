"""批次温度清洗设置：严格校验、版本和签名，历史空设置解释为关闭。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .errors import SensorDatabaseError
from .temperature_quality import temperature_check_settings
from .temperature_cusum import RULE_VERSION as CUSUM_RULE_VERSION
from .temperature_daily_trend import RULE_VERSION as DAILY_RULE_VERSION


RULE_VERSION = "temperature_jump_v1"
REPAIR_RULE_VERSION = "temperature_jump_v2"
CURRENT_REPAIR_RULE_VERSION = DAILY_RULE_VERSION
PERSONAL_FIXED_TEMPERATURE_RULE_VERSION = "temperature_jump_v3"
SUPPORTED_RULE_VERSIONS = (RULE_VERSION, REPAIR_RULE_VERSION, "temperature_jump_v3", CUSUM_RULE_VERSION,
                           CURRENT_REPAIR_RULE_VERSION)


def _digest(document: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CleaningOptions:
    """一份完整设置；签名包含基础 Hampel 策略，但不包含会变化的结果统计。"""

    enabled: bool
    base_policy_sha256: str
    rule_version: str = RULE_VERSION

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise SensorDatabaseError("温度突变检查开关必须为布尔值。")
        if not re.fullmatch(r"[0-9a-f]{64}", self.base_policy_sha256):
            raise SensorDatabaseError("基础清洗策略签名无效。")
        if self.rule_version not in SUPPORTED_RULE_VERSIONS:
            raise SensorDatabaseError("不支持的温度规则版本。")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "temperature_jump_check_enabled": self.enabled,
            "rule_version": self.rule_version,
            "parameters": temperature_check_settings(self.rule_version),
            "base_policy_sha256": self.base_policy_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.document())

    def to_json(self, counts: dict[str, int] | None = None) -> str:
        """输出可审计 JSON；计数是导入时快照，不是后续重算的实时值。"""

        return json.dumps({
            **self.document(), "profile_sha256": self.sha256, "counts": counts or {}
        }, ensure_ascii=False, sort_keys=True)


def parse_cleaning_options(value: object) -> CleaningOptions | None:
    """仅 None/空字符串表示旧批次关闭；损坏配置不得静默降级。"""

    if value is None or value == "":
        return None
    try:
        payload = json.loads(value) if isinstance(value, str) else value
        if not isinstance(payload, dict):
            raise ValueError("配置必须为 JSON 对象")
        if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
            raise ValueError("不支持的设置版本")
        if payload.get("rule_version") not in SUPPORTED_RULE_VERSIONS:
            raise ValueError("不支持的温度规则版本")
        if payload.get("parameters") != temperature_check_settings(payload["rule_version"]):
            raise ValueError("温度规则参数不匹配")
        options = CleaningOptions(
            payload["temperature_jump_check_enabled"], payload["base_policy_sha256"],
            payload["rule_version"],
        )
        if payload.get("profile_sha256") != options.sha256:
            raise ValueError("清洗配置签名不匹配")
        counts = payload.get("counts", {})
        if not isinstance(counts, dict) or any(
            type(count) is not int or count < 0 for count in counts.values()
        ):
            raise ValueError("清洗统计无效")
        return options
    except (KeyError, ValueError, TypeError) as error:
        raise SensorDatabaseError(f"批次清洗设置损坏或不受支持：{error}") from error
