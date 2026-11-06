"""Environment-backed MySQL configuration for the CSG124 application."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


ENVIRONMENT_KEY_PATTERN = re.compile(r"^(?:CSG124|SENSOR_APP)_[A-Z0-9_]+$")


def load_environment_file(path: Path) -> dict[str, str]:
    """Load approved local runtime keys without overriding process settings.

    The parser intentionally accepts only this application's configuration keys.
    Deployment-provided environment variables remain authoritative, and the
    returned mapping never needs to be logged because it may contain credentials.
    """

    if not path.is_file():
        return {}

    loaded: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").lstrip()
        if "=" not in stripped:
            raise ValueError(f"{path.name} 第 {line_number} 行缺少等号。")

        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not ENVIRONMENT_KEY_PATTERN.fullmatch(key):
            continue
        try:
            tokens = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as error:
            raise ValueError(f"{path.name} 第 {line_number} 行的配置值无效。") from error
        if len(tokens) > 1:
            raise ValueError(f"{path.name} 第 {line_number} 行含有未引用的空格。")
        value = tokens[0] if tokens else ""
        if key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


@dataclass(frozen=True)
class MySQLConfig:
    """Connection settings; credentials are never stored in source files."""

    host: str = "127.0.0.1"
    port: int = 3306
    database: str = "csg124_sensor"
    user: str = "root"
    password: str = ""
    pool_size: int = 5
    unix_socket: str | None = None

    @classmethod
    def from_env(cls) -> "MySQLConfig":
        """Read the CSG124_MYSQL_* environment contract."""

        return cls(
            host=os.getenv("CSG124_MYSQL_HOST", cls.host),
            port=int(os.getenv("CSG124_MYSQL_PORT", str(cls.port))),
            database=os.getenv("CSG124_MYSQL_DATABASE", cls.database),
            user=os.getenv("CSG124_MYSQL_USER", cls.user),
            password=os.getenv("CSG124_MYSQL_PASSWORD", ""),
            pool_size=max(1, int(os.getenv("CSG124_MYSQL_POOL_SIZE", str(cls.pool_size)))),
            unix_socket=os.getenv("CSG124_MYSQL_SOCKET") or None,
        )

    @property
    def url(self) -> str:
        """Return a SQLAlchemy URL with safely escaped credentials."""

        from urllib.parse import quote_plus

        return (
            f"mysql+pymysql://{quote_plus(self.user)}:{quote_plus(self.password)}"
            f"@{self.host}:{self.port}/{quote_plus(self.database)}"
            "?charset=utf8mb4"
        )

    def engine(self) -> Engine:
        """Create a pooled engine with bounded connection wait time."""

        connect_args = {"connect_timeout": 10, "read_timeout": 60, "write_timeout": 60}
        if self.unix_socket:
            connect_args["unix_socket"] = self.unix_socket
        return create_engine(
            self.url,
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=self.pool_size,
            max_overflow=max(2, self.pool_size),
            connect_args=connect_args,
        )


def check_mysql_connection(config: MySQLConfig | None = None) -> str:
    """Return the server version or raise a concise connection error."""

    selected = config or MySQLConfig.from_env()
    try:
        with selected.engine().connect() as connection:
            return str(connection.execute(text("SELECT VERSION()")).scalar_one())
    except Exception as error:  # SQLAlchemy normalizes driver-specific errors.
        raise RuntimeError(
            f"无法连接 MySQL {selected.host}:{selected.port}/{selected.database}：{error}"
        ) from error
