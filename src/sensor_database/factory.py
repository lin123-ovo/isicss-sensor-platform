"""Select the explicitly configured persistence backend."""

from __future__ import annotations

from pathlib import Path

from .errors import SensorDatabaseError
from .mysql_database import MySQLSensorDatabase


def get_sensor_database(project_root: Path):
    """Return the App's MySQL database; local CSV storage is not supported."""

    from os import getenv

    backend = getenv("CSG124_STORAGE_BACKEND", "mysql").strip().lower()
    if backend != "mysql":
        raise SensorDatabaseError(
            "CSG124 桌面 App 只支持 MySQL，禁止使用本地 CSV 数据库。"
        )
    database = MySQLSensorDatabase.from_env(project_root)
    try:
        with database.engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1 FROM csg124_schema_meta LIMIT 1")
    except Exception as error:
        raise SensorDatabaseError(
            "MySQL 后端未就绪；请检查连接配置和数据库结构："
            f"{error}"
        ) from error
    return database
