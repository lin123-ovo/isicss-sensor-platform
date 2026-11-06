# ISICSS | Industrial Sensor Calibration and Soft Sensing

ISICSS is a desktop workbench for importing industrial sensor measurements, reviewing data quality, training and versioning models, and estimating a second sensor's two-axis tilt from a first sensor's tilt and temperature.

## Features

- Review measurement trends, import history, and the active model.
- Import CSV or XLSX files with field mapping and batch audit records.
- Review cleaning decisions and export predictions for evaluation.
- Train, save, and activate model versions as separate actions.

The interface uses Streamlit. The macOS launcher opens it in a desktop window. MySQL stores measurements, import history, and model artifacts.

## Local setup

Use Python 3.14 and a local MySQL server. Create an empty database named `isicss_local` and a dedicated account with schema and data access to that database. From the repository root, create a virtual environment and install the dependencies.

```sh
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit the local `.env` with your own MySQL connection details. Then initialize the schema and launch the desktop workbench. The runtime directory keeps this checkout's settings separate from other local installations.

```sh
export SENSOR_APP_DATA_ROOT="$PWD/runtime"
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from sensor_app.mysql_config import load_environment_file
from sensor_database import MySQLSensorDatabase

load_environment_file(Path(".env"))
MySQLSensorDatabase.from_env().ensure_schema()
PY
python desktop_launcher.py
```

The overview starts empty. Import measurements you are authorized to use, then train and activate a local model before running predictions. The legacy file-based experiment requires a local data configuration. Its dated reference window in this export is an example.

## Source distribution

This repository contains the self-use edition's application source, schema initialization code, and macOS build scripts. The distribution excludes measurement files, database exports, project-specific configuration, trained artifacts, credentials, and compiled application bundles. See the [checkpoint setup](assets/tabicl/README.md) for checkpoint-backed models and the complete macOS build.
