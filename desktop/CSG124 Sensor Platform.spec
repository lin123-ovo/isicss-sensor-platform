# PyInstaller configuration for the Apple Silicon desktop build.

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

project_root = Path(SPECPATH).parent
src_root = project_root / "src"

hiddenimports = [
    "prediction_ui",
    "streamlit.web.bootstrap",
    "streamlit.runtime.scriptrunner",
    "webview.platforms.cocoa",
    "pymysql",
    "catboost",
    "catboost._catboost",
    "tabicl",
    "tabpfn",
    "tabm",
    "rtdl_num_embeddings",
    "tabicl._sklearn.classifier",
    "torch",
    "sqlalchemy.dialects.mysql.pymysql",
]
hiddenimports += collect_submodules("streamlit.runtime.scriptrunner")
hiddenimports += collect_submodules("tabpfn")
hiddenimports += collect_submodules("sensor_database")
hiddenimports += collect_submodules("tilt_predictor")
hiddenimports += collect_submodules("streamlit_shadcn_ui")
hiddenimports += collect_submodules("scipy._external.array_api_compat")

datas = [
    (str(project_root / "app.py"), "."),
    (str(src_root), "src"),
]
datas += copy_metadata("streamlit")
datas += copy_metadata("catboost")
for package in ("tabm", "rtdl_num_embeddings", "tabicl", "torch", "scikit-learn", "numpy",
                "einops", "huggingface_hub", "psutil", "tabpfn", "safetensors", "pydantic",
                "pydantic-settings", "skrub", "lightgbm"):
    datas += copy_metadata(package)
# Explicitly require the weight file: a directory containing only README is not a full App.
datas.append((str(project_root / "assets" / "tabicl" / "tabicl-classifier-v2-20260212.ckpt"), "assets/tabicl"))
datas.append((str(project_root / "assets" / "tabicl" / "README.md"), "assets/tabicl"))
datas += collect_data_files("skrub")
datas += collect_data_files("streamlit")
datas += collect_data_files("streamlit_shadcn_ui")
datas += copy_metadata("streamlit_shadcn_ui")
for directory_name in ("configs",):
    directory = project_root / directory_name
    if directory.is_dir():
        datas.append((str(directory), directory_name))

a = Analysis(
    [str(project_root / "desktop_launcher.py")],
    pathex=[str(project_root), str(src_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
executable = EXE(
    pyz,
    a.scripts,
    [],
    name="CSG124 Sensor Platform",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    exclude_binaries=True,
)
collected = COLLECT(
    executable,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="工业传感器智能校准与软测量系统",
)
app = BUNDLE(
    collected,
    name="工业传感器智能校准与软测量系统.app",
    icon=None,
    bundle_identifier="local.csg124.sensor-platform",
    info_plist={
        "CFBundleName": "工业传感器智能校准与软测量系统",
        "CFBundleDisplayName": "工业传感器智能校准与软测量系统",
    },
)
