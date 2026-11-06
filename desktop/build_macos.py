"""Build the self-use App in fixed directories without accumulating bundle copies."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sensor_app.model_assets import verified_tabicl_checkpoint, TABICL_CHECKPOINT_SHA256  # noqa: E402

APP_NAME = "工业传感器智能校准与软测量系统.app"


def validate_model_runtime(app: Path, project_root: Path) -> None:
    """Reject a new bundle unless its own frozen runtime can load and archive TabICL."""
    with (app / 'Contents' / 'Info.plist').open('rb') as stream:
        executable = plistlib.load(stream)['CFBundleExecutable']
    report = project_root / 'build' / 'desktop-selfuse' / 'tabicl_options_runtime.json'
    report.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment.update(CSG124_STREAMLIT_ENTRYPOINT=str(project_root / 'scripts' / 'verify_tabicl_options_package.py'),
                       ISICSS_TABICL_OPTIONS_REPORT=str(report))
    subprocess.run([str(app / 'Contents' / 'MacOS' / executable), '--verify-bundle'],
                   cwd=project_root, env=environment, check=True, timeout=300)
    result = json.loads(report.read_text())
    if (result.get('passed') is not True or result.get('frozen') is not True
            or result.get('checkpoint_loaded') is not True
            or result.get('checkpoint_sha256') != TABICL_CHECKPOINT_SHA256
            or len(result.get('checks', [])) != 4):
        raise ValueError('冻结App内TabICLv2权重加载或保存重载检查失败，保留原App。')


def validate_bundle(app: Path) -> None:
    """Require a launchable bundle layout and a valid local code signature."""
    with (app / "Contents" / "Info.plist").open("rb") as stream:
        info = plistlib.load(stream)
    if not isinstance(info, dict):
        raise ValueError("新 App 的 Info.plist 不是字典，保留当前 App。")
    executable = info.get("CFBundleExecutable")
    if (
        not isinstance(executable, str)
        or not executable
        or Path(executable).name != executable
        or executable in {".", ".."}
    ):
        raise ValueError("新 App 的 CFBundleExecutable 无效，保留当前 App。")
    binary = app / "Contents" / "MacOS" / executable
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("新 App 缺少可执行启动文件，保留当前 App。")
    verified_tabicl_checkpoint(app / "Contents" / "Resources")
    subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)], check=True)


def build_app(project_root: Path, *, clean: bool = False) -> Path:
    """Reuse build/cache directories and replace the App only after validation.

    A process lock rejects concurrent self-use builds. Temporary bundles are removed
    on success and failure; a failed replacement restores the previous App. Business
    data, historical backups and the customer App are outside this function's scope.
    """
    build_root = project_root / "build" / "desktop-selfuse"
    build_root.mkdir(parents=True, exist_ok=True)
    with (build_root / "build.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("自用版正在构建，请等待该构建结束。") from error
        return _build_locked(project_root, build_root, clean=clean)


def _build_locked(project_root: Path, build_root: Path, *, clean: bool) -> Path:
    staging = build_root / "staging"
    previous = build_root / "previous.app"
    target = project_root / "dist" / APP_NAME
    target.parent.mkdir(parents=True, exist_ok=True)

    # Recover an interrupted directory replacement before starting another build.
    if previous.exists():
        if target.exists():
            validate_bundle(target)
            shutil.rmtree(previous)
        else:
            previous.rename(target)
    verified_tabicl_checkpoint(project_root)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    environment = os.environ.copy()
    environment["PYINSTALLER_CONFIG_DIR"] = str(build_root / "cache")
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm"]
    if clean:
        command.append("--clean")
    command.extend([
        "--distpath", str(staging),
        "--workpath", str(build_root / "work"),
        str(project_root / "desktop" / "CSG124 Sensor Platform.spec"),
    ])
    try:
        subprocess.run(command, cwd=project_root, env=environment, check=True)
        new_app = staging / APP_NAME
        validate_bundle(new_app)
        validate_model_runtime(new_app, project_root)
        if target.exists():
            target.rename(previous)
        try:
            new_app.rename(target)
        except OSError:
            if previous.exists():
                previous.rename(target)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    finally:
        # PyInstaller also creates a full COLLECT directory next to the .app.
        # Neither that copy nor a failed bundle needs to survive the build.
        shutil.rmtree(staging)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="固定目录构建自用版，仅保留一个正式 App。")
    parser.add_argument("--clean", action="store_true", help="显式清理缓存后重新构建")
    arguments = parser.parse_args()
    try:
        target = build_app(Path(__file__).resolve().parents[1], clean=arguments.clean)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"构建未完成：{error}", file=sys.stderr)
        return 1
    print(f"已生成：{target}（已清理本次中间程序包）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
