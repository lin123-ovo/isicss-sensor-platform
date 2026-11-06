"""Launch the local Streamlit application inside a native macOS window."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sensor_app.runtime_paths import (
    default_data_root,
    initialize_data_root,
)
from sensor_app.mysql_config import load_environment_file


APP_TITLE = os.environ.get("CSG124_APP_TITLE", "工业传感器智能校准与软测量系统")
STREAMLIT_ENTRYPOINT = os.environ.get("CSG124_STREAMLIT_ENTRYPOINT", "app.py")
ALLOW_PROJECT_ENV = os.environ.get("CSG124_ALLOW_PROJECT_ENV", "1") == "1"
MIN_WIDTH = 1180
MIN_HEIGHT = 760


def find_free_port() -> int:
    """Reserve a currently unused local TCP port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def load_runtime_environment(
    *,
    data_root: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, str]:
    """Load external App configuration before the source-tree fallback.

    Credentials are deliberately not bundled inside the macOS application.
    A packaged App reads ``.env`` from its Application Support directory,
    while source-tree development can still use the project-local file.
    """

    selected_data_root = data_root or default_data_root()
    candidates = [selected_data_root / ".env"]
    if ALLOW_PROJECT_ENV:
        candidates.append(project_root / ".env")
    loaded: dict[str, str] = {}
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        loaded.update(load_environment_file(resolved))
    return loaded


def _health_url_is_ready(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=0.5) as response:
            return response.status == 200 and response.read(32).strip() == b"ok"
    except (OSError, URLError):
        return False


def wait_for_server(port: int, *, timeout_seconds: float = 30.0) -> None:
    """Wait for Streamlit's health endpoint or raise a useful startup error."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _health_url_is_ready(port):
            return
        time.sleep(0.1)
    raise RuntimeError(f"本地传感器服务在 {timeout_seconds:.0f} 秒内没有启动。")


def _streamlit_child(argv: list[str]) -> int:
    """Run Streamlit in a child process, including from a frozen executable."""

    from streamlit.web import bootstrap

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--streamlit-child", action="store_true")
    parser.add_argument("--port", type=int, required=True)
    args, _ = parser.parse_known_args(argv)
    if not args.streamlit_child:
        return 2
    flag_options = {
        "server_address": "127.0.0.1",
        "server_port": args.port,
        "server_headless": True,
        "browser_gatherUsageStats": False,
        "global_developmentMode": False,
    }
    bootstrap.load_config_options(flag_options)
    from streamlit import config

    config.set_option("server.address", "127.0.0.1", "command-line")
    config.set_option("server.port", args.port, "command-line")
    config.set_option("server.headless", True, "command-line")
    config.set_option("browser.gatherUsageStats", False, "command-line")
    config.set_option("global.developmentMode", False, "command-line")
    bootstrap.run(str(PROJECT_ROOT / STREAMLIT_ENTRYPOINT), False, [], flag_options)
    return 0


def _child_command(port: int) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--streamlit-child", "--port", str(port)]
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--streamlit-child",
        "--port",
        str(port),
    ]


def start_streamlit(port: int, data_root: Path) -> subprocess.Popen[bytes]:
    """Start a loopback-only Streamlit child without opening a browser."""

    environment = os.environ.copy()
    environment.update(
        {
            "SENSOR_APP_DATA_ROOT": str(data_root),
            "STREAMLIT_SERVER_HEADLESS": "true",
            "STREAMLIT_SERVER_ADDRESS": "127.0.0.1",
            "STREAMLIT_SERVER_PORT": str(port),
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        }
    )
    kwargs: dict[str, object] = {
        "env": environment,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "posix":
        kwargs["start_new_session"] = True
    return subprocess.Popen(_child_command(port), **kwargs)


def stop_streamlit(process: subprocess.Popen[bytes]) -> None:
    """Terminate Streamlit and its process group, tolerating an already closed child."""

    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.kill()


def _open_window(port: int) -> None:
    import webview

    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.create_window(
        APP_TITLE,
        f"http://127.0.0.1:{port}",
        width=MIN_WIDTH,
        height=MIN_HEIGHT,
        min_size=(MIN_WIDTH, MIN_HEIGHT),
        confirm_close=True,
    )
    webview.start(debug=False)


def run_desktop_app() -> int:
    """Migrate data, host Streamlit, and run the native application window."""

    data_root = default_data_root()
    initialize_data_root(PROJECT_ROOT, data_root)
    port = find_free_port()
    process = start_streamlit(port, data_root)
    try:
        wait_for_server(port)
        _open_window(port)
    finally:
        stop_streamlit(process)
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    load_runtime_environment()
    if "--verify-bundle" in arguments:
        runpy.run_path(
            str(PROJECT_ROOT / STREAMLIT_ENTRYPOINT),
            run_name="bundle_validation",
        )
        return 0
    if "--streamlit-child" in arguments:
        return _streamlit_child(arguments)
    return run_desktop_app()


if __name__ == "__main__":
    # Frozen multiprocessing helpers must not launch a second desktop server.
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
