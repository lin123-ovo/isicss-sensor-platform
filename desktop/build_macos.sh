#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${0:A:h:h}"
cd "$PROJECT_ROOT"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "此构建脚本只支持 Apple Silicon arm64 macOS。" >&2
  exit 2
fi

PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "找不到项目 Python：$PYTHON" >&2
  exit 2
fi

exec "$PYTHON" "$PROJECT_ROOT/desktop/build_macos.py" "$@"
