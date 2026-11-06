"""Upload chooser options shared by the Streamlit entry points."""

from __future__ import annotations

import sys


SUPPORTED_TABULAR_FILE_TYPES = ("csv", "xlsx")


def native_upload_file_types(
    *,
    is_frozen: bool | None = None,
    platform_name: str | None = None,
) -> tuple[str, str] | None:
    """Return safe file chooser filters for the current desktop runtime.

    The macOS pywebview runtime sends Streamlit's CSV/XLSX MIME filter through
    an incompatible native-panel branch. This affects both packaged and
    source-tree desktop launches: the panel greys out valid spreadsheets, so
    macOS deliberately shows all files. Each reader still validates the
    selected filename and content before processing.
    """

    # Keep ``is_frozen`` in the public seam for callers and regression tests.
    # The pywebview bug is platform-specific, not packaging-specific.
    current_platform = sys.platform if platform_name is None else platform_name
    if current_platform == "darwin":
        return None
    return SUPPORTED_TABULAR_FILE_TYPES
