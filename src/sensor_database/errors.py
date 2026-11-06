"""Project-specific failures that can be shown safely in the UI."""


class SensorDatabaseError(RuntimeError):
    """Raised when uploaded data or CSV persistence violates the contract."""
