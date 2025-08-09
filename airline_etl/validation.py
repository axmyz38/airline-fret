from datetime import datetime


def normalize_date(value: str) -> str:
    """Return an ISO formatted datetime string."""
    dt = datetime.fromisoformat(value)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def normalize_weight(value: float, unit: str) -> float:
    """Convert weight to kilograms if necessary."""
    if unit.lower() in {"lb", "lbs"}:
        return value * 0.45359237
    return value
