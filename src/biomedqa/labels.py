"""Answer labels shared by every system and the scorer."""

from typing import Final

LABELS: Final[tuple[str, str, str]] = ("yes", "no", "maybe")
INVALID: Final[str] = "invalid"
PREDICTION_VALUES: Final[tuple[str, ...]] = (*LABELS, INVALID)

LABEL_TO_ID: Final[dict[str, int]] = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL: Final[dict[int, str]] = dict(enumerate(LABELS))


def check_gold(label: str) -> str:
    """Return ``label`` if it is a valid gold label, else raise ``ValueError``."""
    if label not in LABELS:
        raise ValueError(f"gold label must be one of {LABELS}, got {label!r}")
    return label


def check_prediction(label: str) -> str:
    """Return ``label`` if it is a valid prediction (a gold label or ``invalid``)."""
    if label not in PREDICTION_VALUES:
        raise ValueError(f"prediction must be one of {PREDICTION_VALUES}, got {label!r}")
    return label
