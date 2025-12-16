"""
Training issue data classes.

Contains issue types, severity levels, and issue data structure.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional


class IssueType(Enum):
    """Types of training issues that can be detected."""

    # General training issues
    OVERFITTING = auto()
    UNDERFITTING = auto()
    LR_TOO_HIGH = auto()
    LR_TOO_LOW = auto()
    PLATEAU = auto()
    DIVERGENCE = auto()
    GRADIENT_EXPLOSION = auto()

    # Detection-specific issues (YOLO)
    LOW_MAP = auto()
    PRECISION_RECALL_IMBALANCE = auto()
    CLASS_IMBALANCE = auto()

    # Embedding-specific issues (Siamese)
    EMBEDDING_NOT_CONVERGING = auto()
    TRIPLET_LOSS_STUCK = auto()

    # Data issues
    DATA_QUALITY_ISSUE = auto()
    INSUFFICIENT_DATA = auto()

    # Configuration issues
    BATCH_SIZE_TOO_LARGE = auto()
    BATCH_SIZE_TOO_SMALL = auto()


class IssueSeverity(Enum):
    """Severity levels for detected issues."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other: "IssueSeverity") -> bool:
        """Compare severity levels."""
        return self.value < other.value

    def __le__(self, other: "IssueSeverity") -> bool:
        """Compare severity levels."""
        return self.value <= other.value


@dataclass
class Issue:
    """
    Detected training issue.

    Contains issue type, severity, descriptive message, and relevant details.
    """

    issue_type: IssueType
    severity: IssueSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    epoch_detected: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert issue to dictionary.

        Returns:
            Dictionary representation of the issue
        """
        return {
            "issue_type": self.issue_type.name,
            "severity": self.severity.name,
            "message": self.message,
            "details": self.details,
            "epoch_detected": self.epoch_detected,
        }

    def __str__(self) -> str:
        """String representation of the issue."""
        severity_icon = {
            IssueSeverity.LOW: "ℹ️",
            IssueSeverity.MEDIUM: "⚠️",
            IssueSeverity.HIGH: "🔴",
            IssueSeverity.CRITICAL: "🚨",
        }
        icon = severity_icon.get(self.severity, "•")
        epoch_str = f" (epoch {self.epoch_detected})" if self.epoch_detected else ""
        return f"{icon} [{self.severity.name}] {self.issue_type.name}{epoch_str}: {self.message}"
