"""
Training recommendation data classes.

Contains recommendation types and data structure.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional

from .issues import IssueType


class RecommendationType(Enum):
    """Types of recommendations that can be made."""

    # Learning rate adjustments
    REDUCE_LR = auto()
    INCREASE_LR = auto()
    USE_LR_SCHEDULER = auto()
    USE_WARMUP = auto()

    # Regularization
    ADD_REGULARIZATION = auto()
    INCREASE_DROPOUT = auto()
    REDUCE_DROPOUT = auto()
    ADD_WEIGHT_DECAY = auto()

    # Training control
    EARLY_STOPPING = auto()
    INCREASE_EPOCHS = auto()
    REDUCE_EPOCHS = auto()

    # Data augmentation
    INCREASE_DATA_AUGMENTATION = auto()
    REDUCE_DATA_AUGMENTATION = auto()

    # Model architecture
    REDUCE_MODEL_COMPLEXITY = auto()
    INCREASE_MODEL_CAPACITY = auto()

    # Batch size
    ADJUST_BATCH_SIZE = auto()
    INCREASE_BATCH_SIZE = auto()
    REDUCE_BATCH_SIZE = auto()

    # Data handling
    CHECK_DATA_QUALITY = auto()
    BALANCE_CLASSES = auto()
    ADD_MORE_DATA = auto()

    # Detection-specific
    ADJUST_ANCHOR_BOXES = auto()
    MODIFY_LOSS_WEIGHTS = auto()
    ADJUST_IOU_THRESHOLD = auto()
    ADJUST_CONFIDENCE_THRESHOLD = auto()

    # Embedding-specific
    ADJUST_TRIPLET_MARGIN = auto()
    USE_HARD_NEGATIVE_MINING = auto()
    ADJUST_EMBEDDING_DIM = auto()

    # General
    CHECK_CONFIGURATION = auto()
    MANUAL_REVIEW = auto()


@dataclass
class Recommendation:
    """
    Training recommendation.

    Contains recommendation type, source issue, message, and optional
    configuration changes that can be auto-applied.
    """

    rec_type: RecommendationType
    source_issue: IssueType
    message: str
    suggested_value: Optional[Any] = None
    config_key: Optional[str] = None
    auto_applicable: bool = False
    priority: int = 5  # 1 = highest priority, 10 = lowest

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert recommendation to dictionary.

        Returns:
            Dictionary representation of the recommendation
        """
        return {
            "rec_type": self.rec_type.name,
            "source_issue": self.source_issue.name,
            "message": self.message,
            "suggested_value": self.suggested_value,
            "config_key": self.config_key,
            "auto_applicable": self.auto_applicable,
            "priority": self.priority,
        }

    def __str__(self) -> str:
        """String representation of the recommendation."""
        auto_str = " [AUTO-APPLICABLE]" if self.auto_applicable else ""
        value_str = f" → {self.suggested_value}" if self.suggested_value is not None else ""
        return f"[P{self.priority}]{auto_str} {self.rec_type.name}: {self.message}{value_str}"


# Mapping from issue types to recommended actions
ISSUE_TO_RECOMMENDATIONS: Dict[IssueType, list] = {
    IssueType.OVERFITTING: [
        RecommendationType.ADD_REGULARIZATION,
        RecommendationType.INCREASE_DROPOUT,
        RecommendationType.EARLY_STOPPING,
        RecommendationType.INCREASE_DATA_AUGMENTATION,
        RecommendationType.REDUCE_MODEL_COMPLEXITY,
    ],
    IssueType.UNDERFITTING: [
        RecommendationType.INCREASE_MODEL_CAPACITY,
        RecommendationType.REDUCE_REGULARIZATION if hasattr(RecommendationType, 'REDUCE_REGULARIZATION') else RecommendationType.REDUCE_DROPOUT,
        RecommendationType.INCREASE_EPOCHS,
        RecommendationType.INCREASE_LR,
    ],
    IssueType.LR_TOO_HIGH: [
        RecommendationType.REDUCE_LR,
        RecommendationType.USE_LR_SCHEDULER,
        RecommendationType.USE_WARMUP,
    ],
    IssueType.LR_TOO_LOW: [
        RecommendationType.INCREASE_LR,
        RecommendationType.USE_LR_SCHEDULER,
    ],
    IssueType.PLATEAU: [
        RecommendationType.REDUCE_LR,
        RecommendationType.USE_LR_SCHEDULER,
        RecommendationType.INCREASE_DATA_AUGMENTATION,
    ],
    IssueType.DIVERGENCE: [
        RecommendationType.REDUCE_LR,
        RecommendationType.REDUCE_BATCH_SIZE,
        RecommendationType.CHECK_DATA_QUALITY,
    ],
    IssueType.GRADIENT_EXPLOSION: [
        RecommendationType.REDUCE_LR,
        RecommendationType.ADD_REGULARIZATION,
        RecommendationType.REDUCE_BATCH_SIZE,
    ],
    IssueType.LOW_MAP: [
        RecommendationType.ADJUST_ANCHOR_BOXES,
        RecommendationType.MODIFY_LOSS_WEIGHTS,
        RecommendationType.INCREASE_EPOCHS,
        RecommendationType.CHECK_DATA_QUALITY,
    ],
    IssueType.PRECISION_RECALL_IMBALANCE: [
        RecommendationType.ADJUST_CONFIDENCE_THRESHOLD,
        RecommendationType.MODIFY_LOSS_WEIGHTS,
        RecommendationType.BALANCE_CLASSES,
    ],
    IssueType.CLASS_IMBALANCE: [
        RecommendationType.BALANCE_CLASSES,
        RecommendationType.MODIFY_LOSS_WEIGHTS,
        RecommendationType.ADD_MORE_DATA,
    ],
    IssueType.EMBEDDING_NOT_CONVERGING: [
        RecommendationType.ADJUST_TRIPLET_MARGIN,
        RecommendationType.USE_HARD_NEGATIVE_MINING,
        RecommendationType.REDUCE_LR,
    ],
    IssueType.TRIPLET_LOSS_STUCK: [
        RecommendationType.ADJUST_TRIPLET_MARGIN,
        RecommendationType.USE_HARD_NEGATIVE_MINING,
        RecommendationType.INCREASE_LR,
    ],
}
