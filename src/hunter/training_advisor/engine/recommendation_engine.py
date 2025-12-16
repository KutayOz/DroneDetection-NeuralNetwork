"""
Recommendation engine for training advisor.

Generates actionable recommendations based on detected issues.
"""

from typing import Dict, List

from ..domain.issues import Issue, IssueType, IssueSeverity
from ..domain.recommendations import (
    Recommendation,
    RecommendationType,
    ISSUE_TO_RECOMMENDATIONS,
)


class RecommendationEngine:
    """
    Engine for generating training recommendations.

    Maps detected issues to actionable recommendations with
    priority based on issue severity.
    """

    def __init__(self):
        """Initialize recommendation engine."""
        self._issue_mapping = ISSUE_TO_RECOMMENDATIONS

    def generate(self, issues: List[Issue]) -> List[Recommendation]:
        """
        Generate recommendations for detected issues.

        Args:
            issues: List of detected issues

        Returns:
            Prioritized list of recommendations
        """
        recommendations = []

        for issue in issues:
            issue_recs = self._generate_for_issue(issue)
            recommendations.extend(issue_recs)

        # Sort by priority (lower number = higher priority)
        recommendations.sort(key=lambda r: r.priority)

        # Remove duplicates by rec_type
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec.rec_type not in seen:
                seen.add(rec.rec_type)
                unique_recs.append(rec)

        return unique_recs

    def _generate_for_issue(self, issue: Issue) -> List[Recommendation]:
        """
        Generate recommendations for a single issue.

        Args:
            issue: Detected issue

        Returns:
            List of recommendations for this issue
        """
        recommendations = []
        rec_types = self._issue_mapping.get(issue.issue_type, [])

        # Calculate base priority from severity
        base_priority = self._severity_to_priority(issue.severity)

        for i, rec_type in enumerate(rec_types):
            rec = self._create_recommendation(
                rec_type=rec_type,
                issue=issue,
                priority=base_priority + i,
            )
            recommendations.append(rec)

        return recommendations

    def _create_recommendation(
        self,
        rec_type: RecommendationType,
        issue: Issue,
        priority: int,
    ) -> Recommendation:
        """
        Create a recommendation with appropriate message and settings.

        Args:
            rec_type: Type of recommendation
            issue: Source issue
            priority: Priority value

        Returns:
            Recommendation instance
        """
        message, config_key, suggested_value, auto_applicable = self._get_recommendation_details(
            rec_type, issue
        )

        return Recommendation(
            rec_type=rec_type,
            source_issue=issue.issue_type,
            message=message,
            suggested_value=suggested_value,
            config_key=config_key,
            auto_applicable=auto_applicable,
            priority=priority,
        )

    def _get_recommendation_details(
        self, rec_type: RecommendationType, issue: Issue
    ) -> tuple:
        """
        Get details for a recommendation type.

        Returns:
            Tuple of (message, config_key, suggested_value, auto_applicable)
        """
        details = {
            RecommendationType.REDUCE_LR: (
                "Reduce learning rate by 50%",
                "lr0",
                None,  # Calculated from current
                True,
            ),
            RecommendationType.INCREASE_LR: (
                "Increase learning rate by 2x",
                "lr0",
                None,
                True,
            ),
            RecommendationType.ADD_REGULARIZATION: (
                "Add weight decay regularization",
                "weight_decay",
                0.0005,
                True,
            ),
            RecommendationType.INCREASE_DROPOUT: (
                "Increase dropout rate",
                "dropout",
                0.5,
                False,
            ),
            RecommendationType.EARLY_STOPPING: (
                "Enable early stopping with patience=10",
                "patience",
                10,
                True,
            ),
            RecommendationType.INCREASE_DATA_AUGMENTATION: (
                "Increase data augmentation strength",
                "augment",
                True,
                True,
            ),
            RecommendationType.REDUCE_MODEL_COMPLEXITY: (
                "Consider using a smaller model variant",
                None,
                None,
                False,
            ),
            RecommendationType.INCREASE_MODEL_CAPACITY: (
                "Consider using a larger model variant",
                None,
                None,
                False,
            ),
            RecommendationType.USE_LR_SCHEDULER: (
                "Use cosine learning rate scheduler",
                "cos_lr",
                True,
                True,
            ),
            RecommendationType.ADJUST_ANCHOR_BOXES: (
                "Re-calculate anchor boxes for your dataset",
                None,
                None,
                False,
            ),
            RecommendationType.MODIFY_LOSS_WEIGHTS: (
                "Adjust loss component weights",
                None,
                None,
                False,
            ),
            RecommendationType.BALANCE_CLASSES: (
                "Balance classes through oversampling or loss weighting",
                None,
                None,
                False,
            ),
            RecommendationType.ADJUST_TRIPLET_MARGIN: (
                "Adjust triplet loss margin",
                "margin",
                0.3,
                True,
            ),
            RecommendationType.USE_HARD_NEGATIVE_MINING: (
                "Enable hard negative mining for triplet loss",
                "hard_mining",
                True,
                True,
            ),
            RecommendationType.CHECK_DATA_QUALITY: (
                "Review training data for quality issues",
                None,
                None,
                False,
            ),
        }

        return details.get(
            rec_type,
            (f"Apply {rec_type.name.lower().replace('_', ' ')}", None, None, False)
        )

    def _severity_to_priority(self, severity: IssueSeverity) -> int:
        """Convert severity to base priority value."""
        mapping = {
            IssueSeverity.CRITICAL: 1,
            IssueSeverity.HIGH: 3,
            IssueSeverity.MEDIUM: 5,
            IssueSeverity.LOW: 7,
        }
        return mapping.get(severity, 5)
