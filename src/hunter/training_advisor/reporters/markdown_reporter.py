"""
Markdown reporter for training advisor.

Generates markdown-formatted reports.
"""

from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .base import BaseReporter
from ..domain.issues import Issue, IssueSeverity
from ..domain.recommendations import Recommendation


class MarkdownReporter(BaseReporter):
    """
    Markdown reporter for documentation/sharing.

    Generates GitHub-flavored markdown reports.
    """

    @property
    def format(self) -> str:
        """Report format."""
        return "markdown"

    def report(
        self,
        issues: List[Issue],
        recommendations: List[Recommendation],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate markdown report.

        Args:
            issues: Detected issues
            recommendations: Generated recommendations
            output_path: Optional path to write report file

        Returns:
            Markdown string
        """
        lines = []

        # Title
        lines.append("# Training Analysis Report")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Issues found:** {len(issues)}")
        lines.append(f"- **Recommendations:** {len(recommendations)}")

        critical = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)
        high = sum(1 for i in issues if i.severity == IssueSeverity.HIGH)

        if critical > 0:
            lines.append(f"- **🚨 Critical issues:** {critical}")
        if high > 0:
            lines.append(f"- **⚠️ High severity issues:** {high}")
        lines.append("")

        # Issues table
        if issues:
            lines.append("## Issues Detected")
            lines.append("")
            lines.append("| Severity | Type | Message | Epoch |")
            lines.append("|----------|------|---------|-------|")
            for issue in issues:
                severity_icon = self._severity_icon(issue.severity)
                epoch = issue.epoch_detected if issue.epoch_detected else "-"
                lines.append(
                    f"| {severity_icon} {issue.severity.name} | "
                    f"{issue.issue_type.name} | {issue.message} | {epoch} |"
                )
            lines.append("")

        # Recommendations
        if recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for i, rec in enumerate(recommendations, 1):
                auto = " `[AUTO-APPLICABLE]`" if rec.auto_applicable else ""
                lines.append(f"### {i}. {rec.rec_type.name}{auto}")
                lines.append("")
                lines.append(f"{rec.message}")
                if rec.suggested_value is not None:
                    lines.append("")
                    lines.append(f"**Suggested value:** `{rec.suggested_value}`")
                if rec.config_key:
                    lines.append(f"**Config key:** `{rec.config_key}`")
                lines.append("")

        content = "\n".join(lines)

        # Write to file if path provided
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        return content

    def _severity_icon(self, severity: IssueSeverity) -> str:
        """Get icon for severity level."""
        icons = {
            IssueSeverity.CRITICAL: "🚨",
            IssueSeverity.HIGH: "🔴",
            IssueSeverity.MEDIUM: "⚠️",
            IssueSeverity.LOW: "ℹ️",
        }
        return icons.get(severity, "•")
