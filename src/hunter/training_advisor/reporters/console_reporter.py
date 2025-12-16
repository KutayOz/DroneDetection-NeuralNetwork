"""
Console reporter for training advisor.

Generates formatted console output.
"""

from typing import List, Optional
from datetime import datetime

from .base import BaseReporter
from ..domain.issues import Issue, IssueSeverity
from ..domain.recommendations import Recommendation


class ConsoleReporter(BaseReporter):
    """
    Console reporter for terminal output.

    Generates colorized, formatted output for the console.
    """

    def __init__(self, use_colors: bool = True):
        """
        Initialize console reporter.

        Args:
            use_colors: Whether to use ANSI colors
        """
        self._use_colors = use_colors

    @property
    def format(self) -> str:
        """Report format."""
        return "console"

    def report(
        self,
        issues: List[Issue],
        recommendations: List[Recommendation],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate console report.

        Args:
            issues: Detected issues
            recommendations: Generated recommendations
            output_path: Ignored for console output

        Returns:
            Formatted console string
        """
        lines = []

        # Header
        lines.append(self._header("Training Analysis Report"))
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary
        lines.append(self._section("Summary"))
        lines.append(f"  Issues found: {len(issues)}")
        lines.append(f"  Recommendations: {len(recommendations)}")
        critical = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)
        if critical > 0:
            lines.append(f"  {self._color('CRITICAL ISSUES: ' + str(critical), 'red')}")
        lines.append("")

        # Issues
        if issues:
            lines.append(self._section("Issues Detected"))
            for issue in issues:
                lines.append(f"  {self._format_issue(issue)}")
            lines.append("")

        # Recommendations
        if recommendations:
            lines.append(self._section("Recommendations"))
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"  {i}. {self._format_recommendation(rec)}")
            lines.append("")

        return "\n".join(lines)

    def _header(self, text: str) -> str:
        """Format header."""
        line = "=" * 60
        return f"{line}\n{text.center(60)}\n{line}"

    def _section(self, text: str) -> str:
        """Format section header."""
        return self._color(f"[{text}]", "cyan")

    def _format_issue(self, issue: Issue) -> str:
        """Format a single issue."""
        severity_colors = {
            IssueSeverity.CRITICAL: "red",
            IssueSeverity.HIGH: "yellow",
            IssueSeverity.MEDIUM: "blue",
            IssueSeverity.LOW: "white",
        }
        color = severity_colors.get(issue.severity, "white")
        return self._color(str(issue), color)

    def _format_recommendation(self, rec: Recommendation) -> str:
        """Format a single recommendation."""
        auto = " [AUTO]" if rec.auto_applicable else ""
        return f"{rec.message}{auto}"

    def _color(self, text: str, color: str) -> str:
        """Apply ANSI color to text."""
        if not self._use_colors:
            return text

        colors = {
            "red": "\033[91m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "reset": "\033[0m",
        }

        return f"{colors.get(color, '')}{text}{colors['reset']}"
