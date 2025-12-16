"""
Main TrainingAdvisor class.

High-level interface for training analysis and recommendations.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Load .env file for API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .config import AdvisorConfig
from .collectors import get_collector, BaseCollector
from .analyzers import get_all_analyzers, BaseAnalyzer
from .engine import DecisionEngine, RecommendationEngine, ActionRegistry
from .reporters import get_reporter, BaseReporter
from .tuner import AutoTuner
from .llm import OpenAIProvider, StubLLMProvider
from .domain.metrics import TrainingMetrics
from .domain.issues import Issue
from .domain.recommendations import Recommendation


class TrainingAdvisor:
    """
    Main training advisor class.

    Provides a high-level interface for:
    - Collecting training metrics
    - Analyzing for issues
    - Generating recommendations
    - Producing reports
    - Auto-tuning configurations

    Example usage:
        advisor = TrainingAdvisor()
        report = advisor.analyze("path/to/training/results")
        print(report)

        # Or with more control:
        metrics = advisor.collect("path/to/results.csv")
        issues = advisor.detect_issues(metrics)
        recommendations = advisor.recommend(issues)
        report = advisor.generate_report(issues, recommendations)
    """

    def __init__(
        self,
        config: Optional[AdvisorConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize training advisor.

        Args:
            config: Advisor configuration object
            config_path: Path to YAML configuration file
        """
        if config_path:
            self._config = AdvisorConfig.from_yaml(Path(config_path))
        else:
            self._config = config or AdvisorConfig()

        self._decision_engine = DecisionEngine()
        self._recommendation_engine = RecommendationEngine()
        self._action_registry = ActionRegistry()
        self._auto_tuner = AutoTuner(
            action_registry=self._action_registry,
            backup_dir=self._config.auto_tune.backup_dir,
            safe_only=self._config.auto_tune.safe_only,
        )

        # Initialize LLM provider
        self._llm_provider = self._init_llm_provider()

    @property
    def config(self) -> AdvisorConfig:
        """Get current configuration."""
        return self._config

    def analyze(
        self,
        source: Union[str, Path],
        source_type: Optional[str] = None,
        report_format: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Full analysis pipeline - collect, analyze, and report.

        Args:
            source: Path to training logs/results
            source_type: Type of collector ('yolo', 'csv', etc.)
            report_format: Output format ('console', 'markdown')
            output_path: Optional path to write report

        Returns:
            Report string
        """
        # Collect metrics
        metrics = self.collect(source, source_type)

        # Detect issues
        issues = self.detect_issues(metrics)

        # Generate recommendations
        recommendations = self.recommend(issues)

        # Generate report
        return self.generate_report(
            issues,
            recommendations,
            format=report_format,
            output_path=output_path,
        )

    def collect(
        self,
        source: Union[str, Path],
        source_type: Optional[str] = None,
    ) -> TrainingMetrics:
        """
        Collect training metrics from source.

        Args:
            source: Path to training logs/results
            source_type: Type of collector (auto-detect if None)

        Returns:
            Collected training metrics
        """
        collector_type = source_type or self._config.collectors.default_source

        if self._config.collectors.auto_detect and source_type is None:
            collector_type = self._detect_source_type(Path(source))

        collector = get_collector(collector_type)
        return collector.collect(source)

    def detect_issues(self, metrics: TrainingMetrics) -> List[Issue]:
        """
        Detect issues in training metrics.

        Args:
            metrics: Training metrics to analyze

        Returns:
            List of detected issues
        """
        return self._decision_engine.analyze(metrics)

    def recommend(self, issues: List[Issue]) -> List[Recommendation]:
        """
        Generate recommendations for issues.

        Args:
            issues: Detected issues

        Returns:
            List of recommendations
        """
        return self._recommendation_engine.generate(issues)

    def generate_report(
        self,
        issues: List[Issue],
        recommendations: List[Recommendation],
        format: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate analysis report.

        Args:
            issues: Detected issues
            recommendations: Generated recommendations
            format: Report format ('console', 'markdown')
            output_path: Optional path to write report

        Returns:
            Report string
        """
        report_format = format or self._config.reporters.default_format
        reporter = get_reporter(report_format)
        return reporter.report(issues, recommendations, output_path)

    def auto_tune(
        self,
        recommendations: List[Recommendation],
        config_path: str,
        dry_run: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Auto-apply recommendations to configuration.

        Args:
            recommendations: Recommendations to apply
            config_path: Path to training configuration file
            dry_run: Preview changes without applying

        Returns:
            Dict with tuning results
        """
        if not self._config.auto_tune.enabled:
            return {"error": "Auto-tune is disabled in configuration"}

        should_dry_run = dry_run if dry_run is not None else self._config.auto_tune.dry_run

        return self._auto_tuner.tune(
            recommendations,
            config_path,
            create_backup=self._config.auto_tune.create_backup,
            dry_run=should_dry_run,
        )

    def run_full_analysis(
        self,
        source: Union[str, Path],
        training_config_path: Optional[str] = None,
        auto_tune: bool = False,
    ) -> Dict[str, Any]:
        """
        Run complete analysis pipeline with optional auto-tuning.

        Args:
            source: Path to training logs
            training_config_path: Path to training config for auto-tune
            auto_tune: Whether to apply auto-tuning

        Returns:
            Dict with full analysis results
        """
        # Collect and analyze
        metrics = self.collect(source)
        issues = self.detect_issues(metrics)
        recommendations = self.recommend(issues)

        # Generate reports
        console_report = self.generate_report(issues, recommendations, format="console")
        markdown_report = self.generate_report(issues, recommendations, format="markdown")

        # Get LLM analysis if enabled
        llm_analysis = None
        if self._config.llm.enabled:
            llm_analysis = self.llm_analyze(metrics, issues)

        result = {
            "metrics": metrics,
            "issues": issues,
            "recommendations": recommendations,
            "reports": {
                "console": console_report,
                "markdown": markdown_report,
            },
            "llm_analysis": llm_analysis,
            "auto_tune_results": None,
        }

        # Auto-tune if requested
        if auto_tune and training_config_path:
            result["auto_tune_results"] = self.auto_tune(
                recommendations,
                training_config_path,
            )

        return result

    def _detect_source_type(self, source: Path) -> str:
        """
        Auto-detect source type from path.

        Args:
            source: Path to source

        Returns:
            Detected source type
        """
        if source.is_dir():
            # Check for YOLO results
            if (source / "results.csv").exists():
                return "yolo"
            # Check in subdirectories
            for subdir in ["train", "exp"]:
                if (source / subdir / "results.csv").exists():
                    return "yolo"

        if source.suffix == ".csv":
            # Try to detect YOLO format by checking headers
            try:
                with open(source, "r") as f:
                    header = f.readline().lower()
                    if "train/box_loss" in header or "metrics/map50" in header:
                        return "yolo"
            except Exception:
                pass
            return "csv"

        return self._config.collectors.default_source

    def _init_llm_provider(self):
        """Initialize LLM provider based on configuration."""
        if not self._config.llm.enabled:
            return StubLLMProvider()

        provider = self._config.llm.provider.lower()
        api_key = os.getenv(self._config.llm.api_key_env)

        if provider == "openai":
            return OpenAIProvider(
                api_key=api_key,
                model=self._config.llm.model,
                max_tokens=self._config.llm.max_tokens,
                temperature=self._config.llm.temperature,
            )
        else:
            return StubLLMProvider()

    def llm_analyze(
        self,
        metrics: TrainingMetrics,
        issues: List[Issue],
        context: Optional[str] = None,
    ) -> str:
        """
        Get LLM-based analysis of training results.

        Args:
            metrics: Training metrics
            issues: Detected issues
            context: Additional context

        Returns:
            LLM analysis as string
        """
        return self._llm_provider.analyze(metrics, issues, context)
