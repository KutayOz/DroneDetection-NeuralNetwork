"""
OpenAI LLM provider for training advisor.

Provides LLM-based analysis using OpenAI's GPT models.
"""

import os
from typing import List, Optional

from ..domain.metrics import TrainingMetrics
from ..domain.issues import Issue, IssueType, IssueSeverity


class OpenAIProvider:
    """
    OpenAI LLM provider for training analysis.

    Uses GPT-4o-mini by default (cost-effective and fast).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1500,
        temperature: float = 0.3,
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4o-mini - cheap and effective)
            max_tokens: Maximum response tokens
            temperature: Response creativity (0.0-1.0)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Run: pip install openai"
                )
        return self._client

    @property
    def provider_name(self) -> str:
        """Provider name."""
        return "openai"

    def analyze(
        self,
        metrics: TrainingMetrics,
        issues: List[Issue],
        context: Optional[str] = None,
    ) -> str:
        """
        Analyze training results using GPT.

        Args:
            metrics: Training metrics
            issues: Detected issues
            context: Additional context

        Returns:
            LLM analysis as string
        """
        if not self.api_key:
            return "[LLM Analizi devre disi - API key bulunamadi]"

        prompt = self._build_prompt(metrics, issues, context)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Sen bir makine ogrenmesi uzmansin. YOLO nesne tespiti ve "
                            "Siamese network egitim sonuclarini analiz ediyorsun. "
                            "Turkce yanit ver. Kisa ve oneriler odakli ol."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"[LLM Analizi basarisiz: {str(e)}]"

    def _build_prompt(
        self,
        metrics: TrainingMetrics,
        issues: List[Issue],
        context: Optional[str] = None,
    ) -> str:
        """Build analysis prompt from metrics and issues."""
        parts = []

        # Metrics summary
        parts.append("## Egitim Metrikleri\n")

        if metrics.epochs:
            parts.append(f"- Toplam Epoch: {len(metrics.epochs)}")

            # Get first and last epoch data
            first_epoch = metrics.epochs[0]
            last_epoch = metrics.epochs[-1]

            parts.append(f"- Baslangic Train Loss: {first_epoch.train_loss:.4f}")
            parts.append(f"- Son Train Loss: {last_epoch.train_loss:.4f}")
            parts.append(f"- Son Val Loss: {last_epoch.val_loss:.4f}")

            if last_epoch.map50 is not None:
                parts.append(f"- Son mAP@50: {last_epoch.map50:.4f}")

            if last_epoch.precision is not None:
                parts.append(f"- Son Precision: {last_epoch.precision:.4f}")

            if last_epoch.recall is not None:
                parts.append(f"- Son Recall: {last_epoch.recall:.4f}")

        # Issues
        if issues:
            parts.append("\n## Tespit Edilen Sorunlar\n")
            for issue in issues:
                severity_icon = {
                    IssueSeverity.CRITICAL: "!!!!",
                    IssueSeverity.HIGH: "!!!",
                    IssueSeverity.MEDIUM: "!!",
                    IssueSeverity.LOW: "!",
                }.get(issue.severity, "?")

                parts.append(f"[{severity_icon}] {issue.issue_type.name}: {issue.message}")
        else:
            parts.append("\n## Sorun Tespit Edilmedi\n")

        # Context
        if context:
            parts.append(f"\n## Ek Bilgi\n{context}")

        # Request
        parts.append(
            "\n---\n"
            "Bu egitim sonuclarini analiz et ve su konularda yorum yap:\n"
            "1. Genel degerlendirme (basarili mi?)\n"
            "2. Tespit edilen sorunlarin onemi\n"
            "3. Performansi artirmak icin 2-3 somut oneri\n"
            "Kisa ve net yaz."
        )

        return "\n".join(parts)

    def get_recommendations(
        self,
        metrics: TrainingMetrics,
        issues: List[Issue],
    ) -> str:
        """
        Get specific recommendations from LLM.

        Args:
            metrics: Training metrics
            issues: Detected issues

        Returns:
            Recommendations as string
        """
        if not self.api_key:
            return ""

        prompt = (
            f"Asagidaki egitim sorunlari icin cozum oner:\n\n"
        )

        for issue in issues:
            prompt += f"- {issue.issue_type.value}: {issue.message}\n"

        prompt += (
            "\nHer sorun icin:\n"
            "1. Neden olustugunu kisa acikla\n"
            "2. Cozum icin somut parametre degeri oner\n"
            "Kisa tut, kod ornegi verme."
        )

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "YOLO egitimi uzmanisin. Kisa ve net Turkce yanit ver.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=0.2,
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"[Oneri alinamadi: {str(e)}]"


__all__ = ["OpenAIProvider"]
