"""Jinja2 prompt template infrastructure."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound


class PromptTemplateError(Exception):
    """Prompt template errors."""



class PromptManager:
    """Manages Jinja2 prompt templates."""

    def __init__(self, template_dir: str | None = None):
        """Initialize prompt manager.

        Args:
            template_dir: Directory containing .j2 templates (defaults to prompts/)
        """
        if template_dir is None:
            template_dir = str(Path(__file__).parent.parent / "prompts")

        self.template_dir = Path(template_dir)
        self._env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Render a template with given variables.

        Args:
            template_name: Template filename (e.g., "boundary.j2")
            **kwargs: Variables to pass to template

        Returns:
            Rendered prompt string

        Raises:
            PromptTemplateError: If template not found or render fails
        """
        try:
            template = self._env.get_template(template_name)
            return template.render(**kwargs)
        except TemplateNotFound as e:
            raise PromptTemplateError(f"Template not found: {template_name}") from e
        except Exception as e:
            raise PromptTemplateError(f"Template render failed: {e}") from e

    def render_boundary(self, question: str, context: dict | None = None) -> str:
        """Render boundary extraction prompt."""
        return self.render("boundary.j2", question=question, context=context)

    def render_classify(self, question: str, boundary: dict | None = None) -> str:
        """Render classification prompt."""
        return self.render("classify.j2", question=question, boundary=boundary)

    def render_decompose(
        self,
        question: str,
        boundary: dict | None = None,
        classification: dict | None = None,
        indicator_mappings: list[dict] | None = None,
    ) -> str:
        """Render decomposition prompt."""
        return self.render(
            "decompose.j2",
            question=question,
            boundary=boundary,
            classification=classification,
            indicator_mappings=indicator_mappings,
        )

    def render_sentiment(
        self,
        question: str,
        boundary: dict | None = None,
        classification: dict | None = None,
    ) -> str:
        """Render sentiment analysis node prompt."""
        return self.render(
            "sentiment.j2",
            question=question,
            boundary=boundary,
            classification=classification,
        )

    def render_stock_validation(
        self,
        query_value: str,
        query_type: str,
        stocks: list[dict],
    ) -> str:
        """Render the range-mode stock-list validation prompt."""
        return self.render(
            "stock_validation.j2",
            query_value=query_value,
            query_type=query_type,
            stocks=stocks,
        )

    def render_supply_chain(
        self,
        stock_code: str,
        stock_name: str,
        industry_hint: str | None = None,
    ) -> str:
        """Render the supply-chain (upstream/downstream) inference prompt."""
        return self.render(
            "supply_chain.j2",
            stock_code=stock_code,
            stock_name=stock_name,
            industry_hint=industry_hint,
        )

    def render_annual_report(
        self,
        stock_code: str,
        stock_name: str,
        fiscal_year: int,
        report_text: str,
    ) -> str:
        """Render the annual-report customer/supplier extraction prompt."""
        return self.render(
            "annual_report.j2",
            stock_code=stock_code,
            stock_name=stock_name,
            fiscal_year=fiscal_year,
            report_text=report_text,
        )


# Global instance
prompt_manager = PromptManager()