"""Deterministic chart requirement / visualization validator (Step 2, P2).

The LLM classifier (``classify.j2``) is instructed to set ``needs_visualization``
and ``chart_data_requirements`` only when explicit chart wording exists, but it
is not deterministic: the same question can be misclassified (e.g. a price-trend
question tagged as ``indicator_trend``), and ``needs_visualization`` can be
missed. ``ChartValidator`` re-derives both fields from the *question keywords*
(the source of truth per gap #9) and optionally forces ``chart_type`` to match
the requirement (gap #7, default = option A). All corrections are logged so they
are never silently discarded (gap #6).
"""

import logging

from classifier.models import (
    ChartDataRequirement,
    ChartType,
    ClassificationResult,
)

logger = logging.getLogger(__name__)

# Explicit chart / visual wording -> needs_visualization.
_VISUAL_KW = ("圖", "K線", "k線", "長條", "折線", "柱狀")

# Keyword groups, ordered from most specific to least specific.
_OHLC_KW = ("K線", "k線", "日K", "技術分析")
_REVENUE_KW = ("營收",)
_COMPARISON_KW = ("比較", "對比", "vs", "VS", "哪家", "哪一家", "好壞")
_SECTOR_KW = ("產業", "類股", "板塊", "產業分析", "類股分析")
_INDICATOR_KW = ("本益比", "PE", "EPS", "eps", "殖利率", "指標", "財報", "獲利能力")
_PRICE_KW = ("股價", "價格", "收盤", "走勢", "趨勢", "股價走勢", "股價趨勢")

# chart_data_requirements -> chart_type (gap #7 option A: both strongly corrected).
CHART_TYPE_BY_REQUIREMENT: dict[ChartDataRequirement, ChartType] = {
    ChartDataRequirement.PRICE_TREND: ChartType.LINE,
    ChartDataRequirement.PRICE_OHLC: ChartType.KLINE,
    ChartDataRequirement.REVENUE_TREND: ChartType.LINE,
    ChartDataRequirement.REVENUE_COMPARISON: ChartType.BAR,
    ChartDataRequirement.INDICATOR_TREND: ChartType.LINE,
    ChartDataRequirement.SECTOR_ANALYSIS: ChartType.BAR,
}


def derive_visualization(question: str) -> bool:
    """Return True if the question contains explicit chart/visual wording."""
    return any(kw in question for kw in _VISUAL_KW)


def derive_requirement(question: str) -> ChartDataRequirement | None:
    """Derive the chart data requirement from question keywords (gap #9).

    Returns ``None`` when no specific metric keyword is present (caller decides
    the default). Priority: OHLC > revenue > sector > indicator > price.
    """
    if any(kw in question for kw in _OHLC_KW):
        return ChartDataRequirement.PRICE_OHLC
    if any(kw in question for kw in _REVENUE_KW):
        if any(kw in question for kw in _COMPARISON_KW):
            return ChartDataRequirement.REVENUE_COMPARISON
        return ChartDataRequirement.REVENUE_TREND
    if any(kw in question for kw in _SECTOR_KW):
        return ChartDataRequirement.SECTOR_ANALYSIS
    if any(kw in question for kw in _INDICATOR_KW):
        return ChartDataRequirement.INDICATOR_TREND
    if any(kw in question for kw in _PRICE_KW):
        return ChartDataRequirement.PRICE_TREND
    return None


class ChartValidator:
    """Reconcile a ClassificationResult against question keywords."""

    def validate(
        self,
        question: str,
        classification: ClassificationResult,
        force_chart_type: bool = True,
    ) -> tuple[ClassificationResult, list[str]]:
        """Correct ``needs_visualization`` / chart fields from question keywords.

        Args:
            question: Original user question (source of truth for intent).
            classification: LLM classification result (mutated in place).
            force_chart_type: If True (gap #7 option A), also force ``chart_type``
                to match ``chart_data_requirements``. Set False for option B.

        Returns:
            ``(classification, warnings)`` — warnings detail every correction.
        """
        warnings: list[str] = []

        # 1) needs_visualization is derived from keywords, not the LLM guess.
        vis = derive_visualization(question)
        if classification.needs_visualization != vis:
            warnings.append(
                f"needs_visualization corrected {classification.needs_visualization} -> {vis}"
            )
            classification.needs_visualization = vis

        # 2) chart_data_requirements from keywords (#9).
        req = derive_requirement(question)
        if not vis:
            # No chart wording -> no chart fields, regardless of LLM output.
            if classification.chart_data_requirements is not None:
                warnings.append(
                    f"chart_data_requirements cleared (no visual wording): "
                    f"{classification.chart_data_requirements}"
                )
                classification.chart_data_requirements = None
            if classification.chart_type is not None:
                warnings.append(f"chart_type cleared (no visual wording): {classification.chart_type}")
                classification.chart_type = None
        else:
            if req is None:
                req = ChartDataRequirement.PRICE_TREND  # safe default
                warnings.append("chart_data_requirements defaulted to price_trend (no metric keyword)")
            if classification.chart_data_requirements != req:
                warnings.append(
                    f"chart_data_requirements corrected "
                    f"{classification.chart_data_requirements} -> {req}"
                )
                classification.chart_data_requirements = req

            # 3) chart_type forced to match requirement (gap #7 option A).
            if force_chart_type and classification.chart_data_requirements is not None:
                expected = CHART_TYPE_BY_REQUIREMENT.get(classification.chart_data_requirements)
                if expected is not None and classification.chart_type != expected:
                    warnings.append(
                        f"chart_type corrected {classification.chart_type} -> {expected.value} "
                        f"(matches {classification.chart_data_requirements.value})"
                    )
                    classification.chart_type = expected

        for warning in warnings:
            logger.warning("Chart validation: %s", warning)
        return classification, warnings


# Global instance (mirrors stock_resolver / indicator_mapper pattern).
chart_validator = ChartValidator()
