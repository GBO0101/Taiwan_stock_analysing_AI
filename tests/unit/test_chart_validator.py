"""Unit tests for deterministic chart validation (P2, gap #2,#9; #7 option A)."""

import logging

from classifier.chart_validator import ChartValidator, derive_visualization, derive_requirement
from classifier.models import (
    ChartDataRequirement as R,
    ChartType as T,
    ClassificationResult,
)


class TestDerive:
    def test_visual_true_with_graph(self):
        assert derive_visualization("台積電股價走勢圖") is True

    def test_visual_false_no_keyword(self):
        assert derive_visualization("台積電現在股價多少") is False

    def test_requirement_price_trend(self):
        assert derive_requirement("台積電股價走勢圖") == R.PRICE_TREND

    def test_requirement_ohlc(self):
        assert derive_requirement("台積電K線圖") == R.PRICE_OHLC

    def test_requirement_revenue_comparison(self):
        assert derive_requirement("台積電和鴻海營收比較圖") == R.REVENUE_COMPARISON

    def test_requirement_revenue_trend(self):
        assert derive_requirement("台積電營收趨勢圖") == R.REVENUE_TREND

    def test_requirement_indicator(self):
        assert derive_requirement("台積電本益比趨勢圖") == R.INDICATOR_TREND

    def test_requirement_sector(self):
        assert derive_requirement("半導體產業分析圖") == R.SECTOR_ANALYSIS


class TestValidate:
    def _base(self, **kw):
        return ClassificationResult(type="live", confidence=0.9, **kw)

    def test_corrects_misclassified_price_trend(self):
        # 股價走勢圖 wrongly tagged indicator_trend by LLM (gap #2).
        v = ChartValidator()
        c = self._base(needs_visualization=True, chart_type=T.LINE, chart_data_requirements=R.INDICATOR_TREND)
        out, warns = v.validate("台積電股價走勢圖", c)
        assert out.chart_data_requirements == R.PRICE_TREND
        assert out.chart_type == T.LINE
        assert any("price_trend" in w.lower() for w in warns)

    def test_derives_needs_visualization_when_llm_missed(self):
        # needs_visualization derived from keywords when LLM misses it (gap #2,#9).
        v = ChartValidator()
        c = self._base(needs_visualization=False, chart_data_requirements=None, chart_type=None)
        out, warns = v.validate("台積電股價走勢圖", c)
        assert out.needs_visualization is True
        assert out.chart_data_requirements == R.PRICE_TREND

    def test_clears_chart_fields_when_no_visual_keyword(self):
        # No visual wording -> chart fields cleared regardless of LLM (gap #9).
        v = ChartValidator()
        c = self._base(needs_visualization=True, chart_type=T.LINE, chart_data_requirements=R.PRICE_TREND)
        out, warns = v.validate("台積電現在股價多少", c)
        assert out.needs_visualization is False
        assert out.chart_data_requirements is None
        assert out.chart_type is None

    def test_forces_chart_type_to_match_requirement(self):
        # chart_type forced to match requirement (gap #7 option A).
        v = ChartValidator()
        c = self._base(needs_visualization=True, chart_type=T.KLINE, chart_data_requirements=R.INDICATOR_TREND)
        out, warns = v.validate("台積電本益比趨勢圖", c)
        assert out.chart_type == T.LINE

    def test_no_force_chart_type_option_b(self):
        # force_chart_type=False keeps LLM chart_type (gap #7 option B, available if chosen).
        v = ChartValidator()
        c = self._base(needs_visualization=True, chart_type=T.KLINE, chart_data_requirements=R.INDICATOR_TREND)
        out, warns = v.validate("台積電本益比趨勢圖", c, force_chart_type=False)
        assert out.chart_type == T.KLINE
        assert out.chart_data_requirements == R.INDICATOR_TREND

    def test_logs_correction_warning(self, caplog):
        # Corrections are written to log, never silently discarded (gap #6).
        v = ChartValidator()
        c = self._base(needs_visualization=True, chart_type=T.LINE, chart_data_requirements=R.INDICATOR_TREND)
        with caplog.at_level(logging.WARNING, logger="classifier.chart_validator"):
            v.validate("台積電股價走勢圖", c)
        assert any("price_trend" in r.message.lower() for r in caplog.records)
