"""Unit tests for prompt template infrastructure."""

import pytest
from classifier.prompts import PromptManager, PromptTemplateError


class TestPromptManager:
    """Test PromptManager functionality."""

    def test_render_boundary(self):
        """Test boundary prompt rendering."""
        pm = PromptManager()
        result = pm.render_boundary("台積電近30天走勢圖")
        assert "台積電近30天走勢圖" in result
        assert "stock_codes" in result
        assert "confidence" in result

    def test_render_boundary_with_context(self):
        """Test boundary prompt with chat context."""
        pm = PromptManager()
        context = {
            "last_question": "台積電",
            "last_boundary": {"stock_codes": ["2330"], "company_names": ["台積電"]}
        }
        result = pm.render_boundary("它最近股價如何？", context=context)
        assert "它最近股價如何？" in result
        assert "台積電" in result

    def test_render_classify(self):
        """Test classification prompt rendering."""
        pm = PromptManager()
        boundary = {"stock_codes": ["2330"], "company_names": ["台積電"]}
        result = pm.render_classify("台積電現在股價多少", boundary=boundary)
        assert "台積電現在股價多少" in result
        assert "live" in result
        assert "factual" in result
        assert "analytical" in result
        assert "non_financial" in result

    def test_render_decompose(self):
        """Test decomposition prompt rendering."""
        pm = PromptManager()
        boundary = {"stock_codes": ["2330"]}
        classification = {"type": "analytical"}
        indicators = [
            {"indicator": "revenue_growth", "dataset": "TaiwanStockMonthRevenue", "field": "revenue", "derived_calculation": "..."}
        ]
        result = pm.render_decompose("台積電未來展望", boundary=boundary, classification=classification, indicator_mappings=indicators)
        assert "台積電未來展望" in result
        assert "finmind_query" in result
        assert "compute" in result
        assert "sentiment_analysis" in result

    def test_render_sentiment(self):
        """Test sentiment prompt rendering."""
        pm = PromptManager()
        boundary = {"stock_codes": ["2330"]}
        classification = {"type": "analytical"}
        result = pm.render_sentiment("台積電未來展望", boundary=boundary, classification=classification)
        assert "sentiment_analysis" in result
        assert "TaiwanStockNews" in result

    def test_template_not_found(self):
        """Test error on missing template."""
        pm = PromptManager()
        with pytest.raises(PromptTemplateError, match="Template not found"):
            pm.render("nonexistent.j2")

    def test_all_templates_exist(self):
        """Test all required templates can be loaded."""
        pm = PromptManager()
        templates = ["boundary.j2", "classify.j2", "decompose.j2", "sentiment.j2"]
        for t in templates:
            result = pm.render(t, question="test")
            assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])