"""Unit tests for IndicatorMapper."""

import pytest
from classifier.indicator_mapper import IndicatorMapper, IndicatorMappingError


class TestIndicatorMapper:
    """Test IndicatorMapper functionality."""

    def test_load_mappings(self):
        """Test loading mappings from table.csv."""
        mapper = IndicatorMapper()
        mappings = mapper.get_all()
        assert len(mappings) == 6
        indicators = [m["indicator"] for m in mappings]
        assert "revenue_growth" in indicators
        assert "eps" in indicators
        assert "pe" in indicators
        assert "peg" in indicators
        assert "margin" in indicators
        assert "fcf" in indicators

    def test_get_by_indicator(self):
        """Test getting specific indicator mapping."""
        mapper = IndicatorMapper()
        mapping = mapper.get_by_indicator("revenue_growth")
        assert mapping is not None
        assert mapping["indicator"] == "revenue_growth"
        assert mapping["dataset"] == "TaiwanStockMonthRevenue"
        assert mapping["field"] == "revenue"

    def test_get_nonexistent_indicator(self):
        """Test getting non-existent indicator returns None."""
        mapper = IndicatorMapper()
        mapping = mapper.get_by_indicator("nonexistent")
        assert mapping is None

    def test_get_indicators_list(self):
        """Test getting list of indicator names."""
        mapper = IndicatorMapper()
        indicators = mapper.get_indicators_list()
        assert len(indicators) == 6
        assert "revenue_growth" in indicators
        assert "eps" in indicators

    def test_get_datasets_for_indicator(self):
        """Test getting required datasets for indicator."""
        mapper = IndicatorMapper()
        datasets = mapper.get_datasets_for_indicator("revenue_growth")
        assert datasets == ["TaiwanStockMonthRevenue"]

    def test_to_prompt_format(self):
        """Test formatting for prompt template."""
        mapper = IndicatorMapper()
        formatted = mapper.to_prompt_format()
        assert len(formatted) == 6
        for item in formatted:
            assert "indicator" in item
            assert "dataset" in item
            assert "field" in item
            assert "derived_calculation" in item

    def test_missing_file_raises_error(self):
        """Test missing CSV file raises clear error."""
        with pytest.raises(IndicatorMappingError, match="not found"):
            IndicatorMapper("/nonexistent/path/table.csv")

    def test_empty_file_raises_error(self, tmp_path):
        """Test empty CSV file raises error."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        with pytest.raises(IndicatorMappingError, match="No indicator mappings found"):
            IndicatorMapper(str(empty_csv))

    def test_missing_columns_raises_error(self, tmp_path):
        """Test CSV with missing columns raises error."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("indicator,dataset\nrevenue_growth,TaiwanStockMonthRevenue\n")
        with pytest.raises(IndicatorMappingError, match="Missing required columns"):
            IndicatorMapper(str(bad_csv))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])