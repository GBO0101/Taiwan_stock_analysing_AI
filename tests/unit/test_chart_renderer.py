"""Unit tests for ChartRenderer (free TWSE data source)."""

import pytest
from unittest.mock import Mock, patch
from datetime import date
from pathlib import Path

from classifier.chart_renderer import ChartRenderer, ChartRenderError
from classifier.models import ChartRequest, ChartType, ChartDataRequirement
from classifier.data_fetcher import FreeDataFetcher


class TestChartRenderer:
    """Test ChartRenderer functionality."""

    def test_init_with_defaults(self):
        """Test initialization with default settings."""
        with patch.object(FreeDataFetcher, "__init__", return_value=None):
            renderer = ChartRenderer()
            assert renderer.output_dir is not None
            assert renderer.output_dir.exists()

    def test_init_with_custom_output_dir(self, tmp_path):
        """Test initialization with custom output directory."""
        with patch.object(FreeDataFetcher, "__init__", return_value=None):
            renderer = ChartRenderer(output_dir=str(tmp_path))
            assert renderer.output_dir == tmp_path

    @patch("classifier.chart_renderer.FreeDataFetcher")
    def test_render_price_trend(self, mock_fetcher_class):
        """Test rendering a price trend chart."""
        mock_client = Mock()
        mock_fetcher_class.return_value = mock_client
        mock_client.get_stock_price.return_value = [
            {"date": "2024-01-01", "close": 500},
            {"date": "2024-01-02", "close": 510},
            {"date": "2024-01-03", "close": 505},
        ]

        renderer = ChartRenderer(data_fetcher=mock_client)
        request = ChartRequest(
            stock_codes=["2330"],
            chart_data_requirements=ChartDataRequirement.PRICE_TREND,
            chart_type=ChartType.LINE,
        )

        result = renderer.render(request)
        assert result.endswith(".png")
        assert "2330" in result
        assert "price_trend" in result

    @patch("classifier.chart_renderer.FreeDataFetcher")
    def test_render_kline(self, mock_fetcher_class):
        """Test rendering a K-line chart."""
        mock_client = Mock()
        mock_fetcher_class.return_value = mock_client
        mock_client.get_stock_price.return_value = [
            {"date": "2024-01-01", "open": 500, "close": 510, "high": 520, "low": 495},
            {"date": "2024-01-02", "open": 510, "close": 505, "high": 515, "low": 500},
        ]

        renderer = ChartRenderer(data_fetcher=mock_client)
        request = ChartRequest(
            stock_codes=["2330"],
            chart_data_requirements=ChartDataRequirement.PRICE_OHLC,
            chart_type=ChartType.KLINE,
        )

        result = renderer.render(request)
        assert result.endswith(".png")
        assert "kline" in result

    @patch("classifier.chart_renderer.FreeDataFetcher")
    def test_render_revenue_trend_unsupported(self, mock_fetcher_class):
        """Test revenue trend raises clear error on free source."""
        mock_client = Mock()
        mock_fetcher_class.return_value = mock_client
        mock_client.get_month_revenue.side_effect = __import__(
            "classifier.data_fetcher", fromlist=["FreeDataError"]
        ).FreeDataError("月營收資料需 FinMind 付費帳號")

        renderer = ChartRenderer(data_fetcher=mock_client)
        request = ChartRequest(
            stock_codes=["2330"],
            chart_data_requirements=ChartDataRequirement.REVENUE_TREND,
            chart_type=ChartType.LINE,
        )

        with pytest.raises(ChartRenderError, match="月營收資料需 FinMind"):
            renderer.render(request)

    @patch("classifier.chart_renderer.FreeDataFetcher")
    def test_render_indicator_trend(self, mock_fetcher_class):
        """Test rendering an indicator trend chart."""
        mock_client = Mock()
        mock_fetcher_class.return_value = mock_client
        mock_client.get_stock_per.return_value = [
            {"date": "2024-01-01", "pe": 15.5},
            {"date": "2024-02-01", "pe": 16.0},
        ]

        renderer = ChartRenderer(data_fetcher=mock_client)
        request = ChartRequest(
            stock_codes=["2330"],
            chart_data_requirements=ChartDataRequirement.INDICATOR_TREND,
            chart_type=ChartType.LINE,
        )

        result = renderer.render(request)
        assert result.endswith(".png")
        assert "indicator_trend" in result

    @patch("classifier.chart_renderer.FreeDataFetcher")
    def test_render_sector_analysis(self, mock_fetcher_class):
        """Test rendering a sector analysis chart."""
        mock_client = Mock()
        mock_fetcher_class.return_value = mock_client
        mock_client.get_stock_per.return_value = [
            {"date": "2024-01-01", "pe": 15.5},
        ]

        renderer = ChartRenderer(data_fetcher=mock_client)
        request = ChartRequest(
            stock_codes=["2330", "2454"],
            chart_data_requirements=ChartDataRequirement.SECTOR_ANALYSIS,
            chart_type=ChartType.BAR,
        )

        result = renderer.render(request)
        assert result.endswith(".png")
        assert "sector_analysis" in result

    @patch("classifier.chart_renderer.FreeDataFetcher")
    def test_render_no_data_error(self, mock_fetcher_class):
        """Test error when no data is returned."""
        mock_client = Mock()
        mock_fetcher_class.return_value = mock_client
        mock_client.get_stock_price.return_value = []

        renderer = ChartRenderer(data_fetcher=mock_client)
        request = ChartRequest(
            stock_codes=["2330"],
            chart_data_requirements=ChartDataRequirement.PRICE_TREND,
            chart_type=ChartType.LINE,
        )

        with pytest.raises(ChartRenderError, match="No price data"):
            renderer.render(request)

    @patch("classifier.chart_renderer.FreeDataFetcher")
    def test_render_unsupported_requirement(self, mock_fetcher_class):
        """Test error for unsupported chart data requirement."""
        mock_client = Mock()
        mock_fetcher_class.return_value = mock_client

        renderer = ChartRenderer(data_fetcher=mock_client)
        request = ChartRequest(
            stock_codes=["2330"],
            chart_data_requirements=ChartDataRequirement.PRICE_TREND,
            chart_type=ChartType.LINE,
        )

        # Temporarily override to test unsupported case
        request.chart_data_requirements = "unsupported"
        with pytest.raises(ChartRenderError, match="Unsupported chart data requirement"):
            renderer.render(request)

    @patch("classifier.chart_renderer.FreeDataFetcher")
    def test_render_with_custom_dates(self, mock_fetcher_class):
        """Test rendering with custom date range."""
        mock_client = Mock()
        mock_fetcher_class.return_value = mock_client
        mock_client.get_stock_price.return_value = [
            {"date": "2024-06-01", "close": 500},
            {"date": "2024-06-02", "close": 510},
        ]

        renderer = ChartRenderer(data_fetcher=mock_client)
        request = ChartRequest(
            stock_codes=["2330"],
            chart_data_requirements=ChartDataRequirement.PRICE_TREND,
            chart_type=ChartType.LINE,
            start_date=date(2024, 6, 1),
            end_date=date(2024, 6, 30),
        )

        result = renderer.render(request)
        assert result.endswith(".png")
        assert "2024-06-01" in result
        assert "2024-06-30" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
