"""Chart rendering using matplotlib and free TWSE data."""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

# Configure a CJK-capable font so Chinese chart titles render correctly.
_CJK_FONT_CANDIDATES = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK TC",
    "Noto Sans CJK SC",
    "PingFang TC",
    "Heiti TC",
    "AR PL UMing TW",
]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_cjk_font = next((f for f in _CJK_FONT_CANDIDATES if f in _available_fonts), None)
if _cjk_font:
    plt.rcParams["font.sans-serif"] = [_cjk_font, *plt.rcParams["font.sans-serif"]]
    plt.rcParams["axes.unicode_minus"] = False

from classifier.models import ChartRequest, ChartDataRequirement
from classifier.data_fetcher import FreeDataFetcher, FreeDataError


class ChartRenderError(Exception):
    """Chart rendering errors."""

    pass


class ChartRenderer:
    """Renders charts from free TWSE data using matplotlib."""

    def __init__(
        self,
        data_fetcher: FreeDataFetcher | None = None,
        output_dir: str | None = None,
    ):
        """Initialize chart renderer.

        Args:
            data_fetcher: Optional free data fetcher (creates default if not provided)
            output_dir: Directory for chart output (defaults to output/charts/)
        """
        self.data_fetcher = data_fetcher or FreeDataFetcher()
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent / "output" / "charts")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, request: ChartRequest) -> str:
        """Render a chart based on the request.

        Args:
            request: Chart rendering request

        Returns:
            Path to the rendered chart file

        Raises:
            ChartRenderError: If rendering fails
        """
        try:
            if request.chart_data_requirements == ChartDataRequirement.PRICE_TREND:
                return self._render_price_trend(request)
            elif request.chart_data_requirements == ChartDataRequirement.PRICE_OHLC:
                return self._render_kline(request)
            elif request.chart_data_requirements == ChartDataRequirement.REVENUE_TREND:
                return self._render_revenue_trend(request)
            elif (
                request.chart_data_requirements
                == ChartDataRequirement.REVENUE_COMPARISON
            ):
                return self._render_revenue_comparison(request)
            elif request.chart_data_requirements == ChartDataRequirement.INDICATOR_TREND:
                return self._render_indicator_trend(request)
            elif request.chart_data_requirements == ChartDataRequirement.SECTOR_ANALYSIS:
                return self._render_sector_analysis(request)
            else:
                raise ChartRenderError(
                    f"Unsupported chart data requirement: {request.chart_data_requirements}"
                )
        except ChartRenderError:
            raise
        except FreeDataError as e:
            raise ChartRenderError(f"Chart rendering failed: {e}") from e
        except Exception as e:
            raise ChartRenderError(f"Chart rendering failed: {e}") from e

    def _get_date_range(
        self, request: ChartRequest, default_days: int = 365
    ) -> tuple[date, date]:
        """Get start and end dates for the chart.

        Args:
            request: Chart request
            default_days: Default number of days to look back if no start_date specified

        Returns:
            Tuple of (start_date, end_date)
        """
        end_date = request.end_date or datetime.now().date()
        if request.start_date:
            start_date = request.start_date
        else:
            start_date = end_date - timedelta(days=default_days)
        return start_date, end_date

    def _render_price_trend(self, request: ChartRequest) -> str:
        """Render a price trend line chart."""
        start_date, end_date = self._get_date_range(request, default_days=365)
        stock_code = request.stock_codes[0]

        data = self.data_fetcher.get_stock_price(
            stock_id=stock_code,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        if not data:
            raise ChartRenderError(f"No price data for {stock_code}")

        dates = [datetime.strptime(d["date"], "%Y-%m-%d").date() for d in data]
        closes = [d["close"] for d in data]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, closes, linewidth=1.5, color="#2563eb")
        ax.set_title(f"{stock_code} 股價走勢圖", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期")
        ax.set_ylabel("股價 (TWD)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        filename = f"{stock_code}_price_trend_{start_date}_{end_date}.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        return str(filepath)

    def _render_kline(self, request: ChartRequest) -> str:
        """Render a K-line (candlestick) chart."""
        start_date, end_date = self._get_date_range(request, default_days=180)
        stock_code = request.stock_codes[0]

        data = self.data_fetcher.get_stock_price(
            stock_id=stock_code,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        if not data:
            raise ChartRenderError(f"No price data for {stock_code}")

        dates = [datetime.strptime(d["date"], "%Y-%m-%d").date() for d in data]
        opens = [d["open"] for d in data]
        closes = [d["close"] for d in data]
        highs = [d["high"] for d in data]
        lows = [d["low"] for d in data]

        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.6
        colors = ["#ef4444" if c >= o else "#22c55e" for c, o in zip(closes, opens)]

        for i, (date_val, open_val, close_val, high_val, low_val, color) in enumerate(
            zip(dates, opens, closes, highs, lows, colors)
        ):
            ax.vlines(
                date_val, low_val, high_val, color=color, linewidth=1
            )
            ax.hlines(
                open_val,
                date_val - timedelta(days=int(width * 30)),
                date_val + timedelta(days=int(width * 30)),
                color=color,
                linewidth=2,
            )
            ax.hlines(
                close_val,
                date_val - timedelta(days=int(width * 30)),
                date_val + timedelta(days=int(width * 30)),
                color=color,
                linewidth=2,
            )

        ax.set_title(f"{stock_code} K線圖", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期")
        ax.set_ylabel("股價 (TWD)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        filename = f"{stock_code}_kline_{start_date}_{end_date}.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        return str(filepath)

    def _render_revenue_trend(self, request: ChartRequest) -> str:
        """Render a revenue trend line chart (requires FinMind paid tier)."""
        start_date, end_date = self._get_date_range(request, default_days=1095)
        stock_code = request.stock_codes[0]

        data = self.data_fetcher.get_month_revenue(
            stock_id=stock_code,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        if not data:
            raise ChartRenderError(f"No revenue data for {stock_code}")

        dates = [datetime.strptime(d["date"], "%Y-%m-%d").date() for d in data]
        revenues = [d["revenue"] for d in data]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, revenues, linewidth=1.5, color="#10b981", marker="o", markersize=4)
        ax.set_title(f"{stock_code} 營收趨勢圖", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期")
        ax.set_ylabel("營收 (TWD)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        filename = f"{stock_code}_revenue_trend_{start_date}_{end_date}.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        return str(filepath)

    def _render_revenue_comparison(self, request: ChartRequest) -> str:
        """Render a revenue comparison bar chart (requires FinMind paid tier)."""
        start_date, end_date = self._get_date_range(request, default_days=365)
        stock_codes = request.stock_codes

        fig, ax = plt.subplots(figsize=(12, 6))
        all_revenues = []
        labels = []

        for stock_code in stock_codes:
            data = self.data_fetcher.get_month_revenue(
                stock_id=stock_code,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
            if data:
                total_revenue = sum(d["revenue"] for d in data)
                all_revenues.append(total_revenue)
                labels.append(stock_code)

        if not all_revenues:
            raise ChartRenderError("No revenue data for any stock")

        bars = ax.bar(
            labels, all_revenues, color=["#3b82f6", "#ef4444", "#10b981", "#f59e0b"]
        )
        ax.set_title("營收比較圖", fontsize=14, fontweight="bold")
        ax.set_xlabel("股票代碼")
        ax.set_ylabel("總營收 (TWD)")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, revenue in zip(bars, all_revenues):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{revenue:,.0f}",
                ha="center",
                va="bottom",
            )

        fig.tight_layout()
        filename = f"revenue_comparison_{start_date}_{end_date}.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        return str(filepath)

    def _render_indicator_trend(self, request: ChartRequest) -> str:
        """Render an indicator trend chart (e.g., PE) from free TWSE data."""
        start_date, end_date = self._get_date_range(request, default_days=1095)
        stock_code = request.stock_codes[0]

        data = self.data_fetcher.get_stock_per(
            stock_id=stock_code,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        if not data:
            raise ChartRenderError(f"No indicator data for {stock_code}")

        dates = [datetime.strptime(d["date"], "%Y-%m-%d").date() for d in data]
        pe_ratios = [d["pe"] for d in data]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            dates, pe_ratios, linewidth=1.5, color="#8b5cf6", marker="o", markersize=4
        )
        ax.set_title(f"{stock_code} PE 趨勢圖", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期")
        ax.set_ylabel("PE 比率")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        filename = f"{stock_code}_indicator_trend_{start_date}_{end_date}.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        return str(filepath)

    def _render_sector_analysis(self, request: ChartRequest) -> str:
        """Render a sector analysis chart (latest PE per stock)."""
        start_date, end_date = self._get_date_range(request, default_days=365)
        stock_codes = request.stock_codes

        fig, ax = plt.subplots(figsize=(12, 6))
        pe_ratios = []
        labels = []

        for stock_code in stock_codes:
            data = self.data_fetcher.get_stock_per(
                stock_id=stock_code,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
            if data:
                latest_pe = data[-1]["pe"] if data else 0
                pe_ratios.append(latest_pe)
                labels.append(stock_code)

        if not pe_ratios:
            raise ChartRenderError("No sector data available")

        bars = ax.bar(
            labels, pe_ratios, color=["#3b82f6", "#ef4444", "#10b981", "#f59e0b"]
        )
        ax.set_title("產業分析圖 (PE 比率)", fontsize=14, fontweight="bold")
        ax.set_xlabel("股票代碼")
        ax.set_ylabel("PE 比率")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, pe in zip(bars, pe_ratios):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{pe:.2f}",
                ha="center",
                va="bottom",
            )

        fig.tight_layout()
        filename = "sector_analysis.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        return str(filepath)
