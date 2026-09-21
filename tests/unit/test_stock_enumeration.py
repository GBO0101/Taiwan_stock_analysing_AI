"""Unit tests for Step 4 Stock Enumeration (no real LLM / ISIN / network)."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from classifier.annual_report import AnnualReportError
from classifier.isin_client import IsinClientError
from classifier.llm_client import LLMError
from classifier.models import (
    AnnualReportDisclosure,
    BoundaryResult,
    CounterpartyDisclosure,
    DateRange,
    QuerySummary,
    QueryTopicKind,
    RangeQueryType,
    RangeStockResult,
    RelatedStockResult,
    StockItem,
    StockRelation,
    SupplyChainRelation,
    TimeScope,
)
from classifier.stock_enumeration import (
    StockEnumeration,
    StockEnumerationError,
    _RangeValidationModel,
    _SupplyChainEntry,
    _SupplyChainModel,
)


def _stock(code: str, name: str) -> StockItem:
    return StockItem(code=code, name=name)


def _verify(name: str, exists: bool = True) -> dict:
    return {
        "exists": exists,
        "name": name if exists else None,
        "name_matches": True if exists else None,
        "market": "TWSE" if exists else None,
    }


def _supply_chain(
    upstream: tuple[tuple[str, str, float], ...] = (),
    downstream: tuple[tuple[str, str, float], ...] = (),
    confidence: float = 0.5,
) -> _SupplyChainModel:
    return _SupplyChainModel(
        upstream=[_SupplyChainEntry(code=c, name=n, impact=i) for c, n, i in upstream],
        downstream=[_SupplyChainEntry(code=c, name=n, impact=i) for c, n, i in downstream],
        confidence=confidence,
    )


# Canned 2330 annual report (users' own reports list counterparties 2317/3008).
_DISCLOSURE_2330 = AnnualReportDisclosure(
    target_code="2330",
    target_name="台積電",
    fiscal_year=113,
    report_type="F18",
    pdf_name="2330_113.pdf",
    pdf_url="https://doc.twse.com.tw/pdf/2330_113.pdf",
    customers=[
        CounterpartyDisclosure(
            raw_name="鴻海", resolved_code="2317", resolved_name="鴻海", ratio=29.5
        ),
    ],
    suppliers=[
        CounterpartyDisclosure(
            raw_name="大立光", resolved_code="3008", resolved_name="大立光", ratio=12.3
        ),
    ],
    confidence=0.8,
)

# 2317's own report confirming the two-sided link back to 2330.
_DISCLOSURE_2317 = AnnualReportDisclosure(
    target_code="2317",
    target_name="鴻海",
    fiscal_year=113,
    report_type="F18",
    pdf_name="2317_113.pdf",
    pdf_url="https://doc.twse.com.tw/pdf/2317_113.pdf",
    customers=[
        CounterpartyDisclosure(
            raw_name="台積電", resolved_code="2330", resolved_name="台積電", ratio=10.0
        ),
    ],
    confidence=0.7,
)


class TestStockEnumeration:
    """Stock enumeration Step 4 unit tests."""

    def _runner(self, llm_client=None, isin=None, resolver=None):
        return StockEnumeration(llm_client=llm_client, isin=isin, resolver=resolver)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def test_dispatch_single_stock_uses_related(self):
        llm = Mock()
        llm.extract_structured.return_value = _supply_chain(
            upstream=(("3701", "大立光", 0.5),),
            downstream=(("2317", "鴻海", 0.5),),
        )
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台積電"),
            _verify("大立光"),
            _verify("鴻海"),
        ]
        runner = self._runner(llm_client=llm, resolver=resolver)
        boundary = BoundaryResult(
            stock_codes=["2330"], company_names=["台積電"], sectors=[], confidence=0.9
        )

        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            side_effect=AnnualReportError("no report"),
        ):
            result = runner.run("台積電的上下游", boundary)

        assert isinstance(result, RelatedStockResult)
        assert result.stock_code == "2330"
        assert [s.code for s in result.upstream] == ["3701"]
        assert [s.code for s in result.downstream] == ["2317"]

    def test_dispatch_sector_uses_range(self):
        llm = Mock()
        llm.extract_structured.return_value = _RangeValidationModel(
            matched_industry="半導體", confidence=0.8
        )
        isin = Mock()
        isin.by_sector.return_value = [_stock("2330", "台積電")]
        runner = self._runner(llm_client=llm, isin=isin)
        boundary = BoundaryResult(
            stock_codes=[], company_names=[], sectors=["半導體"], confidence=0.8
        )

        result = runner.run("半導體有哪些股票", boundary)

        assert isinstance(result, RangeStockResult)
        assert result.query_type == RangeQueryType.SECTOR
        assert result.query_value == "半導體"

    def test_dispatch_range_raises_without_value(self):
        runner = self._runner()
        boundary = BoundaryResult(stock_codes=[], company_names=[], sectors=[], confidence=0.5)

        with pytest.raises(StockEnumerationError):
            runner.run("沒有範圍的查詢", boundary)

    # ------------------------------------------------------------------
    # Range mode
    # ------------------------------------------------------------------
    def test_range_api_plus_validation_combined(self):
        llm = Mock()
        llm.extract_structured.return_value = _RangeValidationModel(
            matched_industry="半導體",
            to_add=[_stock("2330", "台積電")],
            to_remove=["1234"],
            confidence=0.9,
        )
        isin = Mock()
        isin.by_sector.return_value = [_stock("2454", "聯發科"), _stock("1234", "某公司")]
        runner = self._runner(llm_client=llm, isin=isin)

        result = runner.range_stocks("半導體")

        codes = [s.code for s in result.stocks]
        # to_remove dropped, to_add merged, no dedup
        assert "1234" not in codes
        assert "2330" in codes
        assert "2454" in codes
        assert result.source == "combined"
        # min(0.9 (api ok), validation 0.9)
        assert result.confidence == pytest.approx(0.9)

    def test_range_api_only_when_no_validation(self):
        llm = Mock()
        llm.extract_structured.side_effect = LLMError("llm failed")
        isin = Mock()
        isin.by_sector.return_value = [_stock("2330", "台積電")]
        runner = self._runner(llm_client=llm, isin=isin)

        result = runner.range_stocks("半導體")

        assert result.source == "twse_api"
        assert result.confidence == pytest.approx(0.8)

    def test_range_llm_only_when_api_fails(self):
        llm = Mock()
        llm.extract_structured.return_value = _RangeValidationModel(
            matched_industry="晶圓代工", to_add=[_stock("2330", "台積電")], confidence=0.6
        )
        isin = Mock()
        isin.by_sector.side_effect = IsinClientError("isin unavailable")
        runner = self._runner(llm_client=llm, isin=isin)

        result = runner.range_stocks("晶圓代工")

        assert result.source == "llm"
        assert [s.code for s in result.stocks] == ["2330"]
        assert result.confidence == pytest.approx(0.6)

    def test_range_raises_when_no_source(self):
        llm = Mock()
        llm.extract_structured.side_effect = LLMError("llm failed")
        isin = Mock()
        isin.by_sector.side_effect = IsinClientError("isin unavailable")
        runner = self._runner(llm_client=llm, isin=isin)

        with pytest.raises(StockEnumerationError):
            runner.range_stocks("半導體")

    # ------------------------------------------------------------------
    # Related mode — LLM fallback (annual report unavailable)
    # ------------------------------------------------------------------
    def test_related_filters_invalid_stocks(self):
        llm = Mock()
        llm.extract_structured.return_value = _supply_chain(
            upstream=(("3701", "大立光", 0.5), ("9999", "無效公司", 0.1)),
            downstream=(("2317", "鴻海", 0.5),),
            confidence=0.4,
        )
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台積電"),
            _verify("大立光"),
            _verify("無效公司", exists=False),  # 9999 dropped
            _verify("鴻海"),
        ]
        runner = self._runner(llm_client=llm, resolver=resolver)

        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            side_effect=AnnualReportError("no report"),
        ):
            result = runner.related_stock("2330")

        assert result.stock_name == "台積電"
        assert result.source == "llm"
        assert [s.code for s in result.upstream] == ["3701"]
        assert [s.code for s in result.downstream] == ["2317"]
        assert result.confidence == pytest.approx(0.4)
        # LLM fallback marks every link as llm_inferred without a ratio.
        assert all(e.disclosed_by == "llm_inferred" for e in result.evidence)
        assert all(e.target_ratio is None for e in result.evidence)

    def test_related_filters_code_name_mismatch(self):
        # 止血：LLM 把 2454（實為聯發科）配上「大立光」→ verify_code 回報
        # name_matches=False → 在 _validate_stocks 被剔除。垃圾提名絕不會觸發
        # 第二次年報抓取（extract_annual_report 只為目標公司自身呼叫，沒有
        # 任何一次呼叫針對 2454）。
        llm = Mock()
        llm.extract_structured.return_value = _supply_chain(
            upstream=(("2454", "大立光", 0.5),),
            confidence=0.4,
        )
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台積電"),
            {
                "exists": True,
                "name": "聯發科",
                "name_matches": False,
                "market": "TWSE",
            },
        ]
        runner = self._runner(llm_client=llm, resolver=resolver)

        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            side_effect=AnnualReportError("no report"),
        ) as mock_extract:
            result = runner.related_stock("2330")

        assert result.source == "llm"
        assert [s.code for s in result.upstream] == []
        # 止血核心：碼名錯配的提名被剔除後，不曾為 2454 抓年報。
        assert all(call.args[0] != "2454" for call in mock_extract.call_args_list)
        assert mock_extract.call_count == 1  # 僅目標公司自身之年報嘗試

    def test_related_raises_on_llm_error(self):
        llm = Mock()
        llm.extract_structured.side_effect = LLMError("llm down")
        resolver = Mock()
        resolver.verify_code.return_value = _verify("台積電")
        runner = self._runner(llm_client=llm, resolver=resolver)

        with (
            patch(
                "classifier.stock_enumeration.extract_annual_report",
                side_effect=AnnualReportError("no report"),
            ),
            pytest.raises(StockEnumerationError),
        ):
            runner.related_stock("2330")

    # ------------------------------------------------------------------
    # Related mode — LLM fallback + two-sided annual-report verification
    # ------------------------------------------------------------------
    def test_related_llm_verify_promotes_matching_links(self):
        # Tier 2: 目標年報無法取得 → LLM 提名 3008(上游, impact 0.7) 與
        # 2317(下游, impact 0.7)。被提名公司的年報雙向揭露目標、比例落在
        # LLM impact 帶（0.6-0.8 → 5-20%，容差後 3-22%）→ 升級為 counterparty。
        llm = Mock()
        llm.extract_structured.return_value = _supply_chain(
            upstream=(("3008", "大立光", 0.7),),
            downstream=(("2317", "鴻海", 0.7),),
            confidence=0.4,
        )
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台積電"),
            _verify("大立光"),
            _verify("鴻海"),
        ]
        runner = self._runner(llm_client=llm, resolver=resolver)

        # 3008 是台積電的上游供應商 → 台積電是 3008 的客戶（customers）。
        d3008 = AnnualReportDisclosure(
            target_code="3008",
            target_name="大立光",
            fiscal_year=113,
            report_type="F18",
            pdf_name="3008_113.pdf",
            pdf_url="x",
            customers=[
                CounterpartyDisclosure(
                    raw_name="台積電", resolved_code="2330", resolved_name="台積電", ratio=10.0
                ),
            ],
            confidence=0.8,
        )
        # 2317 是台積電的下游客戶 → 台積電是 2317 的供應商（suppliers）。
        d2317 = AnnualReportDisclosure(
            target_code="2317",
            target_name="鴻海",
            fiscal_year=113,
            report_type="F18",
            pdf_name="2317_113.pdf",
            pdf_url="x",
            suppliers=[
                CounterpartyDisclosure(
                    raw_name="台積電", resolved_code="2330", resolved_name="台積電", ratio=15.0
                ),
            ],
            confidence=0.8,
        )
        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            side_effect=[AnnualReportError("no report"), d3008, d2317],
        ):
            result = runner.related_stock("2330")

        assert result.source == "llm"
        # 雙向揭露且比例一致 → 升級為 counterparty，帶真實比例/來源/年度。
        ev_3008 = next(e for e in result.evidence if e.counterparty_code == "3008")
        assert ev_3008.relation == SupplyChainRelation.UPSTREAM
        assert ev_3008.disclosed_by == "counterparty"
        assert ev_3008.target_ratio == pytest.approx(10.0)
        assert ev_3008.source_pdf == "3008_113.pdf"
        assert ev_3008.fiscal_year == 113

        ev_2317 = next(e for e in result.evidence if e.counterparty_code == "2317")
        assert ev_2317.relation == SupplyChainRelation.DOWNSTREAM
        assert ev_2317.disclosed_by == "counterparty"
        assert ev_2317.target_ratio == pytest.approx(15.0)
        assert ev_2317.source_pdf == "2317_113.pdf"
        assert ev_2317.fiscal_year == 113

        # impact_map 改用真實比例（10/100, 15/100）；confidence 小幅提升。
        assert runner._last_meta["impact_map"][
            ("3008", SupplyChainRelation.UPSTREAM)
        ] == pytest.approx(10.0 / 100)
        assert runner._last_meta["impact_map"][
            ("2317", SupplyChainRelation.DOWNSTREAM)
        ] == pytest.approx(15.0 / 100)
        assert result.confidence == pytest.approx(0.5)
        assert "雙向揭露證實" in " ".join(runner._last_meta["notes"])

    def test_related_llm_verify_keeps_inferred_on_ratio_mismatch(self):
        # 被提名公司年報揭露目標，但比例（40.0%）超出 impact 0.7 帶
        # （5-20%，容差後上限 22%）→ 維持 llm_inferred。
        llm = Mock()
        llm.extract_structured.return_value = _supply_chain(
            upstream=(("3008", "大立光", 0.7),),
            confidence=0.4,
        )
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台積電"),
            _verify("大立光"),
        ]
        runner = self._runner(llm_client=llm, resolver=resolver)

        d3008 = AnnualReportDisclosure(
            target_code="3008",
            target_name="大立光",
            fiscal_year=113,
            report_type="F18",
            pdf_name="3008_113.pdf",
            pdf_url="x",
            customers=[
                CounterpartyDisclosure(
                    raw_name="台積電", resolved_code="2330", resolved_name="台積電", ratio=40.0
                ),
            ],
            confidence=0.8,
        )
        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            side_effect=[AnnualReportError("no report"), d3008],
        ):
            result = runner.related_stock("2330")

        ev_3008 = next(e for e in result.evidence if e.counterparty_code == "3008")
        assert ev_3008.disclosed_by == "llm_inferred"
        assert ev_3008.target_ratio is None
        assert ev_3008.source_pdf is None
        assert result.confidence == pytest.approx(0.4)
        assert "雙向揭露證實" not in " ".join(runner._last_meta["notes"])

    def test_related_llm_verify_keeps_inferred_on_direction_mismatch(self):
        # 被提名公司年報有揭露目標，但方向與推測相反（3008 被提名為上游供應
        # 商應將目標列為客戶，這裡卻列為供應商）→ 維持 llm_inferred。
        llm = Mock()
        llm.extract_structured.return_value = _supply_chain(
            upstream=(("3008", "大立光", 0.7),),
            confidence=0.4,
        )
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台積電"),
            _verify("大立光"),
        ]
        runner = self._runner(llm_client=llm, resolver=resolver)

        d3008 = AnnualReportDisclosure(
            target_code="3008",
            target_name="大立光",
            fiscal_year=113,
            report_type="F18",
            pdf_name="3008_113.pdf",
            pdf_url="x",
            suppliers=[
                CounterpartyDisclosure(
                    raw_name="台積電", resolved_code="2330", resolved_name="台積電", ratio=10.0
                ),
            ],
            confidence=0.8,
        )
        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            side_effect=[AnnualReportError("no report"), d3008],
        ):
            result = runner.related_stock("2330")

        ev_3008 = next(e for e in result.evidence if e.counterparty_code == "3008")
        assert ev_3008.disclosed_by == "llm_inferred"
        assert ev_3008.target_ratio is None
        assert result.confidence == pytest.approx(0.4)

    # ------------------------------------------------------------------
    # Related mode — annual report first
    # ------------------------------------------------------------------
    def test_related_uses_annual_report(self):
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台積電"),
            _verify("大立光"),
            _verify("鴻海"),
        ]
        runner = self._runner(llm_client=Mock(), resolver=resolver)

        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            return_value=_DISCLOSURE_2330,
        ):
            result = runner.related_stock("2330")

        assert result.source == "annual_report"
        assert result.confidence == pytest.approx(0.8)  # max(0.6, disclosure 0.8)
        # suppliers → upstream, customers → downstream, ratios drive impact.
        assert [s.code for s in result.upstream] == ["3008"]
        assert [s.code for s in result.downstream] == ["2317"]

        ev_3008 = next(e for e in result.evidence if e.counterparty_code == "3008")
        assert ev_3008.relation == SupplyChainRelation.UPSTREAM
        assert ev_3008.disclosed_by == "target"
        assert ev_3008.target_ratio == pytest.approx(12.3)
        assert ev_3008.source_pdf == "2330_113.pdf"

        ev_2317 = next(e for e in result.evidence if e.counterparty_code == "2317")
        assert ev_2317.relation == SupplyChainRelation.DOWNSTREAM
        assert ev_2317.target_ratio == pytest.approx(29.5)

    def test_related_cross_validates_two_sided_link(self):
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台積電"),
            _verify("大立光"),
            _verify("鴻海"),
        ]
        runner = self._runner(llm_client=Mock(), resolver=resolver)

        # Order of extract calls: target first, then 2317 (ratio 29.5) before
        # 3008 (ratio 12.3); 3008's own report is unavailable.
        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            side_effect=[
                _DISCLOSURE_2330,
                _DISCLOSURE_2317,
                AnnualReportError("3008 report unavailable"),
            ],
        ):
            result = runner.related_stock("2330")

        ev_2317 = next(e for e in result.evidence if e.counterparty_code == "2317")
        ev_3008 = next(e for e in result.evidence if e.counterparty_code == "3008")
        # 2317's own annual report confirms it, so the link is two-sided.
        assert ev_2317.disclosed_by == "both"
        # 3008's report failed; stays as target-disclosed only.
        assert ev_3008.disclosed_by == "target"

    def test_ratio_less_row_does_not_overwrite_known_ratio(self):
        # A counterparty appears in the raw-material table (ratio=None) in
        # addition to the >10% disclosure row (ratio=13.93). The ratio-less
        # row must NOT overwrite the impact with 0.0 (南亞 1303 年報案例).
        disclosure = AnnualReportDisclosure(
            target_code="2330",
            target_name="台積電",
            fiscal_year=113,
            report_type="F04",
            pdf_name="2330_113.pdf",
            pdf_url="x",
            suppliers=[
                CounterpartyDisclosure(
                    raw_name="台灣化學纖維", resolved_code="1326", resolved_name="台化", ratio=13.93
                ),
                CounterpartyDisclosure(
                    raw_name="台化公司", resolved_code="1326", resolved_name="台化", ratio=None
                ),
                CounterpartyDisclosure(
                    raw_name="台塑石化", resolved_code="6505", resolved_name="台塑化", ratio=10.62
                ),
            ],
            customers=[],
            confidence=0.8,
        )
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台積電"),
            _verify("台化"),
            _verify("台塑化"),
        ]
        runner = self._runner(llm_client=Mock(), resolver=resolver)

        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            return_value=disclosure,
        ):
            result = runner.related_stock("2330")
        summary = runner.build_summary(BoundaryResult(stock_codes=["2330"], confidence=0.9), result)

        # ratio=None raw-material row merges into the ratio-bearing row (max
        # wins), so impact_map keeps the known 13.93 → impact 0.1393, not 0.0.
        assert [s.code for s in result.upstream] == ["1326", "6505"]
        by_code = {s.code: s for s in summary.related_stocks}
        assert by_code["1326"].impact_score == pytest.approx(13.93 / 100)
        assert by_code["6505"].impact_score == pytest.approx(10.62 / 100)
        assert by_code["1326"].ratio == pytest.approx(13.93)
        # The duplicate ratio-less row was deduped: 1326 keeps exactly one
        # evidence row carrying the known 13.93 ratio.
        ev_1326 = [e for e in result.evidence if e.counterparty_code == "1326"]
        assert len(ev_1326) == 1
        assert ev_1326[0].target_ratio == pytest.approx(13.93)
        assert all(e.target_ratio is not None for e in result.evidence)

    # ------------------------------------------------------------------
    # Unified summary (PipelineResult.summary)
    # ------------------------------------------------------------------
    def test_build_summary_related(self):
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台積電"),
            _verify("大立光"),
            _verify("鴻海"),
        ]
        runner = self._runner(llm_client=Mock(), resolver=resolver)
        boundary = BoundaryResult(stock_codes=["2330"], company_names=["臺積電"], confidence=0.9)

        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            return_value=_DISCLOSURE_2330,
        ):
            result = runner.related_stock("2330", boundary)
        summary = runner.build_summary(boundary, result)

        assert isinstance(summary, QuerySummary)
        assert summary.topic_kind == QueryTopicKind.SINGLE_STOCK
        assert summary.topic_code == "2330"
        assert summary.topic_name == "台積電"
        # Used fiscal year = the annual report's (113); newest plays primary.
        assert summary.fiscal_years == [113]
        # Target first (1.0), then downstream 2317 (29.5%), upstream 3008 (12.3%).
        assert [s.code for s in summary.related_stocks] == ["2330", "2317", "3008"]

        target = summary.related_stocks[0]
        assert target.relation == StockRelation.TARGET
        assert target.impact_score == pytest.approx(1.0)
        assert target.detail_source == "annual_report"

        down = summary.related_stocks[1]
        assert down.relation == StockRelation.DOWNSTREAM
        assert down.impact_score == pytest.approx(29.5 / 100)
        assert down.ratio == pytest.approx(29.5)
        assert down.fiscal_year == 113
        assert down.detail_source == "annual_report"

        up = summary.related_stocks[2]
        assert up.relation == StockRelation.UPSTREAM
        assert up.impact_score == pytest.approx(12.3 / 100)

    def test_build_summary_range(self):
        llm = Mock()
        llm.extract_structured.return_value = _RangeValidationModel(
            matched_industry="半導體",
            confidence=0.8,
            relevance={"2330": 0.95, "2454": 0.6},
        )
        isin = Mock()
        isin.by_sector.return_value = [_stock("2330", "台積電"), _stock("2454", "聯發科")]
        runner = self._runner(llm_client=llm, isin=isin)
        boundary = BoundaryResult(
            stock_codes=[], company_names=[], sectors=["半導體"], confidence=0.8
        )

        result = runner.range_stocks("半導體")
        summary = runner.build_summary(boundary, result)

        assert summary.topic_kind == QueryTopicKind.RANGE
        assert summary.range_type == RangeQueryType.SECTOR
        assert summary.range_value == "半導體"
        assert summary.topic_name == "半導體"
        # Sorted by relevance descending.
        assert [s.code for s in summary.related_stocks] == ["2330", "2454"]
        leader = summary.related_stocks[0]
        assert leader.relation == StockRelation.RANGE_MEMBER
        assert leader.impact_score == pytest.approx(0.95)
        assert leader.detail_source == "combined"
        assert summary.fiscal_years == [115]  # current ROC year, no explicit range

    # ------------------------------------------------------------------
    # Related mode — multi-year history (fiscal_year per disclosure)
    # ------------------------------------------------------------------
    def _multi_year_disclosures(self):
        """Canned 2330 annual reports for 民國 112/113/114 (ascending years)."""
        return [
            AnnualReportDisclosure(
                target_code="2330",
                target_name="台積電",
                fiscal_year=112,
                report_type="F04",
                pdf_name="2330_112.pdf",
                pdf_url="https://doc.twse.com.tw/pdf/2330_112.pdf",
                customers=[
                    CounterpartyDisclosure(
                        raw_name="鴻海", resolved_code="2317", resolved_name="鴻海", ratio=20.0
                    ),
                ],
                suppliers=[
                    CounterpartyDisclosure(
                        raw_name="大立光", resolved_code="3008", resolved_name="大立光", ratio=15.0
                    ),
                ],
                confidence=0.7,
            ),
            AnnualReportDisclosure(
                target_code="2330",
                target_name="台積電",
                fiscal_year=113,
                report_type="F04",
                pdf_name="2330_113.pdf",
                pdf_url="https://doc.twse.com.tw/pdf/2330_113.pdf",
                customers=[
                    CounterpartyDisclosure(
                        raw_name="鴻海", resolved_code="2317", resolved_name="鴻海", ratio=25.0
                    ),
                ],
                suppliers=[
                    CounterpartyDisclosure(
                        raw_name="大立光", resolved_code="3008", resolved_name="大立光", ratio=10.0
                    ),
                ],
                confidence=0.7,
            ),
            AnnualReportDisclosure(
                target_code="2330",
                target_name="台積電",
                fiscal_year=114,
                report_type="F04",
                pdf_name="2330_114.pdf",
                pdf_url="https://doc.twse.com.tw/pdf/2330_114.pdf",
                customers=[
                    CounterpartyDisclosure(
                        raw_name="鴻海", resolved_code="2317", resolved_name="鴻海", ratio=29.5
                    ),
                ],
                suppliers=[
                    CounterpartyDisclosure(
                        raw_name="大立光", resolved_code="3008", resolved_name="大立光", ratio=12.3
                    ),
                    CounterpartyDisclosure(
                        raw_name="聯發科", resolved_code="2454", resolved_name="聯發科", ratio=8.0
                    ),
                ],
                confidence=0.8,
            ),
        ]

    def _range_boundary(self, value: str) -> BoundaryResult:
        return BoundaryResult(
            stock_codes=["2330"],
            company_names=["台積電"],
            confidence=0.9,
            date_range=DateRange(type=TimeScope.ABSOLUTE, value=value),
        )

    def test_related_multi_year_primary_is_newest(self):
        d112, d113, d114 = self._multi_year_disclosures()
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台積電"),
            _verify("大立光"),
            _verify("聯發科"),
            _verify("鴻海"),
        ]
        runner = self._runner(llm_client=Mock(), resolver=resolver)

        # extract calls: 3 target years, then 3 cross-validation lookups of the
        # primary year's partners (2317/3008/2454) which all fail best-effort.
        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            side_effect=[
                d112,
                d113,
                d114,
                AnnualReportError("2317 年報無"),
                AnnualReportError("3008 年報無"),
                AnnualReportError("2454 年報無"),
            ],
        ) as mock_extract:
            result = runner.related_stock("2330", self._range_boundary("2023-01-01/2025-12-31"))

        assert result.source == "annual_report"
        # 3 target-year extractions + 3 primary-year cross-validations.
        assert mock_extract.call_count == 6
        assert [c.args[2] for c in mock_extract.call_args_list[:3]] == [112, 113, 114]

        # Stocks de-duplicated across years.
        assert [s.code for s in result.upstream] == ["3008", "2454"]
        assert [s.code for s in result.downstream] == ["2317"]

        # Every historical disclosure keeps its own fiscal_year tag.
        hon = sorted(
            (e.fiscal_year, e.target_ratio)
            for e in result.evidence
            if e.counterparty_code == "2317"
        )
        assert hon == [(112, 20.0), (113, 25.0), (114, 29.5)]
        lar = sorted(
            (e.fiscal_year, e.target_ratio)
            for e in result.evidence
            if e.counterparty_code == "3008"
        )
        assert lar == [(112, 15.0), (113, 10.0), (114, 12.3)]
        mtk = next(e for e in result.evidence if e.counterparty_code == "2454")
        assert mtk.fiscal_year == 114
        assert all(e.disclosed_by == "target" for e in result.evidence)

        # _last_meta: newest-first fiscal years; 114 is the newest available.
        assert runner._last_meta["fiscal_years"] == [114, 113, 112]
        assert "年報未公告" not in " ".join(runner._last_meta["notes"])

    def test_related_multi_year_skips_failed_years(self):
        d113, d114 = self._multi_year_disclosures()[1:]
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台積電"),
            _verify("大立光"),
            _verify("聯發科"),
            _verify("鴻海"),
        ]
        runner = self._runner(llm_client=Mock(), resolver=resolver)

        # 民國 112 fails to fetch; 113/114 succeed (cross-validations all fail).
        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            side_effect=[
                AnnualReportError("民國 112 年報無"),
                d113,
                d114,
                AnnualReportError("2317 年報無"),
                AnnualReportError("3008 年報無"),
                AnnualReportError("2454 年報無"),
            ],
        ):
            result = runner.related_stock("2330", self._range_boundary("2023-01-01/2025-12-31"))

        assert result.source == "annual_report"
        assert [s.code for s in result.upstream] == ["3008", "2454"]
        # 112 missing entirely from evidence and from the fiscal history.
        assert all(e.fiscal_year != 112 for e in result.evidence)
        assert runner._last_meta["fiscal_years"] == [114, 113]
        assert "民國 112 年報無法取得" in " ".join(runner._last_meta["notes"])

    def test_related_all_years_empty_falls_back_to_llm(self):
        llm = Mock()
        llm.extract_structured.return_value = _supply_chain(
            upstream=(("3008", "大立光", 0.5),),
            downstream=(("2317", "鴻海", 0.5),),
        )
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台積電"),
            _verify("大立光"),
            _verify("鴻海"),
        ]
        runner = self._runner(llm_client=llm, resolver=resolver)

        empty = AnnualReportDisclosure(
            target_code="2330",
            target_name="台積電",
            fiscal_year=114,
            report_type="F04",
            pdf_name="2330_114.pdf",
            pdf_url="x",
            customers=[],
            suppliers=[],
            confidence=0.6,
        )
        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            return_value=empty,
        ):
            result = runner.related_stock("2330", self._range_boundary("2023-01-01/2025-12-31"))

        assert result.source == "llm"
        assert all(e.disclosed_by == "llm_inferred" for e in result.evidence)
        assert all(e.fiscal_year is None for e in result.evidence)
        # No usable annual report → no fiscal history at all.
        assert runner._last_meta["fiscal_years"] == []
        assert "未揭露可解析之具名客戶/供應商" in " ".join(runner._last_meta["notes"])

    def test_related_dedupes_duplicate_evidence(self):
        # 台化 1326 案例：同一家公司被 LLM 在年報中重複抽出多筆（>10% 表 +
        # 原料表 + 關係人表），且 target_ratio 不一致——6505 上游同時出現
        # 54.7 與 57.3、下游同時出現 10.2 與 9.9。去重後每 (code, relation)
        # 只留一筆、ratio 取最大值，且 summary 依 relation 分開查 ratio。
        disclosure = AnnualReportDisclosure(
            target_code="1326",
            target_name="台化",
            fiscal_year=113,
            report_type="F04",
            pdf_name="1326_113.pdf",
            pdf_url="x",
            suppliers=[
                CounterpartyDisclosure(
                    raw_name="台塑石化", resolved_code="6505", resolved_name="台塑化", ratio=54.7
                ),
                CounterpartyDisclosure(
                    raw_name="台塑石化(股)公司",
                    resolved_code="6505",
                    resolved_name="台塑化",
                    ratio=57.3,
                ),
                CounterpartyDisclosure(
                    raw_name="台塑石化公司",
                    resolved_code="6505",
                    resolved_name="台塑化",
                    ratio=57.3,
                ),
                CounterpartyDisclosure(
                    raw_name="中石化", resolved_code="1314", resolved_name="中石化", ratio=None
                ),
                CounterpartyDisclosure(
                    raw_name="中石化", resolved_code="1314", resolved_name="中石化", ratio=None
                ),
            ],
            customers=[
                CounterpartyDisclosure(
                    raw_name="台塑石化", resolved_code="6505", resolved_name="台塑化", ratio=10.2
                ),
                CounterpartyDisclosure(
                    raw_name="台塑石化", resolved_code="6505", resolved_name="台塑化", ratio=9.9
                ),
            ],
            confidence=0.8,
        )
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台化"),
            _verify("台塑化"),
            _verify("中石化"),
            _verify("台塑化"),  # downstream validation re-checks 6505
        ]
        runner = self._runner(llm_client=Mock(), resolver=resolver)

        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            return_value=disclosure,
        ):
            result = runner.related_stock("1326")

        # Dedup: one evidence row per (code, relation, fiscal_year), max wins.
        sups = [
            e
            for e in result.evidence
            if e.counterparty_code == "6505" and e.relation == SupplyChainRelation.UPSTREAM
        ]
        assert len(sups) == 1
        assert sups[0].target_ratio == pytest.approx(57.3)
        cust = [
            e
            for e in result.evidence
            if e.counterparty_code == "6505" and e.relation == SupplyChainRelation.DOWNSTREAM
        ]
        assert len(cust) == 1
        assert cust[0].target_ratio == pytest.approx(10.2)
        zhh = [e for e in result.evidence if e.counterparty_code == "1314"]
        assert len(zhh) == 1
        assert zhh[0].target_ratio is None

        # Summary keeps upstream and downstream ratios separate per relation.
        assert [s.code for s in result.upstream] == ["6505", "1314"]
        assert [s.code for s in result.downstream] == ["6505"]
        summary = runner.build_summary(BoundaryResult(stock_codes=["1326"], confidence=0.9), result)
        up = next(
            s
            for s in summary.related_stocks
            if s.code == "6505" and s.relation == StockRelation.UPSTREAM
        )
        down = next(
            s
            for s in summary.related_stocks
            if s.code == "6505" and s.relation == StockRelation.DOWNSTREAM
        )
        assert up.ratio == pytest.approx(57.3)
        assert up.impact_score == pytest.approx(57.3 / 100)
        assert down.ratio == pytest.approx(10.2)
        assert down.impact_score == pytest.approx(10.2 / 100)

    def test_build_summary_multi_year_uses_newest_ratios(self):
        d112, d113, d114 = self._multi_year_disclosures()
        resolver = Mock()
        resolver.verify_code.side_effect = [
            _verify("台積電"),
            _verify("大立光"),
            _verify("聯發科"),
            _verify("鴻海"),
        ]
        runner = self._runner(llm_client=Mock(), resolver=resolver)
        boundary = self._range_boundary("2023-01-01/2025-12-31")

        with patch(
            "classifier.stock_enumeration.extract_annual_report",
            side_effect=[
                d112,
                d113,
                d114,
                AnnualReportError("x"),
                AnnualReportError("x"),
                AnnualReportError("x"),
            ],
        ):
            result = runner.related_stock("2330", boundary)
        summary = runner.build_summary(boundary, result)

        # Newest-first fiscal history; ratios/impacts come from the newest year.
        assert summary.fiscal_years == [114, 113, 112]
        rows = {s.code: s for s in summary.related_stocks}
        assert rows["2317"].impact_score == pytest.approx(29.5 / 100)
        assert rows["3008"].impact_score == pytest.approx(12.3 / 100)
        assert rows["2454"].impact_score == pytest.approx(8.0 / 100)
        assert rows["2317"].fiscal_year == 114
