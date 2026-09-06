"""Unit tests for Step 4 Stock Enumeration (no real LLM / ISIN / network)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from classifier.isin_client import IsinClientError
from classifier.llm_client import LLMError
from classifier.models import (
    BoundaryResult,
    RangeQueryType,
    RangeStockResult,
    RelatedStockResult,
    StockItem,
)
from classifier.stock_enumeration import (
    StockEnumeration,
    StockEnumerationError,
    _RangeValidationModel,
    _SupplyChainModel,
)


def _stock(code: str, name: str) -> StockItem:
    return StockItem(code=code, name=name)


class TestStockEnumeration:
    """Stock enumeration Step 4 unit tests."""

    def _runner(self, llm_client=None, isin=None, resolver=None):
        return StockEnumeration(llm_client=llm_client, isin=isin, resolver=resolver)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def test_dispatch_single_stock_uses_related(self):
        llm = Mock()
        llm.extract_structured.return_value = _SupplyChainModel(
            upstream=[_stock("3701", "大立光")],
            downstream=[_stock("2317", "鴻海")],
            confidence=0.5,
        )
        resolver = Mock()
        resolver.verify_code.side_effect = [
            {"exists": True, "name": "台積電", "name_matches": None, "market": "TWSE"},
            {"exists": True, "name": "大立光", "name_matches": True, "market": "TWSE"},
            {"exists": True, "name": "鴻海", "name_matches": True, "market": "TWSE"},
        ]
        runner = self._runner(llm_client=llm, resolver=resolver)
        boundary = BoundaryResult(
            stock_codes=["2330"], company_names=["台積電"], sectors=[], confidence=0.9
        )

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
    # Related mode
    # ------------------------------------------------------------------
    def test_related_filters_invalid_stocks(self):
        llm = Mock()
        llm.extract_structured.return_value = _SupplyChainModel(
            upstream=[_stock("3701", "大立光"), _stock("9999", "無效公司")],
            downstream=[_stock("2317", "鴻海")],
            confidence=0.4,
        )
        resolver = Mock()
        resolver.verify_code.side_effect = [
            {"exists": True, "name": "台積電", "name_matches": None, "market": "TWSE"},
            {"exists": True, "name": "大立光", "name_matches": True, "market": "TWSE"},
            {"exists": False, "name": None, "name_matches": None, "market": None},  # 9999 dropped
            {"exists": True, "name": "鴻海", "name_matches": True, "market": "TWSE"},
        ]
        runner = self._runner(llm_client=llm, resolver=resolver)

        result = runner.related_stock("2330")

        assert result.stock_name == "台積電"
        assert [s.code for s in result.upstream] == ["3701"]
        assert [s.code for s in result.downstream] == ["2317"]
        assert result.confidence == pytest.approx(0.4)

    def test_related_raises_on_llm_error(self):
        llm = Mock()
        llm.extract_structured.side_effect = LLMError("llm down")
        resolver = Mock()
        resolver.verify_code.return_value = {
            "exists": True,
            "name": "台積電",
            "name_matches": None,
            "market": "TWSE",
        }
        runner = self._runner(llm_client=llm, resolver=resolver)

        with pytest.raises(StockEnumerationError):
            runner.related_stock("2330")
