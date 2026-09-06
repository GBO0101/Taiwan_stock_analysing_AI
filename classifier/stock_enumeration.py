"""Step 4: Stock Enumeration.

Executes after Step 3 and produces either:
- RangeStockResult: all stocks matching a sector/index/market, or
- RelatedStockResult: upstream/downstream of a single stock.

Trigger (checked in the pipeline): boundary has stock codes OR a range/topic,
AND classification type is not NON_FINANCIAL.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from classifier.isin_client import IsinClient, IsinClientError
from classifier.isin_client import isin_client as default_isin
from classifier.llm_client import LLMClient, LLMError
from classifier.models import (
    BoundaryResult,
    RangeQueryType,
    RangeStockResult,
    RelatedStockResult,
    StockItem,
)
from classifier.prompts import prompt_manager
from classifier.stock_resolver import StockResolver
from classifier.stock_resolver import stock_resolver as default_resolver

logger = logging.getLogger(__name__)


class StockEnumerationError(Exception):
    """Step 4 errors."""


class _RangeValidationModel(BaseModel):
    """LLM output for range-mode stock-list validation."""

    matched_industry: str | None = None
    to_add: list[StockItem] = Field(default_factory=list)
    to_remove: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class _SupplyChainModel(BaseModel):
    """LLM output for supply-chain (upstream/downstream) inference."""

    upstream: list[StockItem] = Field(default_factory=list)
    downstream: list[StockItem] = Field(default_factory=list)
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)


class StockEnumeration:
    """Step 4 runner: range enumeration + supply-chain lookup."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        isin: IsinClient | None = None,
        resolver: StockResolver | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.isin = isin or default_isin
        self.resolver = resolver or default_resolver

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    def run(
        self,
        question: str,
        boundary: BoundaryResult,
    ) -> RangeStockResult | RelatedStockResult:
        """Dispatch to range or related based on the boundary.

        Args:
            question: Original user question (unused beyond context/debug).
            boundary: Step 1 boundary extraction output.

        Returns:
            A RangeStockResult or RelatedStockResult.

        Raises:
            StockEnumerationError: If enumeration fails entirely.
        """
        # Single-stock query → related (supply chain).
        if len(boundary.stock_codes) == 1:
            return self.related_stock(boundary.stock_codes[0])

        # Sector/range topic → range enumeration. Use the most specific sector
        # if the LLM/boundary named any; otherwise fall back to company names.
        if boundary.sectors:
            return self.range_stocks(boundary.sectors[0], RangeQueryType.SECTOR)

        # A range query without an explicit sector word: try the first company
        # name as the range value (best effort) or raise.
        if boundary.company_names:
            return self.range_stocks(boundary.company_names[0], RangeQueryType.SECTOR)

        raise StockEnumerationError(
            "Cannot enumerate: no sector/topic and no single stock code found"
        )

    # ------------------------------------------------------------------
    # Range mode
    # ------------------------------------------------------------------
    def range_stocks(
        self,
        query_value: str,
        query_type: RangeQueryType = RangeQueryType.SECTOR,
        market: str = "TWSE",
    ) -> RangeStockResult:
        """Enumerate stocks by sector, falling back to LLM if ISIN fails.

        API-first: filter the ISIN industry column. Then LLM-validate to add
        obvious missing leaders and drop wrong inclusions.
        """
        api_stocks: list[StockItem] = []
        api_ok = False
        try:
            api_stocks = self.isin.by_sector(query_value, market=market)
            api_ok = True
        except IsinClientError as e:
            logger.warning("ISIN lookup failed; falling back to LLM: %s", e)

        # LLM validation/supplement.
        try:
            prompt = prompt_manager.render_stock_validation(
                query_value=query_value,
                query_type=query_type.value,
                stocks=[s.model_dump() for s in api_stocks],
            )
            validation = self.llm_client.extract_structured(prompt, _RangeValidationModel)
        except LLMError as e:
            logger.warning("LLM validation failed; using API list as-is: %s", e)
            validation = None

        if validation is not None:
            final = self._apply_validation(api_stocks, validation)
            source = "combined" if api_ok else "llm"
            confidence = min(0.9 if api_ok else 0.6, validation.confidence or 0.6)
            return RangeStockResult(
                query_type=query_type,
                query_value=query_value,
                stocks=final,
                source=source,
                confidence=confidence,
                matched_industry=validation.matched_industry,
            )

        # No validation available.
        if api_ok:
            return RangeStockResult(
                query_type=query_type,
                query_value=query_value,
                stocks=api_stocks,
                source="twse_api",
                confidence=0.8,
            )

        raise StockEnumerationError(
            f"Range enumeration failed: no authoritative source and no LLM output for '{query_value}'"
        )

    @staticmethod
    def _apply_validation(
        base: list[StockItem],
        validation: _RangeValidationModel,
    ) -> list[StockItem]:
        """Merge validation to_add/to_remove into the base list, de-duplicated."""
        remove_codes = {c for c in validation.to_remove}
        result: list[StockItem] = [s for s in base if s.code not in remove_codes]
        seen = {s.code for s in result}
        for item in validation.to_add:
            if item.code not in seen and item.code:
                result.append(item)
                seen.add(item.code)
        return result

    # ------------------------------------------------------------------
    # Related mode (supply chain)
    # ------------------------------------------------------------------
    def related_stock(self, stock_code: str) -> RelatedStockResult:
        """Infer direct upstream/downstream of a single stock via LLM + validation."""
        info = self.resolver.verify_code(stock_code)
        stock_name = info["name"] or stock_code

        try:
            prompt = prompt_manager.render_supply_chain(
                stock_code=stock_code, stock_name=stock_name
            )
            result = self.llm_client.extract_structured(prompt, _SupplyChainModel)
        except LLMError as e:
            raise StockEnumerationError(f"Supply-chain inference failed: {e}") from e

        upstream = self._validate_stocks(result.upstream)
        downstream = self._validate_stocks(result.downstream)

        return RelatedStockResult(
            stock_code=stock_code,
            stock_name=stock_name,
            upstream=upstream,
            downstream=downstream,
            source="llm",
            confidence=result.confidence,
        )

    def _validate_stocks(self, stocks: list[StockItem]) -> list[StockItem]:
        """Drop stocks whose code doesn't resolve to the given name."""
        valid: list[StockItem] = []
        for s in stocks:
            info = self.resolver.verify_code(s.code)
            if info["exists"] and (info["name"] or s.name):
                valid.append(s)
        return valid


# Convenience function matching the other step modules.
def enumerate_stocks(
    question: str,
    boundary: BoundaryResult,
    llm_client: LLMClient | None = None,
    isin: IsinClient | None = None,
) -> RangeStockResult | RelatedStockResult:
    """Run Step 4 stock enumeration. See StockEnumeration.run()."""
    return StockEnumeration(llm_client=llm_client, isin=isin).run(question, boundary)
