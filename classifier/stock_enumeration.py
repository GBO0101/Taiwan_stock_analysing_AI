"""Step 4: Stock Enumeration.

Executes after Step 3 and produces either:
- RangeStockResult: all stocks matching a sector/index/market, or
- RelatedStockResult: upstream/downstream of a single stock.

Related mode is annual-report-first with a three-tier priority:

1. **Annual report** (股東會年報 from doc.twse.com.tw): the target's report
   discloses top customers/suppliers with real revenue/purchase ratios used
   directly (``disclosed_by="target"``); counterparties' own annual reports
   are cross-checked (Phase 1.5) so two-sided links become ``"both"``.
2. **LLM inference + two-sided verification**: when the target's annual
   report is unavailable or yields nothing resolvable, the LLM nominates
   upstream/downstream companies (supply_chain.j2). Each nominee's own
   annual report is then checked: if it discloses the target in the matching
   direction and the disclosed ratio is consistent with the LLM's impact
   band, the link is promoted to ``disclosed_by="counterparty"`` with the
   real ratio (trusted, not ``llm_inferred``).
3. **Unverified LLM inference**: nominees whose own report cannot confirm
   the link stay ``disclosed_by="llm_inferred"`` (lowest priority).

Trigger (checked in the pipeline): boundary has stock codes OR a range/topic,
AND classification type is not NON_FINANCIAL.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field

from classifier.annual_report import AnnualReportError, extract_annual_report
from classifier.isin_client import IsinClient, IsinClientError
from classifier.isin_client import isin_client as default_isin
from classifier.llm_client import LLMClient, LLMError
from classifier.models import (
    AnnualReportDisclosure,
    BoundaryResult,
    ImpactStock,
    QuerySummary,
    QueryTopicKind,
    RangeQueryType,
    RangeStockResult,
    RelatedStockResult,
    StockItem,
    StockRelation,
    SupplyChainEvidence,
    SupplyChainRelation,
)
from classifier.prompts import prompt_manager
from classifier.stock_resolver import StockResolver
from classifier.stock_resolver import stock_resolver as default_resolver

logger = logging.getLogger(__name__)

# Cap on how many counterparty annual reports we fetch for cross-validation.
_MAX_CROSS_VALIDATE = 3


def _merge_evidence(evidence: list[SupplyChainEvidence]) -> list[SupplyChainEvidence]:
    """Collapse duplicate evidence rows into one per (code, relation, fiscal_year).

    The annual report and its LLM extraction can surface the same counterparty
    multiple times (a >10% disclosure row, a raw-material row, related-party
    rows) with slightly different names/ratios — e.g. 台塑石化 appears 5× for
    1326's upstream with ratios 54.7/57.3. Each group keeps ONE row: the one
    with the highest ``target_ratio`` (a numeric ratio beats ``None``; all-None
    groups keep their first row). First-seen order is preserved.
    """
    merged: dict[tuple[str, SupplyChainRelation, int | None], SupplyChainEvidence] = {}
    order: list[tuple[str, SupplyChainRelation, int | None]] = []
    for ev in evidence:
        key = (ev.counterparty_code, ev.relation, ev.fiscal_year)
        if key not in merged:
            merged[key] = ev
            order.append(key)
            continue
        current = merged[key]
        # Max-row wins: a numeric ratio beats None; among numerics, larger wins.
        if ev.target_ratio is not None and (
            current.target_ratio is None or ev.target_ratio > current.target_ratio
        ):
            merged[key] = ev
    return [merged[k] for k in order]


class StockEnumerationError(Exception):
    """Step 4 errors."""


class _RangeValidationModel(BaseModel):
    """LLM output for range-mode stock-list validation."""

    matched_industry: str | None = None
    to_add: list[StockItem] = Field(default_factory=list)
    to_remove: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    relevance: dict[str, float] = Field(
        default_factory=dict,
        description="Per-stock relevance to the query topic (0-1)",
    )


class _SupplyChainEntry(BaseModel):
    """Single counterparty in the LLM supply-chain inference."""

    code: str = Field(..., description="4-digit stock code")
    name: str = Field(..., description="Chinese company name")
    impact: float = Field(default=0.5, ge=0.0, le=1.0, description="Impact on target")


class _SupplyChainModel(BaseModel):
    """LLM output for supply-chain (upstream/downstream) inference."""

    upstream: list[_SupplyChainEntry] = Field(default_factory=list)
    downstream: list[_SupplyChainEntry] = Field(default_factory=list)
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)


# Taiwan is UTC+8 year-round (no DST); evaluate "today" in that zone.
_TW_OFFSET = timezone(timedelta(hours=8))


def _today() -> date:
    """Current date in the Taiwan timezone."""
    return datetime.now(_TW_OFFSET).date()


def _minguo_year(d: date) -> int:
    """Convert a Gregorian date to the ROC (Minguo) year."""
    return d.year - 1911


def _resolve_fiscal_years(boundary: BoundaryResult | None) -> list[int]:
    """Map the boundary time range to Minguo fiscal years (ascending).

    - No range / unparsable range → [current Minguo year]
    - Sub-1-year range → the fiscal year of the range end (newest value)
    - Multi-year range → every Minguo year spanned
    """
    today_fy = _minguo_year(_today())
    if boundary is None or boundary.date_range is None:
        return [today_fy]
    resolved = boundary.date_range.resolve_dates()
    if resolved is None:
        return [today_fy]
    start, end = resolved
    if (end - start).days < 365:
        return [_minguo_year(end)]
    start_fy, end_fy = _minguo_year(start), _minguo_year(end)
    return list(range(start_fy, end_fy + 1))


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
        # Private debrief for build_summary(): filled by range_stocks/related_stock.
        self._last_meta: dict[str, Any] = {}

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
            return self.related_stock(boundary.stock_codes[0], boundary)

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
        obvious missing leaders and drop wrong inclusions. The LLM also rates
        per-stock ``relevance`` used later by the unified summary.
        """
        api_stocks: list[StockItem] = []
        api_ok = False
        try:
            api_stocks = self.isin.by_sector(query_value, market=market)
            api_ok = True
        except IsinClientError as e:
            logger.warning("ISIN lookup failed; falling back to LLM: %s", e)

        # LLM validation/supplement.
        validation: _RangeValidationModel | None = None
        try:
            prompt = prompt_manager.render_stock_validation(
                query_value=query_value,
                query_type=query_type.value,
                stocks=[s.model_dump() for s in api_stocks],
            )
            validation = self.llm_client.extract_structured(prompt, _RangeValidationModel)
        except LLMError as e:
            logger.warning("LLM validation failed; using API list as-is: %s", e)

        relevance = validation.relevance if validation is not None else {}

        if validation is not None:
            final = self._apply_validation(api_stocks, validation)
            source = "combined" if api_ok else "llm"
            confidence = min(0.9 if api_ok else 0.6, validation.confidence or 0.6)
            self._last_meta = {
                "kind": "range",
                "source": source,
                "relevance": relevance,
            }
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
            self._last_meta = {"kind": "range", "source": "twse_api", "relevance": {}}
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
    def related_stock(
        self,
        stock_code: str,
        boundary: BoundaryResult | None = None,
    ) -> RelatedStockResult:
        """Resolve upstream/downstream of a single stock, annual-report-first.

        Three-tier priority:
        1. Annual report → resolved counterparties (``disclosed_by="target"``,
           two-sided links become ``"both"`` via Phase 1.5 cross-validation).
        2. LLM inference (supply_chain.j2) + two-sided verification: each
           LLM nominee's own annual report is fetched and, if it discloses
           the target in the matching direction with a ratio consistent with
           the LLM's impact band, the link is promoted to
           ``disclosed_by="counterparty"`` with the real ratio.
        3. Unverified LLM nominees stay ``disclosed_by="llm_inferred"``.
        """
        info = self.resolver.verify_code(stock_code)
        stock_name = info["name"] or stock_code

        fiscal_years = _resolve_fiscal_years(boundary)
        notes: list[str] = []
        source = "llm"
        confidence = 0.3
        used_fy: int | None = None

        upstream: list[StockItem] = []
        downstream: list[StockItem] = []
        impact_map: dict[tuple[str, SupplyChainRelation], float] = {}
        evidence: list[SupplyChainEvidence] = []

        # 1) Annual report first — one extraction per requested fiscal year
        #    (ascending, so the newest usable year lands last and wins as primary).
        disclosures: list[AnnualReportDisclosure] = []
        for fy in fiscal_years:
            try:
                disclosure = extract_annual_report(
                    stock_code,
                    stock_name,
                    fy,
                    llm_client=self.llm_client,
                    resolver=self.resolver,
                )
            except AnnualReportError as e:
                notes.append(f"民國 {fy} 年報無法取得，略過：{e}")
                continue
            # A disclosure is only usable if at least one counterparty resolved
            # to a listed stock code; all-anonymous/unresolvable disclosures
            # would otherwise claim source="annual_report" while contributing
            # zero upstream/downstream items.
            resolvable = any(c.resolved_code for c in disclosure.customers) or any(
                s.resolved_code for s in disclosure.suppliers
            )
            if resolvable:
                disclosures.append(disclosure)
            else:
                notes.append(f"民國 {fy} 年報未揭露可解析之具名客戶/供應商")

        if disclosures:
            # Newest successful year is the primary value; older years enrich
            # the history. impact_map keeps the newest known ratio per partner.
            primary = disclosures[-1]
            used_fy = primary.fiscal_year
            source = "annual_report"
            confidence = max(0.6, primary.confidence)
            if used_fy != fiscal_years[-1]:
                notes.append(f"年報未公告，以民國 {used_fy} 年報為最新資料")
            if primary.report_type == "llm":
                notes.append("年報無實體 PDF（LLM 抽取）")

            for d in disclosures:
                d_fy = d.fiscal_year
                # suppliers → upstream
                for s in d.suppliers:
                    code = s.resolved_code
                    if code is None:
                        continue
                    item = StockItem(code=code, name=s.resolved_name or s.raw_name)
                    if code not in {i.code for i in upstream}:
                        upstream.append(item)
                    evidence.append(
                        SupplyChainEvidence(
                            counterparty_code=code,
                            counterparty_name=s.resolved_name or s.raw_name,
                            relation=SupplyChainRelation.UPSTREAM,
                            target_ratio=s.ratio,
                            disclosed_by="target",
                            source_pdf=d.pdf_name,
                            fiscal_year=d_fy,
                        )
                    )
                # customers → downstream
                for c in d.customers:
                    code = c.resolved_code
                    if code is None:
                        continue
                    item = StockItem(code=code, name=c.resolved_name or c.raw_name)
                    if code not in {i.code for i in downstream}:
                        downstream.append(item)
                    evidence.append(
                        SupplyChainEvidence(
                            counterparty_code=code,
                            counterparty_name=c.resolved_name or c.raw_name,
                            relation=SupplyChainRelation.DOWNSTREAM,
                            target_ratio=c.ratio,
                            disclosed_by="target",
                            source_pdf=d.pdf_name,
                            fiscal_year=d_fy,
                        )
                    )

            # Python Phase 1: 同一 (code, relation, fiscal_year) 只保留一筆
            # evidence（年報/LLM 常在 >10% 表、原料表、關係人表重複列出同一
            # 公司，比例還不一致）。合併後才依「每 code 最新已知比例」建
            # impact_map，避免 summary 的 ratio/impact 出現 54.7 與 57.3 並存。
            evidence = _merge_evidence(evidence)
            for ev in evidence:
                if ev.target_ratio is not None:
                    impact_map[(ev.counterparty_code, ev.relation)] = ev.target_ratio / 100.0

            # Phase 1.5: cross-validate only the primary year's links.
            # Counterparty annual reports also cover a single year; checking
            # every year × counterparty would explode doc.twse + LLM calls.
            primary_evidence = [ev for ev in evidence if ev.fiscal_year == used_fy]
            self._cross_validate(primary_evidence, used_fy, stock_code)
        else:
            # 2) LLM fallback.
            try:
                industry_hint = boundary.sectors[0] if boundary and boundary.sectors else None
                prompt = prompt_manager.render_supply_chain(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    industry_hint=industry_hint,
                )
                result = self.llm_client.extract_structured(prompt, _SupplyChainModel)
            except LLMError as e:
                raise StockEnumerationError(f"Supply-chain inference failed: {e}") from e

            confidence = result.confidence
            logger.info(
                "LLM supply-chain nomination: %d upstream, %d downstream, confidence=%.2f",
                len(result.upstream),
                len(result.downstream),
                result.confidence,
            )
            notes.append("資料來源：LLM 推論（年報無可用揭露）")
            for entry in result.upstream:
                upstream.append(StockItem(code=entry.code, name=entry.name))
                impact_map[(entry.code, SupplyChainRelation.UPSTREAM)] = entry.impact
                evidence.append(
                    SupplyChainEvidence(
                        counterparty_code=entry.code,
                        counterparty_name=entry.name,
                        relation=SupplyChainRelation.UPSTREAM,
                        target_ratio=None,
                        disclosed_by="llm_inferred",
                        source_pdf=None,
                        fiscal_year=None,
                    )
                )
            for entry in result.downstream:
                downstream.append(StockItem(code=entry.code, name=entry.name))
                impact_map[(entry.code, SupplyChainRelation.DOWNSTREAM)] = entry.impact
                evidence.append(
                    SupplyChainEvidence(
                        counterparty_code=entry.code,
                        counterparty_name=entry.name,
                        relation=SupplyChainRelation.DOWNSTREAM,
                        target_ratio=None,
                        disclosed_by="llm_inferred",
                        source_pdf=None,
                        fiscal_year=None,
                    )
                )

        upstream = self._validate_stocks(upstream)
        downstream = self._validate_stocks(downstream)

        # Phase 2.0 (tier 2 of the three-tier priority): when NLP fallback was
        # used, verify each LLM-nominated counterparty against its own annual
        # report. A nominee whose report discloses the target in the matching
        # direction with a ratio consistent with the LLM's impact band is
        # promoted from llm_inferred to counterparty (trusted, real ratio).
        if source == "llm" and fiscal_years:
            verified = self._verify_llm_links(
                evidence,
                impact_map,
                {s.code for s in upstream} | {s.code for s in downstream},
                stock_code,
                fiscal_years[-1],
            )
            if verified:
                notes.append(
                    f"其中 {verified} 筆連結經對方(被提名公司)年報雙向揭露證實，比例與推測一致"
                )
                confidence = min(1.0, confidence + 0.1)

        self._last_meta = {
            "kind": "related",
            "source": source,
            "impact_map": impact_map,
            "fiscal_years": [d.fiscal_year for d in reversed(disclosures)] or [],
            "notes": notes,
        }

        return RelatedStockResult(
            stock_code=stock_code,
            stock_name=stock_name,
            upstream=upstream,
            downstream=downstream,
            source=source,
            confidence=confidence,
            evidence=evidence,
        )

    def _cross_validate(
        self,
        evidence: list[SupplyChainEvidence],
        fiscal_year: int,
        target_code: str,
    ) -> None:
        """Phase 1.5: check counterparties' annual reports for a two-sided link.

        For up to ``_MAX_CROSS_VALIDATE`` counterparties (ordered by reported
        ratio desc), fetch their own annual report for the same fiscal year and,
        if they list the target as a customer/supplier, mark the evidence
        ``disclosed_by="both"``. Best-effort: failures are logged and skipped.
        """
        unique: dict[str, SupplyChainEvidence] = {}
        for ev in evidence:
            unique.setdefault(ev.counterparty_code, ev)
        candidates = [ev for ev in unique.values() if ev.counterparty_code != target_code]
        candidates.sort(key=lambda e: e.target_ratio or 0.0, reverse=True)

        checked = 0
        for ev in candidates[:_MAX_CROSS_VALIDATE]:
            try:
                cdisc = extract_annual_report(
                    ev.counterparty_code,
                    ev.counterparty_name,
                    fiscal_year,
                    llm_client=self.llm_client,
                    resolver=self.resolver,
                )
            except AnnualReportError:
                continue
            if cdisc is None:
                continue
            lists_both = cdisc.customers + cdisc.suppliers
            if any(x.resolved_code == target_code for x in lists_both):
                ev.disclosed_by = "both"
                logger.info(
                    "Cross-validated two-sided link: %s <-> %s",
                    target_code,
                    ev.counterparty_code,
                )
            checked += 1

        if checked:
            logger.info("Cross-validated %d counterparty annual reports", checked)

    @staticmethod
    def _impact_matches_ratio(impact: float, ratio: float) -> bool:
        """Check an annual-report ratio (%) matches the LLM impact band.

        Bands from prompts/supply_chain.j2: 0.9+ → >20%, 0.6-0.8 → 5-20%,
        0.3-0.5 → 1-5%, <0.3 → <1%. Each boundary is relaxed by a 2-percentage-
        point tolerance so a borderline ratio still counts as a match.
        """
        tolerance = 2.0
        if impact >= 0.9:
            return ratio > 20.0 - tolerance
        if impact >= 0.6:
            return 5.0 - tolerance <= ratio <= 20.0 + tolerance
        if impact >= 0.3:
            return max(1.0 - tolerance, 0.0) <= ratio <= 5.0 + tolerance
        return ratio < 1.0 + tolerance

    def _verify_llm_links(
        self,
        evidence: list[SupplyChainEvidence],
        impact_map: dict[tuple[str, SupplyChainRelation], float],
        valid_codes: set[str],
        target_code: str,
        fiscal_year: int,
    ) -> int:
        """Phase 2.0: verify LLM-inferred links via nominees' annual reports.

        Fetches each LLM-nominated counterparty's own annual report (up to
        ``_MAX_CROSS_VALIDATE``, ordered by the LLM's impact desc) and, when it
        discloses the target in a direction consistent with the inferred link —
        an UPSTREAM nominee must list the target as its CUSTOMER, a DOWNSTREAM
        nominee as its SUPPLIER — with a ratio matching the LLM's impact band,
        promotes the evidence from ``llm_inferred`` to ``counterparty`` with
        the real ratio/source PDF/fiscal year. Returns the number of links
        promoted.
        """
        unique: dict[tuple[str, SupplyChainRelation], SupplyChainEvidence] = {}
        for ev in evidence:
            if ev.disclosed_by != "llm_inferred":
                continue
            if ev.counterparty_code not in valid_codes:
                continue
            unique.setdefault((ev.counterparty_code, ev.relation), ev)

        candidates = sorted(
            unique.values(),
            key=lambda e: impact_map.get((e.counterparty_code, e.relation), 0.0),
            reverse=True,
        )

        if candidates:
            logger.info(
                "LLM fallback verification: %d unique nominee(s), "
                "cross-checking up to %d through annual reports",
                len(candidates),
                min(len(candidates), _MAX_CROSS_VALIDATE),
            )

        promoted = 0
        for ev in candidates[:_MAX_CROSS_VALIDATE]:
            t0 = time.monotonic()
            logger.info(
                "Fetching nominee %s(%s) annual report (fiscal %d) to verify %s link",
                ev.counterparty_code,
                ev.counterparty_name,
                fiscal_year,
                ev.relation.value,
            )
            try:
                cdisc = extract_annual_report(
                    ev.counterparty_code,
                    ev.counterparty_name,
                    fiscal_year,
                    llm_client=self.llm_client,
                    resolver=self.resolver,
                )
            except AnnualReportError as e:
                logger.info(
                    "Nominee %s annual report unavailable after %.1fs: %s",
                    ev.counterparty_code,
                    time.monotonic() - t0,
                    e,
                )
                continue
            if cdisc is None:
                logger.info(
                    "Nominee %s annual report empty after %.1fs",
                    ev.counterparty_code,
                    time.monotonic() - t0,
                )
                continue
            # Direction-aware: the nominee's own report must list the target in
            # the mirror direction of the inferred link.
            if ev.relation == SupplyChainRelation.UPSTREAM:
                matches = [c for c in cdisc.customers if c.resolved_code == target_code]
            else:
                matches = [s for s in cdisc.suppliers if s.resolved_code == target_code]
            if not matches:
                logger.info(
                    "Nominee %s does not list target %s in the mirrored direction after %.1fs",
                    ev.counterparty_code,
                    target_code,
                    time.monotonic() - t0,
                )
                continue
            real_ratio = matches[0].ratio
            if real_ratio is None or not self._impact_matches_ratio(
                impact_map.get((ev.counterparty_code, ev.relation), 0.0),
                real_ratio,
            ):
                logger.info(
                    "Nominee %s discloses target at ratio=%s which does not match "
                    "the inferred impact band after %.1fs",
                    ev.counterparty_code,
                    real_ratio,
                    time.monotonic() - t0,
                )
                continue

            ev.disclosed_by = "counterparty"
            ev.target_ratio = real_ratio
            ev.source_pdf = cdisc.pdf_name
            ev.fiscal_year = cdisc.fiscal_year
            impact_map[(ev.counterparty_code, ev.relation)] = real_ratio / 100.0
            logger.info(
                "Two-sided LLM link verified: %s (%s, ratio=%.1f%%)",
                ev.counterparty_code,
                ev.relation.value,
                real_ratio,
            )
            promoted += 1

        if promoted:
            logger.info(
                "Promoted %d LLM-inferred link(s) from annual-report verification",
                promoted,
            )
        return promoted

    def _validate_stocks(self, stocks: list[StockItem]) -> list[StockItem]:
        """Drop stocks whose code can't be verified against the claimed name.

        Verification is code+name aware (``name_matches is not False``): a
        nominee whose claimed name does not match the code's real company
        (e.g. the LLM pairs 2454 with "大立光") is garbage and gets dropped
        here, before Phase 2.0, so no annual report is ever fetched for it —
        the 止血 against bad LLM fallback nominations.
        """
        valid: list[StockItem] = []
        for s in stocks:
            info = self.resolver.verify_code(s.code, s.name)
            if info["exists"] and info["name_matches"] is not False and (info["name"] or s.name):
                valid.append(s)
                continue
            logger.info(
                "Dropping nominee %s(%s): exists=%s name_matches=%s",
                s.code,
                s.name,
                info["exists"],
                info["name_matches"],
            )
        return valid

    # ------------------------------------------------------------------
    # Unified summary
    # ------------------------------------------------------------------
    def build_summary(
        self,
        boundary: BoundaryResult,
        enum_result: RangeStockResult | RelatedStockResult,
    ) -> QuerySummary:
        """Build the unified QuerySummary from Step 4 output.

        Related stocks are sorted by ``impact_score`` descending. The target
        itself (single-stock mode) always ranks first with impact 1.0.
        """
        meta = self._last_meta or {}
        if isinstance(enum_result, RelatedStockResult):
            return self._build_related_summary(enum_result, meta)
        return self._build_range_summary(boundary, enum_result, meta)

    @staticmethod
    def _build_related_summary(
        result: RelatedStockResult,
        meta: dict[str, Any],
    ) -> QuerySummary:
        impact_map: dict[tuple[str, SupplyChainRelation], float] = meta.get("impact_map") or {}
        fy_list: list[int] = meta.get("fiscal_years") or []
        notes: list[str] = list(meta.get("notes") or [])
        source = str(meta.get("source") or result.source)
        primary_fy = fy_list[0] if fy_list else None

        # Relation-aware: the same code may sit on BOTH sides (6505 appears as
        # upstream supplier AND downstream customer). Keying by code alone made
        # upstream/downstream share one ratio (upstream 57.3 vs downstream 10.2
        # collapse into whichever row came last).
        ratio_by_code = {
            (e.counterparty_code, e.relation): e.target_ratio
            for e in result.evidence
            if e.target_ratio is not None
        }

        stocks: list[ImpactStock] = [
            ImpactStock(
                code=result.stock_code,
                name=result.stock_name,
                relation=StockRelation.TARGET,
                impact_score=1.0,
                detail_source=source,
            )
        ]

        for s in result.upstream:
            stocks.append(
                ImpactStock(
                    code=s.code,
                    name=s.name,
                    relation=StockRelation.UPSTREAM,
                    impact_score=min(
                        1.0, impact_map.get((s.code, SupplyChainRelation.UPSTREAM), 0.5)
                    ),
                    ratio=ratio_by_code.get((s.code, SupplyChainRelation.UPSTREAM)),
                    fiscal_year=primary_fy,
                    detail_source=source,
                )
            )
        for s in result.downstream:
            stocks.append(
                ImpactStock(
                    code=s.code,
                    name=s.name,
                    relation=StockRelation.DOWNSTREAM,
                    impact_score=min(
                        1.0, impact_map.get((s.code, SupplyChainRelation.DOWNSTREAM), 0.5)
                    ),
                    ratio=ratio_by_code.get((s.code, SupplyChainRelation.DOWNSTREAM)),
                    fiscal_year=primary_fy,
                    detail_source=source,
                )
            )

        stocks.sort(key=lambda x: x.impact_score, reverse=True)

        return QuerySummary(
            topic_kind=QueryTopicKind.SINGLE_STOCK,
            topic_code=result.stock_code,
            topic_name=result.stock_name,
            fiscal_years=fy_list,
            notes=notes,
            related_stocks=stocks,
        )

    @staticmethod
    def _build_range_summary(
        boundary: BoundaryResult,
        result: RangeStockResult,
        meta: dict[str, Any],
    ) -> QuerySummary:
        relevance: dict[str, float] = meta.get("relevance") or {}
        source = str(meta.get("source") or result.source)
        notes: list[str] = list(meta.get("notes") or [])

        stocks = [
            ImpactStock(
                code=s.code,
                name=s.name,
                relation=StockRelation.RANGE_MEMBER,
                impact_score=relevance.get(s.code, 0.5),
                detail_source=source,
            )
            for s in result.stocks
        ]
        stocks.sort(key=lambda x: x.impact_score, reverse=True)

        return QuerySummary(
            topic_kind=QueryTopicKind.RANGE,
            range_type=result.query_type,
            range_value=result.query_value,
            topic_name=result.query_value,
            fiscal_years=_resolve_fiscal_years(boundary),
            notes=notes,
            related_stocks=stocks,
        )


# Convenience function matching the other step modules.
def enumerate_stocks(
    question: str,
    boundary: BoundaryResult,
    llm_client: LLMClient | None = None,
    isin: IsinClient | None = None,
) -> RangeStockResult | RelatedStockResult:
    """Run Step 4 stock enumeration. See StockEnumeration.run()."""
    return StockEnumeration(llm_client=llm_client, isin=isin).run(question, boundary)
