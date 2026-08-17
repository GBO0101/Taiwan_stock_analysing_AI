"""Deterministic stock code resolution from taiwan_stocks.json.

This module backs Step 1 (Boundary Extraction). The LLM is instructed NOT to
guess stock codes, so name-only questions leave ``stock_codes`` empty and the
pipeline downstream breaks. ``StockResolver`` deterministically maps
``company_names`` -> codes (and reverse-verifies any LLM-provided codes) using
the authoritative ``taiwan_stocks.json`` shipped at the project root.

A global ``stock_resolver`` instance is created on import, mirroring
``indicator_mapper.indicator_mapper``. Importing this module raises
``StockResolutionError`` if the mapping file is missing or malformed.
"""

import json
import logging
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)


class StockResolutionError(Exception):
    """Raised when the stock mapping file cannot be loaded."""


class StockResolver:
    """Resolve company names to stock codes and verify codes against names."""

    def __init__(self, json_path: str | None = None):
        """Initialize the resolver.

        Args:
            json_path: Path to taiwan_stocks.json (defaults to project root).
        """
        if json_path is None:
            json_path = str(Path(__file__).parent.parent / "taiwan_stocks.json")

        self.json_path = Path(json_path)
        self._by_code: dict[str, dict] = {}
        self._by_name_norm: dict[str, list[dict]] = {}
        self._load()

    def _load(self) -> None:
        """Load and index the stock mapping from JSON."""
        if not self.json_path.exists():
            raise StockResolutionError(f"Stock mapping file not found: {self.json_path}")

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise StockResolutionError(f"Failed to load stock mapping: {e}") from e

        if not isinstance(data, list) or not data:
            raise StockResolutionError("Stock mapping is empty or not a list")

        for entry in data:
            code = str(entry.get("stock_id", "")).strip()
            name = str(entry.get("stock_name", "")).strip()
            if not code or not name:
                continue
            self._by_code[code] = entry
            norm = self._normalize(name)
            self._by_name_norm.setdefault(norm, []).append(entry)

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize a company name for matching.

        Applies NFKC (full-width -> half-width), strips whitespace, and drops
        trailing corporate suffixes so "台積電股份有限公司" matches "台積電".
        """
        text = unicodedata.normalize("NFKC", text)
        text = "".join(text.split())
        for suffix in ("股份有限公司", "股份", "公司", "有限公司"):
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
        return text

    def resolve_name(self, name: str, market_priority: str = "TWSE") -> str | None:
        """Resolve a company name to a stock code.

        On multiple matches (e.g. same name on TWSE and TPEx), prefers the
        ``market_priority`` board, then the first entry. Returns ``None`` if no
        match is found.
        """
        norm = self._normalize(name)
        candidates = self._by_name_norm.get(norm)
        if not candidates:
            return None
        if len(candidates) == 1:
            return str(candidates[0]["stock_id"])
        for candidate in candidates:
            if str(candidate.get("type", "")).upper() == market_priority.upper():
                return str(candidate["stock_id"])
        return str(candidates[0]["stock_id"])

    def verify_code(self, code: str, name: str | None = None) -> dict:
        """Reverse lookup: confirm a code exists and optionally matches a name.

        Returns a dict with keys: ``exists`` (bool), ``name`` (str|None),
        ``name_matches`` (bool|None), ``market`` (str|None).
        """
        entry = self._by_code.get(str(code).strip())
        if entry is None:
            return {"exists": False, "name": None, "name_matches": None, "market": None}
        resolved_name = str(entry.get("stock_name", ""))
        name_matches: bool | None = None
        if name:
            name_matches = self._normalize(name) == self._normalize(resolved_name)
        return {
            "exists": True,
            "name": resolved_name,
            "name_matches": name_matches,
            "market": str(entry.get("type", "")),
        }

    def resolve(
        self,
        company_names: list[str],
        llm_codes: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Resolve company names and reconcile with LLM-provided codes.

        Returns ``(final_codes, warnings)``. ``final_codes`` is order-preserving
        and de-duplicated. Any LLM-provided code that is not found in the map is
        dropped (and warned); any name->code mismatch is warned.

        Args:
            company_names: Names extracted by the LLM.
            llm_codes: Codes the LLM may have provided (reverse-verified).
        """
        warnings: list[str] = []
        final: list[str] = []
        seen: set[str] = set()

        for code in llm_codes or []:
            code = str(code).strip()
            if not code:
                continue
            info = self.verify_code(code)
            if not info["exists"]:
                warnings.append(f"LLM-provided code '{code}' not found in stock map")
                continue
            if code not in seen:
                final.append(code)
                seen.add(code)

        for name in company_names:
            name = (name or "").strip()
            if not name:
                continue
            resolved_code = self.resolve_name(name)
            if resolved_code is None:
                warnings.append(f"Company name '{name}' could not be resolved to a code")
                continue
            if resolved_code not in seen:
                final.append(resolved_code)
                seen.add(resolved_code)

        # Cross-check: if the LLM gave a code whose name contradicts a query name.
        for name in company_names:
            name = (name or "").strip()
            if not name:
                continue
            for code in llm_codes or []:
                code = str(code).strip()
                info = self.verify_code(code)
                if not info["exists"] or not info["name"]:
                    continue
                if self._normalize(name) == self._normalize(info["name"]):
                    resolved = self.resolve_name(name)
                    if resolved and code != resolved:
                        warnings.append(
                            f"Code mismatch for '{name}': LLM gave '{code}', "
                            f"map says '{resolved}'"
                        )

        return final, warnings


# Global instance (raises StockResolutionError on bad/missing file at import).
stock_resolver = StockResolver()
