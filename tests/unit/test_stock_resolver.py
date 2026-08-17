"""Unit tests for deterministic stock code resolution (P1, gap #1,#3,#4,#5,#6,#10)."""

import json

from classifier.stock_resolver import StockResolver, stock_resolver


class TestStockResolver:
    def test_resolve_name_known(self):
        assert stock_resolver.resolve_name("台積電") == "2330"
        assert stock_resolver.resolve_name("鴻海") == "2317"
        assert stock_resolver.resolve_name("聯發科") == "2454"

    def test_resolve_name_with_corporate_suffix(self):
        # Normalization strips corporate suffixes (gap #3).
        assert stock_resolver.resolve_name("台積電股份有限公司") == "2330"

    def test_resolve_name_unknown_returns_none(self):
        assert stock_resolver.resolve_name("不存在的公司xyz") is None

    def test_verify_code_exists(self):
        info = stock_resolver.verify_code("2330")
        assert info["exists"] is True
        assert info["name"] == "台積電"

    def test_verify_code_reverse_match(self):
        # Reverse lookup: code -> name consistency (gap #10).
        info = stock_resolver.verify_code("2330", "台積電")
        assert info["name_matches"] is True

    def test_verify_code_not_found(self):
        info = stock_resolver.verify_code("9999")
        assert info["exists"] is False

    def test_resolve_drops_unknown_llm_code(self):
        # LLM-guessed code not in map is dropped, name still resolves (gap #1,#6).
        codes, warnings = stock_resolver.resolve(["台積電"], ["9999"])
        assert codes == ["2330"]
        assert any("9999" in w for w in warnings)

    def test_resolve_fills_from_names(self):
        codes, warnings = stock_resolver.resolve(["台積電", "鴻海"], [])
        assert codes == ["2330", "2317"]
        assert warnings == []

    def test_resolve_multi_stock_comparison(self):
        codes, warnings = stock_resolver.resolve(["台積電", "鴻海"], [])
        assert codes == ["2330", "2317"]

    def test_resolve_drops_conflicting_llm_code(self):
        # LLM gives a valid-but-wrong code (e.g. guesses 亞泥/1102 for a 台積電
        # query). Resolver's name wins; only the correct code is emitted. This is
        # the exact reproduction of the "boundary shows two stocks + chart uses
        # the wrong one" bug.
        codes, warnings = stock_resolver.resolve(["台積電"], ["1102"])
        assert codes == ["2330"]
        assert any("1102" in w for w in warnings)

    def test_resolve_keeps_matching_llm_code_deduped(self):
        # LLM gives the correct code alongside the name -> deduplicated to one.
        codes, _ = stock_resolver.resolve(["台積電"], ["2330"])
        assert codes == ["2330"]

    def test_resolve_pure_code_query_kept(self):
        # No company names -> a verified LLM code is kept as a fallback.
        codes, warnings = stock_resolver.resolve([], ["2330"])
        assert codes == ["2330"]
        assert warnings == []

    def test_tie_break_prefers_twse(self, tmp_path):
        # Same name on both boards -> TWSE preferred (gap #4).
        data = [
            {"stock_id": "1234", "stock_name": "測試公司", "type": "TPEx", "date": "2026-01-01"},
            {"stock_id": "5678", "stock_name": "測試公司", "type": "TWSE", "date": "2026-01-01"},
        ]
        p = tmp_path / "stocks.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        r = StockResolver(str(p))
        assert r.resolve_name("測試公司") == "5678"

    def test_load_encoding_utf8(self, tmp_path):
        # Map must load with explicit utf-8 on Windows (gap #5).
        data = [{"stock_id": "2330", "stock_name": "台積電", "type": "twse", "date": "2026-01-01"}]
        p = tmp_path / "stocks.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        r = StockResolver(str(p))
        assert r.resolve_name("台積電") == "2330"
