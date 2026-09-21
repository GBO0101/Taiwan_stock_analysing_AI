"""Unit tests for classifier.annual_report text handling (no network).

Only the pure text-processing helpers are tested here in isolation:
``_extract_text_from_pdf`` (page-cap regression) and
``_locate_customer_supplier_sections`` (regulatory-header windows). The
network/LLM path of ``extract_annual_report`` stays mocked elsewhere.
"""

from __future__ import annotations

import re
from unittest.mock import Mock

import pytest

from classifier.annual_report import (
    _MAX_PDF_PAGES,
    _MAX_SEGMENT_RETRIES,
    _SECTION_HEADER_RE,
    _AnnualReportExtract,
    _extract_segment,
    _extract_text_from_pdf,
    _flatten_table_lines,
    _LLMCounterpartyEntry,
    _locate_customer_supplier_sections,
    _merge_extracts,
    _resolve_entries,
    _scan_raw_material_names,
    _split_report_segments,
)
from classifier.annual_report import (
    _extract as _run_extract,
)
from classifier.llm_client import LLMError
from classifier.stock_resolver import stock_resolver

# The exact regulatory disclosure title observed in real 中鋼 / 台塑 annual
# reports (年報應行記載事項準則章節標題).
_REGULATORY_TITLE = (
    "(七)最近二年度任一年度曾佔進(銷)貨總額百分之十以上之供應商/客戶資訊：\n"
    "1.本公司並無占個體銷貨總額百分之十以上之客戶。\n"
    "2.占本公司個體進貨總額百分之十以上之供應商資料：\n"
    "年度 名稱 金額(千元) 占全年度進貨淨額比率(%) 與發行人之關係\n"
    "112年 A 公司 15,547,041 12.79 供應商\n"
    "其他 106,038,743 87.21 不適用\n"
    "113年 A 公司 15,830,191 13.71 供應商\n"
    "其他 99,665,862 86.29 不適用\n"
)


class TestLocateCustomerSupplierSections:
    def test_finds_regulatory_title_and_window(self):
        # The real annual report layout: cover pages, then 致股東報告書, then
        # deep in 營運概況 the regulatory supplier/customer table.
        filler = "中鋼集團總部大樓地址：812高雄市小港區中鋼路1號。\n" * 50
        header = "貳、公司治理報告\n" * 10
        text = f"{filler}\n{header}\n{_REGULATORY_TITLE}"

        focused = _locate_customer_supplier_sections(text)

        # The window must contain the actual table rows, not just the header.
        assert "供應商/客戶資訊" in focused
        assert "A 公司" in focused
        assert "12.79" in focused
        # And it must be a focused window, not the whole document.
        assert len(focused) < len(text)

    def test_merges_adjacent_customer_and_supplier_headers(self):
        # 「前十大客戶」+「前十大供應商」 sit back-to-back in the same section;
        # their windows must coalesce into one span so the table survives.
        layout = (
            "前十大客戶之名稱及銷貨金額、比例："
            "客戶甲 32.5% 客戶乙 21.0% 客戶丙 12.3%\n"
            "前十大供應商之名稱及進貨金額、比例："
            "供應商A 18.2% 供應商B 14.7% 唯一供應商C 9.1%\n"
        )
        focused = _locate_customer_supplier_sections(layout)

        assert "客戶甲" in focused
        assert "客戶乙" in focused
        assert "供應商A" in focused
        assert "供應商C" in focused
        # Both headers are adjacent, so their windows coalesce into ONE span
        # => no separators (n spans have n-1 separators).
        assert focused.count("---") == 0

    def test_returns_full_text_when_no_header(self):
        text = "公司簡介與財報附註，沒有客戶供應商章節。" * 30
        assert _locate_customer_supplier_sections(text) == text

    def test_related_party_header_matched(self):
        related_party_block = (
            "與關係人之間的重大交易事項：\n進貨：向中鴻鋼鐵購入鋼捲 12,300,000 千元。\n"
        )
        focused = _locate_customer_supplier_sections(related_party_block)
        assert "中鴻鋼鐵" in focused

    def test_raw_material_supplier_table_matched(self):
        # 南亞 113 年報真實案例：「主要原料之供應狀況」表列出各原料的主要
        # 供應廠商，含未上市/海外公司；此表遠比 >10% 揭露表完整，必須納入
        # LLM 視窗，否則 pipeline 只回報揭露表那少數幾家。
        raw_material_block = (
            "(三)主要原料之供應狀況\n"
            "原料種類 單位 數量 主要供應廠商\n"
            "安定劑 公噸 7,237 內部撥轉、佳友化工\n"
            "可塑劑 公噸 28,826 內部撥轉、南通寶峰\n"
            "環氧氯丙烷 公噸 161,379 无棣鑫岳化工集团有限公司、台塑公司\n"
            "玻纖紗 公噸 86,442 台灣必成、必成玻璃纖維(昆山)有限公司\n"
            "銅線 公噸 74,322 金益鼎企業、鏶鑫企業\n"
        )
        focused = _locate_customer_supplier_sections(raw_material_block)

        assert "主要原料之供應狀況" in focused
        assert "佳友化工" in focused
        assert "南通寶峰" in focused
        assert "金益鼎企業" in focused

    @pytest.mark.parametrize(
        "title",
        [
            "最近二年度任一年度曾佔進(銷)貨總額百分之十以上之供應商/客戶資訊",
            "前十大客戶之名稱及銷貨金額、比例",
            "與關係人進銷貨之金額及百分比",
            "關聯企業及主要交易往來",
            "主要原料之供應狀況",
            "主要原料之使用狀況及供應廠商",
        ],
    )
    def test_regulatory_titles_match_regex(self, title: str):
        assert re.search(_SECTION_HEADER_RE, title), f"regex missed: {title}"


class TestFlattenTableLines:
    """Re-flow cell-per-line PDF table output into readable rows.

    PyMuPDF's ``page.get_text()`` emits each table cell on its own line, which
    local LLMs fail to read as a row (台塑 113 real data: "1"/"台塑石化"/
    "53,578,492"/"33.44"/"註3" on five consecutive lines). The helper joins
    short non-sentence lines onto the previous line.
    """

    def test_joins_cell_per_line_rows(self):
        raw = "1 \n台塑石化 \n53,578,492 \n33.44 \n註3 \n"
        assert _flatten_table_lines(raw) == "1 台塑石化 53,578,492 33.44 註3"

    def test_long_prose_lines_stay_separate(self):
        # A normal prose sentence (> join_len chars) must NOT be glued to the
        # previous line.
        raw = (
            "本公司並無占個體銷貨總額百分之十以上之客戶，特此說明。\n"
            "前十大客戶之名稱及銷貨金額、比例：\n"
        )
        out = _flatten_table_lines(raw)
        assert out.splitlines() == [
            "本公司並無占個體銷貨總額百分之十以上之客戶，特此說明。",
            "前十大客戶之名稱及銷貨金額、比例：",
        ]

    def test_sentence_ending_punctuation_starts_new_line(self):
        # A sentence-ending line is never glued onto the previous line; a
        # following short cell line still joins after it (table semantics).
        raw = "這個句子超過二十八個字元所以它必須單獨佔一行不會被合併處理\n短句。\n"
        out = _flatten_table_lines(raw)
        assert out.splitlines() == [
            "這個句子超過二十八個字元所以它必須單獨佔一行不會被合併處理",
            "短句。",
        ]

    def test_empty_lines_preserved(self):
        raw = "A 公司\n\n其他\n"
        out = _flatten_table_lines(raw)
        assert out.splitlines() == ["A 公司", "", "其他"]

    def test_join_len_parameter(self):
        # join_len is the short-line threshold: below it lines stay separate,
        # above it consecutive cells merge into one row.
        raw = "台灣塑膠工業\n台塑石化\n"
        merged = _flatten_table_lines(raw, join_len=10)
        assert merged.splitlines() == ["台灣塑膠工業 台塑石化"]
        split = _flatten_table_lines(raw, join_len=0)
        assert len(split.splitlines()) == 2


class TestExtractTextFromPdf:
    """Page-cap regression: customer tables live ~p140-160 of 200-400 page
    annual reports, so the old default of 80 pages silently starved the LLM.
    """

    def _make_pdf(self, pages: list[str]) -> bytes:
        fitz = pytest.importorskip("fitz", reason="PyMuPDF optional for tests")
        doc = fitz.open()
        for content in pages:
            page = doc.new_page()
            # Built-in CJK font: the default base-14 font cannot encode
            # Traditional Chinese, which would corrupt the extracted text.
            page.insert_text((72, 72), content, fontname="china-t")
        data = doc.tobytes()
        doc.close()
        return data

    def test_default_covers_annual_report_length(self):
        # Simulate a 150-page annual report with the dislosures near the end.
        page_texts = [f"第 {i} 頁" for i in range(150)]
        page_texts[-1] = "前十大客戶：客戶甲 32.5%"
        pdf = self._make_pdf(page_texts)

        text = _extract_text_from_pdf(pdf)
        assert "前十大客戶" in text
        assert _MAX_PDF_PAGES >= 150, "default page cap must cover real reports"

    def test_low_page_cap_vs_default(self):
        # With the old 80-page cap the tail disclosure was invisible; with the
        # new default it is visible.
        page_texts = [f"第 {i} 頁" for i in range(100)]
        page_texts[-1] = "供應商/客戶資訊：A 公司 12.79%"
        pdf = self._make_pdf(page_texts)

        truncated = _extract_text_from_pdf(pdf, max_pages=80)
        assert "A 公司" not in truncated  # old behavior = missed disclosure

        full = _extract_text_from_pdf(pdf, max_pages=200)
        assert "A 公司" in full  # new behavior = disclosure visible

    def test_invalid_bytes_returns_empty(self):
        assert _extract_text_from_pdf(b"not a pdf at all") == ""

    def test_empty_pdf_bytes(self):
        assert _extract_text_from_pdf(b"") == ""


# ---------------------------------------------------------------------------
# Segment-splitting / merge / retry (南亞 113 修復：LLM 非確定性)
# ---------------------------------------------------------------------------


def _entry(raw_name: str, ratio: float | None = None, note: str | None = None):
    return _LLMCounterpartyEntry(raw_name=raw_name, ratio=ratio, note=note)


def _extract(suppliers=(), customers=(), confidence: float = 0.5, notes: str | None = None):
    return _AnnualReportExtract(
        suppliers=list(suppliers),
        customers=list(customers),
        confidence=confidence,
        fiscal_year=113,
        notes=notes,
    )


class TestSplitReportSegments:
    """Group A (disclosure tables) vs Group B (raw-material table) splitting.

    南亞 113 案例：金益鼎企業/鏶鑫企業 只出現在 Group B（主要原料之供應狀況）
    的銅線行。把兩組段落塞進同一張 prompt，local LLM 會隨機漏掉後段；拆分後
    Group B 有自己短而聚焦的 prompt。
    """

    def test_splits_disclosure_and_raw_material_into_two_segments(self):
        focused = (
            "(七)最近二年度任一年度曾佔進(銷)貨總額百分之十以上之供應商/客戶資訊：\n"
            "113年 台塑石化 53,578,492 33.44\n"
            "\n---\n\n"
            "(三)主要原料之供應狀況\n"
            "銅線 公噸 74,322 金益鼎企業、鏶鑫企業\n"
        )
        segments = _split_report_segments(focused)

        assert len(segments) == 2
        # Group A keeps the disclosure table, Group B the raw-material rows.
        assert "供應商/客戶資訊" in segments[0]
        assert "台塑石化" in segments[0]
        assert "主要原料之供應狀況" in segments[1]
        assert "金益鼎企業" in segments[1]
        # 銅線行必須待在 Group B（不能被 Group A 吃掉）。
        assert "銅線" not in segments[0]

    def test_disclosure_only_yields_single_segment(self):
        focused = "(七)供應商/客戶資訊：\n113年 台塑石化 53,578,492 33.44\n"
        assert len(_split_report_segments(focused)) == 1

    def test_raw_material_only_yields_single_segment(self):
        focused = "(三)主要原料之供應狀況\n銅線 公噸 74,322 金益鼎企業、鏶鑫企業\n"
        segments = _split_report_segments(focused)
        assert len(segments) == 1
        assert "金益鼎企業" in segments[0]

    def test_block_carrying_both_table_kinds_goes_to_both_groups(self):
        # 同一 block 同時含供應商表與原料表時，兩邊都要收（不漏資料）。
        focused = "前十名供應商：A公司 18.2%\n主要原料之供應狀況：銅線 公噸 74,322 金益鼎企業\n"
        segments = _split_report_segments(focused)
        assert len(segments) == 2
        assert "A公司" in segments[0]
        assert "金益鼎企業" in segments[1]

    def test_unclassifiable_falls_back_to_full_text(self):
        focused = "公司簡介與財報附註，沒有可辨識的表格標題。"
        assert _split_report_segments(focused) == [focused]


class TestMergeExtracts:
    def test_unions_and_dedups_by_name_note(self):
        a = _extract(
            suppliers=[_entry("台塑石化", 33.44), _entry("南亞塑膠")],
            customers=[_entry("台化", 12.3)],
            confidence=0.6,
        )
        b = _extract(
            suppliers=[_entry("台塑石化", 33.44), _entry("金益鼎企業", note="銅線")],
            confidence=0.8,
        )
        merged = _merge_extracts([a, b], fiscal_year=113)

        # 台塑石化 在兩段都出現 → 並集只留一份；金益鼎企業 只有 B 段有 → 保留。
        names = {s.raw_name for s in merged.suppliers}
        assert names == {"台塑石化", "南亞塑膠", "金益鼎企業"}
        assert merged.confidence == 0.8
        assert merged.fiscal_year == 113

    def test_ratio_row_and_note_row_of_same_name_both_kept(self):
        # 同一公司可同時出現在比率表（ratio）與原料表（note=原料名），
        # 兩行都是有效證據，不因 raw_name 相同而吃掉其中一行。
        a = _extract(suppliers=[_entry("台灣化學纖維", 13.93)])
        b = _extract(suppliers=[_entry("台灣化學纖維", note="環氧氯丙烷")])
        merged = _merge_extracts([a, b], fiscal_year=113)

        assert len(merged.suppliers) == 2

    def test_empty_segments_yield_empty(self):
        merged = _merge_extracts([_extract(), _extract()], fiscal_year=113)
        assert merged.suppliers == []
        assert merged.customers == []
        assert merged.confidence == 0.5


class TestExtractSegmentRetry:
    """_extract_segment: LLM 呼叫失敗或空結果時重試（local LLM 非確定性）。"""

    def _mock_llm(self, *results):
        llm = Mock()
        llm.extract_structured.side_effect = list(results)
        return llm

    def test_succeeds_on_first_attempt(self):
        llm = self._mock_llm(_extract(suppliers=[_entry("金益鼎企業", note="銅線")]))
        result = _extract_segment(
            "銅線 公噸 74,322 金益鼎企業、鏶鑫企業\n",
            stock_code="1303",
            stock_name="南亞",
            fiscal_year=113,
            llm_client=llm,
        )
        assert result.suppliers[0].raw_name == "金益鼎企業"
        assert llm.extract_structured.call_count == 1

    def test_retries_on_llm_error_then_succeeds(self):
        llm = self._mock_llm(
            LLMError("validation failed: is_related_party null"),
            _extract(suppliers=[_entry("鏶鑫企業", note="銅線")]),
        )
        result = _extract_segment(
            "銅線 公噸 74,322 金益鼎企業、鏶鑫企業\n",
            stock_code="1303",
            stock_name="南亞",
            fiscal_year=113,
            llm_client=llm,
        )
        assert result.suppliers[0].raw_name == "鏶鑫企業"
        assert llm.extract_structured.call_count == 2

    def test_raises_after_exhausting_retries(self):
        llm = self._mock_llm(LLMError("boom"), LLMError("boom"), LLMError("boom"), LLMError("boom"))
        with pytest.raises(LLMError):
            _extract_segment(
                "銅線 公噸 74,322 金益鼎企業\n",
                stock_code="1303",
                stock_name="南亞",
                fiscal_year=113,
                llm_client=llm,
            )
        assert llm.extract_structured.call_count == _MAX_SEGMENT_RETRIES

    def test_retries_when_empty_but_table_rows_present(self):
        # 段落有表格內容（含 、）卻回傳全空 → 判定為 LLM 漏抽，重試。
        llm = self._mock_llm(_extract(), _extract(suppliers=[_entry("金益鼎企業")]))
        result = _extract_segment(
            "銅線 公噸 74,322 金益鼎企業、鏶鑫企業\n",
            stock_code="1303",
            stock_name="南亞",
            fiscal_year=113,
            llm_client=llm,
        )
        assert len(result.suppliers) == 1
        assert llm.extract_structured.call_count == 2

    def test_no_retry_on_prose_empty(self):
        # 「本公司並無佔百分之十以上之客戶」是合法空結果（無表格內容標記），
        # 不能因為回傳空就白燒 retry 次數。
        llm = self._mock_llm(_extract())
        result = _extract_segment(
            "本公司並無占個體銷貨總額百分之十以上之客戶。",
            stock_code="1303",
            stock_name="南亞",
            fiscal_year=113,
            llm_client=llm,
        )
        assert result.customers == []
        assert llm.extract_structured.call_count == 1


class TestExtractMultiPassUnion:
    """_extract：同一 segment 多次並集抽取收斂。

    單次 LLM 抽取仍具非確定性（南亞 113 實測同一 segment 13 vs 3 家）；
    並集多 pass，每次都漏掉「不同」子集，union 收斂到完整表。
    """

    _RAW_TEXT = (
        "(三)主要原料之供應狀況\n"
        "原料種類 單位 數量 主要供應廠商\n"
        "安定劑 公噸 7,237 內部撥轉、佳友化工\n"
        "可塑劑 公噸 28,826 內部撥轉、南通寶峰\n"
        "環氧氯丙烷 公噸 161,379 无棣鑫岳化工集团有限公、台塑公司\n"
        "玻纖紗 公噸 86,442 台灣必成\n"
        "銅線 公噸 74,322 金益鼎企業、鏶鑫企業\n"
    )

    def _mock_llm(self, *results):
        llm = Mock()
        llm.extract_structured.side_effect = list(results)
        return llm

    def test_passes_add_missing_rows_until_converged(self):
        # pass1 漏掉原料表（只有 3 家揭露行），pass2 補上金益鼎/鏶鑫，
        # pass3 與 pass2 相同 → 收斂（3 次呼叫）。
        llm = self._mock_llm(
            _extract(suppliers=[_entry("台化", 10.0)]),
            _extract(
                suppliers=[
                    _entry("台化", 10.0),
                    _entry("金益鼎企業", note="銅線"),
                    _entry("鏶鑫企業", note="銅線"),
                ]
            ),
            _extract(
                suppliers=[
                    _entry("台化", 10.0),
                    _entry("金益鼎企業", note="銅線"),
                    _entry("鏶鑫企業", note="銅線"),
                ]
            ),
        )
        result = _run_extract(
            self._RAW_TEXT,
            stock_code="1303",
            stock_name="南亞",
            fiscal_year=113,
            llm_client=llm,
        )

        names = {s.raw_name for s in result.suppliers}
        assert "金益鼎企業" in names
        assert "鏶鑫企業" in names
        assert llm.extract_structured.call_count == 3

    def test_stops_early_when_first_pass_already_complete(self):
        # pass1 就已完整 → pass2 無新增行 → 停在 2 次呼叫。
        complete = _extract(
            suppliers=[
                _entry("台化", 10.0),
                _entry("金益鼎企業", note="銅線"),
                _entry("鏶鑫企業", note="銅線"),
            ]
        )
        llm = self._mock_llm(complete, complete)
        result = _run_extract(
            self._RAW_TEXT,
            stock_code="1303",
            stock_name="南亞",
            fiscal_year=113,
            llm_client=llm,
        )

        assert {s.raw_name for s in result.suppliers} == {
            "台化",
            "金益鼎企業",
            "鏶鑫企業",
        }
        assert llm.extract_structured.call_count == 2

    def test_deduplicated_union_across_passes(self):
        # 每一 pass 回傳重複行時，union 仍只留一份（台化 across passes）。
        llm = self._mock_llm(
            _extract(suppliers=[_entry("台化", 10.0), _entry("台化", note="玻纖紗")]),
            _extract(suppliers=[_entry("台化", 10.0)]),
        )
        result = _run_extract(
            self._RAW_TEXT,
            stock_code="1303",
            stock_name="南亞",
            fiscal_year=113,
            llm_client=llm,
        )

        assert len(result.suppliers) == 2  # (台化/10.0) 與 (台化/玻纖紗) 各一
        assert llm.extract_structured.call_count == 2


class TestScanRawMaterialNames:
    """確定性 backstop：LLM 整行漏掉時，直接從原料表 token 掃回上市供應商。

    台化 113 案例：LLM（又是本地小模型）每一 pass 都漏掉「丙烯腈→台塑公司」
    這一行，多 pass 並集也補不回來。backstop 把段落文字 token 化後逐個
    resolve，上市/上櫃公司（台塑/南亞/台橡/中石化）被補回；未上市/海外
    （長春/日本出光/奇美/ASAHI/SIBUR）resolve 為 None 自然被丟掉。
    """

    # 節錄台化 1326 年報「主要原料之供應狀況」真實內容（flatten 後）。
    _RAW_MATERIAL_TEXT = (
        "主要原料之供應狀況\n"
        "原料種類 單位 數量 主要供應廠商\n"
        "丙烯腈 公噸 53,848 2,090,799 台塑公司\n"
        "丙二酚 公噸 200,985 5,384,624 南亞公司、長春、日本出光\n"
        "橡膠 公噸 357,800 18,213,400 台橡、奇美、ASAHI、SIBUR(LRD)、中石化\n"
        "己內醯胺 公噸 20,900 703,142 中石化、日本UBE\n"
        "木漿 公噸 33,000 352,823 智利Arauco、南非Sappi、日本NPI\n"
    )

    def test_backstop_recovers_row_llm_dropped(self):
        # LLM 完全漏掉丙烯腈→台塑公司行 → backstop 補回台塑/1301。
        found = _scan_raw_material_names(
            self._RAW_MATERIAL_TEXT, stock_resolver, "1326", existing=[]
        )
        names = {e.raw_name for e in found}
        assert "台塑公司" in names  # 丙烯腈行 → strip 公司 → 台塑/1301
        assert "南亞公司" in names  # 丙二酚行分號列表第一項 → 南亞/1303
        assert "台橡" in names  # 橡膠行 → 台橡/2103
        assert "中石化" in names  # 橡膠行 + 己內醯胺行 → 中石化/1314

        codes = {stock_resolver.resolve_name(e.raw_name) for e in found}
        assert codes == {"1301", "1303", "2103", "1314"}

    def test_backstop_never_emits_unlisted_names(self):
        # 長春/日本出光/奇美/ASAHI/SIBUR(LRD)/智利Arauco 等未上市或海外
        # 公司必須被排除（resolve 為 None 直接丟棄）。
        found = _scan_raw_material_names(
            self._RAW_MATERIAL_TEXT, stock_resolver, "1326", existing=[]
        )
        raw_names = {e.raw_name for e in found}
        assert "長春" not in raw_names
        assert "日本出光" not in raw_names
        assert "奇美" not in raw_names
        assert "ASAHI" not in raw_names
        assert "SIBUR(LRD)" not in raw_names
        assert "日本UBE" not in raw_names
        assert "智利Arauco" not in raw_names

    def test_backstop_skips_target_company_and_existing(self):
        # 自己的公司名（台化/1326 = target）不輸出；已由 LLM 抽出的公司
        # （台塑/1301 已在 existing）不再重複補。
        found = _scan_raw_material_names(
            "主要原料之供應狀況\n台化 公噸 100 內部\n台塑公司 公噸 200 台塑\n",
            stock_resolver,
            "1326",
            existing=[_entry("台塑", note="丙烯腈")],
        )
        raw_names = {e.raw_name for e in found}
        assert "台化" not in raw_names  # target code 1326 excluded
        assert "台塑公司" not in raw_names  # code 1301 already in existing_codes

    def test_backstop_dedups_against_enumerated_existing(self):
        # LLM 已用併格 raw_name（「台塑石化公司、台塑公司」）捕捉過 1301/6505
        # 兩家 → existing_codes 必須拆格展開，backstop 不得重複補行。
        found = _scan_raw_material_names(
            "主要原料之供應狀況\n台塑石化(股)公司 公噸 200 台塑公司\n",
            stock_resolver,
            "1326",
            existing=[_entry("台塑石化公司、台塑公司", note="丙烯腈")],
        )
        assert found == []  # 兩家都已在 existing，無新增

    def test_backstop_skips_prose_lines_without_units(self):
        # 年報同一 segment 內含大量散文（「長短期業務發展計畫」「說明」），
        # 其中的「大陸」（中國大陸）不得被掃成 2526 謬誤供應商。只有帶單位
        # 標記（公噸/公斤/千元）的資料列才會被掃描。
        text = (
            "主要原料之供應狀況\n"
            "丙烯腈 公噸 53,848 2,090,799 台塑公司\n"
            "（3）、鄰二甲苯：短期方面，OX產出減少有利大陸OX產品外銷，"
            "以致大陸 OX市場供應相對偏緊\n"
            "說明：113年全球景氣雖維持平穩，但受大陸石化與塑膠產能持續投放等影響\n"
        )
        found = _scan_raw_material_names(text, stock_resolver, "1326", existing=[])
        codes = {stock_resolver.resolve_name(e.raw_name) for e in found}
        assert "2526" not in codes  # 大陸 → 2526 不得出現
        assert "1301" in codes  # 帶單位的資料列仍正常掃回台塑公司/1301

    def test_backstop_ratio_bearing_entries_untouched(self):
        # ratio 非 None（揭露表一行一公司）不拆格 — 拆格只適用於 ratio=None 的
        # 原料表行。backstop 掃描本身不改動、只新增；此處確認揭露表行原樣保留。
        entries = [_entry("台塑石化", ratio=33.44)]
        disclosures, warnings = _resolve_entries(entries, stock_resolver, "supplier")
        assert len(disclosures) == 1
        assert disclosures[0].resolved_code == "6505"
        assert disclosures[0].ratio == 33.44
        assert warnings == []


class TestResolveEntriesEnumerationSplit:
    """_resolve_entries 拆格：LLM 把 、/，分隔的供應商並成一個 raw_name 時，
    確定性拆回個別公司再 resolve。

    台化 113 案例：LLM 把「南亞公司、長春、日本出光」合併成一格 → 南亞/1303
    完全遺失。拆格後南亞/1303 找回；長春/日本出光 未上市 → resolved None。
    """

    def test_split_merged_raw_material_cell(self):
        entries = [_entry("南亞公司、長春、日本出光", note="丙二酚")]
        disclosures, warnings = _resolve_entries(entries, stock_resolver, "supplier")

        codes = {d.resolved_code for d in disclosures}
        assert "1303" in codes  # 南亞公司 → 南亞 → 1303 找回

        by_name = {d.raw_name: d for d in disclosures}
        assert by_name["長春"].resolved_code is None
        assert by_name["日本出光"].resolved_code is None
        # 三個部分各自保留原 ratio=None 與 note。
        assert all(d.ratio is None for d in disclosures)
        assert all(d.note == "丙二酚" for d in disclosures)
        # 未上市部分產生 warning，但南亞/1303 不產生。
        assert any("長春" in w for w in warnings)

    def test_split_full_raw_material_rows(self):
        entries = [
            _entry("台橡、奇美、ASAHI、SIBUR(LRD)、中石化", note="橡膠"),
            _entry("中石化、日本UBE", note="己內醯胺"),
        ]
        disclosures, _ = _resolve_entries(entries, stock_resolver, "supplier")

        # 上市/上櫃恰為 台橡/2103、中石化/1314；其餘部分 resolved_code=None。
        resolved = {d.resolved_code for d in disclosures if d.resolved_code}
        assert resolved == {"2103", "1314"}
        # 每個部分都拆開；未上市部分 resolved_code=None。
        names = {d.raw_name for d in disclosures}
        assert names == {
            "台橡",
            "奇美",
            "ASAHI",
            "SIBUR(LRD)",
            "中石化",
            "日本UBE",
        }

    def test_ratio_row_never_split(self):
        # 揭露表一行一公司（ratio 非 None）——即使 raw_name 含中文逗號也不拆，
        # 避免把「台塑化, 台化」這種合併揭露當成單一一格的先例。
        entries = [_entry("台塑石化", ratio=33.44)]
        disclosures, _ = _resolve_entries(entries, stock_resolver, "supplier")
        assert [d.raw_name for d in disclosures] == ["台塑石化"]

    def test_duplicate_parts_deduped(self):
        # 同一格內重複的公司名只輸出一次。
        entries = [_entry("台橡、台橡", note="橡膠")]
        disclosures, _ = _resolve_entries(entries, stock_resolver, "supplier")
        assert [d.raw_name for d in disclosures if d.resolved_code] == ["台橡"]
