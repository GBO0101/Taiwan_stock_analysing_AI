# ADR-002: Multi-Source Data Strategy — TWSE API + LLM Validation

## Status

Proposed (TWSE endpoints confirmed via live probing, Sep 2026)

## Context

Step 4 needs stock lists by sector/index and supply-chain relationships. No single free data source covers all cases. Live probing of TWSE endpoints (Sep 2026) established what is actually available for free:

### Confirmed data sources

| Need | Source | Endpoint / Method | Auth | Notes |
|------|--------|-------------------|------|-------|
| **Sector → stock list** (上市) | TWSE ISIN 出版列表 | `https://isin.twse.com.tw/isin/C_public.jsp?strMode=2` | Free | HTML table (big5 encoding). Columns: 代號, 名稱, ISIN, 上市日, 市場別, 產業別, CFICode. `strMode=4`=上櫃, `strMode=5`=興櫃. Parse + filter `len(code)==4`. |
| **Index constituents** (台灣50/中型100) | 台灣指數公司 TIP | `https://taiwanindex.com.tw/indexes/TW50` / `TWMC` (download CSV) | Free | TIP 是 TWSE 子公司；提供成分股下載。備援：MOPS ETF(0050/0051) 持股明細當代理。 |
| **Analytical / PE** | TWSE OpenAPI | `https://openapi.twse.com.tw/v1/...` (BWIBBU etc.) | Free | 已有 `FreeDataFetcher` 在用。 |
| **Supply chain (上下游)** | 股東會年報 + LLM fallback | `doc.twse.com.tw` → PDF → PyMuPDF 抽取 → LLM 結構化（見下表） | 無（公開資料） | 年報公告後 F18 優先、F04 次要；未公告或無實體 PDF fallback 到 LLM。 |

### Key finding: supply chain's only authoritative free source is the annual report

TWSE / MOPS 免費公開層面**沒有**結構化的「公司 X 的上游供應商 / 下游客戶」API。唯一相關的是 ESG 供應鏈管理政策揭露（`openapi.twse.com.tw/v1/opendata/t187ap46_L_13`）——敘述性政策文件，非關係資料。

但是股東會年報（doc.twse.com.tw 的 F18/F04）依金管會規定**必須揭露前五大客戶/供應商及佔比**（客戶名稱可匿名），是唯一免費、權威、帶佔比的關係資料。

因此 **Related (上下游) mode 的主來源改為年報（annual-report-first）**，LLM 推論只是 fallback。這與 Range mode 一致——權威來源優先，LLM 僅在權威來源缺漏時接手。

## Decision

### Range mode: API-First + LLM Validation

```
1. 解析 intent → "半導體產業" → type=sector, value="半導體"
2. 抓取 ISIN 出版列表（上市+上櫃），快取 7 天
3. 依 產業別 欄位過濾出相符股票 → 對應到 半導體業=24 (Fugle code) / 半導體(text)
4. 用 LLM 驗證補充：
   - 提供 TWSE 過濾後清單 + 原始產業關鍵字
   - 問 LLM: 有無明顯遺漏(龍頭股)? 有無錯誤混入?
   - LLM 回 { to_add:[{code,name}], to_remove:[code], notes }
   - 套用修正 → 最終清單 (source="combined")
5. 若 LLM 失敗 → 用 TWSE 清單原樣 (source="twse_api")
6. 若 TWSE 失敗(網路/改版) → LLM-only (source="llm", confidence=0.6)
```

### Index mode (台灣50/中型100): TIP CSV → stock list

```
1. 解析 intent → "台灣50成分股" → query_type=index, value="台灣50"
2. 抓取 TIP 下載 CSV (taiwanindex.com.tw) 或使用本地靜態成分股檔，快取 7 天
3. 解析成 [{code,name}]
4. LLM 驗證補充（同 Range 流程）
```

### Related mode (供應鏈): Annual-report-first + LLM fallback

```
1. 確認只有單一目標股票 (或取 boundary 第一支)。
2. 年報路徑（doc.twse.com.tw）：
   a. 搜尋 Step 1：query=代號, mtype=F18(F04 備援), step=1 → 取得最近 3 年 PDF url。
   b. GET PDF URL → PyMuPDF 提取文字。
   c. 將 PDF 文字 + 代號/名稱/民國年 → LLM 結構化抽取（_AnnualReportExtract）：
      - customers: [{raw_name, resolved_code?, resolved_name?, ratio?, is_anonymous?, inferred?}]
      - suppliers: [{raw_name, resolved_code?, resolved_name?, ratio?, is_anonymous?, inferred?}]
   d. 用 StockResolver 交叉驗證每個 counterparty 是否為上市櫃公司：
      - counterparty 可解析 → 併入 upstream/downstream；impact = ratio / 100。
      - counterparty 不可解析 → 跳過。
   e. Phase 1.5：對 up to 3 家 ratio 最高的 counterparty，抓取其自身年報，
      - 若他們也列出目標股票 → disclosed_by = "both"。
   f. 年報未公告（PDF 404）或無可解析 counterparty → fallback 到 LLM。
3. LLM fallback（supply_chain.j2）：
   - prompt 含 stock_code, stock_name, industry_hint（若有）
   - LLM 回 { upstream:[{code,name,impact}], downstream:[{code,name,impact}] }
   - impact 直接使用 LLM 提供之 0–1 數值
4. 用 StockResolver 交叉驗證每個 code/name 是否存在
   - code 無效或 name 不符 → 移除，記入 notes
5. 回傳 source="annual_report" 或 source="llm"；confidence 由年報信心或 LLM 自評
```

### Confidence Scoring

| 情境 | Source | Confidence |
|------|--------|-----------|
| TWSE/TIP + LLM 驗證 | combined | 0.9 |
| TWSE/TIP only (LLM 失敗) | twse_api / tip | 0.8 |
| 上下游 (年報揭露, F18/F04) | annual_report | max(0.6, 年報抽取信心) |
| 上下游 (年報 + 雙邊交叉驗證) | annual_report | 同上（disclosed_by="both" 標記） |
| LLM-only (API 失敗) | llm | 0.6 |
| 上下游 LLM fallback（年報缺漏） | llm | LLM 自評 0.3–0.6 |
| LLM 低信心推論 | llm | 0.3 |

系統取 `min(資料源基數, LLM 自評置信度)`；年報路徑的 impact 以實質佔比（ratio/100）為基礎，LLM 推論的 impact 以 LLM 自評 0–1 為基礎。

### LLM Validation Prompt (`prompts/stock_validation.j2`)

1. 提供資料源清單作為 context
2. 要求 LLM 檢查明顯遺漏的龍頭股、錯誤混入
3. 回傳結構化 `{ to_add, to_remove, confidence, notes }`
4. Temperature 0.0

### 快取 (classifier/stock_cache.py)

- ISIN 全上市/上櫃列表：7 天 TTL
- TIP 成分股：7 天 TTL
- 年報揭露（`AnnualReportDisclosure`）：30 天 TTL，key = `annual_report_{code}_{fiscal_year}`；raw 抽取 JSON 快取，PDF 下載僅在 cache miss 時發生
- 供應鏈 LLM 結果：**不快取**（查詢依賴、易變）

### Error Handling

| 情境 | 行為 |
|------|------|
| ISIN/TIP timeout | Retry once (5s)，再失敗 fall back LLM |
| ISIN/TIP 空 | fall back LLM |
| LLM 無效 JSON | 用資料源原樣 |
| 兩者皆失敗 | 空清單 + confidence=0 + error |
| LLM code 無效 | 從清單移除無效 code |
| 年報 PDF 404 / 尚未公告 | fall back LLM（記入 notes） |
| 年報抽取 LLM 失敗 | fall back LLM |
| 年報無可解析 counterparty | fall back LLM（記入 notes） |
| counterparty 年報抓取失敗 | best-effort，跳過（Phase 1.5 不阻塞主結果） |

## Consequences

### Positive

- **Range mode 有權威免費來源**（ISIN 產業別 + TIP 成分股），LLM 只做驗證補充
- **Related mode 年報-first**：股東會年報是唯一免費、權威、帶實質佔比的供應鏈揭露來源（F18/F04 法定揭露）
- 年報路徑 impact 由實質佔比推導（ratio/100），遠比 LLM 自評可信
- Phase 1.5 雙邊交叉驗證（disclosed_by="both"）進一步強化年報資料可信度
- code 交叉驗證 + 快取（7 天 / 30 天 TTL）避免重複解析與抓取

### Negative

- ISIN 是 HTML 非 JSON — 需解析、big5 編碼、HTML 結構可能改版
- 年報 PDF 解析依賴 PyMuPDF 與檔案版型；PDF 版型異動需調整個案
- 多一次 LLM call（抽取/驗證/推論）→ 成本與延遲增加
- 年報揭露常為匿名客戶（公司A、甲廠商）→ 需 LLM inferred identity，非權威
- 成分股需代理來源（TIP CSV / ETF 持股），非單一權威 JSON

### Risks

- TWSE 網頁改版 → ISIN 解析器、doc.twse.com.tw 協議需更新
- LLM 驗證可能誤加/誤刪 → temperature=0.0、code 交叉驗證
- 年報 PDF 版型或 doc.twse 回應格式變動 → annual_report.py 需監控
- 匿名揭露的家數逐年增加（金管會規範）→ resolved 率可能下降，LLM fallback 比例上升
- TIP 下載格式異動 → 需監控
