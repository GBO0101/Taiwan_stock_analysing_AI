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
| **Supply chain (上下游)** | 無免費直接來源 | — | — | **必須靠 LLM 推論**。TWSE ESG `t187ap46_L_13`(供應鏈管理) 只是政策揭露，非關係資料。 |

### Key finding: supply chain has NO free direct source

TWSE / MOPS 免費公開層面**沒有**提供「公司 X 的上游供應商 / 下游客戶」的關係資料。唯一相關的是 ESG 供應鏈管理政策揭露（`openapi.twse.com.tw/v1/opendata/t187ap46_L_13`），但那是敘述性文件，不是結構化上下游客戶關係。

因此 **Related (上下游) mode 的主要來源是 LLM 推論**，搭配置信度標記。這與 Range mode 不同 — Range 有權威來源，Related 只有 LLM。

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

### Related mode (供應鏈): LLM-first + code validation

```
1. 確認只有單一目標股票 (或取 boundary 第一支)
2. LLM 推論：
   - prompt: "給定台積電(2330)，列出其直接上游供應商與直接下游客戶，各列出 5-8 家台灣上市公司"
   - LLM 回 { upstream:[{code,name}], downstream:[{code,name}] }
3. 用 StockResolver / 全上市列表 交叉驗證每個 code/name 是否存在
   - code 無效或 name 不符 → 移除，記入 notes
4. 回傳 source="llm", confidence 由 LLM 自評
```

### Confidence Scoring

| 情境 | Source | Confidence |
|------|--------|-----------|
| TWSE/TIP + LLM 驗證 | combined | 0.9 |
| TWSE/TIP only (LLM 失敗) | twse_api / tip | 0.8 |
| LLM-only (API 失敗) | llm | 0.6 |
| LLM 低信心推論 | llm | 0.3 |
| 上下游 (LLM-first, code 驗證) | llm | LLM 自評 0.3–0.6 |

系統取 `min(資料源基數, LLM 自評置信度)`。

### LLM Validation Prompt (`prompts/stock_validation.j2`)

1. 提供資料源清單作為 context
2. 要求 LLM 檢查明顯遺漏的龍頭股、錯誤混入
3. 回傳結構化 `{ to_add, to_remove, confidence, notes }`
4. Temperature 0.0

### 快取 (cache/stock_cache.py)

- ISIN 全上市/上櫃列表：7 天 TTL
- TIP 成分股：7 天 TTL
- 供應鏈 LLM 結果：**不快取**（查詢依賴、易變）

### Error Handling

| 情境 | 行為 |
|------|------|
| ISIN/TIP timeout | Retry once (5s)，再失敗 fall back LLM |
| ISIN/TIP 空 | fall back LLM |
| LLM 無效 JSON | 用資料源原樣 |
| 兩者皆失敗 | 空清單 + confidence=0 + error |
| LLM code 無效 | 從清單移除無效 code |

## Consequences

### Positive

- **Range mode 有權威免費來源**（ISIN 產業別 + TIP 成分股），LLM 只做驗證補充
- 供應鏈 LLM-first 雖無權威來源，但 code 交叉驗證降低 hallucination 影響
- 快取避免重複抓取大表（ISIN 全表約數千列）
- 新鮮度有 TTL 保證

### Negative

- ISIN 是 HTML 非 JSON — 需解析、big5 編碼、HTML 結構可能改版
- 供應鏈準確度受限（無免費關係資料源）→ confidence 誠實標記
- 每 Step 4 多一次 LLM call（驗證/推論）
- 成分股需代理來源（TIP CSV / ETF 持股），非單一權威 JSON

### Risks

- TWSE 網頁改版 → ISIN 解析器需更新
- LLM 驗證可能誤加/誤刪 → temperature=0.0、code 交叉驗證
- 供應鏈清單是推論 → 需在 UI 標明「AI 推論結果」
- TIP 下載格式異動 → 需監控
