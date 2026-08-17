# 開始使用 (Getting Started)

本文件說明如何安裝、設定並執行 **classify-twse-query**（台股自然語言查詢 / 分類 / 分解 / 視覺化工具）。

## 1. 安裝依賴

```bash
pip install -r requirements.txt
```

## 2. 設定環境變數

```bash
cp .env.example .env
```

編輯 `.env`：

- **LLM 層是 vendor-neutral**，指向任何 OpenAI 相容的 `/v1/chat/completions` 端點。
  - 本機免費（推薦先試這個，不需任何 key）：先裝 [Ollama](https://ollama.com) 並拉模型，再設：
    ```ini
    LLM_BASE_URL=http://localhost:11434/v1
    LLM_API_KEY=ollama
    LLM_MODEL=qwen2.5:7b
    LLM_TIMEOUT=120
    ```
  - 或用 OpenAI：`LLM_BASE_URL=https://api.openai.com/v1`、`LLM_API_KEY=sk-...`、`LLM_MODEL=gpt-4o-mini`
  - 也接受舊版 `OPENAI_*` 變數（向後相容）
- **圖表資料來自免費 TWSE**（股價 / K 線 / 本益比），不需要任何 key。
- `FINMIND_API_TOKEN` 為**選填**，僅「月營收圖」需要 FinMind 付費帳號時才填。

## 3. 命令列（CLI）

```bash
# 單次查詢（輸出 JSON 步驟追蹤；若分類要求視覺化會產出 PNG 到 output/charts/）
python -m classifier.cli "台積電現在股價多少"

# 對話模式（多輪，自帶代詞解析上下文）
python -m classifier.cli --chat

# 只拿 JSON、不畫圖
python -m classifier.cli "台積電本益比多少" --no-chart
```

## 4. 一鍵啟動（API + 前端）

最簡單的方式：執行啟動腳本，它會同時啟動 API（8000）與前端靜態伺服器（8080），並自動開啟瀏覽器。

```bash
python start.py
```

- API：`http://127.0.0.1:8000`
- 前端：`http://127.0.0.1:8080`（腳本會自動開啟）
- 按 `Ctrl-C` 同時關閉兩個服務

## 5. 手動啟動（分開跑）

若不想用一鍵腳本，可手動分別啟動：

### HTTP API

```bash
python -m classifier.api
```

預設跑在 `http://127.0.0.1:8000`：

| 方法 | 路徑 | 說明 |
|---|---|---|
| POST | `/pipeline` | 本體：`{"question": "台積電未來展望"}` → 回傳步驟追蹤 JSON |
| POST | `/chart` | 本體為 `ChartRequest` JSON → 回傳 `image/png` 圖檔 |
| GET | `/health` | 健康檢查（不呼叫外部服務） |

### 前端介面

API **不會代管**前端靜態檔案，需另開一個靜態伺服器：

```bash
cd frontend
python -m http.server 8080
```

瀏覽器開 `http://127.0.0.1:8080`（前端會呼叫 `http://127.0.0.1:8000` 的 API，已開 CORS）。

## 6. 執行測試

```bash
pytest
```
