# classify-twse-query

Taiwan-stock query understanding, classification, decomposition, and visualization tool.

## 開始使用 (Getting Started)

完整使用說明（安裝、LLM/圖表設定、CLI、一鍵啟動、API、前端、測試）請見 [GETTING_STARTED.md](GETTING_STARTED.md)。

快速開始：

```bash
pip install -r requirements.txt
cp .env.example .env        # 編輯 LLM_BASE_URL / LLM_MODEL（本機可用 Ollama，免 key）
python start.py             # 一鍵啟動 API + 前端，並開啟瀏覽器
```

## Project Structure

```
classify-twse-query/
├── classifier/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── models.py
│   ├── llm_client.py
│   ├── finmind_client.py
│   ├── pipeline.py
│   ├── boundary.py
│   ├── chart_validator.py
│   ├── stock_resolver.py
│   ├── classification.py
│   ├── decomposition.py
│   ├── indicator_mapper.py
│   ├── chart_renderer.py
│   └── prompts.py
├── prompts/
│   ├── boundary.j2
│   ├── classify.j2
│   ├── decompose.j2
│   └── sentiment.j2
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── tests/
│   ├── unit/
│   └── integration/
├── output/
│   └── charts/
├── table.csv
├── pyproject.toml
├── requirements.txt
├── .env.example
└── README.md
```

## Pipeline

1. **Step 1 — Boundary Extraction**: Extract entities, scope, time range, metrics, market, visualization hints
2. **Step 2 — Classification**: Determine query type (live/factual/analytical/non_financial) and visualization needs
3. **Step 3 — Decomposition** (conditional): Convert analytical questions into executable FinMind query DAG

The **boundary output** is the single source of truth for charting: its `chart_data_requirements` (reconciled by `chart_validator` keywords) decides the chart type, not the classification step.

## Configuration

The LLM layer is **vendor-neutral**: it speaks the OpenAI-compatible Chat
Completions API, so any provider exposing `/v1/chat/completions` works — OpenAI,
Ollama, vLLM, Groq, DeepSeek, OpenRouter, LM Studio, and more. Just point
`LLM_BASE_URL` at the provider's `/v1` endpoint.

Required environment variables:
- `LLM_BASE_URL`: OpenAI-compatible endpoint base URL (default: `https://api.openai.com/v1`)
- `LLM_API_KEY` (or legacy `OPENAI_API_KEY`): API key for the provider
- `LLM_MODEL` (or legacy `OPENAI_MODEL`): Model to use (default: `gpt-4o-mini`)
- `LLM_TIMEOUT` (or legacy `OPENAI_TIMEOUT`): Request timeout in seconds (default: 30)

### Chart data source

Charts are rendered from **free TWSE open APIs** (no API key required):

- Price / K-line / volume → TWSE `STOCK_DAY`
- PE ratio / sector analysis → TWSE `BWIBBU`

The K-line (candlestick) chart overlays a **blue dashed daily-average trend line** (mean of each day's open/high/low/close).

Monthly **revenue** charts are NOT available from the free source (MOPS blocks
unauthenticated access). They require a paid FinMind account:

- `FINMIND_API_TOKEN`: FinMind API token (only needed for monthly-revenue charts)
- `FINMIND_BASE_URL`: FinMind API base URL (default: https://api.finmindtrade.com/api/v4)

### Examples

```bash
# OpenAI (default)
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...

# Local Ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama   # Ollama ignores the key, any non-empty value works
LLM_MODEL=llama3.1

# Groq
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...
LLM_MODEL=llama-3.1-70b-versatile
```

## License

MIT