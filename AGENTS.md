# AGENTS.md — classify-twse-query

Taiwan-stock natural-language query **understanding → classification → decomposition → visualization** tool.
Python ≥3.10, FastAPI backend + static `frontend/`, LLM-driven via an OpenAI-compatible (vendor-neutral) client.

## 溝通語言

- **使用中文回應我。**

## Structure

```
classify-twse-query/
├── classifier/            # importable package: the whole pipeline lives here
│   ├── pipeline.py        # Pipeline.run(): STRICT sequential orchestrator (Steps 1-3)
│   ├── boundary.py        # Step 1: entity/scope/time/metric extraction; chart fields reconciled here (chart source of truth)
│   ├── chart_validator.py  # keyword → ChartDataRequirement + enforced chart_type (used by boundary)
│   ├── stock_resolver.py   # authoritative company_name ↔ stock_code resolver (singleton)
│   ├── classification.py  # Step 2: query type + visualization needs
│   ├── decomposition.py   # Step 3 (conditional): analytical → sub-query DAG
│   ├── models.py          # Pydantic contracts + enums (single source of truth)
│   ├── llm_client.py      # vendor-neutral OpenAI-compatible structured-extraction client
│   ├── prompts.py         # Jinja2 PromptManager over prompts/*.j2
│   ├── indicator_mapper.py# loads table.csv → indicator→dataset/field map
│   ├── data_fetcher.py    # FREE TWSE fetcher (STOCK_DAY, BWIBBU); no key
│   ├── finmind_client.py  # paid FinMind client (charts only — see ANTI-PATTERNS)
│   ├── chart_renderer.py  # matplotlib chart writer → output/charts/*.png; K-line draws a daily-average trend line
│   ├── config.py          # pydantic-settings; imports .env
│   ├── cli.py             # `python -m classifier.cli` entry point
│   └── api.py             # FastAPI app (endpoints below)
├── prompts/               # boundary.j2, classify.j2, decompose.j2, sentiment.j2
├── frontend/              # plain index.html / app.js / styles.css (static, calls API :8000)
├── tests/unit/            # one test_*.py per classifier module (LLM/network MOCKED)
├── tests/integration/     # FastAPI TestClient + pipeline-e2e sequential-contract tests
├── table.csv              # indicator→dataset/field mapping (loaded at import)
├── finmind_data/          # standalone scraped datasets + finmind_*.py scripts (NOT part of pipeline)
├── output/charts/         # generated PNGs (gitignored in practice)
├── start.py               # one-click: API :8000 + frontend :8080 + opens browser
└── pyproject.toml         # pytest / ruff / mypy config
```

## Where to look

| Task | Location |
|------|----------|
| Change pipeline order/flow | `classifier/pipeline.py` (`Pipeline.run`, ~L49) |
| Add/change an LLM step | the step module + its `prompts/*.j2` template |
| Add a chart type | `models.py` (`ChartDataRequirement`), `chart_renderer.py` (`_render_*`), `cli.py` (`_build_chart_request`) |
| Change how a chart is decided | `boundary.py` (`_reconcile_chart_fields`) is the source of truth; cli/api read `boundary.chart_data_requirements` (not classification) |
| Change indicator mapping | `table.csv` + `classifier/indicator_mapper.py` |
| Switch LLM provider | `.env` `LLM_BASE_URL` (any OpenAI-compatible `/v1`) — no code change |
| Tune LLM call (temp/format) | `classifier/llm_client.py` (`extract_structured`, ~L90) |
| Edit API routes | `classifier/api.py` (`/pipeline`, `/chart`, `/health`) |

## Code map (key symbols)

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `Pipeline.run` | method | pipeline.py:49 | Strict sequential runner; Step 3 only if `classification.type == ANALYTICAL` |
| `Settings` | class | config.py:7 | Env config; `finmind_api_token` has NO default (required) |
| `LLMClient.extract_structured` | method | llm_client.py:90 | JSON-mode structured extraction, temp 0.0, maps errors→`LLM*` exceptions |
| `FreeDataFetcher` | class | data_fetcher.py:58 | Key-less TWSE price/PE; `get_month_revenue` always raises |
| `FinMindClient` | class | finmind_client.py:32 | Paid FinMind; **charts/optional only** |
| `ChartRenderer.render` | method | chart_renderer.py:61 | Dispatch by `ChartDataRequirement` → PNG in `output/charts/` |
| `PromptManager` | class | prompts.py:16 | Renders `prompts/*.j2` (autoescape off) |
| `IndicatorMapper` | class | indicator_mapper.py:16 | Loads `table.csv`; global `indicator_mapper` at import |
| `extract_boundary` / `classify_query` / `decompose_query` | fn | boundary/classification/decomposition.py | One LLM call each, returns a Pydantic model |
| `StockResolver` | class | stock_resolver.py | Authoritative company_name ↔ stock_code resolver; fills & reverse-verifies `stock_codes` |
| `ChartValidator` | module | chart_validator.py | Keyword logic → `ChartDataRequirement` + enforced `chart_type` |
| `BoundaryResult.chart_data_requirements` | field | models.py:256 | Post-reconciliation authoritative chart spec; cli/api render from this (not classification) |
| `_reconcile_chart_fields` | fn | boundary.py:39 | Validates/corrects chart fields on boundary output — charting source of truth |
| `ChartRenderer._render_kline` | method | chart_renderer.py:153 | Candlestick chart; overlays a blue dashed daily-average (O/H/L/C mean) trend line |

## Conventions (deviations from stock Python)

- **LLM is vendor-neutral.** Call only `LLMClient.extract_structured(prompt, Model)`. Never use the OpenAI-only `beta.chat.completions.parse` (explicit comment in `llm_client.py`).
- **Structured output via Pydantic.** Every LLM step returns a model from `models.py`; add new fields there, not ad-hoc dicts.
- **typing is mandatory.** `pyproject.toml` sets `mypy disallow_untyped_defs = true` and uses `X | None`; new code must be fully typed.
- **ruff** `line-length = 100`, `target-version = "py310"`.
- **Tests never hit the network.** Unit tests `patch` `LLMClient`/`OpenAI` with canned models (`side_effect` lists). Integration tests use `fastapi.testclient.TestClient` and assert "no real network calls". Mock `Pipeline` or `FreeDataFetcher`, never call the live API in CI.
- **Strict sequential contract** (verified in `tests/integration/test_pipeline_e2e.py`): Step 1 fail blocks Step 2; Step 2 non-`ANALYTICAL` skips Step 3; Step 2 fail blocks Step 3. Preserve this ordering.
- Charts use **matplotlib Agg backend** + CJK font auto-detection (see `chart_renderer.py` top). Keep `matplotlib.use("Agg")` before pyplot import.

## Anti-patterns (this project)

- **Do NOT call `FinMindClient` from Steps 1–3.** It is for chart rendering only (`finmind_client.py` states this explicitly). Steps must use only `LLMClient` + `IndicatorMapper`.
- **Do NOT make `Pipeline` steps call the network directly** except through the injected clients. Keep steps pure + injectable for tests.
- **Do NOT widen CORS in production.** `api.py` uses `allow_origins=["*"]` for local dev only — restrict before deploy.
- **Do NOT skip `table.csv`.** `IndicatorMapper()` is instantiated at import; missing/empty/mis-shaped CSV raises `IndicatorMappingError` on startup.

## Notes (gotchas)

- **`.env` MUST define `FINMIND_API_TOKEN`** even though the free path uses TWSE, not FinMind — `config.Settings` has no default for it, so `import classifier.config` raises `ValidationError` without it. (README's "optional" wording is misleading.)
- **Revenue charts don't work on free data.** `ChartRenderer` routes `REVENUE_TREND`/`REVENUE_COMPARISON` through `FreeDataFetcher.get_month_revenue`, which always raises (MOPS blocks unauthenticated access). They need FinMind wired in — currently they fail.
- `sentiment.j2` + `PromptManager.render_sentiment` exist but are **not used** by the strict 3-step pipeline (optional/legacy node).
- Root `finmind_*.py` and `finmind_data/` are standalone data-acquisition utilities, separate from the importable `classifier` package — don't import them into the pipeline.
- Git repo (branch `master`) tracked at `https://github.com/GBO0101/Taiwan_stock_analysing_AI`. No CI/Makefile/Dockerfile present.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env          # set LLM_* AND FINMIND_API_TOKEN (required, see Notes)

# CLI
python -m classifier.cli "台積電現在股價多少"        # one-shot JSON trace (+PNG if viz)
python -m classifier.cli --chat                      # multi-turn, pronoun context
python -m classifier.cli "台積電本益比" --no-chart    # skip rendering

# Services
python start.py                 # API :8000 + frontend :8080, opens browser (Ctrl-C stops both)
python -m classifier.api        # API only (uvicorn, 127.0.0.1:8000)
python -m http.server 8080 --directory frontend   # frontend only

# Quality
pytest                          # unit + integration (no live network)
ruff check . && ruff format .
mypy classifier
```
