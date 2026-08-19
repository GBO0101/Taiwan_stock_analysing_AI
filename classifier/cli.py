"""Command-line interface for the classify-twse-query pipeline.

Modes:
  One-shot:  python -m classifier.cli "台積電現在股價多少"
  Chat:      python -m classifier.cli --chat

The CLI runs the strict sequential pipeline (Step 1 -> Step 2 ->
conditional Step 3) and, when the classification requests a chart,
renders it via ChartRenderer. It never opens a browser and returns a
non-zero exit code on strict LLM failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from classifier.chart_renderer import ChartRenderer, ChartRenderError
from classifier.classification import ClassificationError
from classifier.models import (
    ChartDataRequirement,
    ChartRequest,
    ChartType,
    ClassificationResult,
    DateRange,
    PipelineResult,
)
from classifier.pipeline import Pipeline, PipelineError


def _get_classification_and_boundary(
    result: PipelineResult,
) -> tuple[ClassificationResult | None, dict[str, Any] | None]:
    """Extract the completed classification and boundary steps from a result."""
    classification: ClassificationResult | None = None
    boundary: dict[str, Any] | None = None
    for step in result.steps:
        if step.step == "classification" and step.status == "completed":
            classification = ClassificationResult(**step.output)
        elif step.step == "boundary" and step.status == "completed":
            boundary = step.output
    return classification, boundary


def _build_chart_request(result: PipelineResult) -> ChartRequest | None:
    """Build a ChartRequest from a successful pipeline result, if applicable.

    The boundary output is the single source of truth for charting: its
    ``chart_data_requirements`` / ``chart_type_placeholder`` (validated by
    ChartValidator at Step 1) decide both whether a chart is produced and which
    data source to fetch. Classification only remains the gate for Step 3
    decomposition.
    """
    _, boundary = _get_classification_and_boundary(result)

    if boundary is None:
        return None
    # Source of truth = boundary's validated chart fields.
    raw_req = boundary.get("chart_data_requirements")
    if not raw_req or not boundary.get("stock_codes"):
        return None
    try:
        req_enum = ChartDataRequirement(raw_req)
    except ValueError:
        return None

    chart_type_raw = boundary.get("chart_type") or ChartType.LINE
    try:
        ct_enum = ChartType(chart_type_raw)
    except ValueError:
        ct_enum = ChartType.LINE

    date_range = None
    dr = boundary.get("date_range")
    if isinstance(dr, dict) and dr.get("type") and dr.get("value"):
        try:
            date_range = DateRange(type=dr["type"], value=dr["value"])
        except (ValueError, TypeError):
            date_range = None
    return ChartRequest(
        stock_codes=list(boundary["stock_codes"]),
        chart_data_requirements=req_enum,
        chart_type=ct_enum,
        date_range=date_range,
    )


def _no_chart_reason(result: PipelineResult) -> str | None:
    """Return a human-readable reason when no chart is produced despite a viz request.

    Uses the boundary output as the single source of truth for chart intent
    (its validated ``chart_data_requirements``). Returns None when no chart is
    expected (boundary has no chart requirement, or a chart was actually built),
    so the caller only warns on the silent-failure case: visualization requested
    but no stock code could be resolved.
    """
    _, boundary = _get_classification_and_boundary(result)
    if boundary is None or not boundary.get("chart_data_requirements"):
        return None
    if boundary.get("stock_codes"):
        return None
    return (
        "查詢要求產生圖表，但無法從查詢解析出股票代號（stock_codes 為空），"
        "因此未產生圖表。請確認股票名稱或代號是否正確。"
    )


def _run_once(
    pipeline: Pipeline,
    question: str,
    context: dict[str, Any] | None,
    render_charts: bool,
) -> dict[str, Any]:
    """Run the pipeline once and return a JSON-serializable result dict."""
    result = pipeline.run(question=question, context=context)

    output: dict[str, Any] = result.model_dump()

    chart_path: str | None = None
    if render_charts:
        try:
            request = _build_chart_request(result)
            if request is not None:
                renderer = ChartRenderer()
                chart_path = renderer.render(request)
            else:
                reason = _no_chart_reason(result)
                if reason is not None:
                    print(f"警告：{reason}", file=sys.stderr)
                    output["chart_note"] = reason
        except ChartRenderError as e:
            # Chart failure is non-fatal for the pipeline result itself.
            chart_path = None
            output["chart_error"] = str(e)

    if chart_path is not None:
        output["chart_path"] = chart_path

    return output


def _update_context(context: dict[str, Any], result: PipelineResult) -> dict[str, Any]:
    """Accumulate the latest boundary into the chat context for pronoun resolution."""
    for step in result.steps:
        if step.step == "boundary" and step.status == "completed":
            context.update(step.output)
            break
    return context


def _run_oneshot(question: str, render_charts: bool) -> int:
    """Execute exactly one pipeline and print JSON. Returns process exit code."""
    pipeline = Pipeline()
    try:
        output = _run_once(pipeline, question, context=None, render_charts=render_charts)
    except PipelineError as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _run_chat(render_charts: bool) -> int:
    """Interactive chat loop with process-local context for pronoun resolution."""
    pipeline = Pipeline()
    context: dict[str, Any] = {}
    print("進入對話模式 (輸入 exit / quit 離開)", file=sys.stderr)
    try:
        while True:
            try:
                question = input("User: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            try:
                output = _run_once(
                    pipeline, question, context=context or None, render_charts=render_charts
                )
            except PipelineError as e:
                print(json.dumps({"error": str(e)}, ensure_ascii=False))
                continue
            print(json.dumps(output, ensure_ascii=False, indent=2))
            # Update context from the latest boundary for subsequent turns.
            result = PipelineResult(**{k: v for k, v in output.items() if k in {"question", "steps"}})
            context = _update_context(context, result)
    except PipelineError as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="classifier.cli",
        description="Taiwan-stock query understanding, classification, and visualization CLI.",
    )
    parser.add_argument("question", nargs="?", help="Natural language question (one-shot mode)")
    parser.add_argument("--chat", action="store_true", help="Run interactive chat mode")
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Disable chart rendering even when visualization is requested",
    )
    args = parser.parse_args(argv)

    render_charts = not args.no_chart

    if args.chat:
        return _run_chat(render_charts)
    if args.question:
        return _run_oneshot(args.question, render_charts)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
