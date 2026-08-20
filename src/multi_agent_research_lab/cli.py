"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    BenchmarkMetrics,
    ResearchQuery,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()

_BASELINE_SYSTEM_PROMPT = (
    "You are a single research assistant with no external tools. Answer the user's "
    "research question directly and concisely for the stated audience, using your own "
    "knowledge. Note where you are uncertain."
)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _run_baseline(query: str) -> ResearchState:
    request = _parse_query(query)
    state = ResearchState(request=request)
    llm_client = LLMClient()
    response = llm_client.complete(
        _BASELINE_SYSTEM_PROMPT,
        f"Audience: {request.audience}\n\nQuestion: {request.query}",
        temperature=0.3,
    )
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "cost_usd": response.cost_usd,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
    )
    return state


def _run_multi_agent(query: str) -> ResearchState:
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline: one direct LLM call, no tools."""

    _init()
    try:
        state = _run_baseline(query)
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="LLM Error", style="red"))
        raise typer.Exit(code=1) from exc
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow (Supervisor -> Researcher/Analyst/Writer/Critic)."""

    _init()
    try:
        result = _run_multi_agent(query)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="LLM Error", style="red"))
        raise typer.Exit(code=1) from exc
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    queries: Annotated[
        list[str] | None,
        typer.Option("--query", "-q", help="Query to benchmark (repeatable)."),
    ] = None,
    output: Annotated[
        str, typer.Option("--output", help="Report path relative to reports/")
    ] = "benchmark_report.md",
) -> None:
    """Run baseline and multi-agent on each query and write a comparison report."""

    _init()
    query_list = queries or [
        "Research GraphRAG state-of-the-art and write a 500-word summary",
        "Compare single-agent and multi-agent workflows for customer support",
        "Summarize production guardrails for LLM agents",
    ]

    all_metrics: list[BenchmarkMetrics] = []
    for query in query_list:
        console.print(Panel.fit(query, title="Benchmarking query"))
        _, baseline_metrics = run_benchmark("baseline", query, _run_baseline)
        all_metrics.append(baseline_metrics)
        _, multi_metrics = run_benchmark("multi-agent", query, _run_multi_agent)
        all_metrics.append(multi_metrics)

    report = render_markdown_report(all_metrics)
    path = LocalArtifactStore().write_text(output, report)
    console.print(Panel.fit(f"Wrote {path}", title="Benchmark complete"))


if __name__ == "__main__":
    app()
