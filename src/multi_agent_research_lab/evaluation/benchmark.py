"""Benchmark harness for single-agent vs multi-agent runs."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_-]+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _citation_coverage(state: ResearchState) -> float | None:
    answer = state.final_answer
    if not answer:
        return None
    sentences = [s for s in _SENTENCE_RE.split(answer) if s.strip()]
    claim_sentences = [s for s in sentences if len(s.split()) > 4]
    if not claim_sentences:
        return None
    cited = [s for s in claim_sentences if _CITATION_RE.search(s)]
    return len(cited) / len(claim_sentences)


def _total_cost(state: ResearchState) -> float | None:
    costs: list[float] = [
        float(cost)
        for result in state.agent_results
        if (cost := result.metadata.get("cost_usd")) is not None
    ]
    return sum(costs) if costs else None


def _quality_score(state: ResearchState) -> float | None:
    """0-10 heuristic quality proxy: rewards a non-empty answer with citations and
    no recorded errors. Intended as a placeholder for human/rubric-based peer review
    (see docs/peer_review_rubric.md) — not a substitute for it.
    """

    if not state.final_answer:
        return 0.0
    score = 5.0
    coverage = _citation_coverage(state)
    if coverage is not None:
        score += coverage * 4
    if not state.errors:
        score += 1
    return min(score, 10.0)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run `runner(query)`, measure latency, and derive cost/quality/citation metrics
    from the resulting state.
    """

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001 - benchmark must capture any runner failure
        latency = perf_counter() - started
        failed_state = ResearchState(request=ResearchQuery(query=query))
        failed_state.errors.append(f"runner raised {type(exc).__name__}: {exc}")
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            failure_rate=1.0,
            notes=failed_state.errors[0],
        )
        return failed_state, metrics
    latency = perf_counter() - started

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_total_cost(state),
        quality_score=_quality_score(state),
        citation_coverage=_citation_coverage(state),
        failure_rate=1.0 if state.errors else 0.0,
        notes="; ".join(state.errors) if state.errors else "",
    )
    return state, metrics
