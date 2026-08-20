"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a markdown table plus an aggregate summary."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    lines.extend(["", "## Summary by run name", ""])
    lines.extend(["| Run | Avg latency (s) | Avg quality | Runs |", "|---|---:|---:|---:|"])
    by_name: dict[str, list[BenchmarkMetrics]] = {}
    for item in metrics:
        by_name.setdefault(item.run_name, []).append(item)
    for name, group in by_name.items():
        avg_latency = sum(m.latency_seconds for m in group) / len(group)
        quality_values = [m.quality_score for m in group if m.quality_score is not None]
        avg_quality = f"{sum(quality_values) / len(quality_values):.1f}" if quality_values else ""
        lines.append(f"| {name} | {avg_latency:.2f} | {avg_quality} | {len(group)} |")

    return "\n".join(lines) + "\n"
