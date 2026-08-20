"""Critic agent: fact-check and citation coverage review."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_-]+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class CriticAgent(BaseAgent):
    """Validates the final answer's citation coverage against known source ids."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""

        with trace_span("critic.run", {}) as span:
            answer = state.final_answer or ""
            known_ids = {src.metadata.get("source_id") for src in state.sources}
            sentences = [s for s in _SENTENCE_RE.split(answer) if s.strip()]
            claim_sentences = [s for s in sentences if len(s.split()) > 4]
            cited = [s for s in claim_sentences if _CITATION_RE.search(s)]
            coverage = len(cited) / len(claim_sentences) if claim_sentences else 0.0

            unknown_citations = sorted(
                {cid for cid in _CITATION_RE.findall(answer) if cid not in known_ids}
            )

            findings = [f"citation_coverage={coverage:.0%}"]
            if unknown_citations:
                findings.append(f"unknown_citations={unknown_citations}")
                state.errors.append(f"critic: citations not found in sources: {unknown_citations}")
            if not answer:
                findings.append("final_answer is empty")
                state.errors.append("critic: final_answer is empty")

            span["attributes"]["citation_coverage"] = coverage
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content="; ".join(findings),
                    metadata={
                        "citation_coverage": coverage,
                        "unknown_citations": unknown_citations,
                    },
                )
            )
        state.record_span(span)
        return state
