"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a research analyst. Given research notes, extract the key claims, compare "
    "any conflicting viewpoints, and flag claims with weak or single-source evidence. "
    "Preserve the bracketed source ids from the notes. Be concise and structured."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        with trace_span("analyst.run", {}) as span:
            metadata: dict[str, object] = {}
            if not state.research_notes:
                state.analysis_notes = "No research notes available to analyze."
                state.errors.append("analyst: missing research_notes")
            else:
                user_prompt = (
                    f"Research question: {state.request.query}\n\n"
                    f"Research notes:\n{state.research_notes}\n\n"
                    "Produce:\n"
                    "1. Key claims (bulleted, with source ids)\n"
                    "2. Points of agreement/conflict across sources\n"
                    "3. Claims with weak evidence (single source or low confidence)"
                )
                response = self._llm_client.complete(_SYSTEM_PROMPT, user_prompt, temperature=0.1)
                state.analysis_notes = response.content
                metadata["cost_usd"] = response.cost_usd
                metadata["input_tokens"] = response.input_tokens
                metadata["output_tokens"] = response.output_tokens

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=state.analysis_notes or "",
                    metadata=metadata,
                )
            )
        state.record_span(span)
        return state
