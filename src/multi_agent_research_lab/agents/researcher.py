"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

_SYSTEM_PROMPT = (
    "You are a research assistant. Given source excerpts, write concise, well-organized "
    "research notes. Every factual claim must reference its source using the bracketed "
    "id shown next to that source, e.g. [autogen] or [A03]. Do not invent sources."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self, search_client: SearchClient | None = None, llm_client: LLMClient | None = None
    ) -> None:
        self._search_client = search_client or SearchClient()
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        with trace_span("researcher.run", {"query": state.request.query}) as span:
            sources = self._search_client.search(
                state.request.query, max_results=state.request.max_sources
            )
            state.sources = sources
            span["attributes"]["source_count"] = len(sources)
            metadata: dict[str, object] = {"source_count": len(sources)}

            if not sources:
                state.research_notes = "No sources found for this query."
                state.errors.append("researcher: no sources returned")
            else:
                source_block = "\n\n".join(
                    f"[{src.metadata.get('source_id', idx)}] {src.title}\n{src.snippet}"
                    for idx, src in enumerate(sources)
                )
                user_prompt = (
                    f"Research question: {state.request.query}\n\n"
                    f"Audience: {state.request.audience}\n\n"
                    f"Sources:\n{source_block}\n\n"
                    "Write research notes (bullet points) summarizing what these sources say, "
                    "with a bracketed source id after each claim."
                )
                response = self._llm_client.complete(_SYSTEM_PROMPT, user_prompt, temperature=0.2)
                state.research_notes = response.content
                metadata["cost_usd"] = response.cost_usd
                metadata["input_tokens"] = response.input_tokens
                metadata["output_tokens"] = response.output_tokens

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=state.research_notes or "",
                    metadata=metadata,
                )
            )
        state.record_span(span)
        return state
