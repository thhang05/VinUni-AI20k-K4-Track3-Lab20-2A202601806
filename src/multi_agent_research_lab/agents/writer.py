"""Writer agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a technical writer. Synthesize research notes and analysis into a clear, "
    "well-structured final answer for the given audience. Keep bracketed source ids as "
    "inline citations. End with a 'Sources' list mapping each id to its title."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        with trace_span("writer.run", {}) as span:
            source_list = "\n".join(
                f"[{src.metadata.get('source_id', idx)}] {src.title}"
                for idx, src in enumerate(state.sources)
            )
            user_prompt = (
                f"Research question: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes or '(none)'}\n\n"
                f"Analysis:\n{state.analysis_notes or '(none)'}\n\n"
                f"Available sources:\n{source_list or '(none)'}\n\n"
                "Write the final answer now."
            )
            response = self._llm_client.complete(_SYSTEM_PROMPT, user_prompt, temperature=0.4)
            state.final_answer = response.content

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=state.final_answer,
                    metadata={
                        "cost_usd": response.cost_usd,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    },
                )
            )
        state.record_span(span)
        return state
