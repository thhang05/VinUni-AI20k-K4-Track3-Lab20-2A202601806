"""Supervisor / router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_CRITIC = "critic"
ROUTE_DONE = "done"

_AGENT_ORDER = (ROUTE_RESEARCHER, ROUTE_ANALYST, ROUTE_WRITER, ROUTE_CRITIC)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, max_iterations: int | None = None) -> None:
        self._max_iterations = max_iterations or get_settings().max_iterations

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Policy:
        - researcher runs first (need sources + notes before anything else)
        - analyst runs once research_notes exist
        - writer runs once analysis_notes exist
        - critic runs once after a final_answer exists, to check citation coverage
        - done once critic has run, or when max_iterations is hit (fallback stop),
          or when an agent has already failed twice in a row (avoid infinite retry loop)
        """

        with trace_span("supervisor.run", {"iteration": state.iteration}) as span:
            next_route = self._decide(state)
            state.record_route(next_route)
            span["attributes"]["next_route"] = next_route
        state.record_span(span)
        return state

    def _decide(self, state: ResearchState) -> str:
        if state.iteration >= self._max_iterations:
            return ROUTE_DONE

        if self._repeated_failure(state):
            return ROUTE_DONE

        if not state.sources or not state.research_notes:
            return ROUTE_RESEARCHER
        if not state.analysis_notes:
            return ROUTE_ANALYST
        if not state.final_answer:
            return ROUTE_WRITER
        if not self._has_run(state, "critic"):
            return ROUTE_CRITIC
        return ROUTE_DONE

    @staticmethod
    def _has_run(state: ResearchState, agent_name: str) -> bool:
        return any(result.agent == agent_name for result in state.agent_results)

    @staticmethod
    def _repeated_failure(state: ResearchState) -> bool:
        """Stop instead of looping forever if the last two routes were identical
        worker steps that both failed to advance the state (e.g. researcher kept
        being re-selected because no sources were ever produced)."""

        history = state.route_history
        if len(history) < 2:
            return False
        last_two = history[-2:]
        if last_two[0] != last_two[1] or last_two[0] not in _AGENT_ORDER:
            return False
        return bool(state.errors)
