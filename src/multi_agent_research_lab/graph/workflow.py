"""LangGraph workflow.

Keep orchestration here; keep agent internals in `agents/`.
"""

from collections.abc import Callable

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_CRITIC,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState

_WORKER_NODES = (ROUTE_RESEARCHER, ROUTE_ANALYST, ROUTE_WRITER, ROUTE_CRITIC)


def _make_node(agent: BaseAgent) -> Callable[[ResearchState], ResearchState]:
    def node(state: ResearchState) -> ResearchState:
        try:
            return agent.run(state)
        except AgentExecutionError as exc:
            state.errors.append(f"{agent.name}: {exc}")
            return state

    return node


def _route_after_supervisor(state: ResearchState) -> str:
    last_route = state.route_history[-1] if state.route_history else ROUTE_DONE
    return last_route if last_route in _WORKER_NODES else END


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self._supervisor = supervisor or SupervisorAgent()
        self._researcher = researcher or ResearcherAgent()
        self._analyst = analyst or AnalystAgent()
        self._writer = writer or WriterAgent()
        self._critic = critic or CriticAgent()
        self._compiled: CompiledStateGraph[ResearchState, None, ResearchState, ResearchState] = (
            self.build()
        )

    def build(self) -> CompiledStateGraph[ResearchState, None, ResearchState, ResearchState]:
        """Create a LangGraph graph.

        Nodes: supervisor, researcher, analyst, writer, critic. Supervisor is the hub:
        every worker returns to supervisor, and supervisor conditionally routes to the
        next worker or to END once `done` (or the iteration/failure guardrail) fires.
        """

        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", _make_node(self._supervisor))  # type: ignore[call-overload]
        graph.add_node(ROUTE_RESEARCHER, _make_node(self._researcher))  # type: ignore[call-overload]
        graph.add_node(ROUTE_ANALYST, _make_node(self._analyst))  # type: ignore[call-overload]
        graph.add_node(ROUTE_WRITER, _make_node(self._writer))  # type: ignore[call-overload]
        graph.add_node(ROUTE_CRITIC, _make_node(self._critic))  # type: ignore[call-overload]

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            _route_after_supervisor,
            {
                ROUTE_RESEARCHER: ROUTE_RESEARCHER,
                ROUTE_ANALYST: ROUTE_ANALYST,
                ROUTE_WRITER: ROUTE_WRITER,
                ROUTE_CRITIC: ROUTE_CRITIC,
                END: END,
            },
        )
        for worker in _WORKER_NODES:
            graph.add_edge(worker, "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""

        result = self._compiled.invoke(state, config={"recursion_limit": 100})
        return ResearchState.model_validate(result)
