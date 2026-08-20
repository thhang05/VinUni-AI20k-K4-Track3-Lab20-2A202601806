"""Unit tests for SupervisorAgent's routing policy.

Replaces the original skeleton-guard test now that SupervisorAgent is implemented
(see docs/lab_guide.md, Milestone 2).
"""

from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_CRITIC,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_routes_to_researcher_first() -> None:
    state = SupervisorAgent().run(_state())
    assert state.route_history == [ROUTE_RESEARCHER]


def test_routes_to_analyst_after_research_notes() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "some notes"
    state = SupervisorAgent().run(state)
    assert state.route_history == [ROUTE_ANALYST]


def test_routes_to_writer_after_analysis() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "some notes"
    state.analysis_notes = "some analysis"
    state = SupervisorAgent().run(state)
    assert state.route_history == [ROUTE_WRITER]


def test_routes_to_critic_after_writer() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    state = SupervisorAgent().run(state)
    assert state.route_history == [ROUTE_CRITIC]


def test_routes_to_done_after_critic_ran() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    state.agent_results.append(AgentResult(agent=AgentName.CRITIC, content="ok"))
    state = SupervisorAgent().run(state)
    assert state.route_history == [ROUTE_DONE]


def test_stops_at_max_iterations() -> None:
    state = _state()
    supervisor = SupervisorAgent(max_iterations=2)
    for _ in range(3):
        state = supervisor.run(state)
    assert state.route_history[-1] == ROUTE_DONE


def test_stops_after_repeated_failure_without_sources() -> None:
    state = _state()
    supervisor = SupervisorAgent(max_iterations=10)
    state.route_history = [ROUTE_RESEARCHER, ROUTE_RESEARCHER]
    state.iteration = 2
    state.errors = ["researcher: no sources returned"]
    state = supervisor.run(state)
    assert state.route_history[-1] == ROUTE_DONE
