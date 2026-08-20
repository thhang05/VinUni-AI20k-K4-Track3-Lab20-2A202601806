from multi_agent_research_lab.services.search_client import SearchClient


def test_search_returns_relevant_offline_sources() -> None:
    client = SearchClient()
    results = client.search("multi-agent debate and voting conflict resolution", max_results=3)
    assert results
    assert len(results) <= 3
    for doc in results:
        assert doc.title
        assert doc.snippet
        assert "source_id" in doc.metadata


def test_search_respects_max_results() -> None:
    client = SearchClient()
    results = client.search("agent observability tracing debugging", max_results=2)
    assert len(results) <= 2
