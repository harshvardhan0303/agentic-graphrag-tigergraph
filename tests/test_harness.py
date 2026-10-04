"""Harness behaviour: budgets, stopping, escalation, tracing.

Exercised with the mock LLM so the agent loop is tested independently of prompt
quality or network access.
"""
import pathlib

import pytest

from src.agents.harness import Budget, Evidence, EvidenceLedger, InvestigationState
from src.agents.orchestrator import Orchestrator
from src.agents.tools import build_registry
from src.graph.store import LocalGraphStore
from src.llm.client import UsageMeter, build_client

GRAPH = pathlib.Path("data/graph")
pytestmark = pytest.mark.skipif(not (GRAPH / "event.csv").exists(),
                                reason="run `make ingest` first")


@pytest.fixture(scope="module")
def agent():
    store = LocalGraphStore(GRAPH)
    registry = build_registry(store)
    meter = UsageMeter()
    sports = sorted({e.sport for e in store.events.values()})
    llm = build_client("mock")
    return Orchestrator(llm, registry, meter, sports, budget=Budget(max_steps=6))


def test_evidence_ledger_deduplicates():
    led = EvidenceLedger()
    e = Evidence("t", "claim", ["D1"])
    assert led.add(e) is True
    assert led.add(Evidence("t", "claim", ["D1"])) is False
    assert len(led) == 1
    assert led.doc_ids() == ["D1"]


def test_budget_stops_on_steps():
    st = InvestigationState("q", "question", Budget(max_steps=3))
    assert st.should_stop(0) is None
    assert st.should_stop(3) == "step_budget_exhausted"


def test_budget_stops_on_tokens():
    st = InvestigationState("q", "question", Budget(max_tokens=100))
    st.spend(150)
    assert st.should_stop(0) == "token_budget_exhausted"


def test_no_progress_detector():
    st = InvestigationState("q", "question", Budget(max_barren_steps=2))
    st.barren_streak = 2
    assert st.should_stop(0) == "no_progress"


def test_aggregation_is_answered_in_few_steps(agent):
    a = agent.investigate(
        "t1", "According to the provided corpus, how many biathlon events at the "
              "2018 Winter Olympics had more than 73 competitors?")
    assert a.answer == "5"
    assert a.trace["n_steps"] <= 3
    assert "count_events" in a.trace["tools_called"]
    assert a.trace["stop_reason"] == "evidence_sufficient"
    assert len(a.citations) >= 10  # every event scanned is cited


def test_multi_hop_walks_venue_then_medalists(agent):
    a = agent.investigate(
        "t2", "Who won the gold medal in the event held at Olympic Weightlifting "
              "Gymnasium on 20 September 1988?")
    assert "Süleymanoğlu" in a.answer
    tools = [t for t in a.trace["tools_called"] if t != "route"]
    assert tools[:2] == ["events_at_venue_on_date", "get_medalists"]


def test_temporal_uses_precedes_edge(agent):
    a = agent.investigate(
        "t3", "Who won the gold medal in the men's 20 kilometres walk athletics "
              "event at the Summer Olympics held immediately before 2016?")
    assert a.answer == "Chen Ding"
    assert "previous_games_event" in a.trace["tools_called"]


def test_trace_is_serialisable_and_complete(agent):
    a = agent.investigate("t4", "How many nations competed in Judo at the 2016 "
                                "Summer Olympics – Women's 57 kg?")
    t = a.trace
    for key in ("qid", "pipeline", "n_steps", "tools_called", "agents_invoked",
                "strategy_changed", "stop_reason", "wall_time_s", "steps"):
        assert key in t
    assert a.usage["total_tokens"] > 0
    assert all("rationale" in s for s in t["steps"])
