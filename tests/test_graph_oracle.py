"""Graph correctness: the oracle must stay at 100% on the public set.

This is the regression gate. If a change to parsing, normalisation or the graph
queries breaks a question type, this fails before any LLM is involved.
"""
import json
import pathlib

import pytest

from src.bench.oracle import solve
from src.graph.store import LocalGraphStore
from src.ingest.normalize import norm

GRAPH = pathlib.Path("data/graph")
QUESTIONS = pathlib.Path("data/eval_public.jsonl")

pytestmark = pytest.mark.skipif(
    not (GRAPH / "event.csv").exists() or not QUESTIONS.exists(),
    reason="run `make ingest` first (corpus is not committed)")


@pytest.fixture(scope="module")
def store():
    return LocalGraphStore(GRAPH)


@pytest.fixture(scope="module")
def questions():
    return [json.loads(l) for l in open(QUESTIONS, encoding="utf-8") if l.strip()]


def test_graph_loaded(store):
    assert len(store.events) == 2162
    assert len({e.sport for e in store.events.values()}) == 41


def test_oracle_is_perfect_on_public_set(store, questions):
    failures = []
    for q in questions:
        pred, _evidence, _trace = solve(store, q["question"])
        if pred is None or not any(norm(pred) == norm(g) for g in q["answer"]):
            failures.append((q["qid"], q["qtype"], pred, q["answer"]))
    assert not failures, f"{len(failures)} oracle failures: {failures[:5]}"


def test_evidence_covers_gold_documents(store, questions):
    """Completeness gate: the graph must surface every document the benchmark
    marks as required, not just the one holding the answer."""
    recalls = []
    for q in questions:
        if not q.get("gold_doc_ids"):
            continue
        _pred, evidence, _trace = solve(store, q["question"])
        gold = set(q["gold_doc_ids"])
        recalls.append(len(gold & set(evidence)) / len(gold))
    assert sum(recalls) / len(recalls) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("qtype", ["lookup", "temporal", "multi_hop",
                                   "aggregation", "superlative"])
def test_each_question_type_is_perfect(store, questions, qtype):
    subset = [q for q in questions if q["qtype"] == qtype]
    assert subset
    for q in subset:
        pred, _e, _t = solve(store, q["question"])
        assert pred is not None and any(norm(pred) == norm(g) for g in q["answer"]), \
            f"{q['qid']}: {pred!r} != {q['answer']}"


def test_venue_date_collision_is_disambiguated(store):
    """Two 2014 events share 'Laura Biathlon & Ski Complex' on 22 February.
    The venue name itself is the disambiguating signal."""
    hits = store.events_by_venue_date("Laura Biathlon & Ski Complex",
                                      "22 February 2014")
    assert len(hits) == 1
    assert hits[0].sport == "Biathlon"


def test_precedes_edge_supports_temporal_hop(store):
    assert store.previous_games(2016, "Summer") == "2012 Summer"
    assert store.previous_games(2014, "Winter") == "2010 Winter"
