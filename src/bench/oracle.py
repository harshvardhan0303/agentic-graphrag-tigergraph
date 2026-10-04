"""Graph correctness oracle.

This is NOT an answering pipeline. It is a test harness that drives the graph
store through the same tool calls the agent makes, using regex-extracted
arguments instead of an LLM planner. It answers one question only:

    "Is the graph loaded correctly and does the tool layer return the right facts?"

Because it removes the LLM from the loop, any failure here is a data or query
bug, not a prompting bug. We run it after every ingest and after every
TigerGraph load. It is also how we measure the accuracy ceiling of the graph.

Usage:
    python -m src.bench.oracle --questions data/eval_public.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from ..graph.store import LocalGraphStore
from ..ingest.normalize import norm

Q_AGG = re.compile(
    r"how many (?P<sport>.+?) events at the (?P<year>\d{4}) (?P<season>Summer|Winter) "
    r"Olympics had more than (?P<n>\d+) competitors", re.I)
Q_SUP = re.compile(
    r"which (?P<sport>.+?) event at the (?P<year>\d{4}) (?P<season>Summer|Winter) "
    r"Olympics had the (?P<dir>highest|lowest) number of competitors", re.I)
Q_LOOKUP = re.compile(r"how many nations competed in (?P<title>.+?)\?*$", re.I)
Q_MULTIHOP = re.compile(
    r"event held at (?P<venue>.+?) on (?P<date>.+?)"
    r"(?: at the (?P<year>\d{4}) (?P<season>Summer|Winter) Olympics)?\?*$", re.I)
Q_TEMPORAL = re.compile(
    r"gold medal in the (?P<event>.+?) event at the (?P<season>Summer|Winter) "
    r"Olympics held immediately before (?P<year>\d{4})", re.I)


def solve(store: LocalGraphStore, question: str) -> tuple[str | None, list[str], list[str]]:
    """Return (answer, evidence_doc_ids, tool_trace)."""
    trace: list[str] = []

    if m := Q_AGG.search(question):
        trace.append(f"count_events_where(sport={m['sport']!r}, year={m['year']}, "
                     f"season={m['season']!r}, field='competitors', op='>', value={m['n']})")
        r = store.count_events_where(m["sport"], int(m["year"]), m["season"],
                                     "competitors", ">", int(m["n"]))
        return str(r["count"]), r["evidence_doc_ids"], trace

    if m := Q_SUP.search(question):
        direction = "max" if m["dir"].lower() == "highest" else "min"
        trace.append(f"extreme_event(sport={m['sport']!r}, year={m['year']}, "
                     f"season={m['season']!r}, field='competitors', direction={direction!r})")
        r = store.extreme_event(m["sport"], int(m["year"]), m["season"],
                                "competitors", direction)
        ev = r["event"]
        return (ev["title"] if ev else None), r["evidence_doc_ids"], trace

    if m := Q_LOOKUP.search(question):
        trace.append(f"find_events(title={m['title']!r})")
        hits = store.find_events(title=m["title"])
        if not hits:
            return None, [], trace
        return str(hits[0].nations), [hits[0].doc_id], trace

    if "event held at" in question.lower() and (m := Q_MULTIHOP.search(question)):
        year = int(m["year"]) if m["year"] else None
        trace.append(f"events_by_venue_date(venue={m['venue']!r}, date={m['date']!r}, year={year})")
        hits = store.events_by_venue_date(m["venue"], m["date"], year)
        if not hits:
            return None, [], trace
        trace.append(f"medalists(doc_id={hits[0].doc_id!r}, medal='gold')")
        gold = store.medalists(hits[0].doc_id, "gold")
        return "".join(g["name"] for g in gold), [hits[0].doc_id], trace

    if m := Q_TEMPORAL.search(question):
        year, season = int(m["year"]), m["season"]
        evidence: list[str] = []
        # 1. link the event phrase to the Games named in the question (the anchor)
        trace.append(f"resolve_event_phrase({m['event']!r}, season={season!r})")
        anchors = [e for e in store.resolve_event_phrase(m["event"], season) if e.year == year]
        target = None
        if anchors:
            anchor = anchors[0]
            evidence.append(anchor.doc_id)
            # 2. traverse Games.PRECEDES backwards
            prev_games = store.previous_games(year, season)
            trace.append(f"previous_games({year}, {season!r}) -> {prev_games!r}")
            if prev_games:
                trace.append(f"counterpart_in_games({anchor.doc_id!r}, {prev_games!r})")
                target = store.counterpart_in_games(anchor.doc_id, prev_games)
        if target is None:
            # anchor Games absent from corpus: fall back to most recent prior staging
            trace.append("fallback: resolve_event_phrase(before_year=%d)" % year)
            hits = store.resolve_event_phrase(m["event"], season, year)
            if not hits:
                return None, evidence, trace
            target = hits[0]
        evidence.append(target.doc_id)
        trace.append(f"medalists(doc_id={target.doc_id!r}, medal='gold')")
        gold = store.medalists(target.doc_id, "gold")
        return "".join(g["name"] for g in gold), evidence, trace

    return None, [], trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/graph")
    ap.add_argument("--questions", default="data/eval_public.jsonl")
    ap.add_argument("--show-failures", action="store_true", default=True)
    args = ap.parse_args()

    store = LocalGraphStore(args.graph)
    qs = [json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip()]

    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    doc_recall: list[float] = []
    failures = []

    for q in qs:
        pred, evidence, trace = solve(store, q["question"])
        qt = q["qtype"]
        tally[qt][1] += 1
        if "answer" not in q:
            continue
        ok = pred is not None and any(norm(pred) == norm(g) for g in q["answer"])
        if ok:
            tally[qt][0] += 1
        else:
            failures.append((q["qid"], qt, q["question"], pred, q["answer"], trace))
        if q.get("gold_doc_ids"):
            gold = set(q["gold_doc_ids"])
            doc_recall.append(len(gold & set(evidence)) / len(gold))

    print("=== GRAPH CORRECTNESS GATE (not a pipeline score) ===")
    print("    Checks the graph returns the right facts, with no LLM involved.")
    print("    This number must never be reported as system accuracy.")
    tot = [0, 0]
    for qt in sorted(tally):
        c, n = tally[qt]
        tot[0] += c
        tot[1] += n
        print(f"  {qt:12s} {c:3d}/{n:<3d} {c / n:6.1%}")
    print(f"  {'TOTAL':12s} {tot[0]:3d}/{tot[1]:<3d} {tot[0] / tot[1]:6.1%}"
          "   <- graph correctness, NOT pipeline accuracy")
    if doc_recall:
        print(f"  gold-doc recall (answer-bearing docs): {sum(doc_recall) / len(doc_recall):.1%}")

    if failures and args.show_failures:
        print("\n=== FAILURES ===")
        for qid, qt, text, pred, gold, trace in failures:
            print(f"{qid} [{qt}]\n  Q: {text}\n  pred: {pred!r}\n  gold: {gold}")
            for t in trace:
                print(f"    tool> {t}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
