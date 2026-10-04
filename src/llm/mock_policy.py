"""A rule-based stand-in for the LLM.

Purpose: run the *entire* agentic loop -- router, orchestrator, evaluator,
answerer, judge -- deterministically, offline, with no API key. This is what
CI executes, and it is how we test the harness (budgets, escalation, stopping,
tracing) separately from prompt quality.

It is NOT part of the submitted pipelines. The benchmark numbers in the
dashboard come from the real model; ``--backend mock`` exists so that a judge
cloning the repo can verify the machinery runs before spending a token.
"""
from __future__ import annotations

import json
import re

_SPORTS_HINT = re.compile(r"Sports in the graph: (.+)")
_Q = re.compile(r"QUESTION: (.+)")

AGG = re.compile(r"how many (?P<sport>.+?) events at the (?P<year>\d{4}) "
                 r"(?P<season>Summer|Winter) Olympics had more than (?P<n>\d+) competitors", re.I)
SUP = re.compile(r"which (?P<sport>.+?) event at the (?P<year>\d{4}) "
                 r"(?P<season>Summer|Winter) Olympics had the highest number of competitors", re.I)
LOOK = re.compile(r"how many nations competed in (?P<title>.+?)\?", re.I)
MH = re.compile(r"event held at (?P<venue>.+?) on (?P<date>.+?)"
                r"(?: at the (?P<year>\d{4}) (?P<season>Summer|Winter) Olympics)?\?", re.I)
TEMP = re.compile(r"gold medal in the (?P<event>.+?) event at the "
                  r"(?P<season>Summer|Winter) Olympics held immediately before (?P<year>\d{4})", re.I)


def _question(prompt: str) -> str:
    m = _Q.search(prompt)
    return m.group(1).strip() if m else ""


def _entities(q: str) -> tuple[str, dict]:
    if m := AGG.search(q):
        return "graph_direct", {"sport": m["sport"], "year": int(m["year"]),
                                "season": m["season"], "threshold": int(m["n"]),
                                "field": "competitors", "comparison": ">"}
    if m := SUP.search(q):
        return "graph_direct", {"sport": m["sport"], "year": int(m["year"]),
                                "season": m["season"], "field": "competitors"}
    if m := LOOK.search(q):
        return "graph_direct", {"exact_title": m["title"]}
    if "event held at" in q.lower() and (m := MH.search(q)):
        return "graph_multihop", {"venue": m["venue"], "date": m["date"],
                                  "year": int(m["year"]) if m["year"] else 0,
                                  "season": m["season"] or ""}
    if m := TEMP.search(q):
        return "graph_multihop", {"event_phrase": m["event"], "season": m["season"],
                                  "year": int(m["year"])}
    return "hybrid", {}


def default_handler(prompt: str, system: str | None) -> str:
    sys_txt = system or ""
    q = _question(prompt)

    # ---- router ----
    if "STRATEGY ROUTER" in sys_txt:
        strategy, ent = _entities(q)
        return json.dumps({"strategy": strategy, "confidence": 0.9,
                           "reasoning": "template match", "sub_questions": [],
                           "entities": ent})

    # ---- evidence evaluator ----
    if "EVIDENCE EVALUATOR" in sys_txt:
        body = prompt.split("EVIDENCE:", 1)[-1]
        sufficient = any(k in body for k in
                         ("won by", "resolves to", "of ", "with the highest"))
        return json.dumps({"sufficient": bool(sufficient and body.strip()),
                           "confidence": 0.8,
                           "gaps": [] if sufficient else ["need the answering fact"],
                           "reasoning": "mock"})

    # ---- orchestrator ----
    if "ORCHESTRATOR" in sys_txt:
        ent_m = re.search(r"use these verbatim as tool arguments\): (\{.*?\})\n", prompt, re.S)
        ent = json.loads(ent_m.group(1)) if ent_m else {}
        evidence_block = prompt.split("EVIDENCE SO FAR:", 1)[-1].split("\n\n", 1)[0]
        have = evidence_block.strip() and evidence_block.strip() != "(none)"

        if "threshold" in ent:
            if have:
                return json.dumps({"reasoning": "have the count", "action": "answer"})
            return json.dumps({"reasoning": "exact aggregation in graph",
                               "action": "count_events",
                               "args": {"sport": ent["sport"], "year": ent["year"],
                                        "season": ent["season"], "field": "competitors",
                                        "op": ">", "value": ent["threshold"]}})
        if "exact_title" in ent:
            if have:
                return json.dumps({"reasoning": "have the event", "action": "answer"})
            return json.dumps({"reasoning": "direct lookup", "action": "link_event",
                               "args": {"phrase": "", "exact_title": ent["exact_title"]}})
        if "venue" in ent:
            if "won by" in prompt:
                return json.dumps({"reasoning": "have medalist", "action": "answer"})
            if have:
                m = re.search(r"\(events_at_venue_on_date\).*?<docs: ([^,>]+)", prompt)
                if m:
                    return json.dumps({"reasoning": "read medalists",
                                       "action": "get_medalists",
                                       "args": {"doc_id": m.group(1), "medal": "gold"}})
            return json.dumps({"reasoning": "venue+date traversal",
                               "action": "events_at_venue_on_date",
                               "args": {"venue": ent["venue"], "date": ent["date"],
                                        "year": ent.get("year", 0)}})
        if "event_phrase" in ent:
            if "won by" in prompt:
                return json.dumps({"reasoning": "have medalist", "action": "answer"})
            if "counterpart event is" in prompt:
                m = re.search(r"\(previous_games_event\).*?<docs: [^,>]+,([^,>]+)", prompt)
                if m:
                    return json.dumps({"reasoning": "read medalists",
                                       "action": "get_medalists",
                                       "args": {"doc_id": m.group(1), "medal": "gold"}})
            if have:
                m = re.search(r"\(link_event\).*?<docs: ([^,>]+)", prompt)
                if m:
                    return json.dumps({"reasoning": "hop to previous Games",
                                       "action": "previous_games_event",
                                       "args": {"doc_id": m.group(1)}})
            return json.dumps({"reasoning": "link the event phrase",
                               "action": "link_event",
                               "args": {"phrase": ent["event_phrase"],
                                        "season": ent.get("season", ""),
                                        "at_year": ent.get("year", 0)}})
        if "sport" in ent:
            if have:
                return json.dumps({"reasoning": "have the superlative", "action": "answer"})
            return json.dumps({"reasoning": "superlative in graph", "action": "top_event",
                               "args": {"sport": ent["sport"], "year": ent["year"],
                                        "season": ent["season"], "field": "competitors",
                                        "direction": "max"}})
        if have:
            return json.dumps({"reasoning": "fall back to answering", "action": "answer"})
        return json.dumps({"reasoning": "no structured entities; search text",
                           "action": "similarity_search",
                           "args": {"query": q, "k": 5}})

    # ---- judge ----
    if "grading short factual answers" in sys_txt:
        return json.dumps({"verdict": "FAIL", "reason": "mock judge"})

    # ---- answerers ----
    body = prompt.split("EVIDENCE:", 1)[-1]
    cites = re.findall(r"<docs: ([^>]+)>", body)
    citations: list[str] = []
    for c in cites:
        citations.extend(x.strip() for x in c.replace("...", "").split(",") if x.strip())

    ans = ""
    if m := re.search(r"(\d+) of \d+ \S+ events at the \d{4} \w+ Olympics have", body):
        ans = m.group(1)
    elif m := re.search(r" is '(.+?)' \(competitors=", body):
        ans = m.group(1)
    elif m := re.search(r"won by ([^\n<]+)", body):
        ans = m.group(1).strip()
    elif m := re.search(r"nations=(\d+)", body):
        ans = m.group(1)
    if not ans:
        ans = "unknown"
    return json.dumps({"answer": ans, "citations": citations[:20], "confidence": 0.8})
