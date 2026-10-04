"""Pipeline 2 - GraphRAG (single fixed retrieval pass).

Structure-aware but NOT agentic. One extraction call pulls entities out of the
question; a fixed retrieval sequence then gathers graph facts and supporting
passages; one generation call answers. Crucially, the retrieval sequence is the
same for every question - it never looks at what came back and decides to try
something else. That difference is the whole experiment.
"""
from __future__ import annotations

import json
import time
from typing import Any

from ..agents.harness import Answer
from ..agents.prompts import GRAPHRAG_SYSTEM, ROUTER_SYSTEM
from ..agents.specialists import Answerer, _safe_json
from ..agents.tools import normalize_op
from ..llm.client import UsageMeter


class GraphRAGPipeline:
    name = "graphrag"

    def __init__(self, llm, registry, vectors, meter: UsageMeter,
                 sports: list[str], top_k: int = 4):
        self.llm = llm
        self.registry = registry
        self.vectors = vectors
        self.meter = meter
        self.sports = sports
        self.top_k = top_k
        self.answerer = Answerer(llm, meter, self.name)

    # ------------------------------------------------------------------
    def _extract(self, qid: str, question: str) -> dict[str, Any]:
        prompt = (f"QUESTION: {question}\n\n"
                  f"Sports present in the graph: {', '.join(self.sports)}\n\n"
                  "Classify and extract entities.")
        resp = self.llm.complete(prompt, system=ROUTER_SYSTEM, json_mode=True,
                                 max_output_tokens=400)
        self.meter.record(self.name, qid, "extract", resp)
        out = _safe_json(resp, {"strategy": "graph_direct", "entities": {}})
        out.setdefault("entities", {})
        return out

    def _call(self, name: str, **kw) -> tuple[list, str]:
        tool = self.registry.get(name)
        if tool is None:
            return [], f"tool {name} unavailable"
        try:
            return tool.fn(**kw)
        except Exception as e:
            return [], f"{name} error: {type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    def answer(self, qid: str, question: str) -> Answer:
        started = time.time()
        extraction = self._extract(qid, question)
        ent = extraction.get("entities") or {}

        def s(key: str) -> str:
            v = ent.get(key)
            return "" if v in (None, 0) else str(v)

        def i(key: str) -> int:
            try:
                return int(ent.get(key) or 0)
            except (TypeError, ValueError):
                return 0

        steps: list[dict[str, Any]] = [{
            "n": 1, "agent": "entity_extractor", "action": "extract_entities",
            "args": {}, "rationale": extraction.get("reasoning", ""),
            "result_summary": json.dumps(ent, ensure_ascii=False)[:400],
            "ok": bool(ent), "latency_s": 0.0, "new_evidence": 0,
        }]

        facts: list[str] = []
        doc_ids: list[str] = []

        def absorb(evidence, summary, agent, action, args):
            nonlocal facts, doc_ids
            if summary:
                facts.append(summary)
            for e in evidence:
                for d in e.doc_ids:
                    if d not in doc_ids:
                        doc_ids.append(d)
            steps.append({"n": len(steps) + 1, "agent": agent, "action": action,
                          "args": args, "rationale": "fixed retrieval pass",
                          "result_summary": (summary or "")[:400],
                          "ok": bool(evidence), "latency_s": 0.0,
                          "new_evidence": len(evidence)})

        # --- fixed retrieval sequence (identical for every question) ---
        if s("exact_title"):
            absorb(*self._call("link_event", phrase="", exact_title=s("exact_title")),
                   agent="entity_linker", action="link_event",
                   args={"exact_title": s("exact_title")})

        if s("venue") and s("date"):
            args = {"venue": s("venue"), "date": s("date"), "year": i("year")}
            ev, sm = self._call("events_at_venue_on_date", **args)
            absorb(ev, sm, "graph_traversal_agent", "events_at_venue_on_date", args)
            for e in ev[:1]:
                did = e.doc_ids[0]
                absorb(*self._call("get_medalists", doc_id=did, medal="gold"),
                       agent="graph_traversal_agent", action="get_medalists",
                       args={"doc_id": did, "medal": "gold"})

        if s("sport") and i("year") and s("season"):
            if i("threshold"):
                args = {"sport": s("sport"), "year": i("year"), "season": s("season"),
                        "field": s("field") or "competitors",
                        "op": normalize_op(s("comparison")), "value": i("threshold")}
                absorb(*self._call("count_events", **args),
                       agent="aggregation_agent", action="count_events", args=args)
            args = {"sport": s("sport"), "year": i("year"), "season": s("season"),
                    "field": s("field") or "competitors", "direction": "max"}
            absorb(*self._call("top_event", **args),
                   agent="aggregation_agent", action="top_event", args=args)

        if s("event_phrase"):
            args = {"phrase": s("event_phrase"), "season": s("season"),
                    "at_year": i("year")}
            ev, sm = self._call("link_event", **args)
            absorb(ev, sm, "entity_linker", "link_event", args)
            for e in ev[:2]:
                did = e.doc_ids[0]
                absorb(*self._call("get_medalists", doc_id=did, medal="gold"),
                       agent="graph_traversal_agent", action="get_medalists",
                       args={"doc_id": did, "medal": "gold"})
                # Fixed one-hop neighbourhood expansion. GraphRAG is not
                # adaptive, but it IS structure-aware: having linked an entity
                # it always walks the adjacent Games and reads that event too.
                # Without this the baseline cannot answer temporal questions at
                # all, which would make the comparison a strawman rather than
                # a test of adaptivity.
                prev_ev, prev_sm = self._call("previous_games_event", doc_id=did)
                absorb(prev_ev, prev_sm, "multi_hop_agent",
                       "previous_games_event", {"doc_id": did})
                for pe in prev_ev[:1]:
                    pdid = pe.doc_ids[-1]
                    absorb(*self._call("get_medalists", doc_id=pdid, medal="gold"),
                           agent="graph_traversal_agent", action="get_medalists",
                           args={"doc_id": pdid, "medal": "gold"})

        # supporting passages, always, to keep the comparison with RAG honest
        t0 = time.time()
        hits = self.vectors.search(question, k=self.top_k) if self.vectors else []
        passages = []
        for h in hits:
            passages.append(f"[{h['doc_id']}] {h['title']}\n{h['text'][:700]}")
            if h["doc_id"] not in doc_ids:
                doc_ids.append(h["doc_id"])
        steps.append({"n": len(steps) + 1, "agent": "similarity_search_agent",
                      "action": "similarity_search",
                      "args": {"query": question, "k": self.top_k},
                      "rationale": "supporting passages",
                      "result_summary": f"{len(hits)} chunks",
                      "ok": bool(hits), "latency_s": round(time.time() - t0, 3),
                      "new_evidence": len(hits)})

        context = "GRAPH FACTS:\n" + ("\n".join(f"- {f}" for f in facts) or "(none)")
        if passages:
            context += "\n\nSUPPORTING PASSAGES:\n" + "\n\n".join(passages)

        result = self.answerer.answer(qid, question, evidence=None,  # type: ignore[arg-type]
                                      system=GRAPHRAG_SYSTEM, context=context)

        from .base import empty_trace
        # harness.CITATION_RULE: cite what the fixed pass actually read. For
        # this pipeline that is a superset by construction - the same traversal
        # runs whatever the question is - which is exactly what we want the
        # precision number to expose.
        return Answer(
            qid=qid, pipeline=self.name, question=question,
            answer=str(result.get("answer", "")).strip(),
            citations=doc_ids,
            claimed_citations=list(result.get("citations") or []),
            confidence=float(result.get("confidence", 0.0) or 0.0),
            trace=empty_trace(self.name, qid, steps, "fixed_pipeline_complete", started),
            usage=self.meter.totals(self.name, qid),
        )
