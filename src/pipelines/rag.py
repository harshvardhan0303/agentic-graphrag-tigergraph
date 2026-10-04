"""Pipeline 1 - plain RAG.

One similarity search, one generation. No graph, no planning, no second look.
This is the honest baseline: it is what most production "chat with your docs"
systems actually do, and the benchmark exists to show precisely where it breaks.
"""
from __future__ import annotations

import time

from ..agents.harness import Answer
from ..agents.prompts import RAG_SYSTEM
from ..agents.specialists import Answerer
from ..llm.client import UsageMeter
from .base import empty_trace


class RAGPipeline:
    name = "rag"

    def __init__(self, llm, vectors, meter: UsageMeter, top_k: int = 8,
                 max_context_chars: int = 9000):
        self.llm = llm
        self.vectors = vectors
        self.meter = meter
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.answerer = Answerer(llm, meter, self.name)

    def answer(self, qid: str, question: str) -> Answer:
        started = time.time()
        t0 = time.time()
        hits = self.vectors.search(question, k=self.top_k)
        retrieval_s = time.time() - t0

        blocks, used, doc_ids = [], 0, []
        for h in hits:
            block = f"[{h['doc_id']}] {h['title']}\n{h['text']}"
            if used + len(block) > self.max_context_chars:
                break
            blocks.append(block)
            used += len(block)
            if h["doc_id"] not in doc_ids:
                doc_ids.append(h["doc_id"])
        context = "\n\n---\n\n".join(blocks) if blocks else "(no passages retrieved)"

        result = self.answerer.answer(qid, question, evidence=None,  # type: ignore[arg-type]
                                      system=RAG_SYSTEM, context=context)

        steps = [{
            "n": 1, "agent": "similarity_search_agent", "action": "similarity_search",
            "args": {"query": question, "k": self.top_k},
            "rationale": "fixed single-shot retrieval",
            "result_summary": f"{len(hits)} chunks, {used} context chars",
            "ok": bool(hits), "latency_s": round(retrieval_s, 3),
            "input_tokens": 0, "output_tokens": 0, "new_evidence": len(hits),
        }]
        return Answer(
            qid=qid, pipeline=self.name, question=question,
            answer=str(result.get("answer", "")).strip(),
            citations=doc_ids,                  # harness.CITATION_RULE
            claimed_citations=list(result.get("citations") or []),
            confidence=float(result.get("confidence", 0.0) or 0.0),
            trace=empty_trace(self.name, qid, steps, "fixed_pipeline_complete", started),
            usage=self.meter.totals(self.name, qid),
        )
