"""Specialist agents.

Each one is a thin, single-purpose LLM call with a strict output contract. They
are separate agents rather than one mega-prompt because the benchmark measures
*which* agents fire on *which* question types - that signal is lost if planning,
evaluation and answering share a prompt.

  StrategyRouter    - classify the question, pick the cheapest viable strategy
  EvidenceEvaluator - is what we have sufficient? if not, what is missing?
  Answerer          - turn evidence into the final short answer + citations
"""
from __future__ import annotations

import json
from typing import Any

from ..llm.client import LLMResponse, UsageMeter
from . import prompts
from .harness import EvidenceLedger


def _safe_json(resp: LLMResponse, default: dict[str, Any]) -> dict[str, Any]:
    try:
        out = resp.json()
        return out if isinstance(out, dict) else default
    except (json.JSONDecodeError, ValueError, IndexError):
        return default


class StrategyRouter:
    """Cost-aware routing.

    This is the piece that answers the hackathon's actual research question:
    rather than running a full investigation on every question, the router
    predicts how much machinery a question needs. The orchestrator may still
    escalate - that escalation is recorded as a strategy change in the trace.
    """

    name = "strategy_router"

    def __init__(self, llm, meter: UsageMeter, pipeline: str = "agentic"):
        self.llm, self.meter, self.pipeline = llm, meter, pipeline

    def route(self, qid: str, question: str, sports: list[str]) -> dict[str, Any]:
        prompt = (
            f"QUESTION: {question}\n\n"
            f"Sports present in the graph: {', '.join(sports)}\n\n"
            "Classify and extract entities."
        )
        resp = self.llm.complete(prompt, system=prompts.ROUTER_SYSTEM, json_mode=True,
                                 max_output_tokens=400)
        self.meter.record(self.pipeline, qid, "router", resp)
        out = _safe_json(resp, {"strategy": "graph_multihop", "confidence": 0.3,
                                "reasoning": "router parse failure",
                                "sub_questions": [], "entities": {}})
        out.setdefault("entities", {})
        out.setdefault("sub_questions", [])
        out["tokens"] = resp.total_tokens
        return out


class EvidenceEvaluator:
    name = "evidence_evaluator"

    def __init__(self, llm, meter: UsageMeter, pipeline: str = "agentic"):
        self.llm, self.meter, self.pipeline = llm, meter, pipeline

    def evaluate(self, qid: str, question: str,
                 evidence: EvidenceLedger) -> dict[str, Any]:
        if len(evidence) == 0:
            return {"sufficient": False, "confidence": 1.0,
                    "gaps": ["no evidence collected yet"],
                    "reasoning": "nothing retrieved", "tokens": 0}
        prompt = (f"QUESTION: {question}\n\nEVIDENCE:\n{evidence.summarize()}\n\n"
                  "Is this sufficient to answer exactly?")
        resp = self.llm.complete(prompt, system=prompts.EVALUATOR_SYSTEM,
                                 json_mode=True, max_output_tokens=300)
        self.meter.record(self.pipeline, qid, "evaluator", resp)
        out = _safe_json(resp, {"sufficient": False, "confidence": 0.3,
                                "gaps": ["evaluator parse failure"],
                                "reasoning": ""})
        out["tokens"] = resp.total_tokens
        return out


class Answerer:
    name = "answerer"

    def __init__(self, llm, meter: UsageMeter, pipeline: str = "agentic"):
        self.llm, self.meter, self.pipeline = llm, meter, pipeline

    def answer(self, qid: str, question: str, evidence: EvidenceLedger,
               system: str | None = None, context: str | None = None
               ) -> dict[str, Any]:
        body = context if context is not None else evidence.summarize(max_chars=8000)
        prompt = f"QUESTION: {question}\n\nEVIDENCE:\n{body}\n\nAnswer."
        resp = self.llm.complete(prompt, system=system or prompts.ANSWERER_SYSTEM,
                                 json_mode=True, max_output_tokens=400)
        self.meter.record(self.pipeline, qid, "answerer", resp)
        out = _safe_json(resp, {"answer": "unknown", "citations": [],
                                "confidence": 0.0})
        out.setdefault("citations", [])
        out["tokens"] = resp.total_tokens
        return out
