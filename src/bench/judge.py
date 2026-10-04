"""LLM-as-judge with a free fast path.

Exact match is checked first; the judge is only invoked on disagreements. That
keeps grading cheap and keeps judge tokens out of the pipelines' token budgets
(they are metered separately under the pipeline name ``judge``).
"""
from __future__ import annotations

from typing import Any

from ..agents.prompts import JUDGE_SYSTEM
from ..agents.specialists import _safe_json
from ..llm.client import UsageMeter
from .metrics import exact_match


class Judge:
    def __init__(self, llm, meter: UsageMeter | None = None):
        self.llm = llm
        self.meter = meter or UsageMeter()
        self.calls = 0

    def grade(self, qid: str, question: str, pred: str,
              golds: list[str]) -> dict[str, Any]:
        if exact_match(pred, golds):
            return {"verdict": "PASS", "reason": "exact match", "via": "exact"}
        if not (pred or "").strip():
            return {"verdict": "FAIL", "reason": "empty answer", "via": "exact"}

        prompt = (f"QUESTION: {question}\n"
                  f"GOLD ANSWER: {' | '.join(golds)}\n"
                  f"PREDICTED ANSWER: {pred}\n\nGrade it.")
        resp = self.llm.complete(prompt, system=JUDGE_SYSTEM, json_mode=True,
                                 max_output_tokens=120)
        self.meter.record("judge", qid, "grade", resp)
        self.calls += 1
        out = _safe_json(resp, {"verdict": "FAIL", "reason": "judge parse failure"})
        out["via"] = "llm"
        return out
