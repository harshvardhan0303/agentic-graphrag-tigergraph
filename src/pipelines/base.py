"""Shared pipeline plumbing so the three approaches are compared fairly.

Fairness rules enforced here:
  * identical corpus, identical chunking, identical embedder
  * identical answer-format instructions (only the *context* differs)
  * identical token accounting, through the same UsageMeter
  * every pipeline emits the same Answer shape, including a trace
"""
from __future__ import annotations

import time
from typing import Any, Protocol

from ..agents.harness import Answer
from ..llm.client import UsageMeter


class Pipeline(Protocol):
    name: str
    def answer(self, qid: str, question: str) -> Answer: ...


def empty_trace(pipeline: str, qid: str, steps: list[dict[str, Any]],
                stop_reason: str, started: float) -> dict[str, Any]:
    return {
        "qid": qid,
        "pipeline": pipeline,
        "n_steps": len(steps),
        "tools_called": [s.get("action", "") for s in steps],
        "agents_invoked": sorted({s.get("agent", "") for s in steps}),
        "strategy_changed": False,
        "strategy_changes": [],
        "stop_reason": stop_reason,
        "wall_time_s": round(time.time() - started, 3),
        "steps": steps,
    }
