"""Agent harness: state, tools, evidence, budget and stopping criteria.

The harness is deliberately independent of any particular LLM or graph backend.
It owns four things the guidebook asks for explicitly:

  state      -- what we know, what we tried, what is still missing
  tools      -- a typed registry the orchestrator can choose from
  evidence   -- an append-only, deduplicated, citable record
  stopping   -- budget, sufficiency, and a no-progress detector

Everything the orchestrator does is appended to a :class:`Trace`, which is
serialised with every answer. That trace is the submission artefact that shows
*why* an answer was reached, and is what the "evidence quality and
explainability" criterion is scored on.
"""
from __future__ import annotations

import dataclasses
import json
import time
from typing import Any, Callable


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@dataclasses.dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, str]          # arg name -> human description
    fn: Callable[..., Any]
    cost_hint: str = "cheap"            # cheap | moderate | expensive
    category: str = "retrieval"

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "cost": self.cost_hint,
            "category": self.category,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def catalog(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------
@dataclasses.dataclass
class Evidence:
    """One citable fact. ``doc_ids`` is what grounds it in the corpus."""
    source: str                     # tool that produced it
    claim: str                      # short natural-language statement
    doc_ids: list[str]
    payload: dict[str, Any] = dataclasses.field(default_factory=dict)

    def key(self) -> str:
        return f"{self.source}|{self.claim}"


class EvidenceLedger:
    def __init__(self) -> None:
        self._items: list[Evidence] = []
        self._seen: set[str] = set()

    def add(self, ev: Evidence) -> bool:
        if ev.key() in self._seen:
            return False
        self._seen.add(ev.key())
        self._items.append(ev)
        return True

    @property
    def items(self) -> list[Evidence]:
        return list(self._items)

    def doc_ids(self) -> list[str]:
        out, seen = [], set()
        for e in self._items:
            for d in e.doc_ids:
                if d not in seen:
                    seen.add(d)
                    out.append(d)
        return out

    def summarize(self, max_chars: int = 4000) -> str:
        """Compact view handed back to the orchestrator each turn.

        Kept short on purpose: re-feeding raw tool output is the single biggest
        source of token waste in agentic pipelines.
        """
        lines = []
        for i, e in enumerate(self._items, 1):
            cites = ",".join(e.doc_ids[:6]) + ("..." if len(e.doc_ids) > 6 else "")
            lines.append(f"[{i}] ({e.source}) {e.claim}  <docs: {cites}>")
        text = "\n".join(lines)
        return text[:max_chars]

    def __len__(self) -> int:
        return len(self._items)


# --------------------------------------------------------------------------
# Trace
# --------------------------------------------------------------------------
@dataclasses.dataclass
class Step:
    n: int
    agent: str
    action: str
    args: dict[str, Any]
    rationale: str
    result_summary: str
    ok: bool
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    new_evidence: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class Trace:
    def __init__(self, qid: str, pipeline: str) -> None:
        self.qid = qid
        self.pipeline = pipeline
        self.steps: list[Step] = []
        self.strategy_changes: list[dict[str, Any]] = []
        self.stop_reason: str = ""
        self.started = time.time()

    def add(self, step: Step) -> None:
        self.steps.append(step)

    def note_strategy_change(self, frm: str, to: str, why: str) -> None:
        self.strategy_changes.append({"step": len(self.steps), "from": frm,
                                      "to": to, "why": why})

    def as_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "pipeline": self.pipeline,
            "n_steps": len(self.steps),
            "tools_called": [s.action for s in self.steps],
            "agents_invoked": sorted({s.agent for s in self.steps}),
            "strategy_changed": bool(self.strategy_changes),
            "strategy_changes": self.strategy_changes,
            "stop_reason": self.stop_reason,
            "wall_time_s": round(time.time() - self.started, 3),
            "steps": [s.as_dict() for s in self.steps],
        }


# --------------------------------------------------------------------------
# Budget + state
# --------------------------------------------------------------------------
@dataclasses.dataclass
class Budget:
    max_steps: int = 8
    max_tokens: int = 40_000
    max_seconds: float = 180.0
    # stop if this many consecutive steps add no new evidence
    max_barren_steps: int = 2


@dataclasses.dataclass
class InvestigationState:
    qid: str
    question: str
    budget: Budget = dataclasses.field(default_factory=Budget)
    evidence: EvidenceLedger = dataclasses.field(default_factory=EvidenceLedger)
    plan: str = ""
    gaps: list[str] = dataclasses.field(default_factory=list)
    strategy: str = "unset"
    tokens_used: int = 0
    barren_streak: int = 0
    started: float = dataclasses.field(default_factory=time.time)
    failed_actions: list[str] = dataclasses.field(default_factory=list)

    def spend(self, tokens: int) -> None:
        self.tokens_used += tokens

    def elapsed(self) -> float:
        return time.time() - self.started

    def should_stop(self, n_steps: int) -> str | None:
        """Return a stop reason, or None to keep investigating."""
        if n_steps >= self.budget.max_steps:
            return "step_budget_exhausted"
        if self.tokens_used >= self.budget.max_tokens:
            return "token_budget_exhausted"
        if self.elapsed() >= self.budget.max_seconds:
            return "time_budget_exhausted"
        if self.barren_streak >= self.budget.max_barren_steps:
            return "no_progress"
        return None

    def context(self) -> str:
        """The compact working context the orchestrator reasons over."""
        parts = [f"QUESTION: {self.question}"]
        if self.plan:
            parts.append(f"PLAN: {self.plan}")
        parts.append(f"STRATEGY: {self.strategy}")
        parts.append("EVIDENCE SO FAR:\n" + (self.evidence.summarize() or "(none)"))
        if self.gaps:
            parts.append("KNOWN GAPS:\n- " + "\n- ".join(self.gaps))
        if self.failed_actions:
            parts.append("ACTIONS THAT RETURNED NOTHING (do not repeat verbatim):\n- "
                         + "\n- ".join(self.failed_actions[-5:]))
        return "\n\n".join(parts)


@dataclasses.dataclass
class Answer:
    qid: str
    pipeline: str
    question: str
    answer: str
    citations: list[str]
    confidence: float
    trace: dict[str, Any]
    usage: dict[str, int]
    error: str | None = None
    # What the model *said* it used, kept separately from `citations`, which is
    # what the system actually read. See CITATION_RULE below.
    claimed_citations: list[str] = dataclasses.field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False)


# --------------------------------------------------------------------------
# Citation rule (applied identically by every pipeline)
# --------------------------------------------------------------------------
CITATION_RULE = """\
`Answer.citations` is the set of documents the SYSTEM ACTUALLY READ while
producing the answer - the union of the doc_ids returned by every retrieval it
performed - not the subset the language model claims it used.

This is the only rule under which the three pipelines are comparable. Each one
reads a different number of documents by design:

  rag       k nearest chunks, fixed
  graphrag  a fixed graph traversal plus k chunks, same shape every question
  agentic   whatever the orchestrator chose to call, question by question

Measuring the model's self-reported list instead would score three different
prompts, not three different retrieval strategies, and self-reporting is
unreliable: on the first full run the model named 2 of 14 scanned events on
aggregation questions, understating recall by 72 points.

Consequences, both intended:
  completeness_doc_recall  rewards reading the gold documents at all
  citation_precision       punishes reading documents that were not needed

A pipeline that retrieves a large fixed superset therefore scores high recall
and low precision; an adaptive one should score high on both, and that is
precisely the claim the benchmark is testing. The model's own list is retained
as `Answer.claimed_citations` for inspection.
"""
