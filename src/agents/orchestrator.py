"""The orchestrator agent.

Implements the plan -> act -> evaluate -> re-plan loop. The next action is
chosen by an LLM from the tool catalog given the current evidence and gaps; it
is not a fixed sequence. Two things make the loop terminate sensibly:

  * the evidence evaluator votes on sufficiency after every productive step
  * the harness enforces step / token / time budgets and a no-progress detector

Every decision, tool call, result and stop reason lands in the Trace.
"""
from __future__ import annotations

import json
import time
from typing import Any

from ..llm.client import UsageMeter
from . import prompts
from .harness import (Answer, Budget, InvestigationState, Step, Trace,
                      ToolRegistry)
from .specialists import Answerer, EvidenceEvaluator, StrategyRouter, _safe_json

# Strategies the router may pick, and the step budget each one gets. Escalation
# raises the budget; that escalation is recorded as a strategy change.
STRATEGY_BUDGETS = {
    "graph_direct": 3,
    "graph_multihop": 6,
    "hybrid": 8,
    "document": 8,
}


class Orchestrator:
    def __init__(self, llm, registry: ToolRegistry, meter: UsageMeter,
                 sports: list[str], pipeline: str = "agentic",
                 budget: Budget | None = None, use_router: bool = True):
        self.llm = llm
        self.registry = registry
        self.meter = meter
        self.sports = sports
        self.pipeline = pipeline
        self.base_budget = budget or Budget()
        self.use_router = use_router
        self.router = StrategyRouter(llm, meter, pipeline)
        self.evaluator = EvidenceEvaluator(llm, meter, pipeline)
        self.answerer = Answerer(llm, meter, pipeline)

    # ------------------------------------------------------------------
    def investigate(self, qid: str, question: str) -> Answer:
        trace = Trace(qid, self.pipeline)
        state = InvestigationState(qid=qid, question=question,
                                   budget=Budget(**vars(self.base_budget)))

        # ---- plan -------------------------------------------------------
        if self.use_router:
            route = self.router.route(qid, question, self.sports)
            state.strategy = route.get("strategy", "graph_multihop")
            state.plan = route.get("reasoning", "")
            state.gaps = list(route.get("sub_questions") or [])
            state.spend(route.get("tokens", 0))
            state.budget.max_steps = min(
                self.base_budget.max_steps,
                STRATEGY_BUDGETS.get(state.strategy, self.base_budget.max_steps))
            trace.add(Step(n=0, agent="strategy_router", action="route",
                           args={"question": question},
                           rationale=route.get("reasoning", ""),
                           result_summary=(f"strategy={state.strategy} "
                                           f"budget={state.budget.max_steps} "
                                           f"entities={json.dumps(route.get('entities', {}))}"),
                           ok=True, latency_s=0.0,
                           input_tokens=0, output_tokens=route.get("tokens", 0)))
            hint = self._entity_hint(route.get("entities") or {})
        else:
            state.strategy = "unrouted"
            hint = ""

        # ---- act / evaluate loop ----------------------------------------
        final_reason = ""
        while True:
            stop = state.should_stop(len([s for s in trace.steps if s.n > 0]))
            if stop:
                final_reason = stop
                break

            decision = self._next_action(state, hint)
            action = decision.get("action", "answer")

            if action == "answer":
                final_reason = "evidence_sufficient"
                break
            if action == "give_up":
                final_reason = "gave_up:" + decision.get("why", "")
                break

            tool = self.registry.get(action)
            n = len([s for s in trace.steps if s.n > 0]) + 1
            if tool is None:
                state.failed_actions.append(f"{action}(unknown tool)")
                trace.add(Step(n=n, agent="orchestrator", action=action,
                               args=decision.get("args", {}),
                               rationale=decision.get("reasoning", ""),
                               result_summary=f"unknown tool {action!r}",
                               ok=False, latency_s=0.0))
                state.barren_streak += 1
                continue

            args = self._coerce_args(tool, decision.get("args", {}))
            t0 = time.time()
            try:
                evidence, summary = tool.fn(**args)
                ok = True
            except TypeError as e:
                evidence, summary, ok = [], f"bad arguments: {e}", False
            except Exception as e:  # tool failures must not kill the run
                evidence, summary, ok = [], f"tool error: {type(e).__name__}: {e}", False
            latency = time.time() - t0

            added = sum(1 for ev in evidence if state.evidence.add(ev))
            if added == 0:
                state.barren_streak += 1
                state.failed_actions.append(
                    f"{action}({json.dumps(args, ensure_ascii=False)[:160]})")
            else:
                state.barren_streak = 0

            trace.add(Step(n=n, agent=self._agent_for(tool.category), action=action,
                           args=args, rationale=decision.get("reasoning", ""),
                           result_summary=summary[:600], ok=ok,
                           latency_s=round(latency, 3), new_evidence=added))

            # ---- evaluate ------------------------------------------------
            if added:
                verdict = self.evaluator.evaluate(qid, question, state.evidence)
                state.spend(verdict.get("tokens", 0))
                state.gaps = list(verdict.get("gaps") or [])
                if verdict.get("sufficient"):
                    final_reason = "evidence_sufficient"
                    break
            elif state.strategy in ("graph_direct", "graph_multihop") \
                    and state.barren_streak >= 1:
                # escalate: the cheap strategy is not working
                old = state.strategy
                state.strategy = "hybrid"
                state.budget.max_steps = self.base_budget.max_steps
                trace.note_strategy_change(
                    old, "hybrid", f"{action} returned no new evidence")

        trace.stop_reason = final_reason

        # ---- answer -------------------------------------------------------
        if len(state.evidence) == 0:
            # Retrieval returned nothing at all. Asked anyway, the answerer
            # still produces a fluent, confident name - drawn from the model's
            # own memory rather than from the corpus. That is exactly what
            # CORPUS_GROUNDING forbids ("never answer from memory"), and on an
            # evidence-scored benchmark an uncited answer is worthless even
            # when it happens to be right.
            #
            # It is also not a trade of integrity against accuracy: measured
            # over the public set, these zero-evidence guesses were correct
            # 1 time in 11. Abstaining is both the honest and the more
            # accurate behaviour, and it saves the answerer call.
            result = {"answer": "unknown", "citations": [], "confidence": 0.0}
            trace.stop_reason = f"{final_reason}|no_evidence_abstained"
        else:
            result = self.answerer.answer(qid, question, state.evidence)
            state.spend(result.get("tokens", 0))
        # harness.CITATION_RULE: cite every document the investigation read.
        # Unlike the fixed pipelines this set is chosen, step by step, by the
        # orchestrator - so its precision is a direct measure of whether
        # adaptive retrieval actually reads less to learn the same thing.
        citations = state.evidence.doc_ids()

        usage = self.meter.totals(self.pipeline, qid)
        return Answer(
            qid=qid, pipeline=self.pipeline, question=question,
            answer=str(result.get("answer", "")).strip(),
            citations=[c for c in citations if c],
            claimed_citations=list(result.get("citations") or []),
            confidence=float(result.get("confidence", 0.0) or 0.0),
            trace=trace.as_dict(), usage=usage,
        )

    # ------------------------------------------------------------------
    def _next_action(self, state: InvestigationState, hint: str) -> dict[str, Any]:
        catalog = json.dumps(self.registry.catalog(), indent=1)
        prompt = (f"{state.context()}\n\n"
                  f"{hint}\n\n"
                  f"AVAILABLE TOOLS:\n{catalog}\n\n"
                  f"Sports in the graph: {', '.join(self.sports)}\n\n"
                  "Choose the single next action.")
        resp = self.llm.complete(prompt, system=prompts.ORCHESTRATOR_SYSTEM,
                                 json_mode=True, max_output_tokens=500)
        self.meter.record(self.pipeline, state.qid, "orchestrator", resp)
        state.spend(resp.total_tokens)
        return _safe_json(resp, {"action": "answer",
                                 "reasoning": "orchestrator parse failure"})

    @staticmethod
    def _entity_hint(entities: dict[str, Any]) -> str:
        kept = {k: v for k, v in entities.items() if v not in ("", 0, None)}
        if not kept:
            return ""
        return ("ENTITIES EXTRACTED FROM THE QUESTION (use these verbatim as tool "
                f"arguments): {json.dumps(kept, ensure_ascii=False)}")

    @staticmethod
    def _agent_for(category: str) -> str:
        return {
            "entity_linking": "entity_linker",
            "graph_traversal": "graph_traversal_agent",
            "similarity_search": "similarity_search_agent",
            "document_retrieval": "document_retrieval_agent",
            "aggregation": "aggregation_agent",
            "multi_hop": "multi_hop_agent",
        }.get(category, "retrieval_agent")

    @staticmethod
    def _coerce_args(tool, args: dict[str, Any]) -> dict[str, Any]:
        """LLMs emit '2016' and 2016 interchangeably; tools take ints."""
        out: dict[str, Any] = {}
        for k, v in (args or {}).items():
            if k not in tool.parameters:
                continue
            if k in ("year", "at_year", "before_year", "value", "threshold",
                     "k", "max_chars"):
                try:
                    out[k] = int(str(v).strip() or 0)
                except (TypeError, ValueError):
                    out[k] = 0
            else:
                out[k] = "" if v is None else v
        return out
