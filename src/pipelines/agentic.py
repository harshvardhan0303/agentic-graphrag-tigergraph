"""Pipeline 3 - Agentic GraphRAG.

A thin wrapper: all of the behaviour lives in the orchestrator and the harness.
Kept as its own class so that the benchmark treats all three pipelines
identically.
"""
from __future__ import annotations

from ..agents.harness import Answer, Budget
from ..agents.orchestrator import Orchestrator
from ..llm.client import UsageMeter


class AgenticGraphRAGPipeline:
    name = "agentic"

    def __init__(self, llm, registry, meter: UsageMeter, sports: list[str],
                 budget: Budget | None = None, use_router: bool = True):
        self.orchestrator = Orchestrator(
            llm=llm, registry=registry, meter=meter, sports=sports,
            pipeline=self.name, budget=budget, use_router=use_router)

    def answer(self, qid: str, question: str) -> Answer:
        return self.orchestrator.investigate(qid, question)


class AgenticNoRouterPipeline(AgenticGraphRAGPipeline):
    """Ablation: the same agent with the cost-aware router disabled.

    Reported alongside the main results to isolate how much of the efficiency
    win comes from routing rather than from the graph tools.
    """
    name = "agentic_no_router"

    def __init__(self, llm, registry, meter: UsageMeter, sports: list[str],
                 budget: Budget | None = None):
        super().__init__(llm, registry, meter, sports, budget, use_router=False)
        self.orchestrator.pipeline = self.name
        self.orchestrator.router.pipeline = self.name
        self.orchestrator.evaluator.pipeline = self.name
        self.orchestrator.answerer.pipeline = self.name
