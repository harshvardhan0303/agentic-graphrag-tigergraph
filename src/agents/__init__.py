"""The agent itself: harness, orchestrator, specialists, tools, prompts.

    harness       state, typed tool registry, evidence ledger, budget, trace
    orchestrator  the loop - choose one action, evaluate, stop
    specialists   single-purpose agents (router, evaluator, answerer)
    tools         what the orchestrator is allowed to call, with cost hints
    prompts       every system prompt, written to be read by a judge

The harness is deliberately independent of any LLM or graph backend, which is
what lets the same agent run against TigerGraph or the in-memory store.
"""
