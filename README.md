# Agentic GraphRAG on TigerGraph

An agent that investigates complex questions over a fixed document corpus using
graph, vector and document evidence — benchmarked head-to-head against plain RAG
and single-pass GraphRAG.

Built for the **TigerGraph Agentic GraphRAG Hackathon**, Round 1.

![architecture](docs/architecture.png)

---

## The question this repo answers

The hackathon's stated goal is not "make the agent win". It is:

> figure out which questions need an agent, and which don't.

So the headline result here is not an accuracy number — it is a **cost/benefit
map**. For every question type we report accuracy, completeness, token cost, and
the ratio that actually matters: **accuracy per 1,000 tokens**. Where the agent
pays for itself, we show it. Where it is overkill, we say so.

The repo is organised so that claim can be checked: three pipelines share one
corpus, one graph, one tool layer and one answer contract. **Only the control
flow differs.**

| | Retrieval | Adapts to what it finds? | LLM calls |
|---|---|---|---|
| **1 · RAG** | vector top-k | no | 2 |
| **2 · GraphRAG** | fixed graph pass + vectors | no | 2 |
| **3 · Agentic GraphRAG** | agent-selected, per step | **yes** | 3–9, budgeted |

---

## Results

Benchmarked against a live TigerGraph Savanna workspace, not a local stub:

![TigerGraph Savanna workspace](docs/img/savanna-workspace.png)


100 public questions × 4 configurations = 400 graded runs against a live
TigerGraph Savanna instance. `gemini-2.5-flash-lite`, `all-MiniLM-L6-v2`
embeddings, 0 infrastructure errors. Reproduce with `make bench-public`;
the numbers below come from [`out/public/summary.json`](out/public/summary.json)
and are stable to ±1 point across independent runs.

| Pipeline | Accuracy | Completeness | Evidence precision | Tokens/q | Acc/1k tok |
|---|---|---|---|---|---|
| RAG | 43% | 68% | 40% | 2,020 | 0.213 |
| **GraphRAG** | **97%** | **98%** | 41% | **2,106** | **0.461** |
| Agentic (no router) | 92% | 94% | 94% | 5,393 | 0.171 |
| Agentic (routed) | 93% | 94% | **97%** | 6,442 | 0.144 |

**The agent does not win on accuracy, and we are not going to pretend it does.**
Single-pass GraphRAG is 4 points ahead at a third of the token cost. That is the
honest headline, and the per-type breakdown explains it.

### Where the value actually is

| Question type | n | RAG | GraphRAG | Agentic |
|---|---|---|---|---|
| aggregation | 21 | **4.8%** | 100% | 100% |
| superlative | 10 | 20% | 100% | 100% |
| temporal | 22 | 27% | 100% | 95.5% |
| multi-hop | 28 | 54% | 89% | 82% |
| lookup | 19 | 100% | 100% | 95% |

Read it column by column:

**The graph is worth about fifty points.** Aggregation 4.8% → 100%, superlative
20% → 100%, temporal 27% → 100%. Not a tuning difference — a category
difference. These questions need 10–43 documents held at once, or a date
relation the model would otherwise have to *know*. No value of *k* fixes that;
typed columns and a `PRECEDES` edge do.

**Adaptive retrieval is worth approximately zero here** — and the reason is in
the same table. Every type is answerable by a *fixed* traversal, because all 150
questions come from five templates. The agent has nothing to adapt to. It spends
3–5 extra LLM calls per question rediscovering a route that was never going to
change.

> **Adaptive retrieval pays only when the route is unknown in advance. On a
> benchmark generated from five templates, it never is.**

Lookup at 100% for plain RAG is the control: the vector store is not broken, the
questions simply outgrow it.

### What the agent does buy: attribution

Accuracy is not the only thing the brief asks about. Under a single citation
rule applied identically to all three pipelines
([`harness.CITATION_RULE`](src/agents/harness.py) — *cite every document the
system actually read*, not the subset the model claims), a real and large
difference appears:

| Question type | GraphRAG precision | Agentic precision |
|---|---|---|
| multi-hop | 16% | **100%** |
| temporal | 16% | **100%** |
| lookup | 13% | **85%** |

GraphRAG runs the same traversal whatever the question, so it reaches the gold
documents almost always (98% completeness) while five of every six documents it
pulls on a multi-hop question are irrelevant. The agent reaches 94% of the gold
documents having read **almost nothing it did not need**.

> The agent does not buy accuracy on this benchmark. It buys **attribution** —
> the same answers, with an evidence set 2.4× cleaner, because every retrieval
> was a decision rather than a default.

Which tells you when each design is correct: GraphRAG when you want the answer,
the agent when a human has to check the work.

### The router is a bad trade, and the ablation says so

The cost-aware router buys **one point of accuracy for 19% more tokens** — 93%
against 92% without it. On the metric this project says matters, accuracy per
1,000 tokens, that is a loss: **0.144 routed against 0.171 unrouted**. It also
escalates past its own opening strategy on 11% of questions, which means its
budget was simply wrong there, and escalating costs more than starting at the
right budget would have.

It ships enabled, with `--ablation` turning it off, because a feature you cannot
switch off is a feature you cannot evaluate. The data to fix it — observed
escalations, per strategy — is already in the traces; learning the budgets
instead of guessing them is the obvious next step and is not implemented.

### The agent says "unknown" rather than guessing

On 5 of the 50 held-out questions the agent retrieves nothing — entity linking
fails on the phrasing — and answers `unknown`. That is deliberate.

Asked anyway, the model will happily produce a fluent, confident name drawn from
its own memory rather than from the corpus. An earlier build did exactly that:
`eval-042` answered "Zhang Shan" after two failed lookups, with an empty citation
list. It reads like an answer and is worth nothing, because nothing in the corpus
supports it, and it contradicts this project's own system prompt
(["never answer from memory; every claim must come from a tool result"](src/agents/prompts.py)).

So when the evidence ledger is empty the orchestrator skips the answerer
entirely and returns `unknown`. This is not a trade of accuracy for integrity:
measured over the public set, zero-evidence guesses were correct **1 time in 11**.
Abstaining is the more accurate behaviour as well as the honest one.

The result is that **every answer in `out/hidden/submission.jsonl` that is not
`unknown` carries the documents it was derived from** — 45 answered and cited,
5 abstentions, 0 uncited claims.

### Honest notes on measurement

- **Judge tokens are metered separately** and excluded from pipeline budgets, so
  no pipeline is charged for being graded.
- **Thinking tokens count.** Gemini 2.5+ spends output budget on hidden
  reasoning; `thoughtsTokenCount` is included in every total here. Omitting it
  would understate agent cost by the largest margin of any single choice.
- **Infrastructure failures are reported, not absorbed.** A run that cannot
  reach the database is scored wrong, but counted separately in
  `summary.json → infrastructure_errors`. An earlier run lost 11 of 400 to a
  ~20-second Savanna outage and every pipeline moved 2–3 points; that run is not
  the one reported here.
- **One citation rule, three pipelines.** An earlier version of this benchmark
  let each pipeline define its own, which made the completeness comparison
  meaningless. See `harness.CITATION_RULE` for the rule and why.
- **`tests/test_no_overfitting.py`** asserts that no gold answer or question id
  is reachable from any answering path.

---

## Quick start

```bash
git clone <this repo> && cd agentic-graphrag-tigergraph
make setup                      # installs deps, copies .env.example -> .env
# put the hackathon files in data/: corpus.jsonl, eval_public.jsonl, eval_hidden.jsonl

make ingest                     # corpus -> property graph tables + chunks
make vectors                    # embed 23,497 chunks (~30s on CPU)
make test                       # 28 tests, no API key needed
make oracle                     # graph correctness gate: expects 100/100
```

Then add your `GEMINI_API_KEY` and TigerGraph credentials to `.env` and:

```bash
make deploy                     # schema + data + GSQL queries + vectors -> TigerGraph
make bench-public               # all 3 pipelines x 100 graded questions
make dashboard                  # -> out/dashboard.html
make bench-hidden               # -> out/hidden/submission.jsonl
```

Everything runs offline with `--backend mock`, which replaces the LLM with a
deterministic policy so a reviewer can exercise the full agent loop — router,
orchestrator, evaluator, answerer — before spending a token:

```bash
python -m src.bench.run --backend mock --graph-backend local --limit 40 --out out/mock
python -m src.bench.dashboard --run out/mock
```

---

## What the corpus actually is

Read this before designing anything — it determines the whole engineering
strategy.

```
corpus.jsonl          2,951 documents · ~5.47M tokens
eval_public.jsonl       100 questions with answers + gold_doc_ids
eval_hidden.jsonl        50 questions, held out
```

| Document kind | Count | Role |
|---|---|---|
| `[Infobox Olympic event]` | **2,162** | the answer source |
| `[Infobox film]` | 546 | distractor |
| people (officeholder / person / writer / scientist) | 152 | distractor |
| other (tennis events, companies, football/handball competitions, …) | 91 | distractor |

Every Olympic document opens with a machine-readable block:

```
[Infobox Olympic event]
  event: Men's canoe sprint K-2 1,000 metres
  games: 2012 Summer
  venue: Eton Dorney
  date: 6 to 8 August
  competitors: 24
  nations: 12
  gold: Rudolf DombiRoland Kökény
  prev: 2008
  next: 2016
```

**All 150 questions are answerable from those fields**, in five shapes:

| qtype | public | hidden | needs |
|---|---|---|---|
| `aggregation` | 21 | 15 | an exact count over *every* event of a sport at one Games (10–43 documents) |
| `multi_hop` | 28 | 10 | venue + date → event → medalist |
| `superlative` | 10 | 10 | argmax over the full event set |
| `temporal` | 22 | 8 | a hop along `prev` to the preceding Games |
| `lookup` | 19 | 7 | one attribute of one event |

None of them touch the films or biographies. Those exist to make naive vector
retrieval noisy — and they succeed.

### Why this is the whole story

Aggregation and superlative questions need *the complete set* of events in one
place. `pub-004` requires all **43** athletics events of 2008 simultaneously.
No top-k retriever gets 43 of 43, and no LLM reliably counts them if it did.
Put the same facts in a graph and it is one query, exact, at constant token cost.

**So accuracy here is a data-modelling problem, not a prompting problem.**

---

## The graph

```
POPULATED (what every query actually traverses)

Event ──IN_SPORT──▶ Sport          Event.competitors : INT
  │   ──AT_GAMES──▶ Games          Event.nations     : INT
  │   ──HELD_AT───▶ Venue          Event.date_norm   : STRING
  └── ──WON_MEDAL─▶ Athlete ──REPRESENTS──▶ Nation
                      (WON_MEDAL.medal, WON_MEDAL.rank)

Games ──PRECEDES──▶ Games          ← temporal questions are one hop

DECLARED BUT NOT POPULATED

Event ──EVIDENCE──▶ Document ──HAS_CHUNK──▶ Chunk(chunk_emb)
```

Live counts after `make deploy`: 2,162 Event, 6,220 Athlete, 303 Venue, 136
Nation, 41 Sport, 20 Games; 8,440 WON_MEDAL, 6,256 REPRESENTS, 2,162 AT_GAMES,
2,162 IN_SPORT, 2,081 HELD_AT, 27 PRECEDES.

> **On the second block, in the interest of being checkable.** `Document`,
> `Chunk`, `EVIDENCE` and `HAS_CHUNK` are declared in
> [`gsql/01_schema.gsql`](gsql/01_schema.gsql) but the loading job does not fill
> them, so those edges are empty in the live graph. Document text is served
> instead by the local vector index in
> [`src/vectorstore/`](src/vectorstore/index.py), because the `chunk_emb` vector
> attribute was not available on our Savanna workspace — the deploy step probes
> for it and skips cleanly rather than failing (see
> [`docs/FEEDBACK.md`](docs/FEEDBACK.md) #11). `TigerGraphVectorIndex` is
> implemented and wired for a workspace that does support it. We have left the
> schema in place rather than quietly deleting it, because it is the honest
> record of what this design intends and what the platform actually gave us.

Two decisions do most of the work:

1. **`competitors` and `nations` are typed integers on the vertex**, not text in
   a passage. Aggregation and superlative become exact server-side operations
   (`q_count_events_where`, `q_extreme_event`) instead of arithmetic delegated
   to a language model.
2. **`Games.PRECEDES` is a real edge**, built from the infobox `prev`/`next`
   fields. "The Olympics held immediately before 2016" is a traversal, not a
   guess — and it stays correct for Winter/Summer interleaving and for Games
   missing from the corpus.

Schema, loading and the nine installed queries are in [`gsql/`](gsql/).

The schema as TigerGraph Savanna renders it — `Event` at the centre, with
`Document`/`Chunk` declared but unpopulated as described above:

![TigerGraph Savanna schema](docs/img/savanna-schema.png)

### Three parsing traps

Each of these silently cost accuracy, and each now has a named regression test
in [`tests/test_normalize.py`](tests/test_normalize.py):

| Trap | Example | Cost if missed |
|---|---|---|
| number/unit run-on | `event: Men's 68kg` (2012) vs `Men's 68 kg` (2008) | temporal hop finds the wrong Games |
| stripped `+` | `Men's +80 kg` normalises to `Men's 80 kg` | returns a different event's medalist |
| venue+date collision | `Laura Biathlon & Ski Complex`, 22 Feb 2014 → two events | coin-flip on multi-hop |

The last one is resolved by a signal already in the data: venue names usually
name their sport, so `Biathlon` in the venue string picks the biathlon event.

---

## The agent

### Harness ([`src/agents/harness.py`](src/agents/harness.py))

Owns the four things the guidebook asks for, independent of any LLM or backend:

- **state** — plan, strategy, gaps, failed actions, spend
- **tools** — a typed registry with cost hints the orchestrator can read
- **evidence** — an append-only, deduplicated ledger where every item carries
  its `doc_ids`, so citations are a property of the investigation rather than
  something the answerer invents
- **stopping** — step / token / wall-clock budgets *and* a no-progress detector
  that fires after two barren steps

### Orchestrator ([`src/agents/orchestrator.py`](src/agents/orchestrator.py))

Each turn it sees the question, the evidence summary, the known gaps and the
tool catalog, and picks **one** next action. Not a fixed sequence. After any
productive step the evidence evaluator votes on sufficiency; the loop ends on a
sufficiency verdict, a budget, or a give-up.

When a cheap strategy returns nothing, the orchestrator **escalates** — raises
the budget and widens the toolset — and that escalation is recorded in the trace
as a strategy change, which the dashboard reports as a rate.

### Specialists ([`src/agents/specialists.py`](src/agents/specialists.py))

Separate single-purpose agents rather than one mega-prompt, because the
benchmark measures *which* agents fire on *which* question type — a signal that
disappears if planning, evaluation and answering share a prompt.

`strategy_router` · `entity_linker` · `graph_traversal_agent` ·
`aggregation_agent` · `multi_hop_agent` · `similarity_search_agent` ·
`document_retrieval_agent` · `evidence_evaluator` · `answerer`

### The innovation: a cost-aware router

Before investigating, the router classifies the question into the cheapest
strategy that can answer it *correctly*, and sets the step budget accordingly:

| strategy | budget | typical question |
|---|---|---|
| `graph_direct` | 3 steps | "how many nations competed in …" |
| `graph_multihop` | 6 steps | "…held at *venue* on *date*" |
| `hybrid` | 8 steps | graph gets partway, text finishes |
| `document` | 8 steps | schema cannot express it |

The orchestrator can still escalate. This is the mechanism that lets the system
be agentic **where it matters** and cheap everywhere else — which is the
hackathon's actual research question, implemented rather than described.

`AgenticNoRouterPipeline` is shipped as an **ablation** so the dashboard can
report how much of the efficiency win comes from routing rather than from having
graph tools at all (`make bench-public` runs it).

---

## Evidence and explainability

Every answer ships with its full investigation trace:

```json
{
  "qid": "pub-001",
  "answer": "5",
  "citations": ["Q47091419", "Q47105341", "..."],
  "tokens": {"input": 2411, "output": 138, "total": 2549, "llm_calls": 3},
  "agentic_trace": {
    "n_steps": 1,
    "tools_called": ["route", "count_events"],
    "agents_invoked": ["aggregation_agent", "strategy_router"],
    "strategy_changed": false,
    "stop_reason": "evidence_sufficient",
    "steps": [{"n": 1, "agent": "aggregation_agent", "action": "count_events",
               "args": {"sport": "biathlon", "year": 2018, "season": "Winter",
                        "field": "competitors", "op": ">", "value": 73},
               "rationale": "exact aggregation belongs in the graph",
               "result_summary": "5 of 11 biathlon events ... have competitors > 73",
               "new_evidence": 1}]
  }
}
```

One deliberate decision: an aggregation answer cites **every event the query
scanned**, not only the ones that matched. Those documents were genuinely
consulted, and the benchmark's `gold_doc_ids` agrees. Without this the pipeline
looks accurate but incomplete — it scores 89% completeness instead of 100%.

---

## Verification

Correctness is gated *before* any LLM is involved.

`src/bench/oracle.py` drives the graph through the same tool calls the agent
makes, with regex-extracted arguments instead of an LLM planner. It is not an
answering pipeline — it is the check that the graph is loaded correctly and the
tool layer returns the right facts. Any failure there is a data bug, not a
prompting bug.

```
$ make oracle
=== GRAPH CORRECTNESS GATE (not a pipeline score) ===
    Checks the graph returns the right facts, with no LLM involved.
    This number must never be reported as system accuracy.
  aggregation   21/21  100.0%
  lookup        19/19  100.0%
  multi_hop     28/28  100.0%
  superlative   10/10  100.0%
  temporal      22/22  100.0%
  TOTAL        100/100 100.0%   <- graph correctness, NOT pipeline accuracy
  gold-doc recall (answer-bearing docs): 100.0%
```

```bash
python -m src.graph.deploy --verify     # same oracle, against the LIVE database
```

The nine queries installed on the live workspace:

![GSQL queries installed on Savanna](docs/img/savanna-queries.png)

That second command is the one that matters: it re-runs the gate through
TigerGraph rather than the in-memory store, so a load is never declared
successful on vertex counts alone. It is how the two bugs in
[`docs/BLOG.md`](docs/BLOG.md) were found — 21/100 (attribute alias prefixes),
then 98/100 (unordered vertex sets), then 100/100.

`make test` runs 28 tests covering the normalisation traps, the graph oracle per
question type, and the harness (budgets, no-progress detection, escalation,
trace completeness) — all without a network or an API key.

---

## Metrics

`make dashboard` produces a single self-contained `out/dashboard.html`:

- accuracy by question type (LLM-as-judge with an exact-match fast path)
- **completeness** as gold-document recall, and citation precision
- token cost per question, split input/output, attributed per reasoning step
- **accuracy per 1,000 tokens** — the number that decides whether the agent is
  worth it
- agentic behaviour: tools called, specialists invoked, steps per question type,
  strategy-change rate, and why the loop stopped

Judge tokens are metered separately and excluded from pipeline budgets, so the
comparison is not contaminated by grading cost.

---

## Layout

```
gsql/                   schema, loading job, 9 installed queries
src/ingest/             infobox parser, normalisation (the accuracy-critical part)
src/graph/              store interface, LocalGraphStore, TigerGraphStore, deploy
src/vectorstore/        chunk embeddings; local, Gemini, or TigerGraph Vector DB
src/llm/                metered LLM client + deterministic mock policy
src/agents/             harness, orchestrator, specialists, tools, prompts
src/pipelines/          rag.py, graphrag.py, agentic.py (+ router ablation)
src/bench/              oracle, runner, judge, metrics, dashboard
tests/                  28 tests, no network required
docs/architecture.svg   the diagram above
```

Both graph backends implement the same interface, so the entire system runs
against TigerGraph or against the in-memory store. That is not a fallback for
its own sake — it is what makes the GSQL queries testable, because both must
return the same answers on all 100 public questions.

---

## Reproducing the submission

```bash
make ingest vectors           # deterministic; no API key
make deploy                   # TigerGraph: schema, data, queries, vectors
make bench-public             # graded run + ablation  -> out/public/
make dashboard                # -> out/dashboard.html
make bench-hidden             # -> out/hidden/submission.jsonl
```

`out/hidden/submission.jsonl` is one JSON object per held-out question carrying
the answer, citations, token counts and the complete agentic trace.

## Limitations

- The five question templates are narrow. The graph schema is general over
  Olympic-event infoboxes, but a corpus with different infobox kinds would need
  its own vertex types; `parse_corpus.py` currently ingests only Olympic events
  as structured vertices and treats everything else as text.
- Ties in superlative questions are reported rather than resolved, because the
  corpus gives no basis for choosing between them.
- The hashing embedder fallback is much weaker than `all-MiniLM-L6-v2`; RAG
  numbers produced with it understate what plain RAG can do. Benchmark runs use
  the real encoder.
- Round 2 (conflicting and evolving facts) is not implemented. The evidence
  ledger already carries per-claim provenance, which is where source
  authority and supersession would attach.

## Attribution

Corpus text is derived from English Wikipedia, licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each document
carries its source URL.
