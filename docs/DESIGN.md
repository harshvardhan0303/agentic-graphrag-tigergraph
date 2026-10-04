# Design notes — why this system is built the way it is

This document exists so that you (and a judge, and a teammate) can follow every
decision from the problem statement down to the code. Read it top to bottom; it
is ordered the way the thinking actually went.

---

## 1. What the hackathon is really asking

Three sentences from the guidebook matter more than the rest:

> RAG retrieves text. GraphRAG adds structure. But some questions need more.

> **The headline goal: figure out which questions need an agent, and which don't.**

> The objective is not simply to measure whether Agentic GraphRAG produces a
> better answer. It is to determine whether the additional reasoning and
> retrieval steps are worth the additional complexity and token cost.

Most teams will read that and build a system designed to prove the agent always
wins. That is the wrong target. The organisers are asking a research question
and want an honest answer, including the cases where the agent wastes money.

**So our deliverable is not "an agent". It is a controlled experiment**: three
pipelines that differ in exactly one variable — whether retrieval adapts to what
it finds — plus the measurement apparatus to tell them apart.

Everything in the architecture follows from wanting that comparison to be fair.

---

## 2. What we found in the data, and why it decided the architecture

Before writing any code we read the corpus. Three facts came out of it.

### Fact 1 — the documents are half structured already

2,933 of 2,951 documents open with a machine-readable block:

```
[Infobox Olympic event]
  event: Men's canoe sprint K-2 1,000 metres
  games: 2012 Summer
  venue: Eton Dorney
  competitors: 24
  nations: 12
  gold: Rudolf DombiRoland Kökény
  prev: 2008
```

That is a row of a database wearing a costume. Treating it as prose and
embedding it throws away everything that makes it answerable.

### Fact 2 — a quarter of the corpus is there to mislead you

| kind | count | role |
|---|---|---|
| Olympic event | 2,162 | answers live here |
| film | 546 | distractor |
| people (officeholder, person, writer, scientist) | 152 | distractor |
| other (company, tennis, football, aircraft, conflict…) | 91 | distractor |

No benchmark question touches the films or the biographies. They exist so that a
naive semantic search returns plausible-looking garbage. That is a deliberate
trap, and vector-only retrieval walks straight into it.

### Fact 3 — the questions come in five shapes, and two of them are unanswerable by retrieval

| shape | example | what it needs |
|---|---|---|
| `lookup` | "how many nations competed in X?" | one attribute of one document |
| `multi_hop` | "who won gold at *venue* on *date*?" | find the event, then its medalist |
| `temporal` | "…at the Olympics held immediately before 2016" | identify the preceding Games, then the event |
| `aggregation` | "how many cycling events had more than 30 competitors?" | **every** cycling event at that Games — 10 to 43 documents |
| `superlative` | "which event had the most competitors?" | the same complete set, then a max |

Look at aggregation and superlative carefully. `pub-004` needs all **43**
athletics events of 2008 at once. There is no value of *k* for which top-*k*
retrieval reliably returns 43 specific documents out of 2,951 — and even if it
did, asking a language model to count 43 numbers correctly is a coin flip.

**This is the crux.** Those two question types are not hard because the reasoning
is hard. They are hard because the *retrieval shape* is wrong. A database answers
them exactly, instantly, and at constant cost.

---

## 3. The central design decision

> **Put the facts where they can be computed on, not where they have to be read.**

Concretely: parse the infobox into typed vertex attributes, and let the database
do counting, comparison and traversal.

```
"how many cycling events at 2008 had more than 30 competitors?"

  vector RAG :  retrieve ~8 passages, hope they contain all the relevant
                events, ask the model to count → usually wrong, and it cannot
                know it is wrong

  graph      :  SELECT events WHERE sport=Cycling AND games='2008 Summer'
                AND competitors > 30  →  exact, every time, ~300 tokens
```

This is why the graph layer reaches 100% on the public set while plain RAG
struggles. It is not a smarter model. It is the right data structure.

**And this is the finding the hackathon is asking for**, stated precisely:
the agent is not what fixes aggregation questions — *the graph* is. What the
agent fixes is a different class of problem: questions where you don't know in
advance which retrieval to run, or where the first attempt comes back empty.

---

## 4. The four layers

```
  ┌────────────────────────────────────────────────────────┐
  │ 4. BENCHMARK    run all three, score, draw the dashboard│  domain-agnostic
  ├────────────────────────────────────────────────────────┤
  │ 3. AGENT        plan → act → evaluate → re-plan → stop  │  domain-agnostic
  ├────────────────────────────────────────────────────────┤
  │ 2. GRAPH        vertices, edges, typed attributes, GSQL │  ← domain-specific
  ├────────────────────────────────────────────────────────┤
  │ 1. READER       documents → structured fields           │  ← domain-specific
  └────────────────────────────────────────────────────────┘
```

Layers 3 and 4 do not know or care what the documents are about. Layers 1 and 2
currently do — see §8, that is the thing we are about to fix.

### Layer 1 — the reader (`src/ingest/`)

Reads each document, splits the infobox into `key: value` pairs, and converts
values to their real types. It also carries the three corpus traps, each of
which silently costs accuracy:

| trap | what happens | why it matters |
|---|---|---|
| `Men's 68kg` (2012) vs `Men's 68 kg` (2008) | no space between number and unit | temporal questions match the wrong Games |
| `Men's +80 kg` vs `Men's 80 kg` | naive punctuation stripping merges them | returns a different athlete entirely |
| `Dani KingLaura TrottJoanna Rowsell` | team members concatenated, `<br>` lost in conversion | three athletes look like one |

Each has a named test in `tests/test_normalize.py`. **This file is where accuracy
is won or lost** — more than any prompt in the repo.

### Layer 2 — the graph (`src/graph/`, `gsql/`)

```
Event ──IN_SPORT──▶ Sport
  │   ──AT_GAMES──▶ Games ──PRECEDES──▶ Games
  │   ──HELD_AT───▶ Venue
  │   ──WON_MEDAL─▶ Athlete ──REPRESENTS──▶ Nation
  └── EVIDENCE ───▶ Document ──HAS_CHUNK──▶ Chunk(embedding)
```

Two choices carry most of the weight:

- **`competitors` and `nations` are integers on the vertex**, not numbers inside
  a sentence. Counting and max become exact server-side operations.
- **`PRECEDES` is a real edge**, built from the infobox `prev`/`next` fields.
  "The Olympics held immediately before 2016" becomes one hop instead of a
  guess — and stays correct when a Games is missing from the corpus.

There are **two implementations of the same interface**: `TigerGraphStore`
(the submission) and `LocalGraphStore` (in memory). This is not a fallback for
its own sake. It means the GSQL queries are *testable* — both backends must
return identical answers on all 100 public questions, so a broken query is
caught by the test suite rather than by a wrong benchmark number.

### Layer 3 — the agent (`src/agents/`)

Four parts, mapping directly onto what the guidebook asks for:

**Harness** — owns state, the tool registry, the evidence ledger, and stopping
criteria. Evidence is append-only and every item carries the `doc_ids` it came
from, so citations are a *property of the investigation* rather than something
the final model invents. Stopping is enforced four ways: step budget, token
budget, wall-clock, and a no-progress detector that fires after two steps that
add nothing.

**Orchestrator** — each turn it sees the question, a compact evidence summary,
the known gaps, and the tool catalog with cost hints, and picks **one** next
action. Not a fixed sequence. If a cheap path returns nothing, it escalates —
raises the budget, widens the toolset — and records that escalation as a
strategy change.

**Specialists** — separate single-purpose agents rather than one large prompt:
router, entity linker, graph traversal, aggregation, multi-hop, similarity
search, document retrieval, evidence evaluator, answerer. They are separate
*because the benchmark needs to measure which agent fires on which question
type*, and that signal disappears if they share a prompt.

**Tools** — the seven graph/vector operations, each tagged `cheap` /
`expensive`. The orchestrator can see those costs, which is how it learns that
an exact graph aggregation beats reading forty documents.

Here is a real trace, an aggregation question:

```
step 0  strategy_router      → strategy=graph_direct, budget=3 steps
                               entities={sport: biathlon, year: 2018,
                                         season: Winter, threshold: 73}
step 1  aggregation_agent    → count_events(...)
                               "5 of 11 biathlon events have competitors > 73"
        evidence_evaluator   → sufficient=true
        answerer             → "5", citing all 11 documents
        stop_reason: evidence_sufficient      total: 3 LLM calls
```

And a multi-hop one, where the agent genuinely chains:

```
step 1  graph_traversal_agent → events_at_venue_on_date("Olympic Weightlifting
                                Gymnasium", "20 September 1988")
                                → one event found
        evidence_evaluator    → insufficient, gap: "need the gold medalist"
step 2  graph_traversal_agent → get_medalists(doc_id, "gold")
                                → "Naim Süleymanoğlu"
        evidence_evaluator    → sufficient
```

### Layer 4 — the benchmark (`src/bench/`)

Runs all three pipelines on the same questions with the same answer contract,
and measures:

- **accuracy** — exact match with an LLM-as-judge fallback for diacritics and
  phrasing
- **completeness** — gold-document recall: of the documents the benchmark says
  were needed, how many did the pipeline actually cite? *This is where
  single-shot retrieval is exposed.* A pipeline can get "5" right by luck while
  having seen only 3 of the 11 relevant events.
- **token cost** — input and output, attributed per reasoning step
- **accuracy per 1,000 tokens** — the number that decides whether the agent
  earned its keep

Judge tokens are metered separately so grading cost never contaminates the
comparison.

---

## 5. Why three pipelines, built this specific way

The comparison is only meaningful if the three differ in *one* thing. So they
share the corpus, the chunking, the embedder, the tool layer, the answer format
instructions, and the token accounting. What differs:

| | retrieval | adapts to what it finds? |
|---|---|---|
| **RAG** | vector top-*k*, once | no |
| **GraphRAG** | entity extraction, then a **fixed** graph sequence | no |
| **Agentic** | agent-selected, per step | **yes** |

GraphRAG deserves a note. It would be easy to build a weak strawman here. We
didn't: it gets the same graph, the same tools, and the same entity extraction
the agent gets. Its *only* handicap is that its retrieval sequence is identical
for every question — it never looks at a result and decides to do something
different. That is precisely the variable under test, and it means any win the
agent shows is attributable to adaptivity rather than to having better tools.

---

## 6. The innovation: a cost-aware router

If the agent runs a full investigation on every question, it burns 3–5× the
tokens to answer a one-attribute lookup that a single query would have settled.
That is the "overkill" case the guidebook explicitly asks us to identify.

So before investigating, a router classifies the question into the cheapest
strategy that can answer it *correctly*, and sets the step budget accordingly:

| strategy | budget | when |
|---|---|---|
| `graph_direct` | 3 | one or two structured lookups |
| `graph_multihop` | 6 | two or more linked steps |
| `hybrid` | 8 | graph gets partway, text finishes |
| `document` | 8 | the schema cannot express it |

The orchestrator can still escalate when the cheap path fails — that is recorded,
and reported as a rate.

We also ship the same agent with the router **disabled**, as an ablation. That
is what turns "we added a router" into a measured claim: the dashboard reports
how much accuracy and how many tokens the router actually moved.

**This is the hackathon's research question implemented as a mechanism rather
than described in a slide.** That is the strongest thing in the submission.

---

## 7. Why we verify before we generate

`src/bench/oracle.py` drives the graph through the same tool calls the agent
makes, but extracts the arguments with regexes instead of an LLM.

It is **not** an answering pipeline and it is not what we submit. It answers one
question: *is the graph loaded correctly and does the tool layer return the right
facts?* Because there is no model in the loop, any failure is a data bug, not a
prompting bug. That separation is worth a lot when you are debugging at 2am.

```
=== GRAPH ORACLE ===
  aggregation   21/21  100.0%
  lookup        19/19  100.0%
  multi_hop     28/28  100.0%
  superlative   10/10  100.0%
  temporal      22/22  100.0%
  TOTAL        100/100 100.0%
  gold-doc recall: 100.0%
```

The same oracle runs against the **live TigerGraph database** after deployment,
so a load is never declared successful on vertex counts alone.

One deliberate decision shows up here: an aggregation answer cites **every event
the query scanned**, not only the matches. Those documents genuinely were
consulted, and the benchmark's own `gold_doc_ids` agrees. Without it, the
pipeline looks accurate but incomplete — 89% instead of 100%.

---

## 8. What is Olympic-specific, and the plan to fix it

Honest accounting of where the system would break on an unseen corpus:

| component | generality | if the corpus changed |
|---|---|---|
| infobox key/value reader | **general** | works on any infobox |
| normalisation | **general** | units, accents, dashes are universal |
| vertex types (`Event`, `Games`, `Sport`…) | **hard-coded** | ✗ nothing to load into |
| title pattern `{Sport} at the {Year} {Season} Olympics – {Event}` | **hard-coded** | ✗ no sport/year/season extracted |
| `PRECEDES` from `prev`/`next` | **hard-coded** | ✗ no temporal edge |
| tools (`count_events`, `top_event`, …) | named for Olympics | ✗ argument names are wrong |
| agent harness, orchestrator, specialists | **general** | ✓ unaffected |
| benchmark, judge, metrics, dashboard | **general** | ✓ unaffected |

The fix is to stop naming the schema in advance and **derive it from the corpus**:

1. **Generic vertex typing.** One vertex per document, typed by its infobox kind
   (`Entity(kind="film")`, `Entity(kind="olympic event")`). Every infobox field
   becomes an attribute, typed by inspection: integers stay integers, dates get
   parsed, everything else stays text.
2. **Generic edges by value resolution.** When a field's value matches another
   document's title, that is an edge (`film.director → person`). When many
   documents share a value, it becomes a shared vertex (`venue`, `sport`,
   `studio`). No hand-written mapping.
3. **Generic tools.** `find_entities(kind, filters)`,
   `aggregate(kind, filters, field, op)`, `extreme(kind, filters, field)`,
   `neighbours(entity, relation)`, `sequence_step(entity, direction)` — the same
   five operations, with the *schema description passed to the agent at runtime*
   so it knows which fields exist.
4. **Schema card.** After ingestion, emit a summary of what was found — kinds,
   field names, types, cardinalities — and give it to the orchestrator in its
   prompt. The agent then plans against the actual corpus rather than against
   assumptions baked into the code months earlier.

Result: point it at a new corpus, re-run ingestion, and the agent adapts —
because it reads the schema instead of having memorised one. The Olympic
behaviour we already validated becomes a *special case* that the generic path
must continue to reproduce at 100%, enforced by the existing tests.

That is the next piece of work.
