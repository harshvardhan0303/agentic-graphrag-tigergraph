# I built an agentic GraphRAG system to beat GraphRAG. It lost. Here's why that's the interesting part.

*Built solo for the TigerGraph Agentic GraphRAG Hackathon, Round 1. Six days,
2,951 documents, 400 benchmarked runs, one uncomfortable result.*

---

## The brief, and the trap inside it

The hackathon hands you 2,951 Wikipedia documents and 150 questions, and asks you
to build an agentic GraphRAG system on TigerGraph. The obvious move is to build
the agent, show it beating a baseline, and call it a day.

But the guidebook doesn't actually ask you to win. It asks you to

> figure out which questions need an agent, and which don't.

That is a different task, and taking it literally is what this write-up is about.
To answer it you need a baseline you have genuinely tried to make good — because
a benchmark where your agent beats a strawman tells you nothing, and any judge
who has built one of these will know it.

So I built three pipelines over **one** corpus, **one** graph, **one** tool layer
and **one** answer contract, differing in exactly one variable:

| | Retrieval | Reacts to what it finds? |
|---|---|---|
| **RAG** | vector top-k | no |
| **GraphRAG** | fixed graph traversal + vectors | no |
| **Agentic GraphRAG** | agent-selected, step by step | **yes** |

Same model. Same prompts where they can be shared. Same graph. The only thing
that varies is whether retrieval is allowed to change its mind.

---

## Reading the corpus before writing any code

Five minutes with the data decided the whole architecture.

The corpus is 2,162 Olympic event pages, each carrying a machine-readable
infobox — venue, date, number of competitors, number of nations, and the gold,
silver and bronze medalists — plus 789 distractors (546 films, 152 biographies, 91 other)
that exist purely to punish naive similarity search.

The 150 questions come from five templates. Two of them are fatal to top-k
retrieval:

> *How many wrestling events at the 2008 Summer Olympics had more than 20 competitors?*

> *Which shooting event at the 2016 Summer Olympics had the highest number of competitors?*

Answering either requires holding **10 to 43 documents simultaneously** and
computing over all of them. No value of *k* fixes this. Retrieve 10 and you miss
events; retrieve 50 and the model is counting inside a 40,000-token haystack. The
operation isn't retrieval at all — it's aggregation, and aggregation belongs in a
database.

That gave me the design principle the whole repo follows:

> **Put the facts where they can be computed on, not where they have to be read.**

Every infobox field becomes a typed vertex attribute. "How many events had more
than 20 competitors" becomes a `WHERE` clause over an integer column. The LLM
never counts. It never does arithmetic. It never does date math.

### The traps in the text

Three normalisation details, each of which silently costs accuracy:

- `Men's 68kg` and `Men's 68 kg` are the same event; some infoboxes drop the space.
- `Men's +80 kg` and `Men's 80 kg` are **different events**. Strip punctuation
  naively and you merge them and return the wrong medalist, with total confidence.
- Team medalist fields arrive as `FirstPersonSecondPersonThirdPerson` — the `<br>`
  tags were lost in conversion. The only remaining boundary signal is a
  lower→upper case transition.

All three are unit-tested. None of them are interesting. All of them are the
difference between 93% and something much worse.

---

## What "agentic" has to mean to be worth the name

An agent that always calls the same three tools in the same order is a pipeline
with extra latency. For the comparison to measure anything, the agentic pipeline
had to be able to do something the fixed one structurally cannot:

- **choose** the next tool from a typed registry, given what it has learned so far
- **evaluate** whether the evidence is sufficient, and name the gap if not
- **escalate** when a cheap strategy stalls — the harness tracks a "barren streak"
  and the orchestrator switches strategy when retrieval stops producing new evidence
- **stop**, on sufficiency, budget, or lack of progress

Every step is appended to a trace that ships with every answer: which agent, which
tool, which arguments, what came back, how many tokens, whether it was new.

I also gave it a **cost-aware router**: classify the question, pick the cheapest
strategy that can answer it correctly (3 steps for a direct graph lookup, 8 for a
document fallback), escalate only when that proves insufficient. And then — this
turns out to matter — I shipped the ablation that turns the router off.

---

## The results

100 questions × 4 configurations = 400 graded runs against a live TigerGraph
Savanna instance.

| Pipeline | Accuracy | Tokens/question | **Accuracy per 1k tokens** |
|---|---|---|---|
| RAG | 43% | 2,020 | 0.213 |
| **GraphRAG** | **97%** | **2,106** | **0.461** |
| Agentic (no router) | 92% | 5,393 | 0.171 |
| Agentic (routed) | 93% | 6,442 | 0.144 |

**The agent lost.** It is four points behind single-pass GraphRAG on accuracy
and three times more expensive. On the efficiency metric that the brief actually asks
about, it is beaten by a factor of three.

I spent a day trying to make this result go away. Then I read the per-question-type
breakdown and realised it was the answer.

| Question type | RAG | GraphRAG | Agentic |
|---|---|---|---|
| aggregation | **4.8%** | 100% | 100% |
| superlative | 20% | 100% | 100% |
| temporal | 27% | 100% | 95.5% |
| multi-hop | 54% | 89% | 82% |
| lookup | 100% | 100% | 95% |

Read that table column by column and it tells you exactly where the value is.

**The graph is worth about fifty points.** Aggregation goes from 4.8% to 100%;
superlative from 20% to 100%; temporal from 27% to 100%. That is not a tuning
difference, it is a category difference — these questions are *unanswerable* by
similarity search and *trivial* once the facts are typed columns and the "Games
immediately before 2016" is a single `PRECEDES` edge rather than something the
model has to know.

**The agent is worth approximately zero — on this benchmark.** And the reason is
visible in the same table: every question type here is answerable by a *fixed*
traversal, because all 150 questions come from five known templates. Adaptive
retrieval has nothing to adapt to. The agent spends three to five extra LLM calls
rediscovering, per question, a route that was always going to be the same route.

The honest statement of the result is therefore not "agents don't work". It is:

> **Adaptive retrieval only pays when the route is unknown in advance. On a
> benchmark generated from five templates, it never is.**

Which is, I think, precisely what the brief was asking me to find out.

### Also: my own best feature made things worse

The cost-aware router — the thing I was proudest of — is a **bad trade**: it buys one point of accuracy (93%
against 92%) for 19% more tokens, which on accuracy-per-1k-tokens is a clear
loss — 0.144 against 0.171. It escalates past its own
initial strategy on 14% of questions, which means its budget was simply wrong
there, and the escalation costs more than starting at the right budget would have.

The ablation that disproves it ships in the repo and runs with one flag. A feature
you can't turn off is a feature you can't evaluate.

---

## Two things that broke, and what they taught me

### A graph bug that hid behind a dictionary

Temporal questions resolve "the Games immediately before 2016" by traversing a
`PRECEDES` edge. My first implementation followed the edge and returned the
predecessor. For 2016 it returned 1900.

The cause is that **several** `PRECEDES` edges can point into one Games node,
because sports return to the Olympics on their own cycles — golf came back in
2016 after an absence since 1904, rugby after 1924. "Immediately before" isn't
"whichever predecessor is stored", it's *the greatest year below the target*.

What makes this worth telling: the bug was invisible locally, because my in-memory
store used a dict and last-write-wins happened to pick the right edge. It only
surfaced on the real graph. **Building two backends behind one interface — a
local store for tests and TigerGraph for the real thing — caught a class of bug
that neither would have caught alone.** I had built the second backend for
convenience. It paid for itself as a correctness oracle.

### A serialisation detail that cost 79 points

After the first TigerGraph deploy, my graph correctness gate — a regex-driven
harness that drives the tool layer with no LLM in the loop, so any failure is a
data bug rather than a prompting bug — scored **21/100**. Every query installed.
Every query "worked".

`PRINT E[E.doc_id, E.title] AS events` returns attribute keys named
`"E.doc_id"` — prefixed with the query alias. My client asked for `doc_id`, got a
`KeyError`, and counted it as a miss.

Finding it required writing a tool (`scripts/tg_probe.py`, dumps raw query JSON)
rather than staring harder at code. 21 → 98. Then the medalist-ordering bug above:
98 → **100/100**.

That gate is the single most useful thing in the repo. It separates "the graph is
wrong" from "the prompt is wrong", and those two failures look identical from the
outside.

---

## The thing I nearly got wrong, and am still slightly embarrassed about

Partway through, my evidence-completeness metric showed GraphRAG answering 100%
of aggregation questions while citing only 28% of the required documents. That is
a *great* headline: right answers it couldn't justify.

It was my own bug. The pipeline was letting the model's self-reported citation
list override what the system had actually retrieved, and the model names two or
three documents when the query genuinely scanned fourteen.

When I fixed it, GraphRAG's completeness went to 97% and my headline evaporated.
Then I found that the fix had left the two pipelines measuring citations by
*different rules*, which would have made the comparison meaningless in a way no
reader could have detected.

So the repo now has a written, single definition —
`citations = every document the system actually read` — applied identically to all
three pipelines, with the model's own claim preserved separately for inspection.

I'm including this because benchmark write-ups never include it, and the
difference between a benchmark and a marketing claim is entirely in how the
measurement was arrived at.

---

## What the agent actually buys

With one citation rule applied to all three pipelines, a difference appears that
the bug had been hiding — and it is not an accuracy difference.

| Question type | GraphRAG evidence precision | Agentic evidence precision |
|---|---|---|
| multi-hop | 16% | **100%** |
| temporal | 16% | **100%** |
| lookup | 13% | **85%** |

GraphRAG runs the same traversal whatever you ask it, so it nearly always
reaches the gold documents — 98% completeness — while five of every six
documents it pulls on a multi-hop question are irrelevant. The agent reaches 94%
of the gold documents having read almost nothing it did not need.

So the result is not "the agent is worse". It is:

> The agent does not buy accuracy on this benchmark. It buys **attribution** —
> the same answers with an evidence set 2.4x cleaner, because every retrieval
> was a decision rather than a default.

Which tells you when each design is right: GraphRAG when you want the answer,
the agent when a human has to check the work. If "here is the answer" is enough,
the fixed traversal wins on every axis. If someone has to audit the citations —
compliance, research, anything regulated — reading 2.4x fewer irrelevant
documents to reach the same answer is the entire game.

---

## What I'd do with another week

The result above is a statement about *this* benchmark, and the obvious next
question is where its boundary lies. Multi-hop is the weakest class for every
pipeline — 89% for GraphRAG, 82% for the agent — and it's the class with the most route
variation. That is where I'd look for the regime in which the agent earns its
tokens: questions whose shape isn't known in advance, where a fixed traversal has
nothing to be fixed *to*.

I'd also make the router learn its budgets from observed escalations instead of
guessing them, since the data for that is already sitting in the traces.

---

## Try it

Everything runs offline with `--backend mock`, which swaps the LLM for a
deterministic policy so you can exercise the full agent loop — router,
orchestrator, evaluator, answerer — before spending a token:

```bash
make setup && make ingest && make vectors
make test          # 28 unit tests, no API key, no network
make oracle        # graph correctness gate: expects 100/100
make bench-public  # the table above
```

Repo: https://github.com/harshvardhan0303/agentic-graphrag-tigergraph
Live dashboard: https://harshvardhan0303.github.io/agentic-graphrag-tigergraph/

Built on TigerGraph Savanna with pyTigerGraph and Gemini.

*The detailed developer feedback — eleven specific friction points with repros,
from `CREATE SECRET` to unordered vertex sets — is in `docs/FEEDBACK.md`.*
