# Social post drafts — TigerGraph Agentic GraphRAG Hackathon

The form asks for a public social post tagging **@TigerGraph**. Pick one, post it,
paste the URL into the form. LinkedIn is the safer choice for reach among the
judges; X/Twitter if you prefer brevity.

---

## Option A — LinkedIn (recommended)

> I built an agentic GraphRAG system for the @TigerGraph hackathon to beat a
> GraphRAG baseline. It lost. That turned out to be the finding.
>
> 2,951 Wikipedia documents, 150 questions, three pipelines over one graph —
> plain RAG, single-pass GraphRAG, and an agent that chooses its own retrieval
> steps. 400 benchmarked runs on TigerGraph Savanna.
>
> Results:
> → RAG: 43% accuracy
> → GraphRAG: 97%, 2.1k tokens/question
> → Agentic GraphRAG: 93%, 6.4k tokens/question
>
> Per question type, the story gets sharper. Aggregation questions ("how many
> wrestling events had more than 20 competitors?") go from 4.8% with plain RAG to
> 100% with the graph. Superlatives: 20% → 100%. These aren't tuning differences —
> they're questions that top-k retrieval structurally cannot answer, because
> they need 10–43 documents held simultaneously, and that's an aggregation, not
> a retrieval.
>
> So the graph is worth ~50 points. The agent is worth ~0.
>
> Why? Every question in this benchmark comes from one of five templates, so the
> retrieval route is always knowable in advance. Adaptive retrieval has nothing
> to adapt to. It spends 3–5 extra LLM calls rediscovering a route that was never
> going to change.
>
> The honest version: adaptive retrieval pays when the route is unknown in
> advance. On a templated benchmark, it never is.
>
> I also shipped the ablation that undercuts my own favourite feature — a
> cost-aware router that buys 1 point of accuracy (93% vs 92%) for 19% more
> tokens, which on accuracy-per-token is a clear loss. A feature you can't turn
> off is a feature you can't evaluate.
>
> Full write-up, benchmark harness and eleven pages of developer feedback on
> GSQL and Savanna in the repo: https://github.com/harshvardhan0303/agentic-graphrag-tigergraph
>
> #TigerGraph #GraphRAG #KnowledgeGraphs #AgenticAI #RAG

---

## Option B — X / Twitter (thread)

**1/**
> Built an agentic GraphRAG system for the @TigerGraph hackathon to beat a
> GraphRAG baseline.
>
> It lost.
>
> That's the finding. 🧵

**2/**
> Setup: 2,951 docs, 150 questions, 3 pipelines over ONE graph, ONE tool layer,
> ONE answer contract. The only variable: does retrieval react to what it finds?
>
> 400 graded runs on TigerGraph Savanna.

**3/**
> RAG: 43% · 2.0k tok
> GraphRAG: 97% · 2.1k tok
> Agentic: 93% · 6.4k tok
>
> The agent is 4 points behind and 3× the cost.

**4/**
> Per question type is where it gets interesting.
>
> Aggregation: RAG 4.8% → graph 100%
> Superlative: RAG 20% → graph 100%
> Temporal: RAG 27% → graph 100%
>
> The graph is worth ~50 points. The agent is worth ~0.

**5/**
> Why zero? All 150 questions come from 5 templates. The retrieval route is always
> knowable in advance, so adaptivity has nothing to adapt to.
>
> Adaptive retrieval pays when the route is UNKNOWN. On a templated benchmark,
> it never is.

**6/**
> Bonus: my favourite feature — a cost-aware router — is a bad trade. +1 point of
> accuracy (93% vs 92%) for 19% more tokens. On accuracy-per-token it loses.
>
> I shipped the ablation that disproves it. A feature you can't turn off is a
> feature you can't evaluate.

**7/**
> Repo, benchmark harness, and 11 pages of GSQL/Savanna developer feedback
> (unordered vertex sets cost me 50 minutes and a wrong answer):
>
> https://github.com/harshvardhan0303/agentic-graphrag-tigergraph

---

## Notes before posting

- Replace `https://github.com/harshvardhan0303/agentic-graphrag-tigergraph` with the public GitHub URL.
- Verify the three headline numbers against the final `out/public/summary.json` —
  they shift by a point between runs.
- On LinkedIn, tag the **TigerGraph company page** rather than typing the text
  `@TigerGraph`, or the mention won't register.
- Screenshot `out/dashboard.html` and attach it; the per-qtype table is the part
  people stop scrolling for.
