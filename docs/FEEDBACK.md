# Developer feedback — TigerGraph Savanna, GSQL and pyTigerGraph

Written while building this project solo over six days, from first login to a
2,162-event graph serving eight installed queries. Everything below is a real
thing that cost real time, with the repro and the workaround we shipped. Nothing
here is a complaint about a product being hard; graph databases *are* hard. It is
a list of places where the error message or the docs could have saved an hour.

Environment: TigerGraph Savanna (cloud), free workgroup, `pyTigerGraph` over
HTTPS:443, graph `Tiger_Graph_RAG`.

---

## 1. "Database secret" vs "API key" — two different things, one word

**Cost: ~40 minutes.**

The Savanna UI offers **API keys** prominently in the workspace settings. The
docs for `pyTigerGraph` ask for a **database secret**. These are not the same
credential and are not interchangeable:

| | Created where | Scope | Used by |
|---|---|---|---|
| Savanna API key | Workgroup settings | Control plane — start/stop workspaces, billing | Savanna REST API |
| Database secret | Inside GSQL: `CREATE SECRET` | Data plane — query the graph | `pyTigerGraph`, `gsql` |

As a first-time user I created the API key, put it in `gsqlSecret=`, and got an
authentication failure that did not say *which* kind of credential was wrong.

**What we did:** ran `CREATE SECRET` from the GSQL shell inside the workspace.

**Suggestion:** when `/requesttoken` rejects a credential that parses as a
Savanna API key, say so explicitly — "this looks like a Savanna API key; data
plane access needs a database secret created with `CREATE SECRET`". A one-line
pointer in the Savanna UI next to "API Keys" would also do it.

---

## 2. Creating a "database" does not create a graph

**Cost: ~25 minutes.**

The Savanna onboarding flow asks you to name a **database** (we named ours
`Tiger_Graph_RAG`) and reports status **Ready**. Natural reading: the graph
exists and I can `USE GRAPH Tiger_Graph_RAG`. It does not, and you cannot — the
workspace is provisioned but contains no graph. `SHOW GRAPH` returns nothing and
the UI still says "no graph created", which contradicts the Ready badge.

**Suggestion:** distinguish the two in the UI. "Workspace ready · 0 graphs" is
unambiguous; "Ready" is not.

---

## 3. `CREATE GRAPH` is rejected at global scope on Savanna

**Cost: ~30 minutes.**

```gsql
CREATE GRAPH_IF_NOT_EXIST Tiger_Graph_RAG ()
```
fails with a parse error, as does plain `CREATE GRAPH` when submitted through
the same path that happily accepts `CREATE SCHEMA_CHANGE JOB`. The error is a
generic ANTLR `mismatched input`, which reads like a syntax problem in *our*
script rather than a statement that is not permitted in that context.

**What we did:** removed graph creation from `gsql/01_schema.gsql` entirely and
create the graph from Python in a guarded `try/except` before the schema job
runs (`src/graph/deploy.py`).

**Suggestion:** a scope error rather than a syntax error — "CREATE GRAPH is not
permitted here; create the graph from the Savanna UI or the `gsql` client at
global scope".

---

## 4. No `ADD VERTEX ... IF NOT EXISTS`

**Cost: ~20 minutes, recurring.**

Re-running a schema job on an existing graph fails with:

```
Semantic Check Fails: The vertex name Event is used by another object!
```

Schema DDL is otherwise idempotent-friendly (`CREATE OR REPLACE QUERY`,
`GRAPH_IF_NOT_EXIST`), so the asymmetry is surprising — and the message points at
a *name collision* rather than at "this already exists", which sent us looking
for a stray object of another type.

**What we did:** `deploy.py` reads the live schema first and skips the schema job
when the expected vertex types are already present, unless `--force-schema` is
passed. Worth noting this is exactly the behaviour that makes a deploy script
safe to re-run in CI, so most users will end up writing it.

**Suggestion:** `ADD VERTEX IF NOT EXISTS` / `ADD DIRECTED EDGE IF NOT EXISTS`,
or at minimum a distinct "already exists" error code.

---

## 5. GSQL v1 / v2 syntax mismatch is detected late and reported confusingly

**Cost: ~45 minutes. The single most expensive item on this list.**

Six of our nine queries failed to install with:

```
Query specifies V2 syntax but uses V1 pattern
```

We had written `SYNTAX v2` at the top because the docs we were reading used it,
but our patterns were written in the v1 form `FROM Seed:e -(WON_MEDAL:w)- Athlete:a`.
Two things made this hard:

1. The error names the conflict but not **which line** is v1, and in a 30-line
   query with four SELECT blocks that is a lot of surface to inspect.
2. Public documentation and blog examples mix the two dialects freely without
   labelling them, so copying two snippets from two pages produces a query that
   cannot install.

**What we did:** dropped `SYNTAX v2` and standardised on v1 throughout
`gsql/03_queries.gsql`.

**Suggestion:** report the first offending line and column. Label every code
sample in the docs with its dialect.

---

## 6. `count` is a reserved word, and the error does not say so

**Cost: ~15 minutes.**

```gsql
SumAccum<INT> @@count;     // and `count` as a query parameter name
```

produces `mismatched input 'count'` — again a bare ANTLR message. Since `count`
is the most natural name for a variable in an aggregation query, this is a
likely first-timer trap.

**What we did:** renamed to `matched`.

**Suggestion:** "`count` is a reserved keyword" in the error text, and a reserved
word list in the GSQL language reference that is findable by searching for the
word itself.

---

## 7. `CASE` / `IF` are not permitted inside `ACCUM`

**Cost: ~20 minutes.**

Conditional accumulation is a normal thing to want:

```gsql
ACCUM CASE WHEN w.medal == "gold" THEN a.@golds += 1 END   // rejected
```

**What we did:** split it into two guarded `SELECT` statements, one per branch.
This is fine, but it is not obvious that it is *the* idiom, and it doubles the
traversal.

**Suggestion:** document the supported control flow inside `ACCUM` explicitly,
with the two-SELECT pattern shown as the recommended alternative.

---

## 8. Vertex sets are unordered, and nothing says so where it matters

**Cost: ~50 minutes, and it produced a silently wrong answer first.**

This is the subtlest item here. Our `q_medalists` query returns the athletes on a
gold-medal team. The benchmark expects them **in finishing/listed order**. The
query returned the right three people in a different order on two of a hundred
questions — which is exactly the kind of bug that survives a smoke test and dies
in evaluation.

The cause is that a GSQL vertex set has no order; the order you observe is an
artefact of internal partitioning and is stable enough to look deterministic in
testing. The edge carries the ordering information (`WON_MEDAL.rank`), but a
vertex-set `PRINT` discards it.

**What we did:** carry the edge attribute onto the vertex with an accumulator and
sort client-side.

```gsql
SumAccum<INT> @rank_order;
A = SELECT a FROM E:e -(WON_MEDAL:w)- Athlete:a
    WHERE w.medal == medal
    ACCUM a.@rank_order += w.rank;
PRINT A[A.athlete_id, A.name, A.@rank_order] AS medalists;
```

**Suggestion:** a prominent note in the `PRINT` and `SELECT` documentation:
"vertex sets are unordered; to preserve an ordering carried on an edge, project
it onto the vertex with an accumulator". This is the kind of thing experienced
users know and first-time users ship a bug over.

---

## 9. Projected attributes come back prefixed with the query alias

**Cost: ~35 minutes, and it cost us 79 points of accuracy before we found it.**

```gsql
PRINT E[E.doc_id, E.title] AS events;
```

returns attribute keys `"E.doc_id"`, `"E.title"` — prefixed with the alias — and
vertex accumulators arrive as `"A.@rank_order"`, with the `@` retained. Our
client code did `attrs["doc_id"]`, got `KeyError`, swallowed it as a miss, and
the live-graph correctness gate dropped to **21/100**. Because every query
"worked", nothing pointed at serialisation.

**What we did:** a `_strip_alias()` helper in `src/graph/tigergraph.py` that
removes the alias prefix and a leading `@`. We also wrote `scripts/tg_probe.py`
to dump raw query results — finding the problem took a tool, not a guess.

**Suggestion:** either return unprefixed names, or document the prefixing on the
`PRINT` page with an example of the exact JSON shape. The round-trip from
"`PRINT E[E.doc_id]`" to "`result["events"][0]["attributes"]["E.doc_id"]`" is not
guessable.

---

## 10. `pyTigerGraph` connections are not thread-safe, and fail far from the cause

**Cost: ~30 minutes.**

We run the benchmark with a small thread pool. Sharing one `TigerGraphConnection`
across workers produces `http.client.RemoteDisconnected` partway through a run —
a network-looking error with nothing in it to suggest a concurrency problem.

**What we did:** thread-local connections plus a bounded retry that discards the
connection before retrying (`src/graph/tigergraph.py`).

**Suggestion:** state the thread-safety contract in the `TigerGraphConnection`
docstring. One sentence — "not thread-safe; create one connection per thread" —
would have saved the whole debugging session.

---

## 11. Vector attributes: syntax differs from the rest of schema DDL

**Cost: ~20 minutes.**

`CREATE VECTOR ATTRIBUTE` was rejected inside our schema-change job, and the
accepted form differs between versions. Since vector support is new and its
availability varies by deployment, a script that assumes it will fail on
workspaces where it is absent.

**What we did:** moved it into a separate `gsql/04_vector.gsql` run tolerantly,
and made the vector upload probe for `Chunk.chunk_emb` and skip cleanly when the
attribute does not exist. The pipeline degrades to the local FAISS index rather
than failing the deploy.

**Suggestion:** a version/capability matrix for vector features, and a
`SHOW VECTOR ATTRIBUTE`-style way to detect support at runtime.

---

## What worked well

Worth saying, because this list is otherwise one-sided:

- **Accumulators are genuinely excellent.** Once `SumAccum` clicked, expressing
  "carry this edge attribute onto that vertex during traversal" was a one-liner.
  Nothing in a relational model is that direct.
- **`CREATE OR REPLACE QUERY` + `INSTALL QUERY` is a good iteration loop.** Query
  development felt like editing code, not administering a database.
- **Savanna provisioning was fast and did not need babysitting.** Workspace from
  zero to Ready in a couple of minutes.
- **Loading jobs handled 2,951 documents → 2,162 events, 6,220 athletes and
  8,440 medal edges without tuning.** We never had to think about batch size.
- **`REVERSE_EDGE` as a declaration rather than a second edge type** kept the
  schema honest; several of our queries traverse backwards and none of them
  needed a duplicate.

---

## One non-TigerGraph note, for completeness

Gemini 2.5+ "thinking" models spend the `maxOutputTokens` budget on hidden
reasoning before emitting any text. Our first live call returned `8 input + 0
output` tokens and an empty string, which looked like an API failure. The fix was
`thinkingConfig.thinkingBudget = 0`, a 512-token output floor, and counting
`thoughtsTokenCount` in our own token accounting — otherwise every benchmark
number would have under-reported the true cost. Recorded here because any team
benchmarking token efficiency on a 2.5+ model will hit it.
