"""Deploy the graph to TigerGraph: schema, data, queries, vectors.

    python -m src.graph.deploy --all
    python -m src.graph.deploy --schema --load --install --vectors

Idempotent: safe to re-run. Verifies vertex/edge counts at the end and then
runs the graph oracle against the live database so a load is never declared
successful on counts alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import sys
import time


from ..config import ensure_loaded


def _read_gsql(path: pathlib.Path) -> str:
    """Read a GSQL script and substitute the configured graph name.

    The graph name is a deployment choice, not a property of the schema, so the
    .gsql files carry a $GRAPH_NAME placeholder rather than a hard-coded name.
    Whatever you called the graph in Savanna goes in TG_GRAPH and everything
    downstream follows.
    """
    graph = os.environ.get("TG_GRAPH", "OlympicGraphRAG")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", graph):
        raise SystemExit(
            f"TG_GRAPH={graph!r} is not a valid TigerGraph graph name "
            "(letters, digits and underscores; must start with a letter)")
    return path.read_text(encoding="utf-8").replace("$GRAPH_NAME", graph)


def run_gsql(conn, script: str, label: str) -> None:
    print(f"--- {label} ---")
    out = conn.gsql(script)
    print(out if isinstance(out, str) else json.dumps(out)[:2000])


def upsert_from_csv(conn, path: pathlib.Path, vertex_type: str,
                    id_col: str, batch: int = 5000) -> int:
    """Vertex upsert over REST.

    Used instead of a server-side loading job when the CSVs live on the client
    (which is the normal case for Savanna). Loading jobs remain in
    gsql/02_load.gsql for a self-managed instance with filesystem access.
    """
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    total = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        payload = {}
        for r in chunk:
            attrs = {k: (int(v) if k in ("year", "competitors", "nations") and v
                         not in ("", None) else v)
                     for k, v in r.items() if k != id_col}
            payload[r[id_col]] = {k: (v, "") for k, v in attrs.items()}
        total += conn.upsertVertices(vertex_type, [
            (vid, {k: v[0] for k, v in a.items()}) for vid, a in payload.items()])
    return total


def upsert_edges(conn, path: pathlib.Path, src_type: str, edge_type: str,
                 tgt_type: str, src_col: str, tgt_col: str,
                 attr_cols: tuple[str, ...] = (), batch: int = 10000) -> int:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    total = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        edges = []
        for r in chunk:
            attrs = {c: (int(r[c]) if c == "rank" else r[c]) for c in attr_cols}
            edges.append((r[src_col], r[tgt_col], attrs))
        total += conn.upsertEdges(src_type, edge_type, tgt_type, edges)
    return total


def main() -> int:
    ensure_loaded()
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graph")
    ap.add_argument("--gsql-dir", default="gsql")
    ap.add_argument("--vectors", default="data/vectors")
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--force-schema", action="store_true",
                    help="re-apply the schema even if it already exists")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--load-vectors", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        args.schema = args.load = args.install = args.load_vectors = args.verify = True

    from .tigergraph import TigerGraphStore, connect
    conn = connect()
    print(f"connected to {os.environ.get('TG_HOST')} "
          f"graph={os.environ.get('TG_GRAPH', 'OlympicGraphRAG')}")

    gsql_dir = pathlib.Path(args.gsql_dir)
    gdir = pathlib.Path(args.graph_dir)

    if args.schema:
        # Idempotency: ADD VERTEX on a type that already exists is an error, so
        # re-running --all on a deployed graph would print a wall of failures.
        # Skip when the schema is already present; --force-schema overrides.
        try:
            existing = set(conn.getVertexTypes())
        except Exception:
            existing = set()
        if {"Event", "Games", "Sport"} <= existing and not args.force_schema:
            print("--- schema --- already present "
                  f"({len(existing)} vertex types); skipping. "
                  "Use --force-schema to re-apply.")
        else:
            run_gsql(conn, _read_gsql(gsql_dir / "01_schema.gsql"), "schema")

    if args.load:
        t0 = time.time()
        specs = [
            ("games.csv", "Games", "games_id"),
            ("sport.csv", "Sport", "sport_id"),
            ("venue.csv", "Venue", "venue_id"),
            ("athlete.csv", "Athlete", "athlete_id"),
            ("nation.csv", "Nation", "noc"),
            ("event.csv", "Event", "doc_id"),
        ]
        for fname, vtype, idcol in specs:
            n = upsert_from_csv(conn, gdir / fname, vtype, idcol)
            print(f"  {vtype:<8} {n:>6} vertices")
        edge_specs = [
            ("e_in_sport.csv", "Event", "IN_SPORT", "Sport", "doc_id", "sport_id", ()),
            ("e_at_games.csv", "Event", "AT_GAMES", "Games", "doc_id", "games_id", ()),
            ("e_held_at.csv", "Event", "HELD_AT", "Venue", "doc_id", "venue_id", ()),
            ("e_medal.csv", "Event", "WON_MEDAL", "Athlete", "doc_id", "athlete_id",
             ("medal", "rank")),
            ("e_represents.csv", "Athlete", "REPRESENTS", "Nation", "athlete_id", "noc", ()),
            ("e_precedes.csv", "Games", "PRECEDES", "Games", "from_games", "to_games", ()),
        ]
        for fname, st, et, tt, sc, tc, attrs in edge_specs:
            n = upsert_edges(conn, gdir / fname, st, et, tt, sc, tc, attrs)
            print(f"  {et:<12} {n:>6} edges")
        print(f"  load took {time.time() - t0:.1f}s")

    if args.install:
        run_gsql(conn, _read_gsql(gsql_dir / "03_queries.gsql"), "queries")

    if args.load_vectors:
        import numpy as np
        # Create the vector attribute first. This is run tolerantly and on its
        # own because CREATE VECTOR ATTRIBUTE is not available on every build
        # and its syntax has changed between them (docs/FEEDBACK.md #11). The
        # graph layer - which is what accuracy depends on - must deploy whether
        # or not this succeeds, so a failure here is reported and stepped over
        # rather than raised.
        try:
            run_gsql(conn, _read_gsql(gsql_dir / "04_vector.gsql"), "vector attribute")
        except Exception as e:
            print(f"  CREATE VECTOR ATTRIBUTE not accepted on this build "
                  f"({str(e)[:140]}).")

        vdir = pathlib.Path(args.vectors)
        matrix = np.load(vdir / "chunk_emb.npy")
        meta = [json.loads(l) for l in open(vdir / "chunk_meta.jsonl", encoding="utf-8")]
        print(f"  uploading {len(meta)} chunk vectors (dim={matrix.shape[1]})")
        try:
            conn.upsertVertices("Chunk", [("__probe__", {"doc_id": "probe"})])
            conn.upsertVertices("Chunk", [("__probe__",
                                           {"chunk_emb": matrix[0].tolist()})])
        except Exception as e:
            print(f"  SKIPPING vector upload: the Chunk.chunk_emb vector "
                  f"attribute does not exist on this build ({str(e)[:120]}).\n"
                  f"  The RAG pipeline uses the local vector index instead; the "
                  f"graph layer is unaffected. Record this in docs/FEEDBACK.md.")
            meta = []
        batch = 500
        for i in range(0, len(meta), batch):
            verts = []
            for j, m in enumerate(meta[i:i + batch]):
                verts.append((m["chunk_id"], {
                    "doc_id": m["doc_id"], "title": m["title"],
                    "text": m["text"][:6000]}))
            conn.upsertVertices("Chunk", verts)
            # embeddings go in a second pass so a partial failure retries cheaply
            conn.upsertVertices("Chunk", [
                (m["chunk_id"], {"chunk_emb": matrix[i + j].tolist()})
                for j, m in enumerate(meta[i:i + batch])])
            if i and i % 5000 == 0:
                print(f"    {i} chunks")
        print("  vectors uploaded")

    if args.verify:
        store = TigerGraphStore(conn)
        counts = conn.getVertexCount("*")
        print("vertex counts:", json.dumps(counts, indent=2))
        print("edge counts:", json.dumps(conn.getEdgeCount("*"), indent=2))
        print("\nrunning graph oracle against the live database...")
        from ..bench import oracle
        ok = oracle_against(store)
        return 0 if ok else 1
    return 0


def oracle_against(store) -> bool:
    """Run the deterministic oracle against a live store."""
    from ..bench.oracle import solve
    from ..ingest.normalize import norm
    qs = [json.loads(l) for l in open("data/eval_public.jsonl", encoding="utf-8")
          if l.strip()]
    good = 0
    for q in qs:
        pred, _ev, _tr = solve(store, q["question"])
        if pred is not None and any(norm(pred) == norm(g) for g in q["answer"]):
            good += 1
        else:
            print(f"  MISMATCH {q['qid']}: pred={pred!r} gold={q['answer']}")
    print(f"live-graph oracle: {good}/{len(qs)} = {good / len(qs):.1%}")
    return good == len(qs)


if __name__ == "__main__":
    sys.exit(main())
