"""Build the local vector index from chunks.jsonl.

    python -m src.vectorstore.build --chunks data/graph/chunks.jsonl --out data/vectors
"""
from __future__ import annotations

import argparse
import json
import time

from ..config import ensure_loaded
from .index import LocalVectorIndex, build_embedder


def main() -> None:
    ensure_loaded()
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/graph/chunks.jsonl")
    ap.add_argument("--out", default="data/vectors")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    embedder = build_embedder(args.embedder)
    print(f"embedder: {embedder.name} (dim={embedder.dim})")

    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8") if l.strip()]
    print(f"chunks: {len(chunks)}")

    t0 = time.time()
    idx = LocalVectorIndex(embedder)
    idx.build(chunks, batch_size=args.batch_size)
    idx.save(args.out)
    print(f"built {idx.matrix.shape} in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
