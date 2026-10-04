"""Vector store over document chunks.

Two backends with one interface:

* :class:`TigerGraphVectorIndex` -- queries the ``chunk_emb`` vector attribute
  on the Chunk vertex in TigerGraph Vector DB. Used for the submission.
* :class:`LocalVectorIndex` -- numpy cosine over a memory-mapped matrix. Used
  for tests and as a fallback; identical ranking semantics.

Embedders are pluggable. ``all-MiniLM-L6-v2`` (384-d, local, free) is the
default so the benchmark is reproducible by a judge without an embedding bill;
Gemini ``text-embedding-004`` (768-d) is available for a quality comparison.
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Iterable, Protocol

import numpy as np


class Embedder(Protocol):
    dim: int
    name: str
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    name = "all-MiniLM-L6-v2"
    dim = 384

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy
        self.name = model_name or os.environ.get("EMBED_MODEL", self.name)
        self._m = SentenceTransformer(self.name)
        self.dim = self._m.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return np.asarray(self._m.encode(texts, batch_size=batch_size,
                                         normalize_embeddings=True,
                                         show_progress_bar=False), dtype="float32")


class HashingEmbedder:
    """Dependency-free fallback: hashed character n-grams, L2-normalised.

    Not competitive with a neural encoder, and we say so in the write-up. It
    exists so that ``pytest`` and a cold-start judge can run the whole pipeline
    with zero downloads.
    """
    name = "hashing-ngram"

    def __init__(self, dim: int = 384, ngram: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.lo, self.hi = ngram

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            t = " " + t.lower() + " "
            for n in range(self.lo, self.hi + 1):
                for j in range(len(t) - n + 1):
                    out[i, hash(t[j:j + n]) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-9, None)


class GeminiEmbedder:
    """Google ``text-embedding-004`` via REST batch endpoint (768-d).

    Useful when the machine running the benchmark cannot download model
    weights, and as a quality comparison against the local encoder.
    """
    name = "text-embedding-004"
    dim = 768
    URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "{model}:batchEmbedContents")

    def __init__(self, api_key: str | None = None, model: str | None = None):
        import urllib.request  # noqa: F401  (kept local to the class)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY required for GeminiEmbedder")
        self.name = model or os.environ.get("EMBED_MODEL", self.name)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        import json as _json
        import time as _time
        import urllib.request as _rq
        out: list[list[float]] = []
        url = self.URL.format(model=self.name)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            body = {"requests": [
                {"model": f"models/{self.name}",
                 "content": {"parts": [{"text": t[:8000]}]}} for t in batch]}
            req = _rq.Request(url, data=_json.dumps(body).encode(),
                              headers={"Content-Type": "application/json",
                                       "x-goog-api-key": self.api_key},
                              method="POST")
            for attempt in range(5):
                try:
                    with _rq.urlopen(req, timeout=120) as r:
                        payload = _json.loads(r.read().decode())
                    out.extend(e["values"] for e in payload["embeddings"])
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    _time.sleep(2 ** attempt)
        arr = np.asarray(out, dtype="float32")
        self.dim = arr.shape[1]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-9, None)


class LocalVectorIndex:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.matrix: np.ndarray | None = None
        self.meta: list[dict[str, Any]] = []

    # ---------- build / persist ----------
    def build(self, chunks: Iterable[dict[str, Any]], batch_size: int = 128,
              progress_every: int = 5000) -> None:
        rows, meta = [], []
        buf: list[str] = []
        for i, c in enumerate(chunks):
            buf.append(f"{c['title']}\n{c['text']}")
            meta.append({k: c[k] for k in ("chunk_id", "doc_id", "title")}
                        | {"text": c["text"]})
            if len(buf) >= batch_size:
                rows.append(self.embedder.encode(buf))
                buf = []
            if progress_every and i and i % progress_every == 0:
                print(f"  embedded {i} chunks...", flush=True)
        if buf:
            rows.append(self.embedder.encode(buf))
        self.matrix = np.vstack(rows).astype("float32") if rows else np.zeros((0, self.embedder.dim), "float32")
        self.meta = meta

    def save(self, out_dir: str | pathlib.Path) -> None:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "chunk_emb.npy", self.matrix)
        with open(out / "chunk_meta.jsonl", "w", encoding="utf-8") as fh:
            for m in self.meta:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        (out / "embedder.json").write_text(
            json.dumps({"name": self.embedder.name, "dim": int(self.embedder.dim)}),
            encoding="utf-8")

    @classmethod
    def load(cls, out_dir: str | pathlib.Path, embedder: Embedder) -> "LocalVectorIndex":
        out = pathlib.Path(out_dir)
        idx = cls(embedder)
        idx.matrix = np.load(out / "chunk_emb.npy")
        idx.meta = [json.loads(l) for l in open(out / "chunk_meta.jsonl", encoding="utf-8")]
        return idx

    # ---------- query ----------
    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if self.matrix is None or not len(self.matrix):
            return []
        q = self.embedder.encode([query])[0]
        scores = self.matrix @ q
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [dict(self.meta[i], score=float(scores[i])) for i in top]


class TigerGraphVectorIndex:
    """Vector search delegated to TigerGraph Vector DB.

    Uses pyTigerGraph's vector search over the ``chunk_emb`` attribute so that
    retrieval happens next to the graph, not in the client.
    """

    def __init__(self, conn, embedder: Embedder, vertex: str = "Chunk",
                 attr: str = "chunk_emb"):
        self.conn = conn
        self.embedder = embedder
        self.vertex = vertex
        self.attr = attr

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        vec = self.embedder.encode([query])[0].tolist()
        rows = self.conn.searchVector(
            data=vec, vertexType=self.vertex, vectorAttribute=self.attr, k=k)
        out = []
        for r in rows:
            attrs = r.get("attributes", r)
            out.append({
                "chunk_id": r.get("v_id", attrs.get("chunk_id", "")),
                "doc_id": attrs.get("doc_id", ""),
                "title": attrs.get("title", ""),
                "text": attrs.get("text", ""),
                "score": float(r.get("score", attrs.get("score", 0.0))),
            })
        return out


def build_embedder(kind: str | None = None) -> Embedder:
    kind = (kind or os.environ.get("EMBEDDER", "sentence-transformers")).lower()
    if kind in ("st", "sentence-transformers", "minilm"):
        try:
            return SentenceTransformerEmbedder()
        except Exception as e:  # missing dependency or no network for weights
            print(f"[vectorstore] falling back to hashing embedder: {e}")
            return HashingEmbedder()
    if kind in ("gemini", "text-embedding-004"):
        return GeminiEmbedder()
    if kind in ("hash", "hashing"):
        return HashingEmbedder()
    raise ValueError(f"unknown embedder {kind!r}")
