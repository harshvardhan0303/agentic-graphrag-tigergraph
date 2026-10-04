"""Preflight check: is this machine ready to run the benchmark?

    python scripts/doctor.py

Checks environment, data integrity, build artefacts, and live connectivity to
Gemini and TigerGraph. Every failure prints the exact command that fixes it, so
a setup problem never turns into a debugging session.

Exit code 0 means ready; 1 means at least one required check failed.
"""
from __future__ import annotations

import importlib
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import ensure_loaded, redact, tg_host  # noqa: E402

# SHA-256 of the files as distributed by the organisers. The benchmark is only
# meaningful on the official corpus, so we verify rather than assume.
OFFICIAL = {
    "data/corpus.jsonl":
        "27aef30bbe32474df782f5164262d5d765af14efc9f0288320486cee585e191a",
    "data/eval_public.jsonl":
        "abddb7d18a6d8ed908f514a7e560fe4950ebe479ebb2cdb75a7456887c10c6e5",
    "data/eval_hidden.jsonl":
        "a2742f449765bb66c7ccc87a281560de3327894fe656a3b2f415f68d990c7d0a",
}

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if os.name == "nt" and not os.environ.get("WT_SESSION"):
    GREEN = RED = YELLOW = DIM = RESET = ""

results: list[tuple[str, bool, str, str]] = []   # name, ok, detail, fix


def check(name: str, ok: bool, detail: str = "", fix: str = "") -> bool:
    results.append((name, ok, detail, fix))
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {name}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    if not ok and fix:
        print(f"         {YELLOW}fix:{RESET} {fix}")
    return ok


def warn(name: str, detail: str) -> None:
    print(f"  [{YELLOW}WARN{RESET}] {name}  {DIM}{detail}{RESET}")


def section(title: str) -> None:
    print(f"\n{title}")


def sha256(path: pathlib.Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    print("=" * 68)
    print("  Agentic GraphRAG — preflight doctor")
    print("=" * 68)
    ensure_loaded()

    # ---------------- 1. environment ----------------
    section("1. Python environment")
    v = sys.version_info
    check("Python >= 3.10", v >= (3, 10), f"found {v.major}.{v.minor}.{v.micro}",
          "install Python 3.10 or newer")

    for mod, why, required in [
        ("numpy", "vector maths", True),
        ("pytest", "test suite", True),
        ("sentence_transformers", "local embeddings", False),
        ("pyTigerGraph", "TigerGraph client", True),
    ]:
        try:
            importlib.import_module(mod)
            check(f"import {mod}", True, why)
        except ImportError:
            check(f"import {mod}", required is False, f"{why} — not installed",
                  "pip install -r requirements.txt")

    # ---------------- 2. dataset ----------------
    section("2. Dataset integrity (must be the organisers' files, unmodified)")
    for rel, expected in OFFICIAL.items():
        p = pathlib.Path(rel)
        if not p.exists():
            check(rel, False, "missing",
                  f"copy it from hackathon-resources into {p.parent}/")
            continue
        got = sha256(p)
        check(rel, got == expected,
              f"sha256 {got[:12]}…" + ("" if got == expected else f" (expected {expected[:12]}…)"),
              "re-copy the original file; do not edit it")

    # ---------------- 3. build artefacts ----------------
    section("3. Build artefacts")
    gstats = pathlib.Path("data/graph/stats.json")
    if gstats.exists():
        s = json.loads(gstats.read_text())
        check("graph tables built", s.get("olympic_events") == 2162,
              f"{s.get('olympic_events')} events, "
              f"{s.get('title_parse_failures')} parse failures",
              "python -m src.ingest.parse_corpus --corpus data/corpus.jsonl --out data/graph")
    else:
        check("graph tables built", False, "data/graph/ not found",
              "python -m src.ingest.parse_corpus --corpus data/corpus.jsonl --out data/graph")

    vec = pathlib.Path("data/vectors/chunk_emb.npy")
    if vec.exists():
        import numpy as np
        m = np.load(vec, mmap_mode="r")
        emb_meta = pathlib.Path("data/vectors/embedder.json")
        name = json.loads(emb_meta.read_text())["name"] if emb_meta.exists() else "?"
        check("vector index built", m.shape[0] > 20000,
              f"{m.shape[0]} chunks x {m.shape[1]}d via {name}",
              "python -m src.vectorstore.build")
        if name.startswith("hashing"):
            warn("embedder", "hashing fallback in use — RAG numbers will understate "
                             "what real embeddings achieve. Check your network and "
                             "re-run src.vectorstore.build to get all-MiniLM-L6-v2.")
    else:
        check("vector index built", False, "data/vectors/ not found",
              "python -m src.vectorstore.build")

    # ---------------- 4. credentials ----------------
    section("4. Credentials (.env)")
    check(".env file present", pathlib.Path(".env").exists(), "",
          "copy .env.example to .env and fill it in")
    gkey = os.environ.get("GEMINI_API_KEY", "")
    check("GEMINI_API_KEY set", bool(gkey) and not gkey.startswith("your-key"),
          redact(gkey), "get one at https://aistudio.google.com/apikey")
    check("TG_HOST set", tg_host().startswith("http"), tg_host() or "(not set)",
          "paste your Savanna workspace URL, no trailing slash")
    check("TG_GRAPH set", bool(os.environ.get("TG_GRAPH")),
          os.environ.get("TG_GRAPH", ""), "the graph name you created in Savanna")
    has_auth = bool(os.environ.get("TG_SECRET") or os.environ.get("TG_TOKEN")
                    or os.environ.get("TG_PASSWORD"))
    check("TigerGraph auth set", has_auth,
          "TG_SECRET=" + redact(os.environ.get("TG_SECRET")),
          "run CREATE SECRET in the Savanna GSQL editor")

    # ---------------- 5. live connectivity ----------------
    section("5. Live connectivity")
    if gkey and not gkey.startswith("your-key"):
        try:
            from src.llm.client import GeminiClient
            t0 = time.time()
            r = GeminiClient().complete("Reply with the single word: ready",
                                        max_output_tokens=10)
            check("Gemini reachable", "ready" in r.text.lower(),
                  f"{r.input_tokens}+{r.output_tokens} tokens, "
                  f"{time.time() - t0:.1f}s, model={os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')}",
                  "check the key, and that the model name in .env exists")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            model_problem = any(t in msg.lower() for t in
                                ("not found", "404", "not supported", "unsupported"))
            check("Gemini reachable", False, msg[:160],
                  "python scripts/list_models.py   (the configured model name is "
                  "wrong or retired — this lists what your key can use)"
                  if model_problem else
                  "verify GEMINI_API_KEY in .env (key invalid, or the Generative "
                  "Language API is not enabled for its project)")
    else:
        check("Gemini reachable", False, "skipped — no key", "set GEMINI_API_KEY")

    if tg_host() and has_auth:
        try:
            from src.graph.tigergraph import connect
            conn = connect()
            echo = conn.echo()
            check("TigerGraph reachable", bool(echo), str(echo)[:80],
                  "check TG_HOST and TG_SECRET")
            try:
                ver = conn.getVer()
                major_minor = ".".join(str(ver).split(".")[:2])
                ok = tuple(int(x) for x in major_minor.split(".")) >= (4, 2)
                check("TigerGraph >= 4.2 (vector support)", ok, f"version {ver}",
                      "vector search needs 4.2+; recreate the workspace on a newer version")
            except Exception as e:
                warn("version check", f"could not read version: {e}")
            try:
                vtypes = conn.getVertexTypes()
                if vtypes:
                    counts = conn.getVertexCount("*")
                    check("schema deployed", True,
                          f"{len(vtypes)} vertex types, "
                          f"{sum(counts.values()) if isinstance(counts, dict) else '?'} vertices")
                else:
                    warn("schema deployed", "graph is empty — run: "
                                            "python -m src.graph.deploy --all")
            except Exception as e:
                warn("schema check", str(e)[:120])
        except Exception as e:
            check("TigerGraph reachable", False, f"{type(e).__name__}: {e}"[:160],
                  "check TG_HOST (no trailing slash), TG_GRAPH and TG_SECRET; "
                  "confirm the workspace is Ready and not suspended")
    else:
        check("TigerGraph reachable", False, "skipped — missing host or auth",
              "fill TG_HOST and TG_SECRET in .env")

    # ---------------- verdict ----------------
    failed = [r for r in results if not r[1]]
    print("\n" + "=" * 68)
    if not failed:
        print(f"  {GREEN}All checks passed.{RESET} Next: python -m src.graph.deploy --all")
    else:
        print(f"  {RED}{len(failed)} check(s) failed.{RESET} Fix them top to bottom:")
        for name, _ok, _d, fix in failed:
            print(f"    - {name}" + (f"  →  {fix}" if fix else ""))
    print("=" * 68)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
