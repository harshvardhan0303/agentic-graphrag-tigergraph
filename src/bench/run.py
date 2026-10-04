"""Benchmark runner: every question, every pipeline, one results file.

    python -m src.bench.run --questions data/eval_public.jsonl --out out/public
    python -m src.bench.run --questions data/eval_hidden.jsonl --out out/hidden --no-grade

Writes:
    <out>/results.jsonl   one row per (question, pipeline) with answer+metrics
    <out>/answers.jsonl   per-question answers with full trace and usage
    <out>/summary.json    aggregated tables consumed by the dashboard
    <out>/submission.jsonl  hidden-set format: answer + tokens + agentic trace
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import ensure_loaded
from ..agents.harness import Budget
from ..agents.tools import build_registry
from ..graph.store import LocalGraphStore
from ..llm.client import UsageMeter, build_client
from ..pipelines.agentic import AgenticGraphRAGPipeline, AgenticNoRouterPipeline
from ..pipelines.graphrag import GraphRAGPipeline
from ..pipelines.rag import RAGPipeline
from ..vectorstore.index import LocalVectorIndex, build_embedder
from .judge import Judge
from .metrics import doc_precision, doc_recall, exact_match, summarize


def load_documents(path: str) -> dict[str, dict]:
    docs = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                docs[d["doc_id"]] = d
    return docs


def load_sport_vocabulary(graph_dir: str) -> list[str]:
    """Sport names for the router prompt.

    Read from the ingested table rather than scanned out of the database: it is
    a static vocabulary, identical in both backends, and fetching it per run
    would cost a query for no benefit.
    """
    import csv as _csv
    path = os.path.join(graph_dir, "sport.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return sorted(r["name"] for r in _csv.DictReader(fh))


def build_all(args, meter: UsageMeter):
    if args.graph_backend == "tigergraph":
        from ..graph.tigergraph import TigerGraphStore
        store = TigerGraphStore()
        print(f"graph backend: TigerGraph Savanna ({os.environ.get('TG_HOST')}, "
              f"graph={os.environ.get('TG_GRAPH')})")
    else:
        store = LocalGraphStore(args.graph)
        print("graph backend: in-memory (local)")
    embedder = build_embedder(args.embedder)
    vectors = LocalVectorIndex.load(args.vectors, embedder)
    documents = load_documents(os.path.join(args.graph, "documents.jsonl"))
    registry = build_registry(store, vectors=vectors, documents=documents)
    sports = load_sport_vocabulary(args.graph) or sorted(
        {e.sport for e in store.events.values()})

    llm = build_client(args.backend, model=args.model,
                       min_interval_s=args.min_interval)

    budget = Budget(max_steps=args.max_steps, max_tokens=args.max_tokens)
    pipelines = {
        "rag": RAGPipeline(llm, vectors, meter, top_k=args.top_k),
        "graphrag": GraphRAGPipeline(llm, registry, vectors, meter, sports),
        "agentic": AgenticGraphRAGPipeline(llm, registry, meter, sports, budget),
    }
    if args.ablation:
        pipelines["agentic_no_router"] = AgenticNoRouterPipeline(
            llm, registry, meter, sports, budget)
    if args.only:
        keep = set(args.only.split(","))
        pipelines = {k: v for k, v in pipelines.items() if k in keep}
    return pipelines, llm


def _error_report(rows: list[dict]) -> dict:
    """Which runs died on infrastructure rather than being answered wrongly."""
    errs = [r for r in rows if r.get("error")]
    if not errs:
        return {"n": 0, "by_pipeline": {}, "kinds": {}, "qids": []}
    by_pipeline: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for r in errs:
        by_pipeline[r["pipeline"]] = by_pipeline.get(r["pipeline"], 0) + 1
        kind = str(r["error"]).split(":", 1)[0].strip()
        kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "n": len(errs),
        "by_pipeline": dict(sorted(by_pipeline.items())),
        "kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "qids": sorted({r["qid"] for r in errs}),
    }


def main() -> int:
    ensure_loaded()
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/eval_public.jsonl")
    ap.add_argument("--graph", default="data/graph")
    ap.add_argument("--vectors", default="data/vectors")
    ap.add_argument("--out", default="out/public")
    ap.add_argument("--backend", default=None, help="LLM backend: gemini | mock")
    ap.add_argument("--graph-backend", default="tigergraph",
                    choices=["tigergraph", "local"],
                    help="where graph queries run; the submission reports this")
    ap.add_argument("--model", default=None)
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-interval", type=float, default=0.0,
                    help="seconds between LLM calls (free-tier rate limiting)")
    ap.add_argument("--only", default="", help="comma list of pipelines to run")
    ap.add_argument("--ablation", action="store_true",
                    help="also run the agent with the router disabled")
    ap.add_argument("--no-grade", action="store_true",
                    help="skip grading (hidden set has no answers)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meter = UsageMeter()
    pipelines, llm = build_all(args, meter)
    judge = Judge(llm)

    questions = [json.loads(l) for l in open(args.questions, encoding="utf-8")
                 if l.strip()]
    if args.limit:
        questions = questions[:args.limit]
    print(f"{len(questions)} questions x {len(pipelines)} pipelines "
          f"= {len(questions) * len(pipelines)} runs")

    jobs = [(q, name, p) for q in questions for name, p in pipelines.items()]
    answers: list[dict] = []
    rows: list[dict] = []
    t_start = time.time()

    def run_one(job):
        q, name, pipe = job
        try:
            a = pipe.answer(q["qid"], q["question"])
            return q, name, a, None
        except Exception as e:
            return q, name, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futs):
            q, name, a, err = fut.result()
            done += 1
            if err:
                print(f"  [{done}/{len(jobs)}] {q['qid']}/{name} ERROR {err[:200]}")
                # A failed run still occupies a row; it must carry every key
                # the aggregator reads, or one error takes down the summary
                # after the whole benchmark has already been paid for.
                rows.append({"qid": q["qid"], "pipeline": name,
                             "qtype": q.get("qtype", "?"), "question": q["question"],
                             "answer": "", "citations": [], "confidence": 0.0,
                             "error": err[:500], "exact": False, "judge": False,
                             "input_tokens": 0, "output_tokens": 0,
                             "total_tokens": 0, "llm_calls": 0, "n_steps": 0,
                             "tools_called": [], "agents_invoked": [],
                             "strategy_changed": False, "stop_reason": "error",
                             "wall_time_s": 0.0})
                continue

            d = a.as_dict()
            answers.append(d)
            usage = d["usage"]
            trace = d["trace"]
            row = {
                "qid": q["qid"], "pipeline": name, "qtype": q.get("qtype", "?"),
                "question": q["question"], "answer": d["answer"],
                "citations": d["citations"], "confidence": d["confidence"],
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "llm_calls": usage.get("llm_calls", 0),
                "n_steps": trace.get("n_steps", 0),
                "tools_called": trace.get("tools_called", []),
                "agents_invoked": trace.get("agents_invoked", []),
                "strategy_changed": trace.get("strategy_changed", False),
                "stop_reason": trace.get("stop_reason", ""),
                "wall_time_s": trace.get("wall_time_s", 0.0),
            }
            if not args.no_grade and q.get("answer"):
                golds = q["answer"]
                row["exact"] = exact_match(d["answer"], golds)
                verdict = judge.grade(q["qid"], q["question"], d["answer"], golds)
                row["judge"] = verdict["verdict"] == "PASS"
                row["judge_via"] = verdict.get("via")
                row["judge_reason"] = verdict.get("reason", "")
                row["gold"] = golds
            if q.get("gold_doc_ids"):
                row["doc_recall"] = doc_recall(d["citations"], q["gold_doc_ids"])
                row["doc_precision"] = doc_precision(d["citations"], q["gold_doc_ids"])
            rows.append(row)
            mark = "" if args.no_grade else ("PASS" if row.get("judge") else "FAIL")
            print(f"  [{done}/{len(jobs)}] {q['qid']:<9} {name:<18} "
                  f"{row['total_tokens']:>6}tok {mark}  {d['answer'][:60]!r}")

    elapsed = time.time() - t_start
    rows.sort(key=lambda r: (r["qid"], r["pipeline"]))

    with open(out / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "answers.jsonl", "w", encoding="utf-8") as fh:
        for a in answers:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "llm_model": os.environ.get("GEMINI_MODEL", "mock"),
        "graph_backend": args.graph_backend,
        "tg_host": os.environ.get("TG_HOST", "") if args.graph_backend == "tigergraph" else "",
        "embedder": args.embedder or os.environ.get("EMBEDDER", "sentence-transformers"),
        "questions": len(questions),
        "pipelines": list(pipelines),
        "wall_time_s": round(elapsed, 1),
        "judge_llm_calls": judge.calls,
        "judge_tokens": judge.meter.totals("judge").get("total_tokens", 0),
        # Infrastructure failures are scored as wrong answers, because a system
        # that cannot reach its database did not answer the question. But they
        # are not the same KIND of failure as a wrong answer, and a run with a
        # handful of them is not comparable to a clean one - so they are
        # counted here rather than silently folded into the accuracy number.
        "infrastructure_errors": _error_report(rows),
        "by_pipeline": summarize(rows, key="qtype"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # hidden-set submission format
    agentic_rows = {r["qid"]: r for r in rows if r["pipeline"] == "agentic"}
    ans_by_key = {(a["qid"], a["pipeline"]): a for a in answers}
    with open(out / "submission.jsonl", "w", encoding="utf-8") as fh:
        for q in questions:
            r = agentic_rows.get(q["qid"])
            if not r:
                continue
            a = ans_by_key.get((q["qid"], "agentic"), {})
            fh.write(json.dumps({
                "qid": q["qid"], "question": q["question"],
                "qtype": q.get("qtype"),
                "answer": r["answer"],
                "citations": r.get("citations", []),
                "confidence": r.get("confidence", 0.0),
                "tokens": {"input": r["input_tokens"], "output": r["output_tokens"],
                           "total": r["total_tokens"], "llm_calls": r["llm_calls"]},
                "agentic_trace": a.get("trace", {}),
            }, ensure_ascii=False) + "\n")

    print(f"\nwrote {out}/results.jsonl, answers.jsonl, summary.json, submission.jsonl")
    print(f"total wall time {elapsed:.1f}s")
    ierr = summary["infrastructure_errors"]
    if ierr["n"]:
        print(f"\n  !! {ierr['n']} run(s) failed on infrastructure, not on the answer: "
              f"{ierr['kinds']}")
        print(f"     affected questions: {', '.join(ierr['qids'])}")
        print(f"     per pipeline: {ierr['by_pipeline']}")
        print("     These count as wrong below. Re-run before quoting these numbers.")
    for p, s in summary["by_pipeline"].items():
        o = s["overall"]
        print(f"  {p:<18} acc={o.get('judge_accuracy')} "
              f"exact={o.get('exact_accuracy')} "
              f"tok/q={o.get('avg_total_tokens')} "
              f"acc/1k={o.get('accuracy_per_1k_tokens')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
