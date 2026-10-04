"""Scoring.

Three accuracy notions are reported, because they disagree in informative ways:

  exact      - normalised string equality against the gold answer. Harsh, but
               unambiguous and free.
  judge      - LLM-as-judge PASS/FAIL, which forgives diacritics, separators and
               phrasing. Used as the headline number, as the guidebook allows.
  completeness - gold-document recall: of the documents the benchmark says are
               needed, how many did the pipeline actually cite? A pipeline that
               guesses the right number from one document is accurate but not
               complete, and the distinction is exactly what the aggregation
               questions are designed to expose.
"""
from __future__ import annotations

import re
from typing import Any

from ..ingest.normalize import norm

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}


def canon_answer(s: str) -> str:
    """Normalise for comparison: accents, case, punctuation, number words.

    Team answers are stored in the corpus as concatenated names
    ('Dani KingLaura TrottJoanna Rowsell'); a pipeline may emit them separated.
    Stripping non-alphanumerics makes both forms agree.
    """
    t = norm(s or "")
    if t in _NUM_WORDS:
        return _NUM_WORDS[t]
    m = re.fullmatch(r"(\d+)(?:\s+\w+)?", t)
    if m:
        return m.group(1)
    return t.replace(" ", "")


def exact_match(pred: str, golds: list[str]) -> bool:
    if not pred:
        return False
    p = canon_answer(pred)
    return any(p == canon_answer(g) for g in golds)


def doc_recall(citations: list[str], gold_doc_ids: list[str]) -> float | None:
    if not gold_doc_ids:
        return None
    gold = set(gold_doc_ids)
    return len(gold & set(citations)) / len(gold)


def doc_precision(citations: list[str], gold_doc_ids: list[str]) -> float | None:
    if not gold_doc_ids or not citations:
        return None
    gold = set(gold_doc_ids)
    return len(gold & set(citations)) / len(set(citations))


def summarize(rows: list[dict[str, Any]], key: str = "qtype") -> dict[str, Any]:
    """Aggregate per-question rows into per-pipeline and per-qtype tables."""
    out: dict[str, Any] = {}
    pipelines = sorted({r["pipeline"] for r in rows})
    for p in pipelines:
        prows = [r for r in rows if r["pipeline"] == p]
        out[p] = {"overall": _agg(prows), "by_" + key: {}}
        for k in sorted({r.get(key, "?") for r in prows}):
            out[p]["by_" + key][k] = _agg([r for r in prows if r.get(key) == k])
    return out


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    def mean(field: str, default=0.0) -> float:
        vals = [r[field] for r in rows if r.get(field) is not None]
        return sum(vals) / len(vals) if vals else default

    graded = [r for r in rows if r.get("exact") is not None]
    judged = [r for r in rows if r.get("judge") is not None]
    total_tokens = sum(r.get("total_tokens", 0) for r in rows)
    acc = (sum(1 for r in judged if r["judge"]) / len(judged)) if judged else None
    exact = (sum(1 for r in graded if r["exact"]) / len(graded)) if graded else None
    headline = acc if acc is not None else exact

    return {
        "n": n,
        "exact_accuracy": round(exact, 4) if exact is not None else None,
        "judge_accuracy": round(acc, 4) if acc is not None else None,
        "completeness_doc_recall": round(mean("doc_recall"), 4),
        "citation_precision": round(mean("doc_precision"), 4),
        "avg_total_tokens": round(total_tokens / n, 1),
        "avg_input_tokens": round(sum(r.get("input_tokens", 0) for r in rows) / n, 1),
        "avg_output_tokens": round(sum(r.get("output_tokens", 0) for r in rows) / n, 1),
        "total_tokens": total_tokens,
        "avg_llm_calls": round(mean("llm_calls"), 2),
        "avg_steps": round(mean("n_steps"), 2),
        "avg_latency_s": round(mean("wall_time_s"), 3),
        "strategy_change_rate": round(
            sum(1 for r in rows if r.get("strategy_changed")) / n, 4),
        # the headline efficiency number: correct answers per 1k tokens spent
        "accuracy_per_1k_tokens": (
            round(headline / (total_tokens / n / 1000), 4)
            if headline is not None and total_tokens else None),
        "unknown_rate": round(
            sum(1 for r in rows if canon_answer(r.get("answer", "")) in ("", "unknown"))
            / n, 4),
    }
