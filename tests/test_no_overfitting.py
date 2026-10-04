"""Guard against overfitting to the public question set.

The benchmark is only meaningful if the answering path cannot see the answers.
These tests make that auditable instead of asking a reviewer to take it on
trust: they fail if a gold answer, a question id, or the question-template
regexes ever leak into code that runs during a benchmark.

Two files DO contain question templates, by design:

  src/bench/oracle.py    a data-correctness gate. It checks that the graph
                         loaded correctly, with no LLM in the loop, so that a
                         failure is unambiguously a data bug. It is never
                         imported by a pipeline and never produces a submitted
                         answer.
  src/llm/mock_policy.py a deterministic stand-in used only with
                         `--backend mock`, so the agent loop can be exercised
                         offline in CI without an API key. It is not one of the
                         three benchmarked pipelines.

Both are excluded below, and both are asserted to be unreachable from the
answering path.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Everything that executes while answering a benchmark question.
ANSWERING_PATH = [
    ROOT / "src" / "agents",
    ROOT / "src" / "pipelines",
    ROOT / "src" / "graph",
    ROOT / "src" / "vectorstore",
    ROOT / "src" / "ingest",
    ROOT / "src" / "llm" / "client.py",
]

ALLOWED_TEMPLATE_FILES = {"oracle.py", "mock_policy.py"}


def python_files(paths) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            out.append(p)
        elif p.is_dir():
            out.extend(f for f in p.rglob("*.py") if "__pycache__" not in f.parts)
    return out


@pytest.fixture(scope="module")
def gold_answers() -> list[str]:
    path = ROOT / "data" / "eval_public.jsonl"
    if not path.exists():
        pytest.skip("public question set not present")
    answers: list[str] = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            answers.extend(json.loads(line).get("answer", []))
    return answers


def test_no_gold_answer_appears_in_answering_code(gold_answers):
    """No answer string from the benchmark may be present in code that runs
    while answering. A match means the system could be returning a memorised
    answer rather than retrieving one."""
    # Short answers like "5" or "26" occur legitimately as numbers in code, so
    # only distinctive answers (names, titles) are checked.
    distinctive = [a for a in gold_answers if len(a) >= 8 and not a.isdigit()]
    assert distinctive, "expected some non-numeric gold answers to check"
    offenders = []
    for f in python_files(ANSWERING_PATH):
        text = f.read_text(encoding="utf-8")
        for a in distinctive:
            if a in text:
                offenders.append(f"{f.relative_to(ROOT)} contains gold answer {a!r}")
    assert not offenders, "\n".join(offenders)


def test_no_question_ids_in_answering_code():
    """Question ids (pub-003, eval-017) must not appear in the answering path:
    any per-question special-casing would show up here."""
    pattern = re.compile(r"\b(pub|eval)-\d{3}\b")
    offenders = [
        f"{f.relative_to(ROOT)}:{pattern.search(f.read_text(encoding='utf-8')).group()}"
        for f in python_files(ANSWERING_PATH)
        if pattern.search(f.read_text(encoding="utf-8"))
    ]
    assert not offenders, "\n".join(offenders)


def test_question_templates_confined_to_diagnostics():
    """The five question-shape regexes may live only in the correctness gate
    and the offline mock. If they appear in a pipeline, the system is pattern
    matching the benchmark rather than investigating it."""
    markers = ("Olympics had more than", "held immediately before",
               "How many nations competed in", "event held at")
    offenders = []
    for f in python_files([ROOT / "src"]):
        if f.name in ALLOWED_TEMPLATE_FILES:
            continue
        text = f.read_text(encoding="utf-8")
        for m in markers:
            if m in text:
                offenders.append(f"{f.relative_to(ROOT)} matches template {m!r}")
    assert not offenders, "\n".join(offenders)


def test_diagnostics_are_unreachable_from_pipelines():
    """A pipeline must never import the oracle or the mock policy."""
    offenders = []
    for f in python_files([ROOT / "src" / "pipelines", ROOT / "src" / "agents"]):
        text = f.read_text(encoding="utf-8")
        for bad in ("bench.oracle", "bench import oracle", "mock_policy"):
            if bad in text:
                offenders.append(f"{f.relative_to(ROOT)} imports {bad}")
    assert not offenders, "\n".join(offenders)


def test_hidden_answers_are_not_present_anywhere():
    """We never had the hidden answers; assert the file still has none, so a
    reviewer can confirm the hidden run was genuinely blind."""
    path = ROOT / "data" / "eval_hidden.jsonl"
    if not path.exists():
        pytest.skip("hidden question set not present")
    for line in open(path, encoding="utf-8"):
        if line.strip():
            q = json.loads(line)
            assert "answer" not in q, f"{q.get('qid')} unexpectedly carries an answer"
