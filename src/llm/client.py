"""LLM client with first-class token accounting.

Token cost is 15% of the hackathon score and the whole point of the benchmark,
so every call is metered here rather than estimated afterwards. ``UsageMeter``
attributes tokens to a (pipeline, question, step) triple so the dashboard can
show cost per reasoning step, not just a total.

Backends:
  * ``gemini``   -- Google Generative Language REST API (no SDK dependency)
  * ``mock``     -- deterministic canned responses, used by tests and CI so the
                    full pipeline can be exercised without a network or a key
"""
from __future__ import annotations

import dataclasses
import json
import os
import random
import time
from collections import defaultdict
from typing import Any, Callable

import urllib.error
import urllib.request

GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent")


@dataclasses.dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    raw: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def json(self) -> Any:
        """Parse the response as JSON, tolerating ```json fences."""
        t = self.text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t
            t = t.rsplit("```", 1)[0]
        t = t.strip()
        # tolerate a leading 'json' language tag left behind by the fence strip
        if t.lower().startswith("json\n"):
            t = t[5:]
        start = min([i for i in (t.find("{"), t.find("[")) if i != -1], default=0)
        return json.loads(t[start:])


class UsageMeter:
    """Accumulates token/latency usage keyed by pipeline and step."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, pipeline: str, qid: str, step: str, resp: LLMResponse) -> None:
        self.records.append({
            "pipeline": pipeline, "qid": qid, "step": step,
            "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
            "total_tokens": resp.total_tokens, "latency_s": round(resp.latency_s, 3),
        })

    def totals(self, pipeline: str | None = None, qid: str | None = None) -> dict[str, int]:
        t = defaultdict(int)
        for r in self.records:
            if pipeline and r["pipeline"] != pipeline:
                continue
            if qid and r["qid"] != qid:
                continue
            t["input_tokens"] += r["input_tokens"]
            t["output_tokens"] += r["output_tokens"]
            t["total_tokens"] += r["total_tokens"]
            t["llm_calls"] += 1
        return dict(t)

    def by_step(self, pipeline: str, qid: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r["pipeline"] == pipeline and r["qid"] == qid]


class LLMError(RuntimeError):
    pass


class GeminiClient:
    """Minimal REST client. No SDK so the repo stays reproducible from
    requirements.txt alone and so token accounting is never hidden."""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 temperature: float = 0.0, max_retries: int = 5,
                 min_interval_s: float = 0.0):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set (see .env.example)")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s
        self._last_call = 0.0

    def complete(self, prompt: str, system: str | None = None,
                 json_mode: bool = False, max_output_tokens: int = 1024) -> LLMResponse:
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        url = GEMINI_URL.format(model=self.model)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": self.api_key},
            method="POST",
        )

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            gap = time.time() - self._last_call
            if gap < self.min_interval_s:
                time.sleep(self.min_interval_s - gap)
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    payload = json.loads(r.read().decode("utf-8"))
                self._last_call = time.time()
                usage = payload.get("usageMetadata", {})
                cands = payload.get("candidates", [])
                text = ""
                if cands:
                    parts = cands[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
                return LLMResponse(
                    text=text,
                    input_tokens=usage.get("promptTokenCount", 0),
                    output_tokens=usage.get("candidatesTokenCount", 0),
                    latency_s=time.time() - t0,
                    raw=payload,
                )
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(min(2 ** attempt + random.random(), 30))
                    continue
                raise LLMError(f"Gemini HTTP {e.code}: {e.read()[:500]!r}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(min(2 ** attempt + random.random(), 30))
        raise LLMError(f"Gemini failed after {self.max_retries} attempts: {last_err}")


class MockLLM:
    """Deterministic stand-in. ``handler(prompt, system)`` returns the text."""

    def __init__(self, handler: Callable[[str, str | None], str]):
        self.handler = handler
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None,
                 json_mode: bool = False, max_output_tokens: int = 1024) -> LLMResponse:
        self.calls += 1
        text = self.handler(prompt, system)
        return LLMResponse(
            text=text,
            input_tokens=max(1, len(prompt) // 4 + (len(system) // 4 if system else 0)),
            output_tokens=max(1, len(text) // 4),
            latency_s=0.0,
        )


def build_client(backend: str | None = None, **kw):
    backend = (backend or os.environ.get("LLM_BACKEND", "gemini")).lower()
    if backend == "gemini":
        allowed = ("api_key", "model", "temperature", "max_retries", "min_interval_s")
        return GeminiClient(**{k: v for k, v in kw.items()
                               if k in allowed and v is not None})
    if backend == "mock":
        from .mock_policy import default_handler
        return MockLLM(kw.get("handler") or default_handler)
    raise LLMError(f"unknown LLM backend {backend!r}")
