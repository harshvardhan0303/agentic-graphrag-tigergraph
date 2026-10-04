"""List the Gemini models your API key can actually use.

    python scripts/list_models.py

Model availability changes over time and varies by key, region and tier, so the
authoritative answer comes from Google's ListModels endpoint rather than from
documentation. Prints every model that supports generateContent, marks the one
currently configured in .env, and recommends one.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import ensure_loaded  # noqa: E402

URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Preference order for this benchmark: we need a cheap, fast, high-rate-limit
# model because the full run is ~1,200 calls. Flash-class beats Pro here, and
# the same model must be used by all three pipelines (the submission form
# requires it), so one choice covers everything.
PREFER = ("flash-lite", "flash", "pro")


def main() -> int:
    ensure_loaded()
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key or key.startswith("your-key"):
        print("GEMINI_API_KEY is not set in .env")
        return 1

    req = urllib.request.Request(
        f"{URL}?pageSize=200", headers={"x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:400]
        print(f"HTTP {e.code} from Google: {body}")
        if e.code in (400, 403):
            print("\nThat usually means the key is invalid, or the Generative "
                  "Language API is not enabled for its project.\n"
                  "Check https://aistudio.google.com/apikey")
        return 1

    models = payload.get("models", [])
    usable = [m for m in models
              if "generateContent" in m.get("supportedGenerationMethods", [])]
    if not usable:
        print("No models support generateContent for this key.")
        return 1

    configured = os.environ.get("GEMINI_MODEL", "")
    print(f"{len(usable)} usable model(s) for this key "
          f"(configured: {configured or 'none'})\n")
    print(f"{'MODEL':<45} {'IN':>8} {'OUT':>7}  NAME")
    print("-" * 95)
    rows = []
    for m in sorted(usable, key=lambda m: m["name"]):
        short = m["name"].removeprefix("models/")
        rows.append(short)
        mark = " <-- configured in .env" if short == configured else ""
        print(f"{short:<45} {m.get('inputTokenLimit', '?'):>8} "
              f"{m.get('outputTokenLimit', '?'):>7}  "
              f"{m.get('displayName', '')}{mark}")

    # recommend: newest flash-class model that is not a preview/experimental build
    def score(name: str) -> tuple:
        n = name.lower()
        family = next((i for i, p in enumerate(PREFER) if p in n), len(PREFER))
        stable = 0 if any(t in n for t in ("preview", "exp", "experimental")) else 1
        return (family, -stable, name)

    best = sorted(rows, key=score)[0] if rows else None
    if best:
        print(f"\nRecommended for this benchmark: {best}")
        print(f"Set this in .env:\n    GEMINI_MODEL={best}")
        if configured and configured != best:
            print(f"\n(You currently have GEMINI_MODEL={configured}, which is "
                  f"{'not in the list above' if configured not in rows else 'valid'}.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
