"""Print the RAW result of every installed GSQL query.

    python scripts/tg_probe.py

The Python layer has to parse whatever shape TigerGraph returns, and that shape
differs between query forms (accumulator PRINT vs vertex-set PRINT) and between
builds. When the live oracle disagrees with the local one, this is how we find
out which, in one round trip, instead of guessing.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import ensure_loaded  # noqa: E402
from src.graph.tigergraph import TigerGraphStore, connect  # noqa: E402

# Known-good arguments, taken from questions the local oracle answers correctly.
PROBES = [
    ("q_events_of_sport_games", {"sport_id": "biathlon", "games_id": "2018 Winter"}),
    ("q_count_events_where", {"sport_id": "biathlon", "games_id": "2018 Winter",
                              "field": "competitors", "threshold": 73}),
    ("q_events_by_venue_date", {"venue_id": "olympic weightlifting gymnasium",
                                "date_norm": "20 september 1988"}),
    ("q_medalists", {"doc_id": "Q25239316", "medal": "gold"}),
    ("q_previous_games", {"games_id": "2016 Summer"}),
    ("q_events_of_games", {"games_id": "2012 Summer"}),
    ("q_events_of_season", {"season": "Summer", "max_year": 2015}),
    ("q_event_by_doc_id", {"doc_id": "Q25239316"}),
]


def brief(obj, limit: int = 1200) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + f" …(+{len(s) - limit} chars)"


def main() -> int:
    ensure_loaded()
    conn = connect()
    print("=" * 76)
    print("RAW QUERY RESULTS")
    print("=" * 76)
    for name, params in PROBES:
        print(f"\n### {name}({json.dumps(params)})")
        try:
            res = conn.runInstalledQuery(name, params, timeout=60_000)
            print(f"  type={type(res).__name__} len={len(res) if hasattr(res, '__len__') else '?'}")
            print(f"  {brief(res)}")
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}")

    print("\n" + "=" * 76)
    print("STORE METHODS (what the agent actually calls)")
    print("=" * 76)
    store = TigerGraphStore(conn)
    checks = [
        ("find_events(sport=Biathlon, 2018, Winter)",
         lambda: [e.title for e in store.find_events(sport="Biathlon", year=2018,
                                                     season="Winter")][:4]),
        ("count_events_where(Biathlon 2018 Winter competitors>73)",
         lambda: {k: v for k, v in store.count_events_where(
             "Biathlon", 2018, "Winter", "competitors", ">", 73).items()
             if k != "evidence_doc_ids"}),
        ("extreme_event(Sailing 2000 Summer)",
         lambda: store.extreme_event("Sailing", 2000, "Summer")["event"]),
        ("events_by_venue_date(Olympic Weightlifting Gymnasium, 20 September 1988)",
         lambda: [e.title for e in store.events_by_venue_date(
             "Olympic Weightlifting Gymnasium", "20 September 1988")]),
        ("medalists(Q25239316, gold)", lambda: store.medalists("Q25239316", "gold")),
        ("previous_games(2016, Summer)", lambda: store.previous_games(2016, "Summer")),
        ("find_events(title='Sailing at the 2016 Summer Olympics – Women's RS:X')",
         lambda: [e.nations for e in store.find_events(
             title="Sailing at the 2016 Summer Olympics – Women's RS:X")]),
        ("resolve_event_phrase(\"men's 20 kilometres walk athletics\", Summer, 2016)",
         lambda: [(e.title, e.year) for e in store.resolve_event_phrase(
             "men's 20 kilometres walk athletics", "Summer", 2016)][:3]),
    ]
    for label, fn in checks:
        try:
            print(f"\n### {label}\n  {brief(fn(), 600)}")
        except Exception as e:
            print(f"\n### {label}\n  ERROR {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
