"""Graph access layer.

Two interchangeable backends behind one interface:

* :class:`TigerGraphStore` -- runs installed GSQL queries on TigerGraph
  Savanna / Community Edition. This is the backend used for the submission.
* :class:`LocalGraphStore` -- loads the same CSV tables into memory. Used for
  unit tests, CI, and as a hard fallback so that development never blocks on
  cloud provisioning. It implements identical semantics, which also lets us
  assert that the GSQL queries return what we think they do.

Every method returns plain dicts so that the agent tool layer, the pipelines
and the benchmark all speak the same vocabulary.
"""
from __future__ import annotations

import csv
import dataclasses
import pathlib
from collections import defaultdict
from typing import Any, Iterable, Protocol

from ..ingest.normalize import date_keys, norm, norm_tokens

MISSING = -1


@dataclasses.dataclass(frozen=True)
class EventRecord:
    doc_id: str
    title: str
    event_name: str
    sport: str
    games_id: str
    year: int
    season: str
    venue: str
    date_raw: str
    competitors: int | None
    nations: int | None
    win_value: str
    url: str

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


class GraphStore(Protocol):
    def find_events(self, **kw) -> list[EventRecord]: ...
    def get_event(self, doc_id: str) -> EventRecord | None: ...
    def count_events_where(self, sport: str, year: int, season: str,
                           field: str, op: str, value: float) -> dict[str, Any]: ...
    def extreme_event(self, sport: str, year: int, season: str,
                      field: str, direction: str) -> dict[str, Any]: ...
    def previous_games(self, year: int, season: str) -> str | None: ...
    def medalists(self, doc_id: str, medal: str) -> list[dict[str, str]]: ...
    def events_by_venue_date(self, venue: str, date: str,
                             year: int | None = None) -> list[EventRecord]: ...
    def resolve_event_phrase(self, phrase: str, season: str | None = None,
                             before_year: int | None = None) -> list[EventRecord]: ...


_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


class LocalGraphStore:
    """In-memory property graph over the CSV tables from ``parse_corpus``."""

    def __init__(self, graph_dir: str | pathlib.Path = "data/graph"):
        self.dir = pathlib.Path(graph_dir)
        self.events: dict[str, EventRecord] = {}
        self._by_sport_games: dict[tuple, list[str]] = defaultdict(list)
        self._by_venue_date: dict[tuple, list[str]] = defaultdict(list)
        self._by_venue_day: dict[tuple, list[str]] = defaultdict(list)
        self._by_title: dict[str, str] = {}
        self._precedes: dict[str, str] = {}      # games_id -> successor
        self._preceded_by: dict[str, str] = {}   # games_id -> predecessor
        self._medals: dict[str, list[dict]] = defaultdict(list)
        self._athlete_names: dict[str, str] = {}
        self._load()

    # ---------------- loading ----------------
    def _rows(self, name: str) -> Iterable[dict[str, str]]:
        path = self.dir / name
        if not path.exists():
            return []
        with open(path, encoding="utf-8", newline="") as fh:
            yield from csv.DictReader(fh)

    def _load(self) -> None:
        for r in self._rows("event.csv"):
            comp = int(r["competitors"])
            nat = int(r["nations"])
            rec = EventRecord(
                doc_id=r["doc_id"], title=r["title"], event_name=r["event_name"],
                sport=r["sport"], games_id=r["games_id"], year=int(r["year"]),
                season=r["season"], venue=r["venue"], date_raw=r["date_raw"],
                competitors=None if comp == MISSING else comp,
                nations=None if nat == MISSING else nat,
                win_value=r["win_value"], url=r["url"],
            )
            self.events[rec.doc_id] = rec
            self._by_sport_games[(norm(rec.sport), rec.year, rec.season)].append(rec.doc_id)
            self._by_title[norm(rec.title)] = rec.doc_id
            if rec.venue and rec.date_raw:
                self._by_venue_date[(norm(rec.venue), norm(rec.date_raw))].append(rec.doc_id)
                # Secondary index on (venue, month-day). Used ONLY when the
                # exact date string misses, so it can recover a match without
                # changing any lookup that already succeeds.
                for k in date_keys(rec.date_raw):
                    self._by_venue_day[(norm(rec.venue), k)].append(rec.doc_id)

        for r in self._rows("athlete.csv"):
            self._athlete_names[r["athlete_id"]] = r["name"]
        for r in self._rows("e_medal.csv"):
            self._medals[r["doc_id"]].append({
                "athlete_id": r["athlete_id"],
                "name": self._athlete_names.get(r["athlete_id"], r["athlete_id"]),
                "medal": r["medal"], "rank": int(r["rank"]),
            })
        for r in self._rows("e_precedes.csv"):
            self._precedes[r["from_games"]] = r["to_games"]
            self._preceded_by[r["to_games"]] = r["from_games"]

    # ---------------- queries ----------------
    def get_event(self, doc_id: str) -> EventRecord | None:
        return self.events.get(doc_id)

    def find_events(self, sport: str | None = None, year: int | None = None,
                    season: str | None = None, venue: str | None = None,
                    title: str | None = None, limit: int = 200) -> list[EventRecord]:
        if title:
            hit = self._by_title.get(norm(title))
            if hit:
                return [self.events[hit]]
        if sport and year and season:
            ids = self._by_sport_games.get((norm(sport), year, season), [])
            return [self.events[i] for i in ids][:limit]
        out = []
        for rec in self.events.values():
            if sport and norm(sport) != norm(rec.sport):
                continue
            if year and rec.year != year:
                continue
            if season and rec.season != season:
                continue
            if venue and norm(venue) not in norm(rec.venue):
                continue
            out.append(rec)
            if len(out) >= limit:
                break
        return out

    def count_events_where(self, sport: str, year: int, season: str,
                           field: str = "competitors", op: str = ">",
                           value: float = 0) -> dict[str, Any]:
        if field not in ("competitors", "nations"):
            raise ValueError(f"unsupported field {field!r}")
        cmp = _OPS[op]
        cands = self.find_events(sport=sport, year=year, season=season)
        matched = [e for e in cands
                   if getattr(e, field) is not None and cmp(getattr(e, field), value)]
        return {
            "count": len(matched),
            "considered": len(cands),
            "missing_field": sum(1 for e in cands if getattr(e, field) is None),
            "matched": [{"doc_id": e.doc_id, "title": e.title,
                         field: getattr(e, field)} for e in matched],
            "evidence_doc_ids": [e.doc_id for e in cands],
        }

    def extreme_event(self, sport: str, year: int, season: str,
                      field: str = "competitors", direction: str = "max") -> dict[str, Any]:
        cands = [e for e in self.find_events(sport=sport, year=year, season=season)
                 if getattr(e, field) is not None]
        if not cands:
            return {"event": None, "value": None, "ties": [], "evidence_doc_ids": []}
        pick = max if direction == "max" else min
        best = pick(getattr(e, field) for e in cands)
        tied = [e for e in cands if getattr(e, field) == best]
        return {
            "event": tied[0].as_dict(),
            "value": best,
            "ties": [e.title for e in tied],
            "considered": len(cands),
            "evidence_doc_ids": [e.doc_id for e in cands],
        }

    def previous_games(self, year: int, season: str) -> str | None:
        """Games immediately before ``year`` in the same season, via PRECEDES."""
        # Several PRECEDES edges can point into one Games, because sports
        # return on different cycles (golf came back in 2016 after 1904, rugby
        # after 1924). "Immediately before" is therefore the greatest year
        # below the target, not whichever predecessor happens to be stored.
        years = sorted({e.year for e in self.events.values()
                        if e.season == season and e.year < year})
        return f"{years[-1]} {season}" if years else None

    def counterpart_in_games(self, doc_id: str, games_id: str) -> EventRecord | None:
        """The same event contested at a different Games.

        Used for temporal questions: resolve the event at the anchor year, then
        traverse ``Games.PRECEDES`` backwards and find its counterpart. Matching
        is on the normalised (sport, event_name) pair, falling back to token
        containment for events that were renamed between Games.
        """
        src = self.events.get(doc_id)
        if src is None:
            return None
        key = norm_tokens(f"{src.sport} {src.event_name}")
        cands = [e for e in self.events.values() if e.games_id == games_id]
        for e in cands:
            if norm_tokens(f"{e.sport} {e.event_name}") == key:
                return e
        best, best_score = None, -1.0
        for e in cands:
            if norm(e.sport) != norm(src.sport):
                continue
            other = norm_tokens(f"{e.sport} {e.event_name}")
            score = len(key & other) / max(len(key | other), 1)
            if score > best_score:
                best, best_score = e, score
        return best if best_score >= 0.6 else None

    def medalists(self, doc_id: str, medal: str = "gold") -> list[dict[str, str]]:
        return sorted([m for m in self._medals.get(doc_id, []) if m["medal"] == medal],
                      key=lambda m: m["rank"])

    def events_by_venue_date(self, venue: str, date: str,
                             year: int | None = None) -> list[EventRecord]:
        ids = self._by_venue_date.get((norm(venue), norm(date)), [])
        if not ids:
            # Fallback: compare the day and month rather than the raw string.
            # Questions and infoboxes order dates differently ("August 12,
            # 2008" vs "12 August 2008") and infobox dates are often ranges,
            # so an exact string match misses roughly one multi-hop question
            # in three. Without this the agent retrieves nothing and the
            # answerer falls back on the model's own memory - an ungrounded
            # answer with no citations, which is the worst outcome here.
            seen: set[str] = set()
            for k in date_keys(date):
                for i in self._by_venue_day.get((norm(venue), k), []):
                    if i not in seen:
                        seen.add(i)
                        ids.append(i)
        recs = [self.events[i] for i in ids]
        if year:
            recs = [r for r in recs if r.year == year]
        if len(recs) > 1:
            # Disambiguation: venue names often name their sport
            # ('Laura Biathlon & Ski Complex' -> Biathlon), and two events can
            # legitimately share a venue and a date.
            vn = norm(venue)
            same_sport = [r for r in recs if norm(r.sport) in vn]
            if same_sport:
                recs = same_sport
        return recs

    def resolve_event_phrase(self, phrase: str, season: str | None = None,
                             before_year: int | None = None,
                             limit: int = 25) -> list[EventRecord]:
        """Entity linking for phrases like "men's 20 kilometres walk athletics".

        Scores candidates by token containment against 'sport + event name'.
        Exact token-set equality wins over containment, which is what keeps
        "Men's 80 kg" from matching "Men's +80 kg".
        """
        toks = norm_tokens(phrase)
        if not toks:
            return []
        exact: list[EventRecord] = []
        loose: list[tuple[int, EventRecord]] = []
        for rec in self.events.values():
            if season and rec.season != season:
                continue
            if before_year and rec.year >= before_year:
                continue
            combo = norm_tokens(f"{rec.sport} {rec.event_name}")
            if toks == combo:
                exact.append(rec)
            elif toks <= combo:
                loose.append((len(combo - toks), rec))
        if exact:
            return sorted(exact, key=lambda r: -r.year)[:limit]
        loose.sort(key=lambda t: (t[0], -t[1].year))
        return [r for _s, r in loose[:limit]]

    # introspection used by the orchestrator prompt
    def schema_summary(self) -> dict[str, Any]:
        sports = sorted({e.sport for e in self.events.values()})
        games = sorted({e.games_id for e in self.events.values()})
        return {"events": len(self.events), "sports": sports, "games": games}
