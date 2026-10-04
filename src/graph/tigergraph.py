"""TigerGraph backend.

Implements the same interface as :class:`LocalGraphStore` by calling the
installed GSQL queries in ``gsql/03_queries.gsql``. Set-selection, traversal and
arithmetic run in the database; string normalisation runs here so that both
backends share one definition of it.

Requires ``pyTigerGraph``. Connection is configured through the environment:

    TG_HOST=https://<workspace>.i.tgcloud.io
    TG_GRAPH=OlympicGraphRAG
    TG_USERNAME=... / TG_PASSWORD=...        (or)
    TG_SECRET=...                            (or)  TG_TOKEN=...
"""
from __future__ import annotations

import os
import random
import threading
import time
from typing import Any

from ..ingest.normalize import date_keys, norm, norm_tokens
from .store import MISSING, EventRecord


def connect(host: str | None = None, graph: str | None = None, **kw):
    import pyTigerGraph as tg  # lazy: only needed for the TigerGraph backend

    host = host or os.environ["TG_HOST"]
    graph = graph or os.environ.get("TG_GRAPH", "OlympicGraphRAG")
    username = kw.get("username") or os.environ.get("TG_USERNAME", "tigergraph")
    password = kw.get("password") or os.environ.get("TG_PASSWORD", "")
    conn = tg.TigerGraphConnection(
        host=host, graphname=graph, username=username, password=password,
        gsqlSecret=os.environ.get("TG_SECRET", ""),
        restppPort=os.environ.get("TG_RESTPP_PORT", "14240"),
        gsPort=os.environ.get("TG_GS_PORT", "14240"),
    )
    token = os.environ.get("TG_TOKEN")
    if token:
        conn.apiToken = token
    elif os.environ.get("TG_SECRET"):
        conn.getToken(os.environ["TG_SECRET"])
    return conn


def _strip_alias(attrs: dict[str, Any]) -> dict[str, Any]:
    """TigerGraph names projected columns after the query alias.

    `PRINT E[E.doc_id, E.title] AS events` comes back with keys "E.doc_id" and
    "E.title", not "doc_id"/"title". Strip the alias so every caller sees the
    attribute names the schema declares.
    """
    out = {}
    for k, v in attrs.items():
        name = k.split(".", 1)[1] if "." in k else k
        out[name.lstrip("@")] = v      # vertex accumulators arrive as "A.@acc"
    return out


def _rec(a: dict[str, Any]) -> EventRecord:
    attrs = _strip_alias(a.get("attributes", a))
    comp = int(attrs.get("competitors", MISSING))
    nat = int(attrs.get("nations", MISSING))
    return EventRecord(
        doc_id=attrs.get("doc_id") or a.get("v_id", ""),
        title=attrs.get("title", ""), event_name=attrs.get("event_name", ""),
        sport=attrs.get("sport", ""), games_id=attrs.get("games_id", ""),
        year=int(attrs.get("year", 0)), season=attrs.get("season", ""),
        venue=attrs.get("venue", ""), date_raw=attrs.get("date_raw", ""),
        competitors=None if comp == MISSING else comp,
        nations=None if nat == MISSING else nat,
        win_value=attrs.get("win_value", ""), url=attrs.get("url", ""),
    )


class TigerGraphStore:
    """GraphStore over installed GSQL queries."""

    def __init__(self, conn=None):
        # pyTigerGraph connections are not thread-safe: the benchmark runs
        # pipelines concurrently, and sharing one connection produces
        # RemoteDisconnected mid-run. Each thread therefore gets its own.
        # The caches are plain dicts, which is safe here because entries are
        # idempotent — a racing write stores the same value.
        self._local = threading.local()
        self._seed_conn = conn
        self._seed_lock = threading.Lock()
        self._event_cache: dict[str, EventRecord] = {}
        self._games_cache: dict[str, list[EventRecord]] = {}

    @property
    def conn(self):
        c = getattr(self._local, "conn", None)
        if c is None:
            with self._seed_lock:
                if self._seed_conn is not None:
                    c, self._seed_conn = self._seed_conn, None
            if c is None:
                c = connect()
            self._local.conn = c
        return c

    # ---------- helpers ----------
    def _run(self, name: str, params: dict[str, Any],
             retries: int = 5) -> list[dict[str, Any]]:
        """Run an installed query, surviving a transient server-side outage.

        Savanna occasionally drops connections for a stretch of seconds at a
        time - not one request, a window. An earlier version retried 3 times
        with a linear 0.5s backoff, giving up after ~1.5s; a ~20s blip during
        one benchmark run therefore killed 11 of 400 runs across four
        consecutive questions, and because a failed run is scored as a wrong
        answer it moved every pipeline's accuracy by 2-3 points.

        Exponential backoff with jitter covers ~30s instead. The jitter matters
        because the worker threads fail together, so without it they would all
        wake together and hammer a server that is still recovering.
        """
        last: Exception | None = None
        for attempt in range(retries):
            try:
                return self.conn.runInstalledQuery(name, params, timeout=60_000)
            except Exception as e:  # dropped keep-alive, transient 5xx
                last = e
                if attempt == retries - 1:
                    raise
                # force a fresh connection on the next attempt
                self._local.conn = None
                time.sleep(min(2 ** attempt, 8) + random.uniform(0, 0.5))
        raise last  # unreachable, keeps type checkers happy

    @staticmethod
    def _rows(result: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        for block in result or []:
            if key in block:
                return block[key]
        return []

    # ---------- queries ----------
    def get_event(self, doc_id: str) -> EventRecord | None:
        if doc_id in self._event_cache:
            return self._event_cache[doc_id]
        rows = self._rows(self._run("q_event_by_doc_id", {"doc_id": doc_id}), "events")
        if not rows:
            return None
        rec = _rec(rows[0])
        self._event_cache[doc_id] = rec
        return rec

    def _events_of_sport_games(self, sport: str, year: int, season: str
                               ) -> list[EventRecord]:
        rows = self._rows(self._run("q_events_of_sport_games", {
            "sport_id": norm(sport), "games_id": f"{year} {season}"}), "events")
        recs = [_rec(r) for r in rows]
        for r in recs:
            self._event_cache[r.doc_id] = r
        return recs

    def find_events(self, sport: str | None = None, year: int | None = None,
                    season: str | None = None, venue: str | None = None,
                    title: str | None = None, limit: int = 200) -> list[EventRecord]:
        if title:
            # Titles are unique and the doc_id is the Wikidata QID, so the exact
            # title path goes through the sport/games index rather than a scan.
            t = norm(title)
            for rec in self._event_cache.values():
                if norm(rec.title) == t:
                    return [rec]
            parts = title.split(" at the ")
            if len(parts) == 2:
                sport_guess = parts[0]
                words = parts[1].split()
                if len(words) >= 2 and words[0].isdigit():
                    cands = self._events_of_sport_games(
                        sport_guess, int(words[0]), words[1])
                    for rec in cands:
                        if norm(rec.title) == t:
                            return [rec]
            return []
        if sport and year and season:
            return self._events_of_sport_games(sport, year, season)[:limit]
        return []

    def count_events_where(self, sport: str, year: int, season: str,
                           field: str = "competitors", op: str = ">",
                           value: float = 0) -> dict[str, Any]:
        if op == ">" and field in ("competitors", "nations"):
            res = self._run("q_count_events_where", {
                "sport_id": norm(sport), "games_id": f"{year} {season}",
                "field": field, "threshold": int(value)})
            flat: dict[str, Any] = {}
            for block in res or []:
                flat.update(block)
            matched_ids = flat.get("matched_doc_ids", []) or []
            return {
                "count": int(flat.get("matched", 0)),
                "considered": int(flat.get("considered", 0)),
                "missing_field": int(flat.get("missing_field", 0)),
                "matched": [{"doc_id": d, "title": (self.get_event(d).title
                                                    if self.get_event(d) else d),
                             field: getattr(self.get_event(d), field, None)}
                            for d in matched_ids],
                "evidence_doc_ids": flat.get("evidence_doc_ids", []) or [],
            }
        # other comparisons: fetch the set once and compare client-side
        from .store import _OPS
        cands = self._events_of_sport_games(sport, year, season)
        cmp = _OPS[op]
        matched = [e for e in cands if getattr(e, field) is not None
                   and cmp(getattr(e, field), value)]
        return {"count": len(matched), "considered": len(cands),
                "missing_field": sum(1 for e in cands if getattr(e, field) is None),
                "matched": [{"doc_id": e.doc_id, "title": e.title,
                             field: getattr(e, field)} for e in matched],
                "evidence_doc_ids": [e.doc_id for e in cands]}

    def extreme_event(self, sport: str, year: int, season: str,
                      field: str = "competitors",
                      direction: str = "max") -> dict[str, Any]:
        """Superlative over the event set the database selected.

        The expensive, correctness-critical part — retrieving *every* event of
        that sport at that Games — happens in GSQL. The argmax then runs over
        at most a few dozen rows here, which keeps the query portable across
        TigerGraph builds (CASE/IF inside ACCUM is not reliably supported) at
        no cost to exactness. Ties are reported, never broken silently.
        """
        cands = [e for e in self._events_of_sport_games(sport, year, season)
                 if getattr(e, field) is not None]
        if not cands:
            return {"event": None, "value": None, "ties": [], "evidence_doc_ids": []}
        pick = max if direction == "max" else min
        best = pick(getattr(e, field) for e in cands)
        tied = [e for e in cands if getattr(e, field) == best]
        return {"event": tied[0].as_dict(), "value": best,
                "ties": [e.title for e in tied], "considered": len(cands),
                "evidence_doc_ids": [e.doc_id for e in cands]}

    def events_by_venue_date(self, venue: str, date: str,
                             year: int | None = None) -> list[EventRecord]:
        rows = self._rows(self._run("q_events_by_venue_date", {
            "venue_id": norm(venue), "date_norm": norm(date)}), "events")
        recs = [_rec(r) for r in rows]
        if not recs:
            # Exact date string missed. Re-ask for every event at this venue
            # and match on (month, day) instead, which survives word order and
            # date ranges. Kept as a fallback rather than the primary lookup so
            # that no query which already succeeds changes behaviour.
            want = date_keys(date)
            if want:
                rows = self._rows(self._run("q_events_by_venue",
                                            {"venue_id": norm(venue)}), "events")
                recs = [r for r in (_rec(x) for x in rows)
                        if want & date_keys(r.date_raw)]
        if year:
            recs = [r for r in recs if r.year == year]
        if len(recs) > 1:
            vn = norm(venue)
            same_sport = [r for r in recs if norm(r.sport) in vn]
            if same_sport:
                recs = same_sport
        return recs

    def medalists(self, doc_id: str, medal: str = "gold") -> list[dict[str, str]]:
        rows = self._rows(self._run("q_medalists",
                                    {"doc_id": doc_id, "medal": medal}), "medalists")
        out = []
        for i, r in enumerate(rows):
            attrs = _strip_alias(r.get("attributes", r))
            try:
                rank = int(attrs.get("rank_order", i))
            except (TypeError, ValueError):
                rank = i
            out.append({"athlete_id": attrs.get("athlete_id", r.get("v_id", "")),
                        "name": attrs.get("name", ""), "medal": medal, "rank": rank})
        # Team order is part of the answer, not a presentation detail.
        return sorted(out, key=lambda m: m["rank"])

    def previous_games(self, year: int, season: str) -> str | None:
        """The Games immediately before `year` in the same season.

        Several PRECEDES edges can point into one Games, because sports return
        on different cycles: golf came back in 2016 after 1904, rugby after
        1924. So "immediately before" is the predecessor with the GREATEST year
        below the target, not simply any predecessor.
        """
        rows = self._rows(self._run("q_previous_games",
                                    {"games_id": f"{year} {season}"}), "previous")
        best: tuple[int, str] | None = None
        for r in rows:
            attrs = _strip_alias(r.get("attributes", r))
            gid = attrs.get("games_id") or r.get("v_id", "")
            try:
                y = int(attrs.get("year") or gid.split()[0])
            except (ValueError, IndexError):
                continue
            if attrs.get("season", season) != season or y >= year:
                continue
            if best is None or y > best[0]:
                best = (y, gid)
        if best:
            return best[1]
        # Target Games absent from the corpus, or no PRECEDES edge recorded.
        years = sorted({e.year for e in self._event_cache.values()
                        if e.season == season and e.year < year})
        return f"{years[-1]} {season}" if years else None

    def _events_of_games(self, games_id: str) -> list[EventRecord]:
        if games_id in self._games_cache:
            return self._games_cache[games_id]
        rows = self._rows(self._run("q_events_of_games", {"games_id": games_id}),
                          "events")
        recs = [_rec(r) for r in rows]
        self._games_cache[games_id] = recs
        for r in recs:
            self._event_cache[r.doc_id] = r
        return recs

    def counterpart_in_games(self, doc_id: str, games_id: str) -> EventRecord | None:
        src = self.get_event(doc_id)
        if src is None:
            return None
        key = norm_tokens(f"{src.sport} {src.event_name}")
        cands = self._events_of_games(games_id)
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

    def resolve_event_phrase(self, phrase: str, season: str | None = None,
                             before_year: int | None = None,
                             limit: int = 25) -> list[EventRecord]:
        toks = norm_tokens(phrase)
        if not toks:
            return []
        rows = self._rows(self._run("q_events_of_season", {
            "season": season or "Summer",
            "max_year": int(before_year - 1) if before_year else 9999}), "events")
        recs = [_rec(r) for r in rows]
        exact, loose = [], []
        for rec in recs:
            combo = norm_tokens(f"{rec.sport} {rec.event_name}")
            if toks == combo:
                exact.append(rec)
            elif toks <= combo:
                loose.append((len(combo - toks), rec))
        if exact:
            return sorted(exact, key=lambda r: -r.year)[:limit]
        loose.sort(key=lambda t: (t[0], -t[1].year))
        return [r for _s, r in loose[:limit]]

    @property
    def events(self) -> dict[str, EventRecord]:
        """Warm cache only; used for schema summaries, never for answering."""
        return self._event_cache
