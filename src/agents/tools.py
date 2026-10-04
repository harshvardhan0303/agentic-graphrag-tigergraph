"""The tool surface the orchestrator chooses from.

Each tool returns ``(evidence_list, summary_string)``. The summary is what goes
back into the LLM context; the evidence carries the citable doc ids. Keeping
those separate is what stops raw graph output from being re-tokenised on every
turn.

Cost hints are deliberate and are surfaced to the orchestrator: they are how the
router learns that an exact graph aggregation is cheaper *and* more accurate
than reading 40 documents.
"""
from __future__ import annotations

from typing import Any

from ..graph.store import GraphStore
from ..ingest.normalize import norm
from .harness import Evidence, Tool, ToolRegistry


def _ev(source: str, claim: str, doc_ids: list[str], **payload) -> Evidence:
    return Evidence(source=source, claim=claim, doc_ids=list(doc_ids), payload=payload)


# A language model will write "more than" as readily as ">". Both mean the same
# thing, and a KeyError here silently degrades the pipeline into guessing from
# text, so the tool normalises rather than demanding one spelling.
_OP_WORDS = {
    ">": ">", "gt": ">", "more than": ">", "greater than": ">", "above": ">",
    "over": ">", "exceeds": ">", "exceeding": ">", "strictly greater than": ">",
    ">=": ">=", "gte": ">=", "at least": ">=", "no fewer than": ">=",
    "greater than or equal to": ">=", "minimum": ">=",
    "<": "<", "lt": "<", "fewer than": "<", "less than": "<", "below": "<",
    "under": "<",
    "<=": "<=", "lte": "<=", "at most": "<=", "no more than": "<=",
    "less than or equal to": "<=", "maximum": "<=",
    "==": "==", "=": "==", "eq": "==", "equal to": "==", "exactly": "==",
    "!=": "!=", "ne": "!=", "not equal to": "!=",
}


def normalize_op(op: str, default: str = ">") -> str:
    key = (op or "").strip().lower()
    return _OP_WORDS.get(key, default)


def build_registry(store: GraphStore, vectors=None, documents=None) -> ToolRegistry:
    reg = ToolRegistry()

    # ---------------- entity linking ----------------
    def link_event(phrase: str = "", season: str = "", at_year: int = 0,
                   exact_title: str = "") -> tuple[list[Evidence], str]:
        if exact_title:
            hits = store.find_events(title=exact_title)
            if hits:
                e = hits[0]
                return ([_ev("link_event", f"'{exact_title}' resolves to {e.title}",
                             [e.doc_id], event=e.as_dict())],
                        f"Exact title match: {e.title} (doc {e.doc_id}); "
                        f"competitors={e.competitors}, nations={e.nations}, "
                        f"venue={e.venue}, date={e.date_raw}")
        if not phrase and not exact_title:
            return [], ("link_event needs either `phrase` (a description of the "
                        "event in words) or `exact_title`. Retry with one of "
                        "them filled in.")
        hits = store.resolve_event_phrase(phrase, season or None)
        if at_year:
            exact_year = [h for h in hits if h.year == int(at_year)]
            if exact_year:
                hits = exact_year
        if not hits:
            return [], f"No event matched the phrase {phrase!r}."
        lines = [f"{h.title} (doc {h.doc_id}, {h.games_id})" for h in hits[:8]]
        evs = [_ev("link_event", f"candidate for '{phrase}': {h.title}", [h.doc_id],
                   event=h.as_dict()) for h in hits[:8]]
        return evs, f"{len(hits)} candidate event(s):\n- " + "\n- ".join(lines)

    reg.register(Tool(
        name="link_event",
        description=("Entity linking. Resolve a natural-language event description "
                     "(e.g. \"men's pole vault athletics\") or an exact article title "
                     "to Event nodes in the graph. Use this FIRST when the question "
                     "names an event.\n"
                     "IMPORTANT: at_year must be a year the question states, never "
                     "one you derive. When a question points at a Games relative to "
                     "a stated year, link at the stated year and let "
                     "previous_games_event take the step. Doing both moves twice "
                     "and lands on the wrong Games."),
        parameters={"phrase": "event description in words",
                    "season": "'Summer' or 'Winter' if known, else ''",
                    "at_year": "the Games year NAMED in the question (0 if none named)",
                    "exact_title": "full article title if the question quotes one"},
        fn=link_event, cost_hint="cheap", category="entity_linking"))

    # ---------------- graph traversal ----------------
    def events_at_venue_on_date(venue: str, date: str, year: int = 0
                                ) -> tuple[list[Evidence], str]:
        hits = store.events_by_venue_date(venue, date, year or None)
        if not hits:
            return [], (f"No event found at venue {venue!r} on date {date!r}. "
                        f"Try a different date spelling, or search documents.")
        evs, lines = [], []
        for h in hits:
            evs.append(_ev("events_at_venue_on_date",
                           f"{h.title} was held at {h.venue} on {h.date_raw}",
                           [h.doc_id], event=h.as_dict()))
            lines.append(f"{h.title} (doc {h.doc_id})")
        return evs, f"{len(hits)} event(s) at that venue/date:\n- " + "\n- ".join(lines)

    reg.register(Tool(
        name="events_at_venue_on_date",
        description=("Graph traversal Venue -> Event. Returns the event(s) that "
                     "took place at a given venue on a given date. Pass the venue "
                     "name and the date string exactly as they appear in the "
                     "question; matching is done on normalised forms."),
        parameters={"venue": "venue name as written in the question",
                    "date": "date string as written in the question",
                    "year": "Games year if the question states one (0 = unknown)"},
        fn=events_at_venue_on_date, cost_hint="cheap", category="graph_traversal"))

    def get_medalists(doc_id: str, medal: str = "gold") -> tuple[list[Evidence], str]:
        rec = store.get_event(doc_id)
        people = store.medalists(doc_id, medal)
        if not people:
            return [], f"No {medal} medalist recorded for {doc_id}."
        names = [p["name"] for p in people]
        title = rec.title if rec else doc_id
        claim = (f"{medal} medal in '{title}' (doc_id {doc_id}) won by "
                 f"{', '.join(names)}")
        return ([_ev("get_medalists", claim, [doc_id], names=names, medal=medal)],
                claim)

    reg.register(Tool(
        name="get_medalists",
        description="Graph traversal Event -> Athlete. Returns the gold/silver/bronze "
                    "medalists of one event, in team order.",
        parameters={"doc_id": "Event doc_id from a previous step",
                    "medal": "'gold', 'silver' or 'bronze'"},
        fn=get_medalists, cost_hint="cheap", category="graph_traversal"))

    def previous_games_event(doc_id: str) -> tuple[list[Evidence], str]:
        """Temporal hop: the same event at the immediately preceding Games."""
        rec = store.get_event(doc_id)
        if rec is None:
            return [], f"Unknown event {doc_id!r}."
        prev = store.previous_games(rec.year, rec.season)
        if not prev:
            return [], f"No Games precede {rec.games_id} in the corpus."
        counterpart = store.counterpart_in_games(doc_id, prev)
        if counterpart is None:
            return ([], f"{prev} precedes {rec.games_id}, but '{rec.event_name}' "
                        f"was not contested there (or is titled differently).")
        # The doc_id of the counterpart MUST appear in the summary. Without it
        # the next step has no identifier to act on except the anchor it came
        # from, and the agent reads medalists off the wrong event. This cost
        # several temporal questions before it was spotted in the traces.
        claim = (f"The Games immediately before {rec.games_id} is {prev}. "
                 f"The counterpart event is '{counterpart.title}' "
                 f"with doc_id {counterpart.doc_id}. "
                 f"USE doc_id {counterpart.doc_id} for any further step about "
                 f"that event - NOT {rec.doc_id}, which is the anchor.")
        return ([_ev("previous_games_event", claim,
                     [rec.doc_id, counterpart.doc_id],
                     event=counterpart.as_dict(),
                     answer_doc_id=counterpart.doc_id)], claim)

    reg.register(Tool(
        name="previous_games_event",
        description=("Temporal traversal over the Games.PRECEDES edge. Given an "
                     "event at one Games, returns the same event at the preceding "
                     "Games of the same season. Call it ONCE per step back in time, "
                     "and never compute the earlier year yourself - the graph knows "
                     "which Games precedes which, including when a sport skipped "
                     "one."),
        parameters={"doc_id": "Event doc_id at the anchor Games"},
        fn=previous_games_event, cost_hint="cheap", category="multi_hop"))

    # ---------------- aggregation ----------------
    def count_events(sport: str, year: int, season: str,
                     field: str = "competitors", op: str = ">",
                     value: int = 0) -> tuple[list[Evidence], str]:
        op = normalize_op(op)
        r = store.count_events_where(sport, int(year), season, field, op, int(value))
        if r["considered"] == 0:
            return [], (f"No {sport} events found at the {year} {season} Olympics. "
                        f"Check the sport spelling.")
        claim = (f"{r['count']} of {r['considered']} {sport} events at the "
                 f"{year} {season} Olympics have {field} {op} {value}")
        dash = "–"
        detail = "; ".join(
            "{}={}".format(m["title"].split(dash)[-1].strip(), m[field])
            for m in r["matched"][:12])
        return ([_ev("count_events", claim, r["evidence_doc_ids"], **r)],
                f"{claim}. Matched: {detail}"
                + (f" ({r['missing_field']} events have no {field} recorded)"
                   if r["missing_field"] else ""))

    reg.register(Tool(
        name="count_events",
        description=("Exact aggregation executed in the graph. Counts events of one "
                     "sport at one Games satisfying a numeric condition. Prefer this "
                     "over reading documents: it is exact, and it costs a constant "
                     "number of tokens regardless of how many events are involved."),
        parameters={"sport": "sport name, e.g. 'Cycling'", "year": "Games year",
                    "season": "'Summer' or 'Winter'",
                    "field": "'competitors' or 'nations'",
                    "op": "comparison: > >= < <= == != (words like 'more than' "
                          "are also accepted)", "value": "threshold"},
        fn=count_events, cost_hint="cheap", category="aggregation"))

    def top_event(sport: str, year: int, season: str,
                  field: str = "competitors",
                  direction: str = "max") -> tuple[list[Evidence], str]:
        r = store.extreme_event(sport, int(year), season, field, direction)
        if not r["event"]:
            return [], f"No {sport} events with {field} recorded at {year} {season}."
        ev = r["event"]
        claim = (f"The {sport} event at the {year} {season} Olympics with the "
                 f"{'highest' if direction == 'max' else 'lowest'} {field} is "
                 f"'{ev['title']}' ({field}={r['value']})")
        tie_note = (f" NOTE: {len(r['ties'])} events tie at {r['value']}: "
                    + "; ".join(r["ties"])) if len(r["ties"]) > 1 else ""
        return ([_ev("top_event", claim, r["evidence_doc_ids"], **r)], claim + tie_note)

    reg.register(Tool(
        name="top_event",
        description=("Superlative executed in the graph: the event of one sport at one "
                     "Games with the highest/lowest competitors or nations. Reports "
                     "ties explicitly instead of picking arbitrarily."),
        parameters={"sport": "sport name", "year": "Games year",
                    "season": "'Summer' or 'Winter'",
                    "field": "'competitors' or 'nations'",
                    "direction": "'max' or 'min'"},
        fn=top_event, cost_hint="cheap", category="aggregation"))

    def event_facts(doc_id: str) -> tuple[list[Evidence], str]:
        rec = store.get_event(doc_id)
        if rec is None:
            return [], f"Unknown event {doc_id!r}."
        claim = (f"{rec.title}: competitors={rec.competitors}, nations={rec.nations}, "
                 f"venue={rec.venue}, date={rec.date_raw}, games={rec.games_id}")
        return [_ev("event_facts", claim, [doc_id], event=rec.as_dict())], claim

    reg.register(Tool(
        name="event_facts",
        description="Read the structured attributes of one Event node "
                    "(competitors, nations, venue, date, Games, winning value).",
        parameters={"doc_id": "Event doc_id"},
        fn=event_facts, cost_hint="cheap", category="graph_traversal"))

    # ---------------- vector + document ----------------
    if vectors is not None:
        def similarity_search(query: str, k: int = 5) -> tuple[list[Evidence], str]:
            hits = vectors.search(query, k=int(k))
            if not hits:
                return [], "Vector search returned nothing."
            evs, lines = [], []
            for h in hits:
                snippet = h["text"][:400].replace("\n", " ")
                evs.append(_ev("similarity_search",
                               f"passage from '{h['title']}': {snippet}",
                               [h["doc_id"]], score=h["score"],
                               chunk_id=h["chunk_id"]))
                lines.append(f"({h['score']:.3f}) {h['title']} [{h['doc_id']}]: {snippet}")
            return evs, "\n".join(lines)

        reg.register(Tool(
            name="similarity_search",
            description=("Semantic search over document chunks in the TigerGraph "
                         "vector store. Use when the question mentions something not "
                         "addressable by the graph schema, or when entity linking "
                         "failed. Expensive in tokens: each hit is a text passage."),
            parameters={"query": "search text", "k": "number of chunks (default 5)"},
            fn=similarity_search, cost_hint="expensive", category="similarity_search"))

    if documents is not None:
        def read_document(doc_id: str, max_chars: int = 2500
                          ) -> tuple[list[Evidence], str]:
            doc = documents.get(doc_id)
            if not doc:
                return [], f"No document {doc_id!r}."
            text = doc["text"][:int(max_chars)]
            return ([_ev("read_document", f"text of '{doc['title']}'", [doc_id],
                         chars=len(text))],
                    f"{doc['title']} [{doc_id}]\n{text}")

        reg.register(Tool(
            name="read_document",
            description=("Fetch the raw article text for one document. Use only to "
                         "confirm a fact the graph cannot express - it is the most "
                         "token-expensive tool available."),
            parameters={"doc_id": "document id", "max_chars": "truncate to N chars"},
            fn=read_document, cost_hint="expensive", category="document_retrieval"))

    return reg


def sport_vocabulary(store: GraphStore) -> list[str]:
    return sorted({e.sport for e in store.events.values()})  # type: ignore[attr-defined]
