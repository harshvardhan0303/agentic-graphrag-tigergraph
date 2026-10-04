"""Corpus -> property graph.

Reads ``corpus.jsonl`` (2,951 Wikipedia-derived documents) and emits the vertex
and edge tables that are loaded into TigerGraph, plus a chunked document table
used by the vector store.

Run:
    python -m src.ingest.parse_corpus --corpus data/corpus.jsonl --out data/graph
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import pathlib
import re
from collections import Counter
from typing import Iterator

from .normalize import norm, split_people, to_int

# '{Sport} at the {Year} {Season} Olympics - {Event}'
TITLE_RE = re.compile(
    r"^(?P<sport>.+?)\s+at\s+the\s+(?P<year>\d{4})\s+(?P<season>Summer|Winter)\s+"
    r"Olympics\s*[–—−-]\s*(?P<event>.+)$"
)

MEDALS = ("gold", "silver", "bronze")


@dataclasses.dataclass
class Document:
    doc_id: str
    title: str
    url: str
    kind: str
    approx_tokens: int
    text: str


@dataclasses.dataclass
class Event:
    doc_id: str
    title: str
    url: str
    sport: str
    year: int
    season: str
    games_id: str
    event_name: str
    venue: str | None
    date_raw: str | None
    competitors: int | None
    nations: int | None
    win_value: str | None
    prev_year: int | None
    next_year: int | None
    medalists: dict[str, list[tuple[str, str | None]]]  # medal -> [(name, noc)]


def parse_infobox(text: str) -> tuple[str | None, dict[str, str]]:
    """Return (infobox kind, fields). Fields are the indented 'key: value' lines."""
    lines = text.split("\n")
    if not lines or not lines[0].startswith("[Infobox"):
        return None, {}
    kind = lines[0][len("[Infobox"):].strip(" ]").lower()
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if not line.startswith("  "):
            break
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        fields[key.strip()] = value.strip()
    return kind, fields


def read_corpus(path: str | pathlib.Path) -> Iterator[Document]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            kind, _ = parse_infobox(raw["text"])
            yield Document(
                doc_id=raw["doc_id"],
                title=raw["title"],
                url=raw.get("url", ""),
                kind=kind or "none",
                approx_tokens=raw.get("approx_tokens", 0),
                text=raw["text"],
            )


def build_event(doc: Document, fields: dict[str, str]) -> Event | None:
    m = TITLE_RE.match(doc.title)
    if not m:
        return None
    year = int(m.group("year"))
    season = m.group("season")

    medalists: dict[str, list[tuple[str, str | None]]] = {}
    for medal in MEDALS:
        entries: list[tuple[str, str | None]] = []
        # 'bronze' and 'bronze2' both occur (two bronzes in judo/wrestling/taekwondo).
        for suffix in ("", "2"):
            raw_names = fields.get(f"{medal}{suffix}")
            noc = fields.get(f"{medal}NOC{suffix}")
            for name in split_people(raw_names):
                entries.append((name, noc))
        if entries:
            medalists[medal] = entries

    return Event(
        doc_id=doc.doc_id,
        title=doc.title,
        url=doc.url,
        sport=m.group("sport").strip(),
        year=year,
        season=season,
        games_id=f"{year} {season}",
        event_name=(fields.get("event") or m.group("event")).strip(),
        venue=fields.get("venue"),
        date_raw=fields.get("date") or fields.get("dates"),
        competitors=to_int(fields.get("competitors")),
        nations=to_int(fields.get("nations")),
        win_value=fields.get("win_value"),
        prev_year=to_int(fields.get("prev")),
        next_year=to_int(fields.get("next")),
        medalists=medalists,
    )


def chunk_text(text: str, target_chars: int = 1400, overlap: int = 200) -> list[str]:
    """Paragraph-aware chunking. The infobox block is always its own first chunk
    because it carries every fact the benchmark asks about."""
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not blocks:
        return []
    chunks: list[str] = []
    if blocks[0].startswith("[Infobox"):
        chunks.append(blocks[0])
        blocks = blocks[1:]
    buf = ""
    for block in blocks:
        if len(buf) + len(block) + 2 > target_chars and buf:
            chunks.append(buf.strip())
            buf = buf[-overlap:] if overlap else ""
        buf += ("\n\n" if buf else "") + block
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _write_csv(path: pathlib.Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--out", default="data/graph")
    ap.add_argument("--chunk-chars", type=int, default=1400)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    docs = list(read_corpus(args.corpus))
    events: list[Event] = []
    kinds = Counter(d.kind for d in docs)
    unparsed_titles: list[str] = []

    for doc in docs:
        kind, fields = parse_infobox(doc.text)
        if kind != "olympic event":
            continue
        ev = build_event(doc, fields)
        if ev is None:
            unparsed_titles.append(doc.title)
            continue
        events.append(ev)

    # ---------------- vertices ----------------
    _write_csv(
        out / "event.csv",
        ["doc_id", "title", "event_name", "sport", "games_id", "year", "season",
         "venue", "date_raw", "date_norm", "competitors", "nations", "win_value", "url"],
        [[e.doc_id, e.title, e.event_name, e.sport, e.games_id, e.year, e.season,
          e.venue or "", e.date_raw or "", norm(e.date_raw),
          e.competitors if e.competitors is not None else -1,
          e.nations if e.nations is not None else -1,
          e.win_value or "", e.url] for e in events],
    )

    games = {}
    for e in events:
        games.setdefault(e.games_id, (e.year, e.season, e.prev_year, e.next_year))
    _write_csv(out / "games.csv", ["games_id", "year", "season"],
               [[gid, y, s] for gid, (y, s, _p, _n) in sorted(games.items())])

    sports = sorted({e.sport for e in events})
    _write_csv(out / "sport.csv", ["sport_id", "name"], [[norm(s), s] for s in sports])

    venues = sorted({e.venue for e in events if e.venue})
    _write_csv(out / "venue.csv", ["venue_id", "name"], [[norm(v), v] for v in venues])

    athletes: dict[str, str] = {}
    nations: set[str] = set()
    for e in events:
        for entries in e.medalists.values():
            for name, noc in entries:
                athletes.setdefault(norm(name), name)
                if noc:
                    nations.add(noc)
    _write_csv(out / "athlete.csv", ["athlete_id", "name"],
               [[k, v] for k, v in sorted(athletes.items())])
    _write_csv(out / "nation.csv", ["noc"], [[n] for n in sorted(nations)])

    # documents + chunks (vector store input; also the citation unit)
    with open(out / "documents.jsonl", "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps({
                "doc_id": doc.doc_id, "title": doc.title, "url": doc.url,
                "kind": doc.kind, "text": doc.text,
            }, ensure_ascii=False) + "\n")

    n_chunks = 0
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for doc in docs:
            for i, chunk in enumerate(chunk_text(doc.text, args.chunk_chars)):
                fh.write(json.dumps({
                    "chunk_id": f"{doc.doc_id}#{i}", "doc_id": doc.doc_id,
                    "title": doc.title, "kind": doc.kind, "text": chunk,
                }, ensure_ascii=False) + "\n")
                n_chunks += 1

    # ---------------- edges ----------------
    _write_csv(out / "e_in_sport.csv", ["doc_id", "sport_id"],
               [[e.doc_id, norm(e.sport)] for e in events])
    _write_csv(out / "e_at_games.csv", ["doc_id", "games_id"],
               [[e.doc_id, e.games_id] for e in events])
    _write_csv(out / "e_held_at.csv", ["doc_id", "venue_id"],
               [[e.doc_id, norm(e.venue)] for e in events if e.venue])

    medal_rows = []
    rep_rows = set()
    for e in events:
        for medal, entries in e.medalists.items():
            for rank, (name, noc) in enumerate(entries):
                medal_rows.append([e.doc_id, norm(name), medal, rank])
                if noc:
                    rep_rows.add((norm(name), noc))
    _write_csv(out / "e_medal.csv", ["doc_id", "athlete_id", "medal", "rank"], medal_rows)
    _write_csv(out / "e_represents.csv", ["athlete_id", "noc"], sorted(rep_rows))

    # Games.PRECEDES built from the prev/next fields, restricted to games we hold.
    precedes = set()
    for e in events:
        if e.prev_year:
            prev_id = f"{e.prev_year} {e.season}"
            if prev_id in games:
                precedes.add((prev_id, e.games_id))
        if e.next_year:
            next_id = f"{e.next_year} {e.season}"
            if next_id in games:
                precedes.add((e.games_id, next_id))
    _write_csv(out / "e_precedes.csv", ["from_games", "to_games"], sorted(precedes))

    _write_csv(out / "e_evidence.csv", ["doc_id", "chunk_id"], [])  # populated by embed step

    stats = {
        "documents": len(docs),
        "doc_kinds": dict(kinds.most_common()),
        "olympic_events": len(events),
        "title_parse_failures": len(unparsed_titles),
        "games": len(games),
        "sports": len(sports),
        "venues": len(venues),
        "athletes": len(athletes),
        "nations": len(nations),
        "medal_edges": len(medal_rows),
        "precedes_edges": len(precedes),
        "chunks": n_chunks,
        "missing_competitors": sum(1 for e in events if e.competitors is None),
        "missing_nations": sum(1 for e in events if e.nations is None),
        "missing_venue": sum(1 for e in events if not e.venue),
        "missing_date": sum(1 for e in events if not e.date_raw),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    if unparsed_titles:
        print("\nUNPARSED TITLES:")
        for t in unparsed_titles[:20]:
            print("  ", t)


if __name__ == "__main__":
    main()
