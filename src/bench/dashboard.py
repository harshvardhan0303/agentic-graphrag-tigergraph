"""Metrics dashboard generator.

Reads the benchmark output and emits a single self-contained HTML file
comparing the three pipelines on accuracy, completeness, token cost and
efficiency, plus the agentic behaviour panel the guidebook asks for.

    python -m src.bench.dashboard --run out/public --out out/dashboard.html
"""
from __future__ import annotations

import argparse
import collections
import html
import datetime
import json
import pathlib
from typing import Any

# Categorical slots 1-3 of the validated palette. Three series only: the
# four-slot ordering fails the all-pairs normal-vision floor, so the ablation
# pipeline appears in the table view rather than in the charts.
SERIES = {
    "rag":               {"label": "RAG",             "light": "#2a78d6", "dark": "#3987e5"},
    "graphrag":          {"label": "GraphRAG",        "light": "#eb6834", "dark": "#d95926"},
    "agentic":           {"label": "Agentic GraphRAG", "light": "#1baf7a", "dark": "#199e70"},
    "agentic_no_router": {"label": "Agentic (no router)", "light": "#4a3aa7", "dark": "#9085e9"},
}
CHART_PIPELINES = ["rag", "graphrag", "agentic"]
QTYPE_ORDER = ["lookup", "temporal", "multi_hop", "aggregation", "superlative"]

CSS = """
*,*::before,*::after{box-sizing:border-box}
.viz-root{
  color-scheme:light;
  --surface-0:#f4f4f2; --surface-1:#fcfcfb; --border:#e2e1dc;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#82817b;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#4a3aa7;
  --grid:#eceae5;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
  color-scheme:dark;
  --surface-0:#111110; --surface-1:#1a1a19; --border:#33322f;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8b8a82;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#9085e9;
  --grid:#2a2a27;
}}
:root[data-theme="dark"] .viz-root{
  color-scheme:dark;
  --surface-0:#111110; --surface-1:#1a1a19; --border:#33322f;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8b8a82;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#9085e9;
  --grid:#2a2a27;
}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:28px;line-height:1.2;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:18px;margin:44px 0 4px;letter-spacing:-.01em}
.sub{color:var(--text-secondary);margin:0 0 4px}
.muted{color:var(--text-muted);font-size:13px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;
  padding:20px 22px;margin-top:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:18px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:18px 20px}
.tile .name{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--text-secondary)}
.swatch{width:10px;height:10px;border-radius:3px;flex:none}
.hero{font-size:34px;font-weight:650;letter-spacing:-.02em;margin:8px 0 2px;
  font-variant-numeric:tabular-nums}
.tile .foot{font-size:12.5px;color:var(--text-muted);font-variant-numeric:tabular-nums}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:2px 0 16px;font-size:13px;color:var(--text-secondary)}
.legend span{display:inline-flex;align-items:center;gap:7px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--border);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--text-secondary);font-weight:550;font-size:12.5px;
  text-transform:uppercase;letter-spacing:.04em}
tbody tr:hover{background:var(--surface-0)}
.best{font-weight:650}
.scroll{overflow-x:auto}
.finding{border-left:3px solid var(--s3);padding:2px 0 2px 14px;margin:14px 0;
  color:var(--text-secondary)}
.finding b{color:var(--text-primary)}
svg{display:block;width:100%;height:auto;overflow:visible}
.axis-label{font-size:11.5px;fill:var(--text-muted)}
.tick{font-size:11.5px;fill:var(--text-muted)}
.cat{font-size:12.5px;fill:var(--text-secondary)}
.val{font-size:11px;fill:var(--text-secondary);font-variant-numeric:tabular-nums}
.gridline{stroke:var(--grid);stroke-width:1}
.bar{transition:opacity .12s}
.bar:hover{opacity:.78;cursor:default}
.toggle{position:fixed;top:14px;right:14px;background:var(--surface-1);
  border:1px solid var(--border);color:var(--text-secondary);border-radius:8px;
  padding:6px 12px;font-size:12.5px;cursor:pointer}
@media (max-width:640px){.wrap{padding:24px 16px 60px}h1{font-size:23px}.hero{font-size:28px}}
"""

TOOLTIP_JS = """
(function(){
  var tip=document.createElement('div');
  tip.style.cssText='position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;'+
    'background:var(--surface-1);border:1px solid var(--border);border-radius:8px;'+
    'padding:7px 10px;font:12.5px/1.45 inherit;color:var(--text-primary);'+
    'box-shadow:0 6px 22px rgba(0,0,0,.14);z-index:99;max-width:280px';
  document.body.appendChild(tip);
  document.addEventListener('mouseover',function(e){
    var t=e.target.closest('[data-tip]'); if(!t) return;
    tip.innerHTML=t.getAttribute('data-tip'); tip.style.opacity='1';
  });
  document.addEventListener('mousemove',function(e){
    if(tip.style.opacity!=='1') return;
    var x=e.clientX+14,y=e.clientY+14;
    if(x+tip.offsetWidth>innerWidth-8) x=e.clientX-tip.offsetWidth-14;
    if(y+tip.offsetHeight>innerHeight-8) y=e.clientY-tip.offsetHeight-14;
    tip.style.left=x+'px'; tip.style.top=y+'px';
  });
  document.addEventListener('mouseout',function(e){
    if(e.target.closest('[data-tip]')) tip.style.opacity='0';
  });
  var btn=document.querySelector('.toggle');
  if(btn) btn.addEventListener('click',function(){
    var cur=document.documentElement.getAttribute('data-theme');
    var next = cur==='dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme',next);
    btn.textContent = next==='dark' ? 'Light mode' : 'Dark mode';
  });
})();
"""


def _human_time(iso: str) -> str:
    """'2026-10-04T11:11:25' -> '4 October 2026, 11:11'.

    The raw ISO string is for machines; a report a person reads should carry a
    date a person writes. Falls back to the input if it will not parse.
    """
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.datetime.strptime(iso, fmt)
        except (ValueError, TypeError):
            continue
        day = dt.day
        suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        stamp = f"{day}{suffix} {dt.strftime('%B %Y')}"
        return stamp if fmt == "%Y-%m-%d" else f"{stamp} at {dt.strftime('%H:%M')}"
    return iso or "unknown"


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def grouped_bars(categories: list[str], series: list[dict[str, Any]],
                 fmt=lambda v: f"{v:.0%}", y_label: str = "",
                 height: int = 260) -> str:
    """Grouped vertical bars. 4px rounded data-end, 2px gap between bars,
    direct value labels (relief for the low-contrast slot), hover tooltips."""
    if not categories or not series:
        return "<p class='muted'>No data.</p>"
    W, H = 1000, height
    pad_l, pad_r, pad_t, pad_b = 46, 8, 18, 46
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    vmax = max([v for s in series for v in s["values"] if v is not None] or [1])
    vmax = vmax * 1.18 or 1
    n_cat, n_ser = len(categories), len(series)
    group_w = plot_w / n_cat
    bar_gap = 2
    bar_w = max(6.0, (group_w * 0.74 - bar_gap * (n_ser - 1)) / n_ser)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(y_label)}">']
    for i in range(5):
        y = pad_t + plot_h * i / 4
        val = vmax * (1 - i / 4)
        out.append(f'<line class="gridline" x1="{pad_l}" y1="{y:.1f}" '
                   f'x2="{W - pad_r}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{pad_l - 8}" y="{y + 4:.1f}" '
                   f'text-anchor="end">{_esc(fmt(val))}</text>')

    for ci, cat in enumerate(categories):
        gx = pad_l + group_w * ci
        cx = gx + group_w / 2
        block_w = bar_w * n_ser + bar_gap * (n_ser - 1)
        x0 = cx - block_w / 2
        for si, s in enumerate(series):
            v = s["values"][ci]
            x = x0 + si * (bar_w + bar_gap)
            if v is None:
                out.append(f'<text class="val" x="{x + bar_w / 2:.1f}" '
                           f'y="{pad_t + plot_h - 4}" text-anchor="middle">–</text>')
                continue
            h = max(1.5, plot_h * (v / vmax))
            y = pad_t + plot_h - h
            tip = (f"<b>{_esc(s['label'])}</b><br>{_esc(cat)}: "
                   f"<b>{_esc(fmt(v))}</b>")
            out.append(
                f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{h:.1f}" rx="4" fill="{s["color"]}" data-tip="{tip}"/>')
            out.append(f'<text class="val" x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" '
                       f'text-anchor="middle">{_esc(fmt(v))}</text>')
        out.append(f'<text class="cat" x="{cx:.1f}" y="{pad_t + plot_h + 20}" '
                   f'text-anchor="middle">{_esc(cat)}</text>')
    if y_label:
        out.append(f'<text class="axis-label" x="{pad_l}" y="{H - 8}">{_esc(y_label)}</text>')
    out.append("</svg>")
    return "".join(out)


def legend(pipelines: list[str]) -> str:
    items = []
    for p in pipelines:
        meta = SERIES[p]
        items.append(f'<span><i class="swatch" style="background:var(--{_slot(p)})"></i>'
                     f'{_esc(meta["label"])}</span>')
    return '<div class="legend">' + "".join(items) + "</div>"


def _slot(pipeline: str) -> str:
    return {"rag": "s1", "graphrag": "s2", "agentic": "s3",
            "agentic_no_router": "s4"}.get(pipeline, "s1")


def build(run_dir: pathlib.Path, title: str) -> str:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in open(run_dir / "results.jsonl", encoding="utf-8")
            if l.strip()]
    by_pipe: dict[str, Any] = summary["by_pipeline"]
    # Name the question set on the page. Only the public set carries gold
    # answers, so only it can be scored; a reader should never have to guess
    # whether a reported accuracy came from the held-out 50.
    qset = {"public": "eval_public.jsonl (public)",
            "hidden": "eval_hidden.jsonl (held out)"}.get(
                run_dir.name, f"{run_dir.name}/")
    present = [p for p in CHART_PIPELINES if p in by_pipe]
    all_pipes = [p for p in SERIES if p in by_pipe]
    qtypes = [q for q in QTYPE_ORDER
              if any(q in by_pipe[p].get("by_qtype", {}) for p in all_pipes)]

    def series_for(metric: str, pipes: list[str]) -> list[dict[str, Any]]:
        out = []
        for p in pipes:
            out.append({
                "label": SERIES[p]["label"],
                "color": f"var(--{_slot(p)})",
                "values": [by_pipe[p].get("by_qtype", {}).get(q, {}).get(metric)
                           for q in qtypes],
            })
        return out

    graded = any(by_pipe[p]["overall"].get("judge_accuracy") is not None
                 for p in all_pipes)
    acc_key = "judge_accuracy"

    # ---- hero tiles ----
    tiles = []
    for p in all_pipes:
        o = by_pipe[p]["overall"]
        acc = o.get(acc_key)
        head = f"{acc:.0%}" if acc is not None else f"{o.get('avg_total_tokens', 0):,.0f}"
        head_label = "accuracy" if acc is not None else "tokens / question"
        tiles.append(
            f'<div class="tile"><div class="name">'
            f'<i class="swatch" style="background:var(--{_slot(p)})"></i>'
            f'{_esc(SERIES[p]["label"])}</div>'
            f'<div class="hero">{head}</div>'
            f'<div class="foot">{head_label} &middot; '
            f'{o.get("avg_total_tokens", 0):,.0f} tok/q &middot; '
            f'{o.get("avg_steps", 0)} steps</div></div>')

    # ---- agentic behaviour ----
    ag_rows = [r for r in rows if r["pipeline"] == "agentic"]
    tool_counts = collections.Counter(
        t for r in ag_rows for t in (r.get("tools_called") or []))
    agent_counts = collections.Counter(
        a for r in ag_rows for a in (r.get("agents_invoked") or []) if a)
    stop_counts = collections.Counter(r.get("stop_reason", "") for r in ag_rows)
    steps_by_type = collections.defaultdict(list)
    for r in ag_rows:
        steps_by_type[r.get("qtype", "?")].append(r.get("n_steps", 0))

    def count_table(counter: collections.Counter, head: str) -> str:
        total = sum(counter.values()) or 1
        body = "".join(
            f"<tr><td>{_esc(k or '(none)')}</td><td>{v}</td>"
            f"<td>{v / total:.0%}</td></tr>"
            for k, v in counter.most_common(12))
        return (f"<table><thead><tr><th>{_esc(head)}</th><th>count</th>"
                f"<th>share</th></tr></thead><tbody>{body}</tbody></table>")

    # ---- main table ----
    metric_cols = [
        ("judge_accuracy", "Accuracy", lambda v: f"{v:.0%}", True),
        ("exact_accuracy", "Exact", lambda v: f"{v:.0%}", True),
        ("completeness_doc_recall", "Completeness", lambda v: f"{v:.0%}", True),
        ("citation_precision", "Cite prec.", lambda v: f"{v:.0%}", True),
        ("avg_total_tokens", "Tokens / q", lambda v: f"{v:,.0f}", False),
        ("avg_llm_calls", "LLM calls", lambda v: f"{v:.1f}", False),
        ("avg_steps", "Steps", lambda v: f"{v:.1f}", False),
        ("accuracy_per_1k_tokens", "Acc / 1k tok", lambda v: f"{v:.3f}", True),
        ("unknown_rate", "Unknown", lambda v: f"{v:.0%}", False),
    ]
    head = "".join(f"<th>{_esc(c[1])}</th>" for c in metric_cols)
    body_rows = []
    for p in all_pipes:
        o = by_pipe[p]["overall"]
        best_flags = {}
        for key, _lbl, _f, higher in metric_cols:
            vals = [by_pipe[q]["overall"].get(key) for q in all_pipes]
            vals = [v for v in vals if v is not None]
            if vals:
                best_flags[key] = (max(vals) if higher else min(vals))
        cells = []
        for key, _lbl, f, _h in metric_cols:
            v = o.get(key)
            if v is None:
                cells.append("<td>–</td>")
                continue
            cls = " class='best'" if best_flags.get(key) == v else ""
            cells.append(f"<td{cls}>{_esc(f(v))}</td>")
        body_rows.append(
            f'<tr><td><i class="swatch" style="display:inline-block;'
            f'background:var(--{_slot(p)})"></i> {_esc(SERIES[p]["label"])}</td>'
            + "".join(cells) + "</tr>")

    # per-qtype detail table
    detail = []
    for q in qtypes:
        for p in all_pipes:
            d = by_pipe[p].get("by_qtype", {}).get(q)
            if not d:
                continue
            acc = d.get("judge_accuracy")
            acc_cell = f"{acc:.0%}" if acc is not None else "&ndash;"
            swatch = (f'<i class="swatch" style="display:inline-block;'
                      f'background:var(--{_slot(p)})"></i>')
            detail.append(
                f"<tr><td>{_esc(q)}</td>"
                f"<td style='text-align:left'>{swatch} {_esc(SERIES[p]['label'])}</td>"
                f"<td>{d['n']}</td><td>{acc_cell}</td>"
                f"<td>{d.get('completeness_doc_recall', 0):.0%}</td>"
                f"<td>{d.get('avg_total_tokens', 0):,.0f}</td>"
                f"<td>{d.get('avg_steps', 0):.1f}</td></tr>")

    findings = derive_findings(by_pipe, all_pipes, qtypes, acc_key)

    charts = []
    if graded:
        charts.append(("Investigation accuracy by question type",
                       "Share of questions answered correctly (LLM-as-judge, "
                       "exact match fast-path).",
                       grouped_bars(qtypes, series_for(acc_key, present))))
        charts.append(("Completeness: gold-document recall",
                       "Of the documents the benchmark marks as required, how many "
                       "the pipeline actually cited. This is where single-shot "
                       "retrieval breaks on aggregation and superlative questions.",
                       grouped_bars(qtypes, series_for("completeness_doc_recall", present))))
    charts.append(("Token cost per question",
                   "Total LLM tokens (input + output) spent per answered question.",
                   grouped_bars(qtypes, series_for("avg_total_tokens", present),
                                fmt=lambda v: f"{v:,.0f}", y_label="tokens")))
    if graded:
        charts.append(("Efficiency: accuracy per 1,000 tokens",
                       "The number that decides whether the agent is worth it. "
                       "Higher is better.",
                       grouped_bars(qtypes, series_for("accuracy_per_1k_tokens", present),
                                    fmt=lambda v: f"{v:.2f}")))

    chart_html = "".join(
        f'<h2>{_esc(t)}</h2><p class="sub muted">{_esc(sub)}</p>'
        f'<div class="card">{legend(present)}{svg}</div>'
        for t, sub, svg in charts)

    steps_tbl = "".join(
        f"<tr><td>{_esc(k)}</td><td>{len(v)}</td>"
        f"<td>{sum(v) / len(v):.2f}</td><td>{max(v)}</td></tr>"
        for k, v in sorted(steps_by_type.items()))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title><style>{CSS}</style></head>
<body class="viz-root"><button class="toggle">Dark mode</button><div class="wrap">
<h1>{_esc(title)}</h1>
<p class="sub">RAG vs GraphRAG vs Agentic GraphRAG on TigerGraph &mdash;
{summary['questions']} questions, {len(all_pipes)} pipelines,
{summary.get('wall_time_s', 0):,.0f}s wall time.</p>
<p class="muted"><b>Question set:</b> {_esc(qset)} &mdash; the
{summary['questions']} questions that ship <em>with</em> gold answers, so they
can be scored. The 50 held-out questions carry no answers; our answers, citations
and traces for those are in <code>out/hidden/submission.jsonl</code> for the
organisers to score. No accuracy is self-reported on the held-out set.</p>
<p class="muted">Generated {_esc(_human_time(summary.get('generated_at', '')))} &middot;
model {_esc(summary.get('llm_model', '?'))} &middot;
graph backend {_esc(summary.get('graph_backend', '?'))} &middot;
{summary.get('infrastructure_errors', {}).get('n', 0)} infrastructure errors &middot;
judge made {summary.get('judge_llm_calls', 0)} calls
({summary.get('judge_tokens', 0):,} tokens, excluded from pipeline budgets).</p>

<div class="tiles">{''.join(tiles)}</div>

<h2>What the numbers say</h2>
<div class="card">{findings}</div>

{chart_html}

<h2>All metrics</h2>
<div class="card scroll"><table>
<thead><tr><th>Pipeline</th>{head}</tr></thead>
<tbody>{''.join(body_rows)}</tbody></table>
<p class="muted" style="margin-top:12px">Bold = best in column.
Completeness is gold-document recall. Accuracy per 1k tokens =
accuracy &divide; (mean tokens per question &divide; 1000).</p></div>

<h2>Agentic behaviour</h2>
<p class="sub muted">Trace statistics for the Agentic GraphRAG pipeline:
what it called, how far it went, and why it stopped.</p>
<div class="card scroll" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:26px">
<div><h3 style="font-size:14px;margin:0 0 8px">Tools called</h3>{count_table(tool_counts, 'tool')}</div>
<div><h3 style="font-size:14px;margin:0 0 8px">Specialist agents invoked</h3>{count_table(agent_counts, 'agent')}</div>
<div><h3 style="font-size:14px;margin:0 0 8px">Why it stopped</h3>{count_table(stop_counts, 'stop reason')}</div>
<div><h3 style="font-size:14px;margin:0 0 8px">Steps per question type</h3>
<table><thead><tr><th>type</th><th>n</th><th>avg</th><th>max</th></tr></thead>
<tbody>{steps_tbl}</tbody></table></div>
</div>

<h2>Per question type</h2>
<div class="card scroll"><table>
<thead><tr><th>Question type</th><th>Pipeline</th><th>n</th><th>Accuracy</th>
<th>Completeness</th><th>Tokens / q</th><th>Steps</th></tr></thead>
<tbody>{''.join(detail)}</tbody></table></div>

</div><script>{TOOLTIP_JS}</script></body></html>"""


def derive_findings(by_pipe: dict, pipes: list[str], qtypes: list[str],
                    acc_key: str) -> str:
    """Write the comparison conclusions from the data, not from hope."""
    out = []
    ov = {p: by_pipe[p]["overall"] for p in pipes}
    if "agentic" in ov and "rag" in ov and ov["agentic"].get(acc_key) is not None:
        a, r = ov["agentic"][acc_key], ov["rag"].get(acc_key) or 0
        cost = (ov["agentic"]["avg_total_tokens"]
                / max(ov["rag"]["avg_total_tokens"], 1))
        out.append(f"<p class='finding'><b>Overall:</b> Agentic GraphRAG answers "
                   f"{a:.0%} of questions correctly against {r:.0%} for plain RAG, "
                   f"at {cost:.1f}&times; the token cost.</p>")
    if "agentic" in by_pipe and "graphrag" in by_pipe:
        deltas = []
        for q in qtypes:
            ag = by_pipe["agentic"].get("by_qtype", {}).get(q, {}).get(acc_key)
            gr = by_pipe["graphrag"].get("by_qtype", {}).get(q, {}).get(acc_key)
            if ag is None or gr is None:
                continue
            deltas.append((ag - gr, q, ag, gr))
        deltas.sort(reverse=True)
        if deltas:
            d, q, ag, gr = deltas[0]
            if d > 0.02:
                out.append(f"<p class='finding'><b>Where the agent earns its cost:</b> "
                           f"on <b>{q}</b> questions it scores {ag:.0%} against "
                           f"{gr:.0%} for single-pass GraphRAG &mdash; a "
                           f"{d:+.0%} swing that comes from re-planning after "
                           f"the first retrieval.</p>")
            d2, q2, ag2, gr2 = deltas[-1]
            if abs(d2) <= 0.02:
                tok_a = by_pipe["agentic"]["by_qtype"][q2]["avg_total_tokens"]
                tok_g = by_pipe["graphrag"]["by_qtype"][q2]["avg_total_tokens"]
                out.append(f"<p class='finding'><b>Where it is overkill:</b> on "
                           f"<b>{q2}</b> questions both reach {ag2:.0%}, but the "
                           f"agent spends {tok_a:,.0f} tokens against "
                           f"{tok_g:,.0f}. A single graph lookup is enough; the "
                           f"planning loop buys nothing.</p>")
    if "agentic" in ov and "agentic_no_router" in ov:
        a, b = ov["agentic"], ov["agentic_no_router"]
        out.append(f"<p class='finding'><b>Router ablation:</b> cost-aware routing "
                   f"changes accuracy by {(a.get(acc_key) or 0) - (b.get(acc_key) or 0):+.0%} "
                   f"while changing token spend by "
                   f"{a['avg_total_tokens'] / max(b['avg_total_tokens'], 1) - 1:+.0%}.</p>")
    return "".join(out) or "<p class='muted'>Run the graded benchmark to populate this.</p>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/public")
    ap.add_argument("--out", default="out/dashboard.html")
    ap.add_argument("--title", default="Agentic GraphRAG Benchmark")
    args = ap.parse_args()
    html_text = build(pathlib.Path(args.run), args.title)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(html_text, encoding="utf-8")
    print(f"wrote {args.out} ({len(html_text):,} bytes)")


if __name__ == "__main__":
    main()
