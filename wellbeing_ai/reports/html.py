"""
Self-contained HTML reports. No CDN, no JS libraries, no network at render or
at view time -- open the file in six months on a plane and it still works.

Charts are inline SVG built here in Python. Two series (mental, physical) using
validated categorical slots 1 and 2, legend plus direct labels so identity is
never colour-alone, one shared y-axis, and a table view underneath for anyone
who would rather read the numbers.
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..safety import ClinicalGuard

# --------------------------------------------------------------- design tokens
CSS = """
:root {
  color-scheme: light;
  --plane:#f9f9f7; --surface:#fcfcfb; --surface-2:#f2f1ed;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:#e1e0d9;
  --mental:#2a78d6; --physical:#eb6834;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --radius:14px;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19; --surface-2:#232321;
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:#2c2c2a;
    --mental:#3987e5; --physical:#d95926;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --plane:#0d0d0d; --surface:#1a1a19; --surface-2:#232321;
  --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:#2c2c2a;
  --mental:#3987e5; --physical:#d95926;
}
* { box-sizing: border-box; }
body {
  margin:0; background:var(--plane); color:var(--ink);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,
       "Noto Sans",sans-serif;
  -webkit-font-smoothing:antialiased;
}
.wrap { max-width:1000px; margin:0 auto; padding:32px 16px 72px; }
header { margin-bottom:28px; }
h1 { font-size:1.45rem; margin:0 0 4px; letter-spacing:-0.01em; }
.sub { color:var(--muted); font-size:0.86rem; }
h2 { font-size:0.78rem; text-transform:uppercase; letter-spacing:0.09em;
     color:var(--muted); margin:36px 0 12px; font-weight:600; }
.card { background:var(--surface); border:1px solid var(--border);
        border-radius:var(--radius); padding:20px 22px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
         gap:12px; }
.tile { background:var(--surface); border:1px solid var(--border);
        border-radius:var(--radius); padding:16px 18px; }
.tile .k { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.07em;
           color:var(--muted); }
.tile .v { font-size:2rem; font-weight:650; letter-spacing:-0.02em;
           margin-top:4px; font-variant-numeric:tabular-nums; }
.tile .d { font-size:0.8rem; color:var(--ink-2); margin-top:2px;
           font-variant-numeric:tabular-nums; }
.badge { display:inline-flex; align-items:center; gap:5px; font-size:0.74rem;
         font-weight:600; padding:3px 9px; border-radius:999px;
         border:1px solid currentColor; }
.narr { font-size:1.02rem; }
.narr .head { font-size:1.22rem; font-weight:620; letter-spacing:-0.01em;
              margin-bottom:10px; line-height:1.35; }
.narr p { margin:0 0 12px; color:var(--ink-2); }
.narr p.lead { color:var(--ink); }
.rec { border:1px solid var(--border); border-radius:var(--radius);
       background:var(--surface); padding:16px 18px; margin-bottom:10px; }
.rec .t { font-weight:620; margin-bottom:5px; display:flex; gap:8px;
          align-items:baseline; flex-wrap:wrap; }
.rec .a { color:var(--ink-2); margin-bottom:7px; }
.rec .w { color:var(--muted); font-size:0.87rem; border-left:2px solid var(--grid);
          padding-left:10px; }
.tag { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.06em;
       color:var(--muted); font-weight:600; border:1px solid var(--border);
       border-radius:5px; padding:1px 6px; }
.alert { border-radius:var(--radius); padding:14px 16px; margin-bottom:10px;
         border:1px solid var(--border); background:var(--surface);
         border-left-width:3px; }
.alert .t { font-weight:620; display:flex; gap:7px; align-items:center; }
.alert .b { color:var(--ink-2); font-size:0.93rem; margin-top:4px;
            white-space:pre-wrap; }
.alert.red { border-left-color:var(--critical); }
.alert.support { border-left-color:var(--critical); background:var(--surface-2); }
.alert.amber { border-left-color:var(--warning); }
.alert.info { border-left-color:var(--muted); }
.alert.praise { border-left-color:var(--good); }
.legend { display:flex; gap:16px; font-size:0.82rem; color:var(--ink-2);
          margin:2px 0 8px; flex-wrap:wrap; }
.legend i { width:11px; height:11px; border-radius:3px; display:inline-block;
            margin-right:6px; vertical-align:-1px; }
.bars { display:grid; gap:7px; }
.bar { display:grid; grid-template-columns:118px 1fr 42px; align-items:center;
       gap:10px; font-size:0.86rem; }
.bar .lbl { color:var(--ink-2); }
.bar .track { height:9px; background:var(--surface-2); border-radius:999px;
              overflow:hidden; display:block; }
.bar .fill { height:100%; border-radius:999px; display:block; }
.bar .num { text-align:right; font-variant-numeric:tabular-nums;
            color:var(--ink-2); }
table { width:100%; border-collapse:collapse; font-size:0.85rem;
        font-variant-numeric:tabular-nums; }
th,td { text-align:right; padding:6px 8px; border-bottom:1px solid var(--grid); }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:600; font-size:0.74rem;
     text-transform:uppercase; letter-spacing:0.06em; }
details { margin-top:14px; }
summary { cursor:pointer; color:var(--muted); font-size:0.84rem;
          padding:6px 0; }
.foot { margin-top:44px; padding-top:18px; border-top:1px solid var(--border);
        color:var(--muted); font-size:0.79rem; }
.chart-wrap { position:relative; }
.tt { position:absolute; pointer-events:none; opacity:0; transition:opacity .1s;
      background:var(--surface); border:1px solid var(--border);
      border-radius:9px; padding:8px 11px; font-size:0.8rem;
      box-shadow:0 4px 14px rgba(0,0,0,.14); white-space:nowrap; z-index:5;
      font-variant-numeric:tabular-nums; }
.tt b { font-weight:620; }
.tt i { width:9px; height:9px; border-radius:2px; display:inline-block;
        margin-right:5px; }
svg { display:block; width:100%; height:auto; }
@media (max-width:560px) {
  .bar { grid-template-columns:96px 1fr 38px; }
  .narr .head { font-size:1.08rem; }
}
"""

TOOLTIP_JS = """
(function(){
  document.querySelectorAll('[data-chart]').forEach(function(wrap){
    var svg=wrap.querySelector('svg'), tt=wrap.querySelector('.tt');
    if(!svg||!tt) return;
    var pts=JSON.parse(svg.getAttribute('data-points')||'[]');
    var cross=svg.querySelector('.crosshair');
    var dots=svg.querySelectorAll('.hoverdot');
    if(!pts.length) return;
    function hide(){ tt.style.opacity=0; if(cross) cross.setAttribute('opacity',0);
      dots.forEach(function(d){ d.setAttribute('opacity',0); }); }
    svg.addEventListener('mousemove', function(e){
      var r=svg.getBoundingClientRect();
      var x=(e.clientX-r.left)/r.width*svg.viewBox.baseVal.width;
      var best=pts[0], bd=1e9;
      pts.forEach(function(p){ var d=Math.abs(p.x-x); if(d<bd){bd=d;best=p;} });
      if(cross){ cross.setAttribute('x1',best.x); cross.setAttribute('x2',best.x);
                 cross.setAttribute('opacity',1); }
      dots.forEach(function(d,i){
        var key=d.getAttribute('data-series');
        if(best[key]==null){ d.setAttribute('opacity',0); return; }
        d.setAttribute('cx',best.x); d.setAttribute('cy',best[key+'_y']);
        d.setAttribute('opacity',1);
      });
      var rows='<b>'+best.label+'</b>';
      if(best.mental!=null) rows+='<br><i style="background:var(--mental)"></i>Mental '+best.mental;
      if(best.physical!=null) rows+='<br><i style="background:var(--physical)"></i>Physical '+best.physical;
      tt.innerHTML=rows; tt.style.opacity=1;
      var px=(best.x/svg.viewBox.baseVal.width)*r.width;
      tt.style.left=Math.min(Math.max(px-tt.offsetWidth/2,0),r.width-tt.offsetWidth)+'px';
      tt.style.top='4px';
    });
    svg.addEventListener('mouseleave', hide);
  });
})();
"""


# ------------------------------------------------------------------ helpers
def esc(s: Any) -> str:
    return html.escape(str(s if s is not None else ""))


def band_of(v: Optional[float]) -> Tuple[str, str, str]:
    """(status role, label, icon) -- colour never carries meaning alone."""
    if v is None or pd.isna(v):
        return "muted", "no data", "·"
    if v >= 70:
        return "good", "good", "●"
    if v >= 58:
        return "good", "steady", "●"
    if v >= 45:
        return "warning", "below your usual", "▲"
    if v >= 38:
        return "serious", "low", "▲"
    return "critical", "very low", "■"


def badge(v: Optional[float]) -> str:
    role, label, icon = band_of(v)
    color = f"var(--{role})" if role != "muted" else "var(--muted)"
    return (f'<span class="badge" style="color:{color}">{icon} {esc(label)}'
            f'</span>')


def delta_str(v: Optional[float]) -> str:
    if v is None or pd.isna(v):
        return ""
    if abs(v) < 0.5:
        return "level vs last week"
    arrow = "▲" if v > 0 else "▼"
    return f"{arrow} {abs(v):.1f} vs last week"


# ------------------------------------------------------------------- charts
def line_chart(series: Dict[str, pd.Series], height: int = 230,
               width: int = 920, days: int = 60) -> str:
    """Two-series line chart, one y-axis, crosshair tooltip, direct labels."""
    keys = [k for k in ("mental", "physical") if k in series]
    if not keys:
        return ""
    frames = {k: series[k].dropna().tail(days) for k in keys}
    idx = sorted(set().union(*[set(f.index) for f in frames.values()]))
    if len(idx) < 2:
        return ""
    pad_l, pad_r, pad_t, pad_b = 38, 62, 16, 26
    W, H = width, height
    iw, ih = W - pad_l - pad_r, H - pad_t - pad_b

    ymin = max(0, min(f.min() for f in frames.values()) - 8)
    ymax = min(100, max(f.max() for f in frames.values()) + 8)
    if ymax - ymin < 20:
        ymin, ymax = max(0, ymin - 10), min(100, ymax + 10)

    def X(i: int) -> float:
        return pad_l + iw * i / max(1, len(idx) - 1)

    def Y(v: float) -> float:
        return pad_t + ih * (1 - (v - ymin) / max(1e-9, ymax - ymin))

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Wellbeing scores over the last {len(idx)} days" '
             f'data-points="{{POINTS}}">']

    # Gridlines, recessive.
    step = 20 if (ymax - ymin) > 45 else 10
    tick = int(np.ceil(ymin / step) * step)
    while tick <= ymax:
        y = Y(tick)
        parts.append(f'<line x1="{pad_l}" x2="{W - pad_r}" y1="{y:.1f}" '
                     f'y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="var(--muted)">{tick}</text>')
        tick += step

    points: List[Dict[str, Any]] = []
    for i, d in enumerate(idx):
        p: Dict[str, Any] = {"x": round(X(i), 1),
                             "label": d.strftime("%a %d %b")}
        for k in keys:
            v = frames[k].get(d)
            if v is not None and pd.notna(v):
                p[k] = round(float(v), 1)
                p[f"{k}_y"] = round(Y(float(v)), 1)
            else:
                p[k] = None
        points.append(p)

    ends: List[Tuple[str, float, float]] = []
    for k in keys:
        pts = [(X(i), Y(float(frames[k][d])))
               for i, d in enumerate(idx)
               if d in frames[k].index and pd.notna(frames[k][d])]
        if len(pts) < 2:
            continue
        path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        parts.append(f'<path d="{path}" fill="none" stroke="var(--{k})" '
                     f'stroke-width="2" stroke-linejoin="round" '
                     f'stroke-linecap="round"/>')
        lx, ly = pts[-1]
        parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.5" '
                     f'fill="var(--{k})" stroke="var(--surface)" '
                     f'stroke-width="2"/>')
        ends.append((k, lx, ly))

    # Direct labels at the line ends -- identity never rests on colour alone.
    # When the two series finish close together the labels would collide, so
    # nudge them apart around their midpoint.
    ends.sort(key=lambda e: e[2])
    if len(ends) == 2 and abs(ends[1][2] - ends[0][2]) < 15:
        mid = (ends[0][2] + ends[1][2]) / 2
        ends = [(ends[0][0], ends[0][1], mid - 7.5),
                (ends[1][0], ends[1][1], mid + 7.5)]
    for k, lx, ly in ends:
        ly = min(max(ly, pad_t + 6), H - pad_b - 2)
        parts.append(f'<text x="{lx + 9:.1f}" y="{ly + 4:.1f}" font-size="11.5" '
                     f'font-weight="600" fill="var(--{k})">'
                     f'{k.capitalize()}</text>')

    # Date axis: first, middle, last.
    for i in (0, len(idx) // 2, len(idx) - 1):
        anchor = "start" if i == 0 else ("end" if i == len(idx) - 1 else "middle")
        parts.append(f'<text x="{X(i):.1f}" y="{H - 7}" text-anchor="{anchor}" '
                     f'font-size="11" fill="var(--muted)">'
                     f'{idx[i].strftime("%d %b")}</text>')

    parts.append(f'<line class="crosshair" x1="0" x2="0" y1="{pad_t}" '
                 f'y2="{pad_t + ih}" stroke="var(--axis)" stroke-width="1" '
                 f'stroke-dasharray="3 3" opacity="0"/>')
    for k in keys:
        parts.append(f'<circle class="hoverdot" data-series="{k}" r="4.5" '
                     f'cx="0" cy="0" fill="var(--{k})" stroke="var(--surface)" '
                     f'stroke-width="2" opacity="0"/>')
    parts.append("</svg>")

    svg = "".join(parts).replace(
        "{POINTS}", esc(json.dumps(points, separators=(",", ":"))))
    legend = ('<div class="legend">'
              + "".join(f'<span><i style="background:var(--{k})"></i>'
                        f'{k.capitalize()} wellbeing</span>' for k in keys)
              + "</div>")
    return (f'{legend}<div class="chart-wrap" data-chart>{svg}'
            f'<div class="tt"></div></div>')


def subscore_bars(scores: pd.Series, cfg: Config) -> str:
    """Horizontal bars: magnitude comparison across sub-scores."""
    groups = [("Physical", list(cfg.weights.physical), "physical"),
              ("Mental", list(cfg.weights.mental), "mental")]
    out = []
    for title, keys, color in groups:
        rows = []
        for k in keys:
            v = scores.get(f"sub_{k}")
            if v is None or pd.isna(v):
                continue
            role, label, icon = band_of(float(v))
            rows.append(
                f'<div class="bar"><span class="lbl">'
                f'{esc(k.replace("_", " ").title())}</span>'
                f'<span class="track"><span class="fill" '
                f'style="width:{max(2, min(100, float(v))):.0f}%;'
                f'background:var(--{color})"></span></span>'
                f'<span class="num">{float(v):.0f}</span></div>')
        if rows:
            out.append(f'<h2>{title} breakdown</h2>'
                       f'<div class="card"><div class="bars">'
                       + "".join(rows) + "</div></div>")
    return "".join(out)


def score_table(scores: pd.DataFrame, days: int = 14) -> str:
    w = scores.tail(days)
    rows = []
    for idx, r in w[::-1].iterrows():
        arch = r.get("archetype")
        rows.append(
            f"<tr><td>{idx.strftime('%a %d %b')}</td>"
            f"<td>{r['mental']:.0f}</td><td>{r['physical']:.0f}</td>"
            f"<td>{r['overall']:.0f}</td>"
            f"<td>{esc(arch) if isinstance(arch, str) else '—'}</td></tr>")
    return (f'<details><summary>Table view -- last {days} days</summary>'
            f'<table><thead><tr><th>Day</th><th>Mental</th><th>Physical</th>'
            f'<th>Overall</th><th>Day type</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></details>")


# ------------------------------------------------------------------ sections
def tiles_html(rep, hist: Dict[str, Any]) -> str:
    s = rep.scores
    seven = hist.get("last_7d", {})
    thirty = hist.get("last_30d", {})
    items = [
        ("Mental", s.get("mental"), delta_str(seven.get("mental_change"))),
        ("Physical", s.get("physical"), delta_str(seven.get("physical_change"))),
        ("Overall", s.get("overall"),
         f"30-day average {thirty.get('mental', '—')}"),
    ]
    out = []
    for k, v, d in items:
        val = f"{v:.0f}" if isinstance(v, (int, float)) and pd.notna(v) else "—"
        out.append(f'<div class="tile"><div class="k">{esc(k)}</div>'
                   f'<div class="v">{val}</div>'
                   f'<div class="d">{badge(v)} {esc(d)}</div></div>')
    risk = s.get("risk")
    if risk is not None and pd.notna(risk):
        pct = float(risk) * 100
        role = "critical" if pct >= 65 else "warning" if pct >= 40 else "good"
        icon = "▲" if pct >= 40 else "●"
        out.append(f'<div class="tile"><div class="k">Next 3 days</div>'
                   f'<div class="v">{pct:.0f}%</div>'
                   f'<div class="d"><span class="badge" '
                   f'style="color:var(--{role})">{icon} dip risk</span></div>'
                   f'</div>')
    return f'<div class="tiles">{"".join(out)}</div>'


def narrative_html(coach) -> str:
    parts = [f'<div class="head">{esc(coach.headline)}</div>',
             f'<p class="lead">{esc(coach.assessment)}</p>']
    for text in (coach.whats_working, coach.watch, coach.encouragement):
        if text:
            parts.append(f"<p>{esc(text)}</p>")
    return f'<div class="card narr">{"".join(parts)}</div>'


def recs_html(recs, notes: Dict[str, str]) -> str:
    if not recs:
        return ""
    out = []
    for r in recs:
        note = notes.get(r.id)
        out.append(
            f'<div class="rec"><div class="t">{esc(r.title)}'
            f'<span class="tag">{esc(r.category.replace("_", " "))}</span>'
            f'<span class="tag">{esc(r.effort)} effort</span>'
            f'<span class="tag">{esc(r.when)}</span></div>'
            f'<div class="a">{esc(r.action)}</div>'
            f'<div class="w">{esc(note or r.why)}</div></div>')
    return f'<h2>What would help</h2>{"".join(out)}'


def alerts_html(alerts) -> str:
    if not alerts:
        return ""
    icons = {"red": "■", "amber": "▲", "info": "●", "praise": "★",
             "support": "■"}
    out = []
    for a in alerts:
        out.append(f'<div class="alert {esc(a.level)}">'
                   f'<div class="t">{icons.get(a.level, "●")} {esc(a.title)}'
                   f'</div><div class="b">{esc(a.body)}</div></div>')
    return f'<h2>Alerts</h2>{"".join(out)}'


def history_html(hist: Dict[str, Any]) -> str:
    if not hist:
        return ""
    rows = []
    for key, label in [("last_7d", "Last 7 days"), ("last_30d", "Last 30 days"),
                       ("last_90d", "Last 90 days")]:
        w = hist.get(key)
        if not w:
            continue
        mc, pc = w.get("mental_change"), w.get("physical_change")
        rows.append(
            f"<tr><td>{label}</td><td>{w.get('mental', '—')}</td>"
            f"<td>{_ch(mc)}</td><td>{w.get('physical', '—')}</td>"
            f"<td>{_ch(pc)}</td></tr>")
    months = hist.get("monthly", [])[-6:]
    for m in months:
        rows.append(f"<tr><td>{esc(m['month'])}</td><td>{m['mental']}</td>"
                    f"<td>—</td><td>{m['physical']}</td><td>—</td></tr>")
    if not rows:
        return ""
    return (f'<h2>Progress</h2><div class="card"><table><thead><tr>'
            f'<th>Period</th><th>Mental</th><th>Δ</th><th>Physical</th>'
            f'<th>Δ</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
            f"</div>")


def _ch(v) -> str:
    if v is None:
        return "—"
    role = "good" if v >= 3 else "critical" if v <= -3 else "muted"
    icon = "▲" if v > 0 else ("▼" if v < 0 else "–")
    return f'<span style="color:var(--{role})">{icon} {abs(v):.1f}</span>'


def patterns_html(models) -> str:
    if not getattr(models, "drivers", None):
        return ""
    rows = []
    for d in models.drivers[:8]:
        rel = d["relationship"]
        role = ("good" if rel == "protective"
                else "warning" if rel == "risk" else "muted")
        icon = "●" if rel == "protective" else "▲" if rel == "risk" else "·"
        rows.append(f"<tr><td>{esc(d['label'])}</td>"
                    f'<td><span style="color:var(--{role})">{icon} {esc(rel)}'
                    f"</span></td><td>{d['importance']:.3f}</td></tr>")
    return (f'<h2>What your own data predicts</h2><div class="card">'
            f'<p style="color:var(--ink-2);margin:0 0 12px;font-size:0.92rem">'
            f"These are the behaviours that best predict where your mood goes "
            f"over the next few days -- learned from your history, not from "
            f"general advice.</p>"
            f'<table><thead><tr><th>Behaviour</th><th>Relationship</th>'
            f"<th>Weight</th></tr></thead><tbody>{''.join(rows)}</tbody>"
            f"</table></div>")


# ------------------------------------------------------------------- shells
def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style></head>
<body><div class="wrap">{body}
<div class="foot">{esc(ClinicalGuard.disclaimer())}<br>
Generated {datetime.now():%d %b %Y, %H:%M} · all processing local</div>
</div><script>{TOOLTIP_JS}</script></body></html>"""


def render_report(cfg: Config, rep, analysis) -> str:
    period_label = {"daily": "Daily check-in", "weekly": "Weekly review",
                    "monthly": "Monthly review"}.get(rep.period, rep.period)
    hist = rep.history or {}
    notes = rep.coach.recommendation_notes or {}
    chart_days = {"daily": 45, "weekly": 90, "monthly": 180}.get(rep.period, 60)

    body = f"""
<header>
  <h1>{esc(period_label)} · {esc(cfg.profile.name)}</h1>
  <div class="sub">{esc(rep.date)} · scores are relative to your own baseline,
  not to anyone else</div>
</header>
{tiles_html(rep, hist)}
<h2>What the data says</h2>
{narrative_html(rep.coach)}
{alerts_html(rep.alerts)}
{recs_html(rep.recommendations, notes)}
<h2>Trend</h2>
<div class="card">{line_chart({"mental": analysis.scores["mental"],
                               "physical": analysis.scores["physical"]},
                              days=chart_days)}
{score_table(analysis.scores)}</div>
{subscore_bars(pd.Series(rep.scores), cfg)}
{history_html(hist)}
{patterns_html(analysis.models)}
"""
    return _page(f"{period_label} — {rep.date}", body)


def render_dashboard(cfg: Config, analysis, rep=None) -> str:
    s = analysis.scores
    latest = s.iloc[-1]
    hist = rep.history if rep else {}
    counts = ""
    if "archetype" in s:
        vc = s["archetype"].tail(30).value_counts()
        if len(vc):
            items = "".join(
                f'<div class="bar"><span class="lbl">{esc(k)}</span>'
                f'<span class="track"><span class="fill" '
                f'style="width:{v / max(vc) * 100:.0f}%;'
                f'background:var(--mental)"></span></span>'
                f'<span class="num">{v}</span></div>' for k, v in vc.items())
            counts = (f'<h2>Your last 30 days, by day type</h2>'
                      f'<div class="card"><div class="bars">{items}</div></div>')

    body = f"""
<header>
  <h1>Wellbeing dashboard · {esc(cfg.profile.name)}</h1>
  <div class="sub">{len(s)} days tracked ·
  {s.index[0]:%d %b %Y} to {s.index[-1]:%d %b %Y}</div>
</header>
{tiles_html(rep, hist) if rep else ""}
<h2>Six-month trend</h2>
<div class="card">{line_chart({"mental": s["mental"],
                               "physical": s["physical"]}, days=180)}
{score_table(s, days=21)}</div>
{subscore_bars(latest, cfg)}
{counts}
{history_html(hist)}
{patterns_html(analysis.models)}
"""
    return _page(f"Wellbeing dashboard — {cfg.profile.name}", body)
