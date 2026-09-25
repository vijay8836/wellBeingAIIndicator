"""Terminal rendering -- the same report, for people who live in a shell."""
from __future__ import annotations

import shutil
import textwrap
from typing import Any, Dict, List, Optional

import pandas as pd

BLOCKS = " ▁▂▃▄▅▆▇█"


def width() -> int:
    return min(96, max(60, shutil.get_terminal_size((88, 24)).columns))


def rule(char: str = "─") -> str:
    return char * width()


def wrap(text: str, indent: str = "") -> str:
    return textwrap.fill(text, width=width() - len(indent),
                         initial_indent=indent, subsequent_indent=indent)


def sparkline(series: pd.Series, n: int = 48) -> str:
    s = series.dropna().tail(n)
    if len(s) < 2:
        return ""
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return BLOCKS[4] * len(s)
    idx = ((s - lo) / (hi - lo) * (len(BLOCKS) - 1)).round().astype(int)
    return "".join(BLOCKS[i] for i in idx)


def bar(value: float, width_chars: int = 26) -> str:
    filled = int(round(max(0, min(100, value)) / 100 * width_chars))
    return "█" * filled + "░" * (width_chars - filled)


def band(v: Optional[float]) -> str:
    if v is None or pd.isna(v):
        return "no data"
    if v >= 70:
        return "good"
    if v >= 58:
        return "steady"
    if v >= 45:
        return "below your usual"
    if v >= 38:
        return "low"
    return "very low"


def render(rep, analysis, cfg) -> str:
    L: List[str] = []
    s = rep.scores
    title = {"daily": "DAILY CHECK-IN", "weekly": "WEEKLY REVIEW",
             "monthly": "MONTHLY REVIEW"}.get(rep.period, rep.period.upper())
    L += [rule("═"), f"  {title} · {cfg.profile.name} · {rep.date}", rule("═"), ""]

    for key, label in (("mental", "Mental  "), ("physical", "Physical")):
        v = s.get(key)
        if v is None:
            continue
        L.append(f"  {label}  {bar(v)}  {v:5.1f}   {band(v)}")
    if "risk" in s and s["risk"] is not None:
        L.append(f"  Dip risk (next 3 days): {s['risk'] * 100:.0f}%")
    if "archetype" in s:
        L.append(f"  Day type: {s['archetype']}")
    L.append("")

    spark = sparkline(analysis.scores["mental"])
    if spark:
        L += [f"  Mental, last {min(48, len(analysis.scores))} days:",
              f"  {spark}", ""]

    L += [rule(), f"  {rep.coach.headline}", rule(), ""]
    L.append(wrap(rep.coach.assessment, "  "))
    for extra in (rep.coach.whats_working, rep.coach.watch,
                  rep.coach.encouragement):
        if extra:
            L += ["", wrap(extra, "  ")]
    L.append("")

    if rep.alerts:
        icons = {"red": "[!]", "amber": "[~]", "info": "[i]", "praise": "[+]",
                 "support": "[!]"}
        L += [rule(), "  ALERTS", rule()]
        for a in rep.alerts:
            L.append(f"  {icons.get(a.level, '[.]')} {a.title}")
            L.append(wrap(a.body, "      "))
            L.append("")

    if rep.recommendations:
        L += [rule(), "  WHAT WOULD HELP", rule(), ""]
        for i, r in enumerate(rep.recommendations, 1):
            note = rep.coach.recommendation_notes.get(r.id) or r.why
            L.append(f"  {i}. {r.title}  "
                     f"[{r.category.replace('_', ' ')} · {r.effort} · {r.when}]")
            L.append(wrap(r.action, "     "))
            L.append(wrap(f"why: {note}", "     "))
            L.append("")

    hist = rep.history or {}
    if hist:
        L += [rule(), "  PROGRESS", rule()]
        for key, label in (("last_7d", "Last 7 days "),
                           ("last_30d", "Last 30 days"),
                           ("last_90d", "Last 90 days")):
            w = hist.get(key)
            if not w:
                continue
            mc = w.get("mental_change")
            arrow = ("▲" if mc and mc > 0 else "▼" if mc and mc < 0 else "–")
            chg = f"{arrow} {abs(mc):.1f}" if mc is not None else "  —"
            L.append(f"  {label}   mental {w.get('mental', '—'):>5}  {chg:>7}"
                     f"   physical {w.get('physical', '—'):>5}")
        if hist.get("direction"):
            L.append(f"\n  Direction: {hist['direction']}")
        L.append("")

    if analysis.models.drivers:
        L += [rule(), "  WHAT YOUR OWN DATA PREDICTS", rule()]
        for d in analysis.models.drivers[:6]:
            mark = "+" if d["relationship"] == "protective" else "-"
            L.append(f"   {mark} {d['label']:<38} {d['relationship']}")
        L.append("")

    from ..safety import ClinicalGuard
    L += [rule("═"), wrap(ClinicalGuard.disclaimer(), "  "), rule("═")]
    return "\n".join(L)
