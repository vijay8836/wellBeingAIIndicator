"""
Composite wellbeing scores.

Each score is built from declared `Signal` specs rather than a black box, so
that every point lost is attributable to a named behaviour. That matters: an
alert that says "your mood score fell 14" is useless; "you slept 5h10m for
four nights and your bedtime moved 2h later" is actionable.

Scores are 0-100 where 50 is "your own typical day". They are *relative to the
person*, not to a population.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config


# --------------------------------------------------------------- transforms
def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def higher_better(v: float, low: float, high: float) -> float:
    """low -> 0, high -> 1, linear between."""
    if high == low:
        return 0.5
    return _clip01((v - low) / (high - low))


def lower_better(v: float, low: float, high: float) -> float:
    """low -> 1, high -> 0."""
    return 1.0 - higher_better(v, low, high)


def band(v: float, lo_bad: float, lo_ok: float, hi_ok: float,
         hi_bad: float) -> float:
    """Inverted-U: full marks inside [lo_ok, hi_ok], tapering to 0 outside."""
    if lo_ok <= v <= hi_ok:
        return 1.0
    if v < lo_ok:
        return _clip01((v - lo_bad) / max(1e-9, lo_ok - lo_bad))
    return _clip01((hi_bad - v) / max(1e-9, hi_bad - hi_ok))


def z_better(z: float, direction: str = "up", span: float = 2.0) -> float:
    """Map a personal z-score to 0-1. 0 z -> 0.5."""
    if pd.isna(z):
        return np.nan
    zz = z if direction == "up" else -z
    return _clip01(0.5 + zz / (2 * span))


@dataclass
class Signal:
    key: str                     # feature column
    label: str                   # human phrasing
    kind: str                    # higher | lower | band | z_up | z_down
    params: Tuple[float, ...] = ()
    weight: float = 1.0
    good_phrase: str = ""
    bad_phrase: str = ""

    def evaluate(self, row: pd.Series) -> Optional[float]:
        if self.key not in row.index:
            return None
        v = row[self.key]
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        v = float(v)
        try:
            if self.kind == "higher":
                return higher_better(v, *self.params)
            if self.kind == "lower":
                return lower_better(v, *self.params)
            if self.kind == "band":
                return band(v, *self.params)
            if self.kind == "z_up":
                return z_better(v, "up")
            if self.kind == "z_down":
                return z_better(v, "down")
        except TypeError:
            return None
        return None


@dataclass
class SubScore:
    name: str
    value: float                 # 0-100
    confidence: float            # 0-1, share of signals with data
    parts: Dict[str, float] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)


@dataclass
class DayScore:
    date: str
    physical: float
    mental: float
    overall: float
    subscores: Dict[str, SubScore]
    drivers: List[Dict[str, Any]]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "physical": round(self.physical, 1),
            "mental": round(self.mental, 1),
            "overall": round(self.overall, 1),
            "confidence": round(self.confidence, 2),
            "subscores": {k: {"value": round(v.value, 1),
                              "confidence": round(v.confidence, 2),
                              "parts": {p: round(s, 3)
                                        for p, s in v.parts.items()},
                              "missing": v.missing}
                          for k, v in self.subscores.items()},
            "drivers": self.drivers,
        }


class Scorer:
    """Turns a feature row into physical/mental scores plus named drivers."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        p = cfg.profile
        ts = p.target_sleep_hours

        self.signals: Dict[str, List[Signal]] = {
            # ------------------------------------------------ physical
            "sleep": [
                Signal("sleep_hours", "sleep duration", "band",
                       (ts - 3.5, ts - 0.75, ts + 1.0, ts + 3.0), 2.0,
                       "you're hitting your sleep target",
                       "you're running short on sleep"),
                Signal("sleep_efficiency", "sleep efficiency", "higher",
                       (70, 92), 1.0, "sleep quality is solid",
                       "sleep is broken even when it's long enough"),
                Signal("sleep_latency", "time to fall asleep", "lower",
                       (10, 50), 0.8, "", "you're lying awake before sleeping"),
                Signal("awakenings", "night wakings", "lower", (0, 6), 0.6,
                       "", "you're waking repeatedly in the night"),
                Signal("sleep_debt_7d", "7-day sleep debt", "lower", (0, 10),
                       1.2, "", "sleep debt has built up over the week"),
                Signal("deep_pct", "deep sleep share", "higher", (8, 22), 0.5),
            ],
            "activity": [
                Signal("steps", "daily steps", "higher",
                       (1500, p.target_steps), 2.0,
                       "you're moving well", "you've barely moved"),
                Signal("active_min", "active minutes", "higher",
                       (5, p.target_active_min), 1.5, "", "very little activity"),
                Signal("workout_days_7d", "workouts this week", "higher",
                       (0, 4), 1.2, "you're training consistently",
                       "no real workouts this week"),
                Signal("days_since_workout", "days since a workout", "lower",
                       (1, 9), 0.8, "", "it's been a while since you trained"),
                Signal("outdoor_min", "time outdoors", "higher", (5, 60), 1.0,
                       "you're getting outside", "almost no time outdoors"),
            ],
            "nutrition": [
                Signal("protein_g", "protein", "higher", (25, 90), 1.0),
                Signal("fiber_g", "fibre", "higher", (8, 30), 0.8),
                Signal("sugar_g", "added sugar", "lower", (20, 90), 1.0,
                       "", "sugar intake is high"),
                Signal("water_ml", "hydration", "higher",
                       (700, p.target_water_ml), 1.0, "", "you're under-hydrated"),
                Signal("ultraprocessed_ratio", "ultra-processed share", "lower",
                       (0.1, 0.7), 1.2, "", "a lot of packaged/processed food"),
                Signal("alcohol_units", "alcohol", "lower", (0, 4), 1.0,
                       "", "alcohol intake is up"),
                Signal("caffeine_mg", "caffeine", "band",
                       (0, 0, 300, 600), 0.8, "", "caffeine is high"),
                Signal("meal_count", "regular meals", "band", (0, 3, 5, 8), 0.6),
            ],
            "recovery": [
                Signal("hrv_z", "heart-rate variability vs your baseline",
                       "z_up", (), 2.0, "recovery markers look good",
                       "HRV is below your normal -- your body is under load"),
                Signal("rhr_z", "resting heart rate vs your baseline", "z_down",
                       (), 1.5, "", "resting heart rate is elevated for you"),
                Signal("sleep_debt_7d", "accumulated sleep debt", "lower",
                       (0, 12), 1.0),
                Signal("spo2", "blood oxygen", "higher", (92, 97), 0.4),
            ],
            "medical_load": [
                Signal("medical_pressure_7d", "health-seeking this week",
                       "lower", (0, 6), 2.0, "no health flags this week",
                       "a lot of health searches, clinic or pharmacy activity"),
                Signal("search_physical_health", "symptom searches", "lower",
                       (0, 5), 1.0, "", "you've been searching symptoms"),
            ],
            # -------------------------------------------------- mental
            "mood": [
                Signal("msg_sentiment", "tone of your messages", "higher",
                       (-0.45, 0.30), 2.0, "your messages sound upbeat",
                       "your messages have turned negative in tone"),
                Signal("note_sentiment", "tone of your notes", "higher",
                       (-0.5, 0.3), 1.0, "", "your own notes sound low"),
                Signal("social_sentiment", "tone of what you post/like",
                       "higher", (-0.4, 0.35), 0.6),
                Signal("self_mood", "your own mood rating", "higher", (1, 5),
                       2.5, "you rated your mood well",
                       "you rated your own mood low"),
                Signal("media_comfort_min", "comfort-watching", "lower",
                       (10, 120), 0.6, "",
                       "a lot of comfort/soothing content"),
            ],
            "stress": [
                Signal("work_overload", "workload", "lower", (0.15, 1.2), 2.0,
                       "workload looks sustainable",
                       "your workload is running hot"),
                Signal("distress_index", "stress language in your text",
                       "lower", (-0.1, 0.55), 2.0, "",
                       "stress language has risen across your messages and searches"),
                Signal("self_stress", "your own stress rating", "lower",
                       (1, 5), 2.0, "", "you rated your stress high"),
                Signal("night_digital_load", "late-night screen use", "lower",
                       (0.15, 1.1), 1.2, "",
                       "you're on your phone late into the night"),
                Signal("spend_night", "late-night spending", "lower", (0, 2500),
                       0.5, "", "some late-night impulse spending"),
                Signal("context_switches", "task switching", "lower", (40, 250),
                       0.6),
            ],
            "social": [
                Signal("social_connection", "contact with people", "higher",
                       (0.1, 0.8), 2.5, "you're staying connected",
                       "you've pulled back from people"),
                Signal("unique_contacts", "people you talked to", "higher",
                       (1, 7), 1.5, "", "you spoke to very few people"),
                Signal("days_since_social", "days since real contact", "lower",
                       (1, 6), 1.5, "", "it's been days since real contact"),
                Signal("reply_ratio", "two-way conversation", "band",
                       (0, 0.6, 2.0, 5.0), 0.8),
                Signal("visit_social", "going out", "higher", (0, 1), 0.8),
            ],
            "rumination": [
                Signal("note_rumination", "absolutist language in notes",
                       "lower", (0.05, 0.4), 1.5, "",
                       "your notes show a lot of all-or-nothing thinking"),
                Signal("night_searches", "3am searching", "lower", (0, 6), 1.5,
                       "", "you've been searching things in the middle of the night"),
                Signal("search_mental_health", "searches about how you feel",
                       "lower", (0, 3), 2.0, "",
                       "you've been looking up anxiety/low-mood topics"),
                Signal("media_doomscroll_min", "doomscrolling", "lower",
                       (15, 120), 1.2, "", "a lot of doomscrolling"),
                Signal("pickups", "phone pickups", "lower", (40, 180), 0.8),
            ],
            "circadian": [
                Signal("sleep_regularity", "consistency of your sleep timing",
                       "lower", (0.4, 2.2), 2.0, "your sleep timing is steady",
                       "your sleep timing is all over the place"),
                Signal("bedtime_drift", "bedtime drift", "lower", (0.4, 2.5),
                       1.5, "", "your bedtime keeps sliding later"),
                Signal("ambient_light", "daylight exposure", "higher",
                       (200, 3000), 1.0, "", "very little bright-light exposure"),
                Signal("days_since_outdoor", "days since being outdoors",
                       "lower", (1, 5), 1.0, "", "you haven't been outside much"),
            ],
            "work_balance": [
                Signal("after_hours_min", "after-hours work", "lower",
                       (20, 180), 2.0, "you're logging off on time",
                       "you keep working after hours"),
                Signal("weekend_work_min", "weekend work", "lower", (0, 240),
                       1.5, "", "you worked through the weekend"),
                Signal("work_span_hours", "length of your working day", "lower",
                       (8.5, 13), 1.5, "", "your working day is very long"),
                Signal("meetings_min", "meeting load", "lower", (120, 420), 1.0,
                       "", "your calendar is packed with meetings"),
                Signal("days_since_restorative",
                       "days since anything restorative", "lower", (2, 10), 1.0,
                       "", "no yoga/temple/park time in a while"),
            ],
        }

    # ------------------------------------------------------------ scoring
    def _subscore(self, name: str, row: pd.Series) -> SubScore:
        sigs = self.signals[name]
        total_w, acc = 0.0, 0.0
        parts, missing = {}, []
        for s in sigs:
            val = s.evaluate(row)
            if val is None or pd.isna(val):
                missing.append(s.key)
                continue
            parts[s.key] = float(val)
            acc += float(val) * s.weight
            total_w += s.weight
        if total_w == 0:
            return SubScore(name, 50.0, 0.0, {}, missing)
        possible = sum(s.weight for s in sigs)
        return SubScore(name, 100.0 * acc / total_w, total_w / possible,
                        parts, missing)

    def score_row(self, row: pd.Series, date: str) -> DayScore:
        subs = {name: self._subscore(name, row) for name in self.signals}

        def composite(weights: Dict[str, float]) -> Tuple[float, float]:
            num = den = conf = cw = 0.0
            for k, w in weights.items():
                s = subs[k]
                if s.confidence <= 0:
                    continue
                # Down-weight a sub-score we barely have data for.
                eff = w * (0.35 + 0.65 * s.confidence)
                num += s.value * eff
                den += eff
                conf += s.confidence * w
                cw += w
            if den == 0:
                return 50.0, 0.0
            return num / den, (conf / cw if cw else 0.0)

        phys, cphys = composite(self.cfg.weights.physical)
        ment, cment = composite(self.cfg.weights.mental)
        overall = 0.45 * phys + 0.55 * ment
        drivers = self._drivers(subs, row)
        return DayScore(date, phys, ment, overall, subs, drivers,
                        (cphys + cment) / 2)

    # ------------------------------------------------------------ drivers
    def _drivers(self, subs: Dict[str, SubScore], row: pd.Series,
                 top: int = 6) -> List[Dict[str, Any]]:
        """Which specific behaviours moved the needle, and in which direction."""
        weights = {**self.cfg.weights.physical, **self.cfg.weights.mental}
        scored: List[Dict[str, Any]] = []
        for name, sub in subs.items():
            if sub.confidence <= 0:
                continue
            domain = "physical" if name in self.cfg.weights.physical else "mental"
            for sig in self.signals[name]:
                if sig.key not in sub.parts:
                    continue
                val = sub.parts[sig.key]
                impact = (val - 0.5) * sig.weight * weights.get(name, 0.1)
                raw = row.get(sig.key)
                scored.append({
                    "domain": domain,
                    "subscore": name,
                    "feature": sig.key,
                    "label": sig.label,
                    "score": round(val, 3),
                    "impact": round(impact, 4),
                    "value": (None if raw is None or pd.isna(raw)
                              else round(float(raw), 2)),
                    "phrase": (sig.good_phrase if val >= 0.6 else sig.bad_phrase),
                    "direction": "up" if val >= 0.6 else
                                 ("down" if val <= 0.4 else "flat"),
                })
        negative = sorted([d for d in scored if d["impact"] < 0],
                          key=lambda d: d["impact"])[:top]
        positive = sorted([d for d in scored if d["impact"] > 0],
                          key=lambda d: -d["impact"])[:3]
        return negative + positive

    # -------------------------------------------------------------- series
    def score_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score every day. Returns a DataFrame with scores + subscores."""
        rows = []
        for idx, row in features.iterrows():
            d = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
            ds = self.score_row(row, d)
            rec = {"date": d, "physical": ds.physical, "mental": ds.mental,
                   "overall": ds.overall, "confidence": ds.confidence}
            for k, s in ds.subscores.items():
                rec[f"sub_{k}"] = s.value
                rec[f"conf_{k}"] = s.confidence
            rows.append(rec)
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["date"] = pd.to_datetime(out["date"])
        return out.set_index("date").sort_index()
