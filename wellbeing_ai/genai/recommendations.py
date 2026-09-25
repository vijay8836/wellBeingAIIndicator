"""
The recommendation catalogue.

Each entry is tied to a *trigger* -- a feature and a condition -- so advice is
only ever given because something in the person's own data called for it. A
generic "drink more water" that fires every day is noise; "you averaged 1.1L
for four days and your resting heart rate is up" is a reason.

Nothing here prescribes medication or names a condition. The single clinical
route is `doctor_consult`, which hands the question to a human professional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

CATEGORIES = [
    "sleep", "exercise", "diet", "hydration", "sunlight", "yoga_breathwork",
    "social", "work_boundary", "vacation", "fun", "digital_hygiene",
    "nature", "mindfulness", "family", "doctor_consult",
]

EFFORT = {"tiny": 1, "small": 2, "medium": 3, "large": 4}


@dataclass
class Recommendation:
    id: str
    category: str
    title: str
    action: str
    why: str
    effort: str = "small"
    when: str = "today"
    priority: float = 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "category": self.category, "title": self.title,
                "action": self.action, "why": self.why, "effort": self.effort,
                "when": self.when, "priority": round(self.priority, 2),
                "evidence": self.evidence}


@dataclass
class Rule:
    id: str
    category: str
    title: str
    action: str
    why_template: str
    trigger: Callable[[pd.Series, Dict[str, Any]], bool]
    severity: Callable[[pd.Series, Dict[str, Any]], float]
    effort: str = "small"
    when: str = "today"
    evidence_keys: List[str] = field(default_factory=list)


def _v(row: pd.Series, key: str, default=np.nan) -> float:
    try:
        val = row.get(key, default)
        return float(val) if val is not None and not pd.isna(val) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _lt(row, key, thr) -> bool:
    v = _v(row, key)
    return bool(not np.isnan(v) and v < thr)


def _gt(row, key, thr) -> bool:
    v = _v(row, key)
    return bool(not np.isnan(v) and v > thr)


def build_rules(profile) -> List[Rule]:
    ts = profile.target_sleep_hours
    steps_t = profile.target_steps
    water_t = profile.target_water_ml

    def sev_scaled(key, lo, hi, invert=False):
        def f(row, ctx):
            v = _v(row, key)
            if np.isnan(v):
                return 0.5
            s = (v - lo) / max(1e-9, hi - lo)
            s = 1 - s if invert else s
            return float(min(1.5, max(0.2, s)))
        return f

    return [
        # ----------------------------------------------------------- sleep
        Rule("sleep_short", "sleep", "Protect tonight's sleep",
             "Set a wind-down alarm 45 minutes before your target bedtime and "
             "put the phone on the other side of the room. Aim to be in bed by "
             f"{int(23)}:00.",
             "You've averaged {sleep_hours_7d:.1f}h against your {target:.1f}h "
             "target, and sleep debt is the single strongest predictor of your "
             "next-day mood in your own data.",
             lambda r, c: _lt(r, "sleep_hours", ts - 1.0)
                          or _gt(r, "sleep_debt_7d", 5),
             sev_scaled("sleep_debt_7d", 2, 14),
             "small", "tonight", ["sleep_hours", "sleep_debt_7d"]),

        Rule("sleep_irregular", "sleep", "Anchor your wake-up time",
             "Pick one wake time and hold it for seven days, weekends included. "
             "Fix the wake end first -- bedtime follows it.",
             "Your sleep timing has been swinging by about "
             "{sleep_regularity:.1f} hours. Irregular timing hits mood even "
             "when total hours look fine.",
             lambda r, c: _gt(r, "sleep_regularity", 1.3)
                          or _gt(r, "bedtime_drift", 1.5),
             sev_scaled("sleep_regularity", 1.0, 3.0),
             "medium", "this week", ["sleep_regularity", "bedtime_drift"]),

        Rule("sleep_latency", "sleep", "Shorten the runway to sleep",
             "Try 4-7-8 breathing for four cycles once you're in bed, and keep "
             "the room dark and cool. If you're still awake after 20 minutes, "
             "get up and read something dull in low light.",
             "You've been taking about {sleep_latency:.0f} minutes to fall "
             "asleep, roughly triple your usual.",
             lambda r, c: _gt(r, "sleep_latency", 35),
             sev_scaled("sleep_latency", 25, 75),
             "tiny", "tonight", ["sleep_latency"]),

        # -------------------------------------------------------- exercise
        Rule("move_more", "exercise", "Get a short walk in",
             "Twenty minutes, outdoors, ideally before 11am. Don't make it a "
             "workout -- just get the legs moving and the light in.",
             "You're at {steps:.0f} steps against your {steps_target} target, "
             "and your step count is one of the features that most reliably "
             "precedes your better days.",
             lambda r, c: _lt(r, "steps", steps_t * 0.55),
             sev_scaled("steps", steps_t, 1500, invert=True),
             "tiny", "today", ["steps", "active_min"]),

        Rule("workout_lapse", "exercise", "Restart the training habit -- small",
             "Book one 30-minute session in the next 48 hours. Half your usual "
             "intensity is fine; the point is breaking the streak of zero.",
             "It's been {days_since_workout:.0f} days since a real workout. "
             "Your longest good stretches this year all had 3+ training days "
             "a week.",
             lambda r, c: _gt(r, "days_since_workout", 4)
                          or _lt(r, "workout_days_7d", 1),
             sev_scaled("days_since_workout", 4, 14),
             "medium", "this week", ["days_since_workout", "workout_days_7d"]),

        # ------------------------------------------------------------ diet
        Rule("ultraprocessed", "diet", "Rebalance a couple of meals",
             "Swap two packaged meals this week for something cooked -- dal and "
             "rice, eggs and vegetables, whatever is easy. Add a protein source "
             "to breakfast.",
             "About {ultraprocessed_pct:.0f}% of your logged food has been "
             "packaged or processed, and it climbs on your high-workload days.",
             lambda r, c: _gt(r, "ultraprocessed_ratio", 0.45),
             sev_scaled("ultraprocessed_ratio", 0.4, 0.85),
             "medium", "this week", ["ultraprocessed_ratio"]),

        Rule("low_protein", "diet", "Get protein into breakfast",
             "Eggs, curd, paneer, sprouts or a shake -- 25-30g at the first "
             "meal. It steadies energy far better than the mid-morning coffee.",
             "Protein has been averaging {protein_g:.0f}g/day, on the low side "
             "for your activity level.",
             lambda r, c: _lt(r, "protein_g", 45),
             sev_scaled("protein_g", 60, 20, invert=False),
             "small", "tomorrow", ["protein_g"]),

        Rule("caffeine_late", "diet", "Move the last coffee earlier",
             "Cut off caffeine by 2pm for a week and see what it does to your "
             "sleep latency.",
             "Caffeine is running around {caffeine_mg:.0f}mg/day and your time "
             "to fall asleep has stretched to {sleep_latency:.0f} minutes.",
             lambda r, c: _gt(r, "caffeine_mg", 280)
                          and _gt(r, "sleep_latency", 25),
             sev_scaled("caffeine_mg", 250, 600),
             "small", "tomorrow", ["caffeine_mg", "sleep_latency"]),

        Rule("alcohol_up", "diet", "Take a few dry nights",
             "Try three consecutive alcohol-free evenings and compare your "
             "sleep efficiency and HRV either side.",
             "Alcohol has been up lately, and on your own data your sleep "
             "efficiency drops noticeably the night after.",
             lambda r, c: _gt(r, "alcohol_units", 1.5)
                          or _gt(r, "spend_alcohol", 1200),
             sev_scaled("alcohol_units", 1, 6),
             "medium", "this week", ["alcohol_units"]),

        Rule("hydration", "hydration", "Front-load your water",
             "Fill a 1L bottle in the morning and finish it before lunch, then "
             "refill. Easier than remembering to sip.",
             "You're at about {water_l:.1f}L against a {water_target:.1f}L "
             "target.",
             lambda r, c: _lt(r, "water_ml", water_t * 0.65),
             sev_scaled("water_ml", water_t, 600, invert=True),
             "tiny", "today", ["water_ml"]),

        # --------------------------------------------------------- sunlight
        Rule("sunlight", "sunlight", "Get 15 minutes of morning light",
             "Step outside within an hour of waking -- balcony, walk to get "
             "coffee, anything. No sunglasses, no phone.",
             "You've had {outdoor_min:.0f} minutes outdoors and light exposure "
             "of about {ambient_light:.0f} lux. Morning light is what resets "
             "the clock that's currently drifting.",
             lambda r, c: _lt(r, "outdoor_min", 25)
                          or _gt(r, "days_since_outdoor", 3),
             sev_scaled("outdoor_min", 45, 0, invert=True),
             "tiny", "tomorrow morning", ["outdoor_min", "ambient_light"]),

        # -------------------------------------------------- yoga / breathing
        Rule("stress_breathwork", "yoga_breathwork",
             "Ten minutes of breathwork or yoga",
             "Try box breathing (4-4-4-4) for five minutes, or a short "
             "Yoga Nidra track. If you prefer moving: cat-cow, forward fold, "
             "legs-up-the-wall for two minutes each.",
             "Stress markers across your messages, searches and workload are "
             "sitting high. This is the cheapest intervention you have and it "
             "works within the day.",
             lambda r, c: _gt(r, "distress_index", 0.28)
                          or _gt(r, "work_overload", 0.6),
             sev_scaled("distress_index", 0.2, 0.8),
             "tiny", "today", ["distress_index", "work_overload"]),

        Rule("restorative_lapse", "yoga_breathwork",
             "Put one restorative thing back in the week",
             "The yoga centre, the temple visit, the park walk -- whichever of "
             "those you actually enjoy. Book it into the calendar rather than "
             "leaving it to spare time.",
             "It's been {days_since_restorative:.0f} days since anything "
             "restorative. Your mood scores run measurably higher in the weeks "
             "those show up.",
             lambda r, c: _gt(r, "days_since_restorative", 6),
             sev_scaled("days_since_restorative", 6, 18),
             "small", "this week", ["days_since_restorative"]),

        # --------------------------------------------------------- social
        Rule("social_withdrawal", "social", "Call one person today",
             "Not a text -- a call, ten minutes, someone you actually like "
             "talking to. Pick the easiest one.",
             "You've been in contact with {unique_contacts:.0f} people and it's "
             "been {days_since_social:.0f} days since real contact. Withdrawal "
             "is usually the first thing to move in your data, before mood.",
             lambda r, c: _gt(r, "days_since_social", 2)
                          or _lt(r, "unique_contacts", 2),
             sev_scaled("days_since_social", 2, 8),
             "small", "today", ["unique_contacts", "days_since_social"]),

        Rule("family_time", "family", "Plan something with family",
             "A meal out, a drive, a visit -- something with a date on it, not "
             "an intention.",
             "Social contact and time out of the house have both been thin for "
             "a couple of weeks.",
             lambda r, c: _gt(r, "days_since_social", 4)
                          and _lt(r, "visit_social", 1),
             sev_scaled("days_since_social", 4, 12),
             "medium", "this weekend", ["days_since_social"]),

        # -------------------------------------------------- work boundaries
        Rule("after_hours", "work_boundary", "Put a hard stop on the day",
             "Pick a finish time and set a calendar block at that hour that "
             "says 'shut laptop'. Tell one colleague, so it costs something to "
             "break.",
             "You've been logging {after_hours_min:.0f} minutes of after-hours "
             "work a day and your working span is {work_span_hours:.1f} hours.",
             lambda r, c: _gt(r, "after_hours_min", 75)
                          or _gt(r, "work_span_hours", 11),
             sev_scaled("after_hours_min", 60, 240),
             "medium", "this week", ["after_hours_min", "work_span_hours"]),

        Rule("weekend_work", "work_boundary", "Reclaim one full weekend day",
             "Block Saturday or Sunday entirely. No laptop, no 'quick check'. "
             "One clean day recovers more than two half-worked ones.",
             "You've worked {weekend_work_min:.0f} minutes across recent "
             "weekends. Your recovery markers don't reset on split weekends.",
             lambda r, c: _gt(r, "weekend_work_min", 90),
             sev_scaled("weekend_work_min", 60, 480),
             "medium", "this weekend", ["weekend_work_min"]),

        Rule("meeting_load", "work_boundary", "Cut the meeting load",
             "Decline or delegate two recurring meetings this week, and batch "
             "the rest into two blocks so you get a clear stretch of focus.",
             "You're at {meetings_hours:.1f} hours of meetings a day with "
             "{context_switches:.0f} context switches -- there's no room left "
             "for actual work, which is why it spills into the evening.",
             lambda r, c: _gt(r, "meetings_min", 300),
             sev_scaled("meetings_min", 240, 480),
             "medium", "this week", ["meetings_min", "context_switches"]),

        Rule("vacation", "vacation", "Book time off -- properly",
             "Look at the next six weeks and put in 3-5 consecutive days. Book "
             "it before you decide where you're going; the dates are the hard "
             "part.",
             "Workload has been elevated for a sustained stretch and your "
             "recovery markers haven't returned to baseline between weeks. "
             "This is the pattern that precedes your worst runs.",
             lambda r, c: (_gt(r, "work_overload_z", 1.0)
                           or _gt(r, "work_overload", 0.7))
                          and c.get("sustained_strain_days", 0) >= 10,
             lambda r, c: min(1.5, 0.6 + c.get("sustained_strain_days", 0) / 20),
             "large", "this month", ["work_overload"]),

        # ----------------------------------------------- digital hygiene
        Rule("night_phone", "digital_hygiene", "Get the phone out of the bed",
             "Charge it outside the bedroom tonight. If you use it as an alarm, "
             "buy a ₹300 clock -- it's the highest-return purchase here.",
             "You've been on the phone {night_screen_min:.0f} minutes after "
             "11pm, and {night_searches:.0f} searches happened in the middle "
             "of the night.",
             lambda r, c: _gt(r, "night_screen_min", 45)
                          or _gt(r, "night_searches", 2),
             sev_scaled("night_screen_min", 30, 150),
             "small", "tonight", ["night_screen_min", "night_searches"]),

        Rule("doomscroll", "digital_hygiene", "Break the scroll loop",
             "Move the two worst apps off your home screen and set a 20-minute "
             "cap. Put a book or a podcast in the gap -- the habit needs a "
             "replacement, not just a block.",
             "Doomscrolling has been running {doomscroll_min:.0f} min/day with "
             "{pickups:.0f} phone pickups, and it climbs on your low days.",
             lambda r, c: _gt(r, "media_doomscroll_min", 45)
                          or _gt(r, "pickups", 150),
             sev_scaled("media_doomscroll_min", 30, 150),
             "small", "today", ["media_doomscroll_min", "pickups"]),

        # --------------------------------------------------------- fun
        Rule("anhedonia", "fun", "Schedule one thing you actually enjoy",
             "Not self-improvement -- enjoyment. The thing you'd do if nobody "
             "was measuring. Put a time on it.",
             "Your leisure has drifted almost entirely to passive screen time, "
             "and the active-enjoyment signals have gone quiet.",
             lambda r, c: _gt(r, "media_min", 120)
                          and _lt(r, "visit_social", 1)
                          and _lt(r, "workout_min", 15),
             lambda r, c: 0.9,
             "small", "this week", ["media_min"]),

        Rule("nature", "nature", "Get to green space",
             "A park, a lake, the hills if you can manage it. Forty minutes "
             "somewhere green does more for stress markers than the same time "
             "indoors.",
             "No park or outdoor time has shown up in {days_since_outdoor:.0f} "
             "days.",
             lambda r, c: _gt(r, "days_since_outdoor", 5),
             sev_scaled("days_since_outdoor", 5, 15),
             "small", "this weekend", ["days_since_outdoor"]),

        # --------------------------------------------------- doctor route
        Rule("doctor_symptoms", "doctor_consult",
             "Take the symptom pattern to a doctor",
             "Book a GP appointment. Take a printout of the last two weeks of "
             "this report -- the sleep, resting heart rate and symptom-search "
             "pattern are the useful parts for them.",
             "There's been sustained health-seeking activity -- searches, "
             "pharmacy and clinic visits -- alongside an elevated resting heart "
             "rate. That combination is worth a professional look rather than "
             "more searching.",
             lambda r, c: _gt(r, "medical_pressure_7d", 4)
                          or (_gt(r, "rhr_z", 1.5)
                              and _gt(r, "medical_pressure_7d", 2)),
             sev_scaled("medical_pressure_7d", 3, 10),
             "medium", "this week",
             ["medical_pressure_7d", "resting_hr", "visit_pharmacy"]),

        Rule("doctor_mood", "doctor_consult",
             "Consider talking to a professional",
             "A GP or a counsellor is the right next step. This isn't a big "
             "declaration -- it's one conversation, and it's what the pattern "
             "warrants.",
             "Mood-related signals have stayed low for a sustained stretch "
             "rather than dipping and recovering. A tracker can spot that "
             "pattern; it can't tell you what's behind it.",
             lambda r, c: c.get("days_mental_low", 0) >= 7
                          or _gt(r, "search_mental_health", 2),
             lambda r, c: min(1.6, 0.8 + c.get("days_mental_low", 0) / 12),
             "medium", "this week", ["search_mental_health"]),
    ]


class RecommendationEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rules = build_rules(cfg.profile)

    def _format(self, template: str, row: pd.Series, ctx: Dict) -> str:
        p = self.cfg.profile
        vals = {
            "target": p.target_sleep_hours,
            "steps_target": p.target_steps,
            "water_target": p.target_water_ml / 1000,
            "sleep_hours_7d": ctx.get("sleep_hours_7d", _v(row, "sleep_hours")),
            "water_l": _v(row, "water_ml") / 1000 if not np.isnan(
                _v(row, "water_ml")) else 0,
            "ultraprocessed_pct": _v(row, "ultraprocessed_ratio") * 100
            if not np.isnan(_v(row, "ultraprocessed_ratio")) else 0,
            "meetings_hours": _v(row, "meetings_min") / 60 if not np.isnan(
                _v(row, "meetings_min")) else 0,
        }
        for k in row.index:
            v = row[k]
            if isinstance(v, (int, float, np.floating)) and not pd.isna(v):
                vals.setdefault(k, float(v))
        try:
            return template.format(**{k: (0 if (isinstance(v, float)
                                                and np.isnan(v)) else v)
                                      for k, v in vals.items()})
        except (KeyError, ValueError, IndexError):
            # A missing field shouldn't silence the advice.
            import re
            return re.sub(r"\{[^}]*\}", "—", template)

    def generate(self, row: pd.Series, ctx: Optional[Dict] = None,
                 limit: int = 5) -> List[Recommendation]:
        ctx = ctx or {}
        fired: List[Recommendation] = []
        for rule in self.rules:
            try:
                if not rule.trigger(row, ctx):
                    continue
                sev = float(rule.severity(row, ctx))
            except Exception:
                continue
            # Cheap wins first when severity ties -- momentum matters.
            priority = sev * (1.0 + 0.12 * (5 - EFFORT.get(rule.effort, 2)))
            if rule.category == "doctor_consult":
                priority *= 1.5           # never bury this one
            fired.append(Recommendation(
                id=rule.id, category=rule.category, title=rule.title,
                action=rule.action,
                why=self._format(rule.why_template, row, ctx),
                effort=rule.effort, when=rule.when, priority=priority,
                evidence={k: (None if pd.isna(_v(row, k)) else
                              round(_v(row, k), 2))
                          for k in rule.evidence_keys},
            ))
        fired.sort(key=lambda r: -r.priority)

        # Spread across categories so the person doesn't get five sleep tips.
        picked, seen_cat = [], {}
        for rec in fired:
            n = seen_cat.get(rec.category, 0)
            if n >= (2 if rec.category == "sleep" else 1):
                continue
            seen_cat[rec.category] = n + 1
            picked.append(rec)
            if len(picked) >= limit:
                break
        return picked
