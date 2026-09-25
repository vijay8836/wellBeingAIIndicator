"""
The GenAI layer: turns numbers into something worth reading.

What leaves the machine
-----------------------
Only the aggregated payload built by `build_payload`: scores, deltas, named
feature labels and rounded values. No message text, no contact names, no
search queries, no place names, no coordinates. You can print the payload
before it is sent (`--show-payload`) and check.

If the API is unavailable, disabled, or returns something unusable, the
`TemplateCoach` writes the narrative locally instead. The system never goes
quiet just because the network did.
"""
from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..safety import ClinicalGuard, CrisisFlag
from .recommendations import Recommendation

SYSTEM_PROMPT = """\
You are the writing voice of a personal wellbeing tracker. You are given a \
numeric summary of one person's own behavioural data -- sleep, movement, \
workload, social contact, mood signals -- plus the recommendations a rules \
engine has already selected.

Your job is to write the narrative around those numbers. You are a thoughtful \
friend who happens to have the data, not a doctor and not a life coach with a \
script.

Hard rules:
- Never diagnose. Never name a medical or psychiatric condition as something \
the person has. "Your mood scores have been low for nine days" is fine; \
"you're depressed" is not.
- Never recommend, name, or dose any medication or supplement regimen. If the \
data warrants clinical attention, the recommendation is to see a clinician.
- Do not invent numbers. Use only the figures in the payload. If something \
isn't there, don't mention it.
- Do not moralise, guilt-trip, or use the word "should" about the person's \
choices. They already know. Describe what changed and what might help.
- Be specific and concrete. "Your bedtime moved 1.8 hours later over two \
weeks" beats "your sleep has been poor".
- Match length to what actually happened. A quiet, ordinary day gets three \
sentences, not five paragraphs. Save the length for days that earned it.

Tone: warm, direct, unsentimental. Write like a person, not a wellness app. \
No emoji. No exclamation marks. Never open with "Great news!" or "I noticed \
that". British-leaning plain English.

Return ONLY a JSON object, no prose around it, with these keys:
{
  "headline": "one line, under 12 words, specific to today",
  "assessment": "2-5 sentences on what the data shows and what changed",
  "whats_working": "1-2 sentences naming something genuinely going well, or \
null if there is honestly nothing",
  "watch": "1-2 sentences on the single thing most worth attention, or null",
  "encouragement": "1-2 sentences. Earned, specific, not a platitude. Null if \
it would be hollow.",
  "recommendation_notes": {"<recommendation id>": "one sentence making this \
one feel relevant to today specifically"}
}"""


@dataclass
class CoachOutput:
    headline: str
    assessment: str
    whats_working: Optional[str] = None
    watch: Optional[str] = None
    encouragement: Optional[str] = None
    recommendation_notes: Dict[str, str] = field(default_factory=dict)
    source: str = "template"
    model: Optional[str] = None
    guarded: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"headline": self.headline, "assessment": self.assessment,
                "whats_working": self.whats_working, "watch": self.watch,
                "encouragement": self.encouragement,
                "recommendation_notes": self.recommendation_notes,
                "source": self.source, "model": self.model,
                "guarded": self.guarded}

    def as_text(self) -> str:
        parts = [self.headline, "", self.assessment]
        for extra in (self.whats_working, self.watch, self.encouragement):
            if extra:
                parts += ["", extra]
        return "\n".join(parts)


# --------------------------------------------------------------------------
def build_payload(cfg: Config, period: str, date: str,
                  scores: pd.DataFrame, features: pd.DataFrame,
                  drivers: List[Dict[str, Any]],
                  recs: List[Recommendation],
                  trend: Optional[Dict[str, Any]] = None,
                  model_drivers: Optional[List[Dict[str, Any]]] = None,
                  anomaly: Optional[float] = None,
                  risk: Optional[float] = None,
                  archetype: Optional[str] = None,
                  history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the numbers-only bundle that the model is allowed to see."""
    latest = scores.iloc[-1] if not scores.empty else pd.Series(dtype=float)

    def delta(col: str, days: int) -> Optional[float]:
        if col not in scores or len(scores) < days + 1:
            return None
        recent = scores[col].tail(days).mean()
        prior = scores[col].tail(days * 2).head(days).mean()
        if pd.isna(recent) or pd.isna(prior):
            return None
        return round(float(recent - prior), 1)

    payload: Dict[str, Any] = {
        "period": period,
        "date": date,
        "person": {"name": cfg.profile.name,
                   "context": cfg.profile.context_notes or None},
        "scores": {
            "physical": round(float(latest.get("physical", np.nan)), 1)
            if not scores.empty else None,
            "mental": round(float(latest.get("mental", np.nan)), 1)
            if not scores.empty else None,
            "overall": round(float(latest.get("overall", np.nan)), 1)
            if not scores.empty else None,
            "confidence": round(float(latest.get("confidence", 0)), 2)
            if not scores.empty else None,
        },
        "subscores": {k.replace("sub_", ""): round(float(v), 1)
                      for k, v in latest.items()
                      if str(k).startswith("sub_") and pd.notna(v)},
        "change": {
            "mental_7d_vs_prior": delta("mental", 7),
            "physical_7d_vs_prior": delta("physical", 7),
            "mental_30d_vs_prior": delta("mental", 30),
        },
        "drivers_today": [
            {"label": d["label"], "direction": d["direction"],
             "value": d["value"], "domain": d["domain"]}
            for d in drivers[:8]
        ],
        "recommendations": [
            {"id": r.id, "category": r.category, "title": r.title,
             "why": r.why, "effort": r.effort}
            for r in recs
        ],
    }
    if trend:
        payload["trend"] = {k: trend.get(k) for k in
                            ("trend", "delta", "changepoint", "recent_mean",
                             "prior_mean", "best_day", "worst_day")}
    if model_drivers:
        payload["learned_patterns"] = [
            {"behaviour": d["label"], "relationship": d["relationship"]}
            for d in model_drivers[:6]
        ]
    if anomaly is not None and pd.notna(anomaly):
        payload["unusualness_today"] = round(float(anomaly), 2)
    if risk is not None and pd.notna(risk):
        payload["low_stretch_risk_next_3d"] = round(float(risk), 2)
    if archetype:
        payload["day_type"] = archetype
    if history:
        payload["history"] = history

    # Key raw figures, rounded -- helps the model be concrete.
    if not features.empty:
        row = features.iloc[-1]
        keep = ["sleep_hours", "sleep_debt_7d", "sleep_regularity", "steps",
                "workout_days_7d", "work_hours", "after_hours_min",
                "unique_contacts", "days_since_social", "outdoor_min",
                "night_screen_min", "days_since_workout", "hrv_z",
                "medical_pressure_7d", "alcohol_units", "water_ml"]
        payload["today_figures"] = {
            k: round(float(row[k]), 1) for k in keep
            if k in row.index and pd.notna(row[k])
        }
    return payload


# --------------------------------------------------------------------------
class TemplateCoach:
    """Local fallback. Plain, honest, and always available."""

    def write(self, payload: Dict[str, Any],
              recs: List[Recommendation]) -> CoachOutput:
        s = payload.get("scores", {})
        ch = payload.get("change", {})
        name = payload.get("person", {}).get("name", "there")
        mental, phys = s.get("mental"), s.get("physical")
        d7 = ch.get("mental_7d_vs_prior")

        def band(v):
            if v is None:
                return "unclear"
            return ("low" if v < 45 else "below your usual" if v < 58
                    else "steady" if v < 72 else "good")

        direction = ""
        if d7 is not None:
            if d7 <= -4:
                direction = f" That's {abs(d7):.0f} points below the week before."
            elif d7 >= 4:
                direction = f" That's {d7:.0f} points up on the week before."
            else:
                direction = " Roughly level with last week."

        headline = {
            "low": f"A hard stretch, {name} -- worth slowing down",
            "below your usual": f"Running below your own baseline",
            "steady": f"Holding steady",
            "good": f"A good day by your own numbers",
        }[band(mental)]

        down = [d for d in payload.get("drivers_today", [])
                if d["direction"] == "down"][:3]
        up = [d for d in payload.get("drivers_today", [])
              if d["direction"] == "up"][:2]

        assess = [f"Mental wellbeing is at {mental:.0f} and physical at "
                  f"{phys:.0f}, on your own 0-100 scale." if mental is not None
                  and phys is not None else "Not enough data to score today."]
        if direction:
            assess.append(direction.strip())
        if down:
            assess.append("The weight today is coming from "
                          + ", ".join(d["label"] for d in down) + ".")
        if payload.get("trend", {}).get("changepoint"):
            assess.append("This run looks like it started around "
                          f"{payload['trend']['changepoint']}.")

        working = ("Still holding: " + ", ".join(d["label"] for d in up) + "."
                   ) if up else None
        watch = (f"The one to watch is {down[0]['label']}." if down else None)

        enc = None
        if d7 is not None and d7 >= 4:
            enc = ("Whatever you changed last week is showing up in the "
                   "numbers. Worth keeping.")
        elif band(mental) == "low":
            enc = ("Bad stretches in your data have ended before, and they "
                   "ended after small changes, not big ones. Pick one thing "
                   "below.")
        return CoachOutput(headline=headline, assessment=" ".join(assess),
                           whats_working=working, watch=watch,
                           encouragement=enc, source="template")


class ClaudeCoach:
    """Calls Claude for the narrative, then puts the result through the guard."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = None

    def available(self) -> bool:
        return bool(self.cfg.llm.enabled and self.cfg.api_key)

    def _client_or_none(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.cfg.api_key)
        except Exception:
            return None
        return self._client

    def write(self, payload: Dict[str, Any],
              recs: List[Recommendation]) -> CoachOutput:
        client = self._client_or_none()
        if client is None:
            return TemplateCoach().write(payload, recs)

        user = (
            f"Here is the {payload['period']} summary for "
            f"{payload['date']}.\n\n```json\n"
            f"{json.dumps(payload, indent=2, default=str)}\n```\n\n"
            "Write the narrative. Return only the JSON object."
        )
        try:
            resp = client.messages.create(
                model=self.cfg.llm.model,
                max_tokens=self.cfg.llm.max_tokens,
                temperature=self.cfg.llm.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content
                           if getattr(b, "type", "") == "text")
            data = _extract_json(text)
            if not data:
                raise ValueError("no JSON in response")
        except Exception as exc:
            out = TemplateCoach().write(payload, recs)
            out.guarded.append(f"LLM unavailable ({type(exc).__name__}); "
                               "wrote locally instead")
            return out

        out = CoachOutput(
            headline=str(data.get("headline", ""))[:160],
            assessment=str(data.get("assessment", "")),
            whats_working=_opt(data.get("whats_working")),
            watch=_opt(data.get("watch")),
            encouragement=_opt(data.get("encouragement")),
            recommendation_notes={str(k): str(v) for k, v in
                                  (data.get("recommendation_notes") or {}).items()},
            source="claude", model=self.cfg.llm.model,
        )
        # The guard is not optional, even on our own model's output.
        blob = " ".join(filter(None, [out.headline, out.assessment,
                                      out.whats_working, out.watch,
                                      out.encouragement]))
        issues = ClinicalGuard.violates(blob)
        if issues:
            fallback = TemplateCoach().write(payload, recs)
            fallback.guarded = [f"model output filtered: {i}" for i in issues]
            fallback.source = "template (filtered)"
            return fallback
        return out


def _opt(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ("null", "none", "n/a") else None


def _extract_json(text: str) -> Optional[Dict]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def get_coach(cfg: Config):
    claude = ClaudeCoach(cfg)
    if claude.available():
        return claude
    return TemplateCoach()
