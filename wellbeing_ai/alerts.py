"""
The alerting engine.

Principles that matter more than the rules:

* An alert must be *earned*. A single bad night is not an alert; four in a row
  is. Everything here requires persistence or a genuine baseline departure.
* Cooldowns are enforced per kind, so the same message can't arrive daily.
* Quiet hours are respected -- waking someone up to tell them they're not
  sleeping enough would be its own punchline.
* Good news is alertable too. A system that only ever speaks up to criticise
  gets muted within a week.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .safety import CrisisDetector, CrisisFlag
from .store import Store

LEVELS = {"info": 0, "praise": 1, "amber": 2, "red": 3, "support": 4}


@dataclass
class Alert:
    kind: str
    level: str
    title: str
    body: str
    date: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "level": self.level, "title": self.title,
                "body": self.body, "date": self.date, "payload": self.payload}


class AlertEngine:
    def __init__(self, cfg: Config, store: Store):
        self.cfg = cfg
        self.store = store
        self.crisis = CrisisDetector(cfg.profile.locale)

    # ------------------------------------------------------------ helpers
    def _recent(self, kind: str) -> Optional[datetime]:
        df = self.store.read_df(
            "SELECT ts FROM alerts WHERE kind = ? ORDER BY ts DESC LIMIT 1",
            (kind,))
        if df.empty:
            return None
        try:
            return datetime.fromisoformat(df["ts"].iloc[0])
        except (ValueError, TypeError):
            return None

    def _cooled_down(self, kind: str) -> bool:
        last = self._recent(kind)
        if last is None:
            return True
        return (datetime.now() - last) >= timedelta(
            hours=self.cfg.alerts.cooldown_hours)

    @staticmethod
    def _streak(series: pd.Series, predicate) -> int:
        """Length of the current run at the end of the series."""
        n = 0
        for v in reversed(series.tolist()):
            if pd.isna(v) or not predicate(v):
                break
            n += 1
        return n

    # ------------------------------------------------------------ the rules
    def evaluate(self, scores: pd.DataFrame, features: pd.DataFrame,
                 model_out=None, texts: Optional[List[str]] = None
                 ) -> List[Alert]:
        if scores.empty:
            return []
        a = self.cfg.alerts
        today = scores.index[-1]
        date = today.strftime("%Y-%m-%d")
        latest = scores.iloc[-1]
        out: List[Alert] = []

        mental = scores["mental"]
        physical = scores["physical"]
        row = features.iloc[-1] if not features.empty else pd.Series(dtype=float)

        # ---------------------------------------------------- safety first
        days_low = self._streak(mental.tail(21), lambda v: v < 40)
        sleep_bad = self._streak(
            features["sleep_hours"].tail(14), lambda v: v < 6) \
            if "sleep_hours" in features else 0
        iso = self._streak(
            features["social_connection"].tail(14), lambda v: v < 0.2) \
            if "social_connection" in features else 0

        flag = self.crisis.evaluate(
            texts=texts, mental_score=float(latest["mental"]),
            days_below_40=days_low, sleep_debt_nights=sleep_bad,
            social_isolation_days=iso)
        if flag.triggered:
            out.append(Alert(
                kind="support", level="support",
                title="Checking in",
                body=self.crisis.support_message(self.cfg.profile.name, flag),
                date=date, payload=flag.to_dict()))
            # When this fires, it is the only thing that gets said.
            return out

        # ------------------------------------------------ absolute severity
        if latest["mental"] < a.red_score:
            streak = self._streak(mental, lambda v: v < a.red_score)
            if streak >= a.persistence_days and self._cooled_down("mental_red"):
                out.append(Alert(
                    "mental_red", "red",
                    f"Mood signals have been low for {streak} days",
                    f"Your mental wellbeing score has sat at "
                    f"{latest['mental']:.0f} or below for {streak} days "
                    "running. That's a pattern rather than a bad day.",
                    date, {"score": round(float(latest["mental"]), 1),
                           "streak": streak}))
        elif latest["mental"] < a.amber_score:
            streak = self._streak(mental, lambda v: v < a.amber_score)
            if streak >= a.persistence_days and self._cooled_down("mental_amber"):
                out.append(Alert(
                    "mental_amber", "amber",
                    f"Running below your baseline for {streak} days",
                    f"Mental wellbeing is at {latest['mental']:.0f}. Not alarming "
                    "on its own, but it hasn't bounced back the way it usually "
                    "does.", date, {"score": round(float(latest["mental"]), 1),
                                    "streak": streak}))

        if latest["physical"] < a.red_score and self._cooled_down("physical_red"):
            out.append(Alert(
                "physical_red", "red", "Physical markers are well down",
                f"Physical score is {latest['physical']:.0f}. Sleep, movement "
                "and recovery markers are all pulling in the same direction.",
                date, {"score": round(float(latest["physical"]), 1)}))

        # ------------------------------------------- baseline-relative drop
        for col, label, kind in [("mental", "mood", "mental_drop"),
                                 ("physical", "physical", "physical_drop")]:
            z = _z_last(scores[col], self.cfg.models.baseline_window_days)
            if z is None:
                continue
            if z <= a.z_drop_red and self._cooled_down(kind):
                out.append(Alert(
                    kind, "red", f"Sharp drop in your {label} signals",
                    f"Today is {abs(z):.1f} standard deviations below your own "
                    f"recent {label} baseline -- the biggest departure in "
                    "several weeks.", date, {"z": round(z, 2)}))
            elif z <= a.z_drop_amber and self._cooled_down(kind):
                out.append(Alert(
                    kind, "amber", f"Your {label} signals have dipped",
                    f"Today sits {abs(z):.1f} SD below your usual range.",
                    date, {"z": round(z, 2)}))

        # ------------------------------------------------- specific drivers
        specifics = [
            ("sleep_debt", "sleep_debt_7d", lambda v: v > 8, "amber",
             "Sleep debt has built up",
             "You're carrying {v:.0f} hours of sleep debt over the last week."),
            ("work_overload", "work_overload", lambda v: v > 0.75, "amber",
             "Workload has been running hot",
             "Your workload index is {v:.2f}, well into the range that "
             "precedes your bad stretches."),
            ("withdrawal", "days_since_social", lambda v: v >= 4, "amber",
             "You've gone quiet on people",
             "It's been {v:.0f} days since meaningful contact with anyone. "
             "In your data this usually moves before mood does."),
            ("medical", "medical_pressure_7d", lambda v: v > 5, "amber",
             "A lot of health-seeking this week",
             "Symptom searches, pharmacy and clinic activity are all elevated "
             "({v:.0f} on the index). Worth a doctor rather than more "
             "searching."),
            ("night_phone", "night_digital_load", lambda v: v > 0.9, "info",
             "Late-night screen use is climbing",
             "Your night-time digital load is at {v:.2f}. It's the most "
             "reliable early warning in your own history."),
        ]
        for kind, feat, test, level, title, body in specifics:
            if feat not in row.index or pd.isna(row[feat]):
                continue
            if test(float(row[feat])) and self._cooled_down(kind):
                out.append(Alert(kind, level, title,
                                 body.format(v=float(row[feat])), date,
                                 {feat: round(float(row[feat]), 2)}))

        # ------------------------------------------------------- preventive
        if model_out is not None and len(getattr(model_out, "risk", [])) > 0:
            risk = model_out.risk.iloc[-1]
            if pd.notna(risk) and risk >= 0.65 and self._cooled_down("forecast"):
                out.append(Alert(
                    "forecast", "amber", "The next few days look like a dip",
                    f"Based on your last three days, there's a "
                    f"{risk * 100:.0f}% chance the coming stretch lands in your "
                    "bottom quartile. This is the point where a small change is "
                    "cheapest.", date, {"risk": round(float(risk), 2)}))

        if model_out is not None and len(getattr(model_out, "anomaly", [])) > 0:
            an = model_out.anomaly.iloc[-1]
            if pd.notna(an) and an >= 0.85 and self._cooled_down("anomaly"):
                out.append(Alert(
                    "anomaly", "info", "Today doesn't look like your usual days",
                    "Several of your daily patterns are outside their normal "
                    "range at once. Not necessarily bad -- just unusual.",
                    date, {"anomaly": round(float(an), 2)}))

        # ----------------------------------------------------- the good news
        z_up = _z_last(mental, self.cfg.models.baseline_window_days)
        if z_up is not None and z_up >= a.praise_z_gain \
                and self._cooled_down("praise"):
            good = _what_improved(features)
            out.append(Alert(
                "praise", "praise", "Good stretch -- and here's why",
                "Your mood signals are "
                f"{z_up:.1f} SD above your recent baseline."
                + (f" The change is coming from {good}." if good else "")
                + " Worth noticing what you did differently.",
                date, {"z": round(z_up, 2)}))

        up_streak = self._streak(mental, lambda v: v >= 65)
        if up_streak >= 5 and self._cooled_down("streak"):
            out.append(Alert(
                "streak", "praise", f"{up_streak} solid days in a row",
                f"That's your longest run above 65 in a while. Whatever the "
                "current routine is, it's working.", date,
                {"streak": up_streak}))

        out.sort(key=lambda al: -LEVELS.get(al.level, 0))
        return out

    # -------------------------------------------------------------- output
    def persist(self, alerts: List[Alert]) -> int:
        rows = [{"ts": datetime.now().isoformat(timespec="seconds"),
                 "date": al.date, "level": al.level, "kind": al.kind,
                 "title": al.title, "body": al.body,
                 "payload": json.dumps(al.payload)} for al in alerts]
        return self.store.insert_events("alerts", rows)

    def deliver(self, alerts: List[Alert]) -> Dict[str, Any]:
        """Send to the configured channels. Never raises."""
        res = {"desktop": 0, "webhook": 0, "skipped_quiet_hours": 0}
        hour = datetime.now().hour
        for al in alerts:
            if hour in self.cfg.alerts.quiet_hours and al.level != "support":
                res["skipped_quiet_hours"] += 1
                continue
            if "desktop" in self.cfg.alerts.channels and _notify(al):
                res["desktop"] += 1
            if self.cfg.alerts.webhook_url:
                if _webhook(self.cfg.alerts.webhook_url, al):
                    res["webhook"] += 1
        return res


# ------------------------------------------------------------------ helpers
#: Scores live on a 0-100 scale, so a stretch of unusually consistent days can
#: drive the MAD near zero and turn a 4-point move into "3.4 SD". Flooring the
#: scale keeps the z honest -- and keeps the alerts from crying wolf in both
#: directions.
_MIN_SCORE_SCALE = 3.0


def _z_last(series: pd.Series, window: int) -> Optional[float]:
    s = series.dropna()
    if len(s) < 10:
        return None
    hist = s.iloc[:-1].tail(window)
    if len(hist) < 7:
        return None
    med = hist.median()
    mad = (hist - med).abs().median() * 1.4826
    scale = mad if mad > 1e-6 else hist.std()
    if not scale or pd.isna(scale):
        return None
    return float((s.iloc[-1] - med) / max(float(scale), _MIN_SCORE_SCALE))


#: (feature, good direction, phrasing when it moved the good way)
_MOVEMENTS = [
    ("sleep_hours", 1, "more sleep"),
    ("steps", 1, "more walking"),
    ("outdoor_min", 1, "more time outdoors"),
    ("unique_contacts", 1, "talking to more people"),
    ("workout_days_7d", 1, "more training days"),
    ("social_connection", 1, "more contact with people"),
    ("hrv_ms", 1, "better recovery markers"),
    ("after_hours_min", -1, "less after-hours work"),
    ("night_screen_min", -1, "less late-night phone use"),
    ("work_overload", -1, "a lighter workload"),
    ("media_doomscroll_min", -1, "less doomscrolling"),
    ("sleep_debt_7d", -1, "less sleep debt"),
    ("alcohol_units", -1, "less alcohol"),
]


def _what_improved(features: pd.DataFrame) -> str:
    """Name the two things that moved most favourably, in plain words."""
    if features.empty or len(features) < 14:
        return ""
    moves = []
    for col, sign, phrase in _MOVEMENTS:
        if col not in features:
            continue
        recent = features[col].tail(7).mean()
        prior = features[col].tail(14).head(7).mean()
        if pd.isna(recent) or pd.isna(prior) or abs(prior) < 1e-6:
            continue
        rel = sign * (recent - prior) / (abs(prior) + 1e-9)
        if rel > 0.12:
            moves.append((rel, phrase))
    moves.sort(reverse=True)
    return " and ".join(p for _, p in moves[:2])


def _notify(alert: Alert) -> bool:
    """Best-effort desktop notification across the three platforms."""
    title = alert.title
    body = alert.body[:400]
    try:
        system = platform.system()
        if system == "Darwin":
            script = (f'display notification {json.dumps(body)} '
                      f'with title {json.dumps("Wellbeing")} '
                      f'subtitle {json.dumps(title)}')
            subprocess.run(["osascript", "-e", script], check=False,
                           capture_output=True, timeout=10)
            return True
        if system == "Linux" and shutil.which("notify-send"):
            urgency = "critical" if alert.level in ("red", "support") else "normal"
            subprocess.run(["notify-send", "-u", urgency, title, body],
                           check=False, capture_output=True, timeout=10)
            return True
        if system == "Windows":
            ps = (f'[Windows.UI.Notifications.ToastNotificationManager, '
                  f'Windows.UI.Notifications, ContentType=WindowsRuntime]'
                  f'| Out-Null; '
                  f'Write-Output {json.dumps(title + ": " + body)}')
            subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           check=False, capture_output=True, timeout=15)
            return True
    except Exception:
        return False
    return False


def _webhook(url: str, alert: Alert) -> bool:
    """ntfy.sh / Slack / Pushover style POST. Uses stdlib only."""
    import urllib.error
    import urllib.request
    try:
        if "ntfy" in url:
            data = alert.body.encode("utf-8")
            req = urllib.request.Request(
                url, data=data,
                headers={"Title": alert.title,
                         "Priority": "high" if alert.level in ("red", "support")
                         else "default"})
        else:
            payload = json.dumps({"text": f"*{alert.title}*\n{alert.body}",
                                  **alert.to_dict()}).encode("utf-8")
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception:
        return False
