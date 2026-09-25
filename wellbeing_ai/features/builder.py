"""
Feature engineering: every table -> one row per day, ~70 columns.

Design notes
------------
* Missing sources produce NaN, never 0. Downstream scoring treats NaN as
  "no evidence" and lowers its confidence rather than inventing a low score.
* Circular quantities (bedtime, wake time) are handled as angles so that
  23:50 and 00:10 are ten minutes apart, not 23 hours.
* Everything comparative (`*_z`, `*_delta`) is computed against the person's
  *own* rolling baseline, not a population norm. A 5-hour sleeper who has
  always slept 5 hours is not flagged; a 7-hour sleeper who drops to 5 is.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..store import Store

PLACE_CATS = ["hospital", "pharmacy", "gym", "yoga", "worship", "park",
              "office", "home", "restaurant", "bar", "social", "transit"]


# ---------------------------------------------------------------- utilities
def _hour_of(ts: Optional[str]) -> float:
    if not ts or (isinstance(ts, float) and pd.isna(ts)):
        return np.nan
    try:
        dt = datetime.fromisoformat(str(ts))
        return dt.hour + dt.minute / 60
    except ValueError:
        try:
            parts = str(ts).split(":")
            return int(parts[0]) + int(parts[1]) / 60
        except (ValueError, IndexError):
            return np.nan


def _circ_mean(hours: pd.Series) -> float:
    h = hours.dropna()
    if h.empty:
        return np.nan
    ang = h * 2 * math.pi / 24
    return (math.atan2(np.sin(ang).mean(), np.cos(ang).mean())
            * 24 / (2 * math.pi)) % 24


def _circ_std(hours: pd.Series) -> float:
    """Circular SD in hours -- the core of 'sleep regularity'."""
    h = hours.dropna()
    if len(h) < 2:
        return np.nan
    ang = h * 2 * math.pi / 24
    R = math.hypot(np.sin(ang).mean(), np.cos(ang).mean())
    R = min(max(R, 1e-9), 1.0)
    return math.sqrt(-2 * math.log(R)) * 24 / (2 * math.pi)


def _circ_diff(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    d = (a - b) % 24
    return d - 24 if d > 12 else d


def robust_z(series: pd.Series, window: int, min_periods: int = 7) -> pd.Series:
    """Rolling median/MAD z-score -- resistant to the odd wild day."""
    med = series.shift(1).rolling(window, min_periods=min_periods).median()
    mad = (series.shift(1)
           .rolling(window, min_periods=min_periods)
           .apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True))
    scale = mad * 1.4826
    scale = scale.replace(0, np.nan)
    fallback = series.shift(1).rolling(window, min_periods=min_periods).std()
    scale = scale.fillna(fallback).replace(0, np.nan)
    return (series - med) / scale


class FeatureBuilder:
    def __init__(self, cfg: Config, store: Store):
        self.cfg = cfg
        self.store = store
        self.coverage: Dict[str, float] = {}

    # ------------------------------------------------------------ pipeline
    def build(self, since: Optional[str] = None,
              until: Optional[str] = None) -> pd.DataFrame:
        start, end = self.store.date_range()
        if start is None:
            return pd.DataFrame()
        since = since or start
        until = until or end
        idx = pd.date_range(since, until, freq="D")
        df = pd.DataFrame(index=idx)
        df.index.name = "date"

        for part in (self._sleep(), self._activity(), self._nutrition(),
                     self._work(), self._screen(), self._sensors(),
                     self._places(), self._messages(), self._searches(),
                     self._media(), self._spending(), self._social(),
                     self._notes(), self._checkins()):
            if not part.empty:
                df = df.join(part, how="left")

        df = self._calendar(df.copy())
        df = self._derived(df).copy()   # defragment after the many inserts
        df = self._baselines(df).copy()
        self.coverage = {c: float(df[c].notna().mean()) for c in df.columns}
        return df

    # ------------------------------------------------------------- sources
    def _daily(self, table: str) -> pd.DataFrame:
        d = self.store.table(table)
        if d.empty or "date" not in d:
            return pd.DataFrame()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        return d.dropna(subset=["date"]).set_index("date").sort_index()

    def _events(self, table: str) -> pd.DataFrame:
        d = self.store.table(table)
        if d.empty or "date" not in d:
            return pd.DataFrame()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        return d.dropna(subset=["date"])

    # -------------------------------------------------------------- sleep
    def _sleep(self) -> pd.DataFrame:
        s = self._daily("sleep")
        if s.empty:
            return pd.DataFrame()
        out = pd.DataFrame(index=s.index)
        out["sleep_hours"] = s.get("duration_min", np.nan) / 60
        out["sleep_efficiency"] = s.get("efficiency", np.nan)
        out["sleep_latency"] = s.get("latency_min", np.nan)
        out["awakenings"] = s.get("awakenings", np.nan)
        dur = s.get("duration_min", pd.Series(index=s.index, dtype=float))
        out["deep_pct"] = (s.get("deep_min", np.nan) / dur.replace(0, np.nan)) * 100
        out["rem_pct"] = (s.get("rem_min", np.nan) / dur.replace(0, np.nan)) * 100
        out["bed_hour"] = s.get("bedtime", pd.Series(dtype=object)).map(_hour_of) \
            if "bedtime" in s else np.nan
        out["wake_hour"] = s.get("waketime", pd.Series(dtype=object)).map(_hour_of) \
            if "waketime" in s else np.nan
        # Midpoint of sleep -- the single best circadian marker.
        bed = out["bed_hour"].where(out["bed_hour"] < 12,
                                    out["bed_hour"] - 24)  # 23:00 -> -1
        out["sleep_midpoint"] = bed + (out["sleep_hours"] / 2)
        return out

    # ------------------------------------------------------------ activity
    def _activity(self) -> pd.DataFrame:
        a = self._daily("activity")
        if a.empty:
            return pd.DataFrame()
        out = pd.DataFrame(index=a.index)
        for src, dst in [("steps", "steps"), ("active_min", "active_min"),
                         ("workout_min", "workout_min"),
                         ("calories_out", "calories_out"),
                         ("resting_hr", "resting_hr"), ("hrv_ms", "hrv_ms"),
                         ("spo2", "spo2"), ("weight_kg", "weight_kg")]:
            out[dst] = a.get(src, np.nan)
        out["did_workout"] = (out["workout_min"].fillna(0) >= 15).astype(float)
        return out

    # ----------------------------------------------------------- nutrition
    def _nutrition(self) -> pd.DataFrame:
        f = self._events("food")
        if f.empty:
            return pd.DataFrame()
        g = f.groupby("date")
        out = pd.DataFrame(index=g.size().index)
        out["calories_in"] = g["calories"].sum(min_count=1)
        out["protein_g"] = g["protein_g"].sum(min_count=1)
        out["fiber_g"] = g["fiber_g"].sum(min_count=1)
        out["sugar_g"] = g["sugar_g"].sum(min_count=1)
        out["sodium_mg"] = g["sodium_mg"].sum(min_count=1)
        out["caffeine_mg"] = g["caffeine_mg"].sum(min_count=1)
        out["alcohol_units"] = g["alcohol_units"].sum(min_count=1)
        out["water_ml"] = g["water_ml"].sum(min_count=1)
        out["meal_count"] = g["meal"].nunique()
        if "ultraprocessed" in f:
            out["ultraprocessed_ratio"] = g["ultraprocessed"].mean()
        return out

    # ---------------------------------------------------------------- work
    def _work(self) -> pd.DataFrame:
        w = self._daily("work")
        if w.empty:
            return pd.DataFrame()
        out = pd.DataFrame(index=w.index)
        out["work_hours"] = w.get("active_hours", np.nan)
        out["meetings_min"] = w.get("meetings_min", np.nan)
        out["after_hours_min"] = w.get("after_hours_min", np.nan)
        out["weekend_work_min"] = w.get("weekend_min", np.nan)
        out["context_switches"] = w.get("context_switches", np.nan)
        if "first_activity" in w and "last_activity" in w:
            fh = w["first_activity"].map(_hour_of)
            lh = w["last_activity"].map(_hour_of)
            out["work_span_hours"] = (lh - fh).where(lh > fh)
            out["work_start_hour"] = fh
            out["work_end_hour"] = lh
        return out

    # -------------------------------------------------------------- screen
    def _screen(self) -> pd.DataFrame:
        s = self._daily("screen")
        if s.empty:
            return pd.DataFrame()
        out = pd.DataFrame(index=s.index)
        out["screen_min"] = s.get("total_min", np.nan)
        out["night_screen_min"] = s.get("night_min", np.nan)
        out["pickups"] = s.get("pickups", np.nan)
        out["social_app_min"] = s.get("social_min", np.nan)
        out["entertainment_min"] = s.get("entertainment_min", np.nan)
        return out

    # ------------------------------------------------------------- sensors
    def _sensors(self) -> pd.DataFrame:
        s = self._daily("sensors")
        if s.empty:
            return pd.DataFrame()
        out = pd.DataFrame(index=s.index)
        out["outdoor_min"] = s.get("outdoor_min", np.nan)
        out["ambient_light"] = s.get("ambient_light_lux", np.nan)
        out["movement_entropy"] = s.get("movement_entropy", np.nan)
        out["location_variance"] = s.get("location_variance", np.nan)
        out["ambient_noise"] = s.get("ambient_noise_db", np.nan)
        out["unlocks"] = s.get("unlocks", np.nan)
        return out

    # -------------------------------------------------------------- places
    def _places(self) -> pd.DataFrame:
        v = self._events("visits")
        if v.empty:
            return pd.DataFrame()
        counts = (v.pivot_table(index="date", columns="place_category",
                                values="id", aggfunc="count")
                  .reindex(columns=PLACE_CATS, fill_value=0)
                  .fillna(0))
        counts.columns = [f"visit_{c}" for c in counts.columns]
        dwell = (v.pivot_table(index="date", columns="place_category",
                               values="dwell_min", aggfunc="sum")
                 .reindex(columns=["office", "home", "park", "gym"],
                          fill_value=np.nan))
        dwell.columns = [f"dwell_{c}_min" for c in dwell.columns]
        out = counts.join(dwell, how="outer")
        out["places_unique"] = v.groupby("date")["place_category"].nunique()
        out["medical_visit"] = ((out.get("visit_hospital", 0)
                                 + out.get("visit_pharmacy", 0)) > 0).astype(float)
        out["restorative_visit"] = ((out.get("visit_yoga", 0)
                                     + out.get("visit_worship", 0)
                                     + out.get("visit_park", 0)) > 0).astype(float)
        return out

    # ------------------------------------------------------------ messages
    def _messages(self) -> pd.DataFrame:
        m = self._events("messages")
        if m.empty:
            return pd.DataFrame()
        g = m.groupby("date")
        out = pd.DataFrame(index=g.size().index)
        out["msg_total"] = g.size()
        out["msg_out"] = m[m["direction"] == "out"].groupby("date").size()
        out["msg_in"] = m[m["direction"] == "in"].groupby("date").size()
        out["msg_out"] = out["msg_out"].fillna(0)
        out["msg_in"] = out["msg_in"].fillna(0)
        out["unique_contacts"] = g["contact_hash"].nunique()
        out["msg_sentiment"] = g["sentiment"].mean()
        out["msg_words"] = g["word_count"].mean()
        neg = m[m["sentiment"] < -0.2].groupby("date").size()
        out["neg_msg_ratio"] = (neg / out["msg_total"]).fillna(0)
        if "hour" in m:
            late = m[(m["hour"] >= 23) | (m["hour"] <= 4)].groupby("date").size()
            out["late_night_msgs"] = late.reindex(out.index).fillna(0)
        # Reciprocity: <1 means he is talking more than he is being answered.
        out["reply_ratio"] = (out["msg_in"] / out["msg_out"].replace(0, np.nan))
        return out

    # ------------------------------------------------------------ searches
    def _searches(self) -> pd.DataFrame:
        s = self._events("searches")
        if s.empty:
            return pd.DataFrame()
        g = s.groupby("date")
        out = pd.DataFrame(index=g.size().index)
        out["search_count"] = g.size()
        out["search_sentiment"] = g["sentiment"].mean()
        for cat in ("mental_health", "physical_health", "work", "finance",
                    "relationships", "travel", "fitness", "sleep"):
            sub = s[s["category"] == cat].groupby("date").size()
            out[f"search_{cat}"] = sub.reindex(out.index).fillna(0)
        out["health_search_ratio"] = ((out["search_physical_health"]
                                       + out["search_mental_health"])
                                      / out["search_count"].replace(0, np.nan))
        if "hour" in s:
            night = s[(s["hour"] >= 23) | (s["hour"] <= 4)].groupby("date").size()
            out["night_searches"] = night.reindex(out.index).fillna(0)
        return out

    # --------------------------------------------------------------- media
    def _media(self) -> pd.DataFrame:
        m = self._events("media")
        if m.empty:
            return pd.DataFrame()
        g = m.groupby("date")
        out = pd.DataFrame(index=g.size().index)
        out["media_min"] = g["minutes"].sum(min_count=1)
        out["media_items"] = g.size()
        for cat in ("doomscroll", "comfort", "selfhelp", "news", "fitness",
                    "learning"):
            sub = m[m["category"] == cat].groupby("date")["minutes"].sum()
            out[f"media_{cat}_min"] = sub.reindex(out.index).fillna(0)
        if "hour" in m:
            late = (m[(m["hour"] >= 23) | (m["hour"] <= 3)]
                    .groupby("date")["minutes"].sum())
            out["late_media_min"] = late.reindex(out.index).fillna(0)
        return out

    # ------------------------------------------------------------ spending
    def _spending(self) -> pd.DataFrame:
        s = self._events("spending")
        if s.empty:
            return pd.DataFrame()
        g = s.groupby("date")
        out = pd.DataFrame(index=g.size().index)
        out["spend_total"] = g["amount"].sum(min_count=1)
        out["spend_txns"] = g.size()
        for cat in ("medical", "food_delivery", "alcohol", "fitness", "travel",
                    "shopping", "entertainment"):
            sub = s[s["category"] == cat].groupby("date")["amount"].sum()
            out[f"spend_{cat}"] = sub.reindex(out.index).fillna(0)
        if "ts" in s:
            hours = pd.to_datetime(s["ts"], errors="coerce").dt.hour
            night = s[(hours >= 23) | (hours <= 4)].groupby("date")["amount"].sum()
            out["spend_night"] = night.reindex(out.index).fillna(0)
        return out

    # -------------------------------------------------------------- social
    def _social(self) -> pd.DataFrame:
        s = self._events("social")
        if s.empty:
            return pd.DataFrame()
        g = s.groupby("date")
        out = pd.DataFrame(index=g.size().index)
        out["social_actions"] = g.size()
        out["social_sentiment"] = g["sentiment"].mean()
        posts = s[s["action"] == "post"].groupby("date").size()
        out["social_posts"] = posts.reindex(out.index).fillna(0)
        return out

    # --------------------------------------------------------------- notes
    def _notes(self) -> pd.DataFrame:
        n = self._events("notes")
        if n.empty:
            return pd.DataFrame()
        g = n.groupby("date")
        out = pd.DataFrame(index=g.size().index)
        out["note_count"] = g.size()
        out["note_sentiment"] = g["sentiment"].mean()
        health = n[n["kind"] == "health"].groupby("date").size()
        out["health_notes"] = health.reindex(out.index).fillna(0)
        if "keywords" in n:
            from ..utils.textanalysis import RUMINATION
            def rum(series):
                kws = ",".join(str(x) for x in series if x)
                toks = [t for t in kws.split(",") if t]
                if not toks:
                    return np.nan
                return sum(1 for t in toks if t in RUMINATION) / len(toks)
            out["note_rumination"] = g["keywords"].apply(rum)
        return out

    # ------------------------------------------------------------ checkins
    def _checkins(self) -> pd.DataFrame:
        c = self._daily("checkins")
        if c.empty:
            return pd.DataFrame()
        out = pd.DataFrame(index=c.index)
        out["self_mood"] = c.get("mood", np.nan)
        out["self_energy"] = c.get("energy", np.nan)
        out["self_stress"] = c.get("stress", np.nan)
        return out

    # ------------------------------------------------------------ calendar
    def _calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        df["dow"] = df.index.dayofweek
        df["is_weekend"] = (df["dow"] >= 5).astype(float)
        df["month"] = df.index.month
        return df

    # ------------------------------------------------------------- derived
    def _derived(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.cfg.profile

        # Sleep debt: cumulative shortfall against target over 7 days.
        if "sleep_hours" in df:
            deficit = (p.target_sleep_hours - df["sleep_hours"]).clip(lower=0)
            df["sleep_debt_7d"] = deficit.rolling(7, min_periods=3).sum()
            df["sleep_regularity"] = (df["sleep_midpoint"]
                                      .rolling(7, min_periods=4).std()
                                      if "sleep_midpoint" in df else np.nan)
            if "bed_hour" in df:
                df["bedtime_drift"] = df["bed_hour"].rolling(
                    7, min_periods=4).apply(_circ_std, raw=False)

        # Movement streaks.
        if "did_workout" in df:
            df["workout_days_7d"] = df["did_workout"].rolling(7,
                                                              min_periods=3).sum()
            df["days_since_workout"] = _days_since(df["did_workout"] > 0)
        if "outdoor_min" in df:
            df["days_since_outdoor"] = _days_since(df["outdoor_min"] > 20)
        if "restorative_visit" in df:
            df["days_since_restorative"] = _days_since(df["restorative_visit"] > 0)

        # Medical-seeking pressure.
        med = pd.Series(0.0, index=df.index)
        for c, w in [("visit_hospital", 2.0), ("visit_pharmacy", 1.0),
                     ("search_physical_health", 0.4), ("health_notes", 0.8),
                     ("spend_medical", 0.0)]:
            if c in df:
                med = med + df[c].fillna(0) * w
        if "spend_medical" in df:
            med = med + (df["spend_medical"].fillna(0) > 0).astype(float) * 0.8
        df["medical_pressure"] = med
        df["medical_pressure_7d"] = med.rolling(7, min_periods=3).sum()

        # Work overload composite.
        over = pd.Series(np.nan, index=df.index)
        parts = []
        if "work_hours" in df:
            parts.append((df["work_hours"] - 8.5).clip(lower=0) / 3.0)
        if "after_hours_min" in df:
            parts.append(df["after_hours_min"].fillna(0) / 120.0)
        if "weekend_work_min" in df:
            parts.append(df["weekend_work_min"].fillna(0) / 180.0)
        if "meetings_min" in df:
            parts.append((df["meetings_min"].fillna(0) - 180).clip(lower=0) / 180)
        if parts:
            over = pd.concat(parts, axis=1).mean(axis=1)
        df["work_overload"] = over.clip(0, 2)

        # Social connection composite (higher = more connected).
        conn = []
        if "unique_contacts" in df:
            conn.append((df["unique_contacts"] / 6).clip(0, 1.5))
        if "msg_out" in df:
            conn.append((df["msg_out"] / 25).clip(0, 1.5))
        if "visit_social" in df:
            conn.append((df["visit_social"].fillna(0) > 0).astype(float))
        if "visit_restaurant" in df:
            conn.append((df["visit_restaurant"].fillna(0) > 0).astype(float) * 0.5)
        df["social_connection"] = (pd.concat(conn, axis=1).mean(axis=1)
                                   if conn else np.nan)
        if "social_connection" in df:
            df["days_since_social"] = _days_since(df["social_connection"] > 0.3)

        # Night-time digital load -- a strong sleep + mood antecedent.
        night = []
        for c, scale in [("night_screen_min", 90.0), ("late_media_min", 60.0),
                         ("late_night_msgs", 10.0), ("night_searches", 5.0)]:
            if c in df:
                night.append((df[c].fillna(0) / scale).clip(0, 2))
        df["night_digital_load"] = (pd.concat(night, axis=1).mean(axis=1)
                                    if night else np.nan)

        # Distress-signal composite from text sources.
        distress = []
        if "msg_sentiment" in df:
            distress.append((-df["msg_sentiment"]).clip(-1, 1))
        if "search_sentiment" in df:
            distress.append((-df["search_sentiment"]).clip(-1, 1))
        if "note_sentiment" in df:
            distress.append((-df["note_sentiment"]).clip(-1, 1))
        if "social_sentiment" in df:
            distress.append((-df["social_sentiment"]).clip(-1, 1) * 0.6)
        if "search_mental_health" in df:
            distress.append((df["search_mental_health"] / 3).clip(0, 1.5))
        df["distress_index"] = (pd.concat(distress, axis=1).mean(axis=1)
                                if distress else np.nan)

        # Recovery: HRV up + RHR down relative to the person's own baseline.
        w = self.cfg.models.baseline_window_days
        if "hrv_ms" in df:
            df["hrv_z"] = robust_z(df["hrv_ms"], w)
        if "resting_hr" in df:
            df["rhr_z"] = robust_z(df["resting_hr"], w)
        return df

    # ----------------------------------------------------------- baselines
    def _baselines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add `_z` columns for the signals the scorer and alerts rely on."""
        w = self.cfg.models.baseline_window_days
        for c in ["sleep_hours", "steps", "active_min", "work_hours",
                  "screen_min", "msg_out", "unique_contacts", "msg_sentiment",
                  "spend_total", "media_min", "search_count", "outdoor_min",
                  "social_connection", "distress_index", "work_overload",
                  "calories_in", "night_digital_load", "medical_pressure"]:
            if c in df and df[c].notna().sum() >= 7:
                df[f"{c}_z"] = robust_z(df[c], w)
        return df


def _days_since(mask: pd.Series) -> pd.Series:
    """For each day, how many days since `mask` was last True (inclusive=0)."""
    out, count = [], np.nan
    for val in mask.fillna(False).astype(bool):
        if val:
            count = 0
        elif not pd.isna(count):
            count += 1
        out.append(count)
    return pd.Series(out, index=mask.index)
