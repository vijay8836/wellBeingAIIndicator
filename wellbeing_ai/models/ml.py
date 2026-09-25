"""
The machine-learning layer.

Four models, all trained *per person* on their own history -- there is no
population model, because "normal" only means anything relative to your own
last few weeks.

1. AnomalyDetector   IsolationForest over the daily feature vector. Answers
                     "is today unlike any day you've had recently?"
2. Forecaster        Gradient boosting on lagged features. Answers "given the
                     last few days, where is mood heading?" and -- more
                     usefully -- "which of your behaviours precede your dips?"
3. RiskModel         Classifier for "tomorrow is likely to be a bad day",
                     which is what makes the alerts preventive rather than
                     a post-mortem.
4. DayArchetypes     KMeans over behaviour, auto-named, so weekly reports can
                     say "you had 4 grind days and 1 restorative day".

Plus TrendAnalyzer: Mann-Kendall + EWMA + CUSUM for change detection.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor, IsolationForest)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=UserWarning)

from ..config import Config

# Features the models are allowed to look at. Excludes score columns (which
# would leak the target) and calendar noise.
MODEL_FEATURES = [
    "sleep_hours", "sleep_efficiency", "sleep_latency", "awakenings",
    "sleep_debt_7d", "sleep_regularity", "bedtime_drift", "sleep_midpoint",
    "steps", "active_min", "workout_min", "workout_days_7d",
    "days_since_workout", "resting_hr", "hrv_ms", "hrv_z", "rhr_z",
    "calories_in", "protein_g", "sugar_g", "water_ml", "alcohol_units",
    "caffeine_mg", "ultraprocessed_ratio",
    "work_hours", "meetings_min", "after_hours_min", "weekend_work_min",
    "work_span_hours", "work_overload", "context_switches",
    "screen_min", "night_screen_min", "pickups", "social_app_min",
    "outdoor_min", "ambient_light", "movement_entropy", "location_variance",
    "msg_total", "msg_out", "unique_contacts", "msg_sentiment",
    "neg_msg_ratio", "late_night_msgs", "reply_ratio",
    "search_count", "search_mental_health", "search_physical_health",
    "health_search_ratio", "night_searches", "search_sentiment",
    "media_min", "media_doomscroll_min", "media_comfort_min", "late_media_min",
    "spend_total", "spend_medical", "spend_food_delivery", "spend_alcohol",
    "spend_night", "social_connection", "days_since_social",
    "night_digital_load", "distress_index", "medical_pressure_7d",
    "days_since_outdoor", "days_since_restorative", "note_rumination",
    "visit_hospital", "visit_pharmacy", "visit_gym", "visit_yoga",
    "visit_worship", "visit_park", "places_unique", "is_weekend",
]

READABLE = {
    "sleep_hours": "hours slept", "sleep_debt_7d": "7-day sleep debt",
    "sleep_regularity": "irregular sleep timing", "bedtime_drift": "bedtime drift",
    "night_digital_load": "late-night phone use", "work_overload": "workload",
    "after_hours_min": "after-hours work", "distress_index": "stress language",
    "social_connection": "contact with people", "days_since_social":
        "days since real contact", "steps": "daily steps",
    "outdoor_min": "time outdoors", "media_doomscroll_min": "doomscrolling",
    "night_searches": "late-night searching", "hrv_z": "HRV vs your baseline",
    "alcohol_units": "alcohol", "unique_contacts": "people you spoke to",
    "msg_sentiment": "tone of your messages", "meetings_min": "meeting load",
    "search_mental_health": "searches about how you feel",
    "days_since_workout": "days since a workout",
    "medical_pressure_7d": "health-seeking activity",
}


def readable(feature: str) -> str:
    return READABLE.get(feature, feature.replace("_", " "))


def _matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    cols = [c for c in MODEL_FEATURES if c in df.columns
            and df[c].notna().sum() >= max(5, 0.15 * len(df))]
    if not cols:
        return np.zeros((len(df), 0)), []
    return df[cols].astype(float).values, cols


# --------------------------------------------------------------------------
@dataclass
class ModelOutputs:
    anomaly: pd.Series = field(default_factory=pd.Series)
    archetype: pd.Series = field(default_factory=pd.Series)
    archetype_names: Dict[int, str] = field(default_factory=dict)
    forecast: pd.Series = field(default_factory=pd.Series)
    risk: pd.Series = field(default_factory=pd.Series)
    drivers: List[Dict[str, Any]] = field(default_factory=list)
    trained_on: int = 0
    notes: List[str] = field(default_factory=list)


class AnomalyDetector:
    """'Today doesn't look like you.' Unsupervised, so it needs no labels."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pipe: Optional[Pipeline] = None
        self.cols: List[str] = []

    def fit_score(self, features: pd.DataFrame) -> pd.Series:
        X, cols = _matrix(features)
        self.cols = cols
        if X.shape[1] == 0 or len(features) < self.cfg.models.min_history_days:
            return pd.Series(np.nan, index=features.index)
        self.pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("iso", IsolationForest(
                n_estimators=300,
                contamination=self.cfg.models.anomaly_contamination,
                random_state=self.cfg.models.random_state)),
        ])
        self.pipe.fit(X)
        raw = self.pipe.decision_function(X)      # higher = more normal
        # Map to 0-1 where 1 = most unusual.
        lo, hi = np.nanpercentile(raw, 2), np.nanpercentile(raw, 98)
        norm = 1.0 - np.clip((raw - lo) / max(1e-9, hi - lo), 0, 1)
        return pd.Series(norm, index=features.index)

    def explain(self, features: pd.DataFrame, date_idx) -> List[Dict[str, Any]]:
        """Which features are furthest from the person's own median today."""
        if not self.cols:
            return []
        sub = features[self.cols].astype(float)
        med = sub.median()
        mad = (sub - med).abs().median().replace(0, np.nan)
        row = sub.loc[date_idx]
        dev = ((row - med) / (mad * 1.4826)).abs().sort_values(ascending=False)
        out = []
        for feat, z in dev.head(5).items():
            if pd.isna(z):
                continue
            out.append({"feature": feat, "label": readable(feat),
                        "z": round(float(z), 2),
                        "value": (None if pd.isna(row[feat])
                                  else round(float(row[feat]), 2)),
                        "direction": "above" if row[feat] > med[feat] else "below"})
        return out


class Forecaster:
    """
    Predicts the mental score `h` days ahead from today's behaviour.

    The point isn't the number -- it's the feature importances, which tell the
    person which of their own habits actually precede their bad stretches.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Optional[Pipeline] = None
        self.cols: List[str] = []
        self.importances: List[Dict[str, Any]] = []
        self.cv_mae: Optional[float] = None

    @staticmethod
    def _lagged(features: pd.DataFrame, cols: List[str],
                lags: Tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
        frames = []
        for lag in lags:
            f = features[cols].shift(lag)
            f.columns = [f"{c}__lag{lag}" for c in cols]
            frames.append(f)
        # 3-day rolling means smooth single-day noise.
        roll = features[cols].rolling(3, min_periods=1).mean()
        roll.columns = [f"{c}__r3" for c in cols]
        frames.append(roll)
        return pd.concat(frames, axis=1)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> bool:
        _, cols = _matrix(features)
        self.cols = cols
        h = self.cfg.models.forecast_horizon_days
        if not cols or len(features) < self.cfg.models.min_history_days + h + 5:
            return False
        X = self._lagged(features, cols)
        y = target.shift(-h)                       # predict h days ahead
        mask = y.notna() & X.notna().any(axis=1)
        if mask.sum() < 20:
            return False
        self.model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("gb", GradientBoostingRegressor(
                n_estimators=250, max_depth=3, learning_rate=0.05,
                subsample=0.85, random_state=self.cfg.models.random_state)),
        ])
        Xm, ym = X[mask], y[mask]
        self.model.fit(Xm, ym)

        # Honest-ish error estimate: walk-forward on the last 25%.
        split = int(len(Xm) * 0.75)
        if split > 15 and len(Xm) - split >= 5:
            probe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("gb", GradientBoostingRegressor(
                    n_estimators=250, max_depth=3, learning_rate=0.05,
                    subsample=0.85,
                    random_state=self.cfg.models.random_state))])
            probe.fit(Xm.iloc[:split], ym.iloc[:split])
            pred = probe.predict(Xm.iloc[split:])
            self.cv_mae = float(np.mean(np.abs(pred - ym.iloc[split:].values)))

        gb = self.model.named_steps["gb"]
        imp = pd.Series(gb.feature_importances_, index=X.columns)
        # Collapse lag/rolling variants back to the underlying behaviour.
        base = imp.groupby(lambda c: c.split("__")[0]).sum().sort_values(
            ascending=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = features[cols].corrwith(target.shift(-h))
        self.importances = [
            {"feature": f, "label": readable(f),
             "importance": round(float(v), 4),
             "relationship": ("protective" if corr.get(f, 0) > 0.05
                              else "risk" if corr.get(f, 0) < -0.05
                              else "mixed"),
             "corr": round(float(corr.get(f, np.nan)), 3)
             if pd.notna(corr.get(f, np.nan)) else None}
            for f, v in base.head(12).items() if v > 0
        ]
        return True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model is None or not self.cols:
            return pd.Series(np.nan, index=features.index)
        X = self._lagged(features, self.cols)
        return pd.Series(self.model.predict(X), index=features.index).clip(0, 100)


class RiskModel:
    """P(the next few days are a low stretch). Turns alerts preventive."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Optional[Pipeline] = None
        self.cols: List[str] = []
        self.base_rate: float = 0.0

    def fit(self, features: pd.DataFrame, mental: pd.Series) -> bool:
        _, cols = _matrix(features)
        self.cols = cols
        if not cols or len(features) < self.cfg.models.min_history_days + 8:
            return False
        h = self.cfg.models.forecast_horizon_days
        # A "low stretch": the h-day forward mean falls into the person's
        # own bottom 25%.
        fwd = mental.shift(-1).rolling(h, min_periods=1).mean().shift(-(h - 1))
        thresh = mental.quantile(0.25)
        y = (fwd < thresh).astype(int)
        X = Forecaster._lagged(features, cols)
        mask = fwd.notna()
        if mask.sum() < 25 or y[mask].nunique() < 2:
            return False
        self.base_rate = float(y[mask].mean())
        self.model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("gb", GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.06,
                subsample=0.85, random_state=self.cfg.models.random_state)),
        ])
        self.model.fit(X[mask], y[mask])
        return True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model is None:
            return pd.Series(np.nan, index=features.index)
        X = Forecaster._lagged(features, self.cols)
        return pd.Series(self.model.predict_proba(X)[:, 1], index=features.index)


class DayArchetypes:
    """Clusters days into a handful of recognisable kinds, and names them."""

    NAME_RULES = [
        ("Grind day", {"work_overload": "high", "steps": "low"}),
        ("Restorative day", {"outdoor_min": "high", "work_overload": "low"}),
        ("Withdrawn day", {"social_connection": "low", "screen_min": "high"}),
        ("Active day", {"steps": "high", "workout_min": "high"}),
        ("Wired-and-tired day", {"night_digital_load": "high",
                                 "sleep_hours": "low"}),
        ("Social day", {"social_connection": "high"}),
    ]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Optional[Pipeline] = None
        self.names: Dict[int, str] = {}
        self.cols: List[str] = []

    def fit_predict(self, features: pd.DataFrame) -> pd.Series:
        X, cols = _matrix(features)
        self.cols = cols
        k = self.cfg.models.cluster_k
        if X.shape[1] == 0 or len(features) < max(k * 4, 16):
            return pd.Series(np.nan, index=features.index)
        self.model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("km", KMeans(n_clusters=k, n_init=10,
                          random_state=self.cfg.models.random_state)),
        ])
        labels = self.model.fit_predict(X)
        self._name(features, cols, labels)
        return pd.Series(labels, index=features.index)

    def _name(self, features: pd.DataFrame, cols: List[str],
              labels: np.ndarray) -> None:
        df = features[cols].copy()
        df["_c"] = labels
        overall = df[cols].median()
        used = set()
        for c in sorted(set(labels)):
            centroid = df[df["_c"] == c][cols].median()
            best, best_hits = None, 0
            for name, rules in self.NAME_RULES:
                if name in used:
                    continue
                hits = 0
                for feat, want in rules.items():
                    if feat not in centroid or pd.isna(centroid[feat]):
                        continue
                    hi = centroid[feat] > overall[feat]
                    if (want == "high") == hi:
                        hits += 1
                if hits > best_hits:
                    best, best_hits = name, hits
            if best and best_hits >= 1:
                used.add(best)
                self.names[int(c)] = best
            else:
                self.names[int(c)] = f"Pattern {c + 1}"


class TrendAnalyzer:
    """Direction, significance, and where the change actually started."""

    @staticmethod
    def mann_kendall(series: pd.Series) -> Dict[str, Any]:
        x = series.dropna().values
        n = len(x)
        if n < 8:
            return {"trend": "insufficient data", "p": None, "slope": None}
        s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
        var = n * (n - 1) * (2 * n + 5) / 18
        z = 0.0 if s == 0 else (s - np.sign(s)) / math.sqrt(var)
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        slopes = [(x[j] - x[i]) / (j - i)
                  for i in range(n - 1) for j in range(i + 1, n)]
        slope = float(np.median(slopes)) if slopes else 0.0
        if p >= 0.10:
            trend = "stable"
        else:
            trend = "improving" if slope > 0 else "declining"
        return {"trend": trend, "p": round(p, 4), "slope": round(slope, 3),
                "z": round(z, 2), "n": n}

    @staticmethod
    def ewma(series: pd.Series, span: int = 7) -> pd.Series:
        return series.ewm(span=span, min_periods=2).mean()

    @staticmethod
    def changepoint(series: pd.Series, window: int = 60,
                    min_seg: int = 5, min_t: float = 2.5) -> Optional[str]:
        """
        The most recent day the level actually shifted.

        Binary segmentation by maximum two-sample t-statistic over a trailing
        window. This answers "when did this stretch start?", which is the
        question that matters -- a CUSUM walked from the beginning just finds
        the oldest wobble in the record.
        """
        x = series.dropna().tail(window)
        if len(x) < 2 * min_seg + 2:
            return None
        vals = x.values.astype(float)
        best_t, best_i = 0.0, None
        for i in range(min_seg, len(vals) - min_seg):
            a, b = vals[:i], vals[i:]
            va, vb = a.var(ddof=1), b.var(ddof=1)
            se = math.sqrt(va / len(a) + vb / len(b))
            if se <= 1e-9:
                continue
            t = abs(b.mean() - a.mean()) / se
            if t > best_t:
                best_t, best_i = t, i
        if best_i is None or best_t < min_t:
            return None
        idx = x.index[best_i]
        return idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)

    @staticmethod
    def summarize(series: pd.Series, window: int = 14) -> Dict[str, Any]:
        recent = series.tail(window)
        prior = series.tail(window * 2).head(window)
        mk = TrendAnalyzer.mann_kendall(recent)
        delta = (recent.mean() - prior.mean()) if len(prior) >= 3 else np.nan
        return {
            **mk,
            "recent_mean": round(float(recent.mean()), 1)
            if recent.notna().any() else None,
            "prior_mean": round(float(prior.mean()), 1)
            if prior.notna().any() else None,
            "delta": round(float(delta), 1) if pd.notna(delta) else None,
            "changepoint": TrendAnalyzer.changepoint(series),
            "best_day": (series.idxmax().strftime("%Y-%m-%d")
                         if series.notna().any() else None),
            "worst_day": (series.idxmin().strftime("%Y-%m-%d")
                          if series.notna().any() else None),
        }


class ModelSuite:
    """Runs all four models and returns one bundle."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.anomaly = AnomalyDetector(cfg)
        self.forecaster = Forecaster(cfg)
        self.risk = RiskModel(cfg)
        self.archetypes = DayArchetypes(cfg)

    def run(self, features: pd.DataFrame, scores: pd.DataFrame) -> ModelOutputs:
        out = ModelOutputs(trained_on=len(features))
        if features.empty:
            out.notes.append("no features to model")
            return out

        out.anomaly = self.anomaly.fit_score(features)
        out.archetype = self.archetypes.fit_predict(features)
        out.archetype_names = self.archetypes.names

        mental = scores["mental"] if "mental" in scores else pd.Series(dtype=float)
        if not mental.empty:
            if self.forecaster.fit(features, mental):
                out.forecast = self.forecaster.predict(features)
                out.drivers = self.forecaster.importances
            else:
                out.notes.append(
                    f"forecaster needs ~{self.cfg.models.min_history_days + 8} "
                    "days of history; skipped")
            if self.risk.fit(features, mental):
                out.risk = self.risk.predict(features)
            else:
                out.notes.append("risk model needs more history; skipped")
        if self.forecaster.cv_mae is not None:
            out.notes.append(
                f"forecast holdout MAE {self.forecaster.cv_mae:.1f} points")
        return out
