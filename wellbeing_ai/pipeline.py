"""
The orchestrator. Ingest -> features -> scores -> models -> coach -> alerts.

`WellbeingSystem` is the one object the CLI, the scheduler and any UI talk to.
Analysis is cached per run so a daily job doesn't rebuild the feature matrix
four times.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .alerts import Alert, AlertEngine
from .config import Config
from .connectors import ingest_all
from .features import FeatureBuilder
from .genai import (CoachOutput, RecommendationEngine, build_payload, get_coach)
from .genai.recommendations import Recommendation
from .models import ModelOutputs, ModelSuite, Scorer, TrendAnalyzer
from .safety import ClinicalGuard
from .store import Store


@dataclass
class Analysis:
    features: pd.DataFrame
    scores: pd.DataFrame
    models: ModelOutputs
    coverage: Dict[str, float] = field(default_factory=dict)

    @property
    def latest_date(self) -> Optional[str]:
        if self.scores.empty:
            return None
        return self.scores.index[-1].strftime("%Y-%m-%d")


@dataclass
class Report:
    period: str                      # daily | weekly | monthly
    period_key: str
    date: str
    coach: CoachOutput
    recommendations: List[Recommendation]
    alerts: List[Alert]
    scores: Dict[str, Any]
    trend: Dict[str, Any]
    payload: Dict[str, Any]
    history: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period, "period_key": self.period_key,
            "date": self.date, "coach": self.coach.to_dict(),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "alerts": [a.to_dict() for a in self.alerts],
            "scores": self.scores, "trend": self.trend,
            "history": self.history,
            "disclaimer": ClinicalGuard.disclaimer(),
        }


class WellbeingSystem:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config.load()
        self.cfg.ensure_dirs()
        self.store = Store(self.cfg.db_path)
        self.scorer = Scorer(self.cfg)
        self.recs = RecommendationEngine(self.cfg)
        self.alerts = AlertEngine(self.cfg, self.store)
        self._analysis: Optional[Analysis] = None

    # ------------------------------------------------------------- ingest
    def ingest(self, root: Optional[Path] = None,
               only: Optional[List[str]] = None,
               force: bool = False) -> List:
        results = ingest_all(self.cfg, self.store, root, only, force=force)
        self._analysis = None
        if self.cfg.privacy.retention_days_detail:
            cutoff = (datetime.now()
                      - timedelta(days=self.cfg.privacy.retention_days_detail)
                      ).strftime("%Y-%m-%d")
            self.store.prune(cutoff)
        return results

    # ----------------------------------------------------------- analysis
    def analyze(self, force: bool = False) -> Analysis:
        if self._analysis is not None and not force:
            return self._analysis
        fb = FeatureBuilder(self.cfg, self.store)
        features = fb.build()
        if features.empty:
            self._analysis = Analysis(features, pd.DataFrame(), ModelOutputs())
            return self._analysis
        scores = self.scorer.score_frame(features)
        models = ModelSuite(self.cfg).run(features, scores)
        for col, series in [("anomaly", models.anomaly),
                            ("risk", models.risk),
                            ("forecast_mental", models.forecast)]:
            if len(series):
                scores[col] = series.reindex(scores.index)
        if len(models.archetype):
            scores["archetype"] = models.archetype.reindex(scores.index).map(
                lambda c: models.archetype_names.get(int(c), None)
                if pd.notna(c) else None)
        self._analysis = Analysis(features, scores, models, fb.coverage)
        self._persist_scores(self._analysis)
        return self._analysis

    def _persist_scores(self, an: Analysis) -> None:
        if an.scores.empty:
            return
        rows = []
        now = datetime.now().isoformat(timespec="seconds")
        for idx, r in an.scores.iterrows():
            date = idx.strftime("%Y-%m-%d")
            subs = {k.replace("sub_", ""): round(float(v), 1)
                    for k, v in r.items()
                    if str(k).startswith("sub_") and pd.notna(v)}
            rows.append({
                "date": date,
                "physical": float(r["physical"]), "mental": float(r["mental"]),
                "overall": float(r["overall"]),
                "subscores": json.dumps(subs), "drivers": None,
                "anomaly": _f(r.get("anomaly")), "risk": _f(r.get("risk")),
                "forecast_mental": _f(r.get("forecast_mental")),
                "archetype": (r.get("archetype")
                              if isinstance(r.get("archetype"), str) else None),
                "computed_at": now,
            })
        self.store.upsert_daily("scores", rows)

    # ------------------------------------------------------------ context
    def _context(self, an: Analysis) -> Dict[str, Any]:
        """Extra facts the rules engine needs that aren't single-day features."""
        ctx: Dict[str, Any] = {}
        s, f = an.scores, an.features
        if s.empty:
            return ctx
        mental = s["mental"]
        ctx["days_mental_low"] = self.alerts._streak(
            mental.tail(30), lambda v: v < 50)
        if "sleep_hours" in f:
            ctx["sleep_hours_7d"] = float(f["sleep_hours"].tail(7).mean())
        if "work_overload" in f:
            hi = f["work_overload"].tail(21) > 0.55
            ctx["sustained_strain_days"] = int(hi.sum())
        ctx["history_days"] = len(f)
        return ctx

    def _texts_for_safety(self, days: int = 5) -> List[str]:
        """Recent free text, only if the person chose to store it."""
        if not self.cfg.privacy.store_raw_text:
            return []
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        out: List[str] = []
        for table in ("messages", "notes", "searches"):
            df = self.store.read_df(
                f"SELECT text FROM {table} WHERE date >= ?", (since,))
            if not df.empty and "text" in df:
                out += [t for t in df["text"].dropna().tolist() if t]
        return out

    # ------------------------------------------------------------ reports
    def daily_report(self, deliver: bool = False) -> Optional[Report]:
        an = self.analyze()
        if an.scores.empty:
            return None
        date = an.latest_date
        row = an.features.iloc[-1]
        ctx = self._context(an)

        # Guard every recommendation before it is shown.
        recs = [_guard(r) for r in self.recs.generate(row, ctx, limit=5)]

        alerts = self.alerts.evaluate(an.scores, an.features, an.models,
                                      self._texts_for_safety())
        trend = TrendAnalyzer.summarize(an.scores["mental"], window=14)

        payload = build_payload(
            self.cfg, "daily", date, an.scores, an.features,
            self.scorer.score_row(row, date).drivers, recs, trend,
            an.models.drivers,
            anomaly=_f(an.scores["anomaly"].iloc[-1])
            if "anomaly" in an.scores else None,
            risk=_f(an.scores["risk"].iloc[-1]) if "risk" in an.scores else None,
            archetype=an.scores["archetype"].iloc[-1]
            if "archetype" in an.scores else None,
            history=self._history_block(an, "daily"),
        )
        # If safety has fired, the coach steps aside entirely.
        support = [a for a in alerts if a.level == "support"]
        if support:
            coach = CoachOutput(
                headline="Let's pause the tracking for a moment",
                assessment=support[0].body, source="safety")
            recs = []
        else:
            coach = get_coach(self.cfg).write(payload, recs)

        rep = Report("daily", date, date, coach, recs, alerts,
                     _score_block(an.scores.iloc[-1]), trend, payload,
                     self._history_block(an, "daily"))
        self._save(rep)
        if alerts:
            self.alerts.persist(alerts)
            if deliver:
                self.alerts.deliver(alerts)
        return rep

    def period_report(self, period: str = "weekly") -> Optional[Report]:
        an = self.analyze()
        if an.scores.empty:
            return None
        days = 7 if period == "weekly" else 30
        window = an.scores.tail(days)
        prior = an.scores.tail(days * 2).head(days)
        date = an.latest_date
        key = (window.index[0].strftime("%Y-W%V") if period == "weekly"
               else window.index[-1].strftime("%Y-%m"))

        feat_window = an.features.tail(days)
        row = feat_window.mean(numeric_only=True)
        ctx = self._context(an)
        recs = [_guard(r) for r in self.recs.generate(row, ctx, limit=6)]
        trend = TrendAnalyzer.summarize(an.scores["mental"], window=days)

        hist = self._history_block(an, period)
        payload = build_payload(
            self.cfg, period, date, an.scores, an.features,
            self.scorer.score_row(row, date).drivers, recs, trend,
            an.models.drivers, history=hist)
        payload["period_summary"] = {
            "days": days,
            "mental_mean": round(float(window["mental"].mean()), 1),
            "physical_mean": round(float(window["physical"].mean()), 1),
            "mental_prior": round(float(prior["mental"].mean()), 1)
            if len(prior) else None,
            "physical_prior": round(float(prior["physical"].mean()), 1)
            if len(prior) else None,
            "best_day": window["mental"].idxmax().strftime("%Y-%m-%d"),
            "worst_day": window["mental"].idxmin().strftime("%Y-%m-%d"),
            "day_types": (window["archetype"].value_counts().to_dict()
                          if "archetype" in window else {}),
        }
        coach = get_coach(self.cfg).write(payload, recs)
        rep = Report(period, key, date, coach, recs, [],
                     _score_block(window.mean(numeric_only=True)), trend,
                     payload, hist)
        self._save(rep)
        return rep

    # ------------------------------------------------------------ history
    def _history_block(self, an: Analysis, period: str) -> Dict[str, Any]:
        """Progress-or-decline, which is the whole point of tracking."""
        s = an.scores
        if s.empty:
            return {}

        def window_stats(days: int) -> Dict[str, Any]:
            w = s.tail(days)
            p = s.tail(days * 2).head(days)
            out = {"days": len(w),
                   "mental": round(float(w["mental"].mean()), 1),
                   "physical": round(float(w["physical"].mean()), 1)}
            if len(p) >= max(3, days // 2):
                out["mental_change"] = round(
                    float(w["mental"].mean() - p["mental"].mean()), 1)
                out["physical_change"] = round(
                    float(w["physical"].mean() - p["physical"].mean()), 1)
            return out

        hist = {"last_7d": window_stats(7), "last_30d": window_stats(30)}
        if len(s) >= 60:
            hist["last_90d"] = window_stats(90)
        # Monthly series -- the long view.
        monthly = s[["mental", "physical"]].resample("ME").mean().round(1)
        hist["monthly"] = [
            {"month": idx.strftime("%Y-%m"), "mental": float(r["mental"]),
             "physical": float(r["physical"])}
            for idx, r in monthly.iterrows() if pd.notna(r["mental"])
        ]
        hist["direction"] = _direction(hist["last_7d"].get("mental_change"))
        return hist

    # --------------------------------------------------------------- save
    def _save(self, rep: Report) -> None:
        self.store.insert_events("reports", [{
            "ts": datetime.now().isoformat(timespec="seconds"),
            "period": rep.period, "period_key": rep.period_key,
            "headline": rep.coach.headline,
            "body": rep.coach.as_text(),
            "payload": json.dumps(rep.to_dict(), default=str),
            "path": None,
        }])

    # ------------------------------------------------------------ run all
    def run_daily(self, deliver: bool = True,
                  html: bool = True) -> Dict[str, Any]:
        from .reports.html import render_dashboard, render_report
        res: Dict[str, Any] = {"ingested": []}
        res["ingested"] = [str(r) for r in self.ingest()]
        an = self.analyze(force=True)
        if an.scores.empty:
            return {**res, "error": "no data to analyse"}
        rep = self.daily_report(deliver=deliver)
        out = Path(self.cfg.out_dir)
        paths = {}
        if rep and html:
            p = out / f"daily_{rep.date}.html"
            p.write_text(render_report(self.cfg, rep, an), encoding="utf-8")
            paths["daily_html"] = str(p)
            d = out / "dashboard.html"
            d.write_text(render_dashboard(self.cfg, an, rep), encoding="utf-8")
            paths["dashboard"] = str(d)
        if rep:
            j = out / f"daily_{rep.date}.json"
            j.write_text(json.dumps(rep.to_dict(), indent=2, default=str))
            paths["daily_json"] = str(j)
        # Weekly on Mondays, monthly on the 1st.
        today = datetime.now()
        if today.weekday() == 0:
            wk = self.period_report("weekly")
            if wk and html:
                p = out / f"weekly_{wk.period_key}.html"
                p.write_text(render_report(self.cfg, wk, an), encoding="utf-8")
                paths["weekly_html"] = str(p)
        if today.day == 1:
            mo = self.period_report("monthly")
            if mo and html:
                p = out / f"monthly_{mo.period_key}.html"
                p.write_text(render_report(self.cfg, mo, an), encoding="utf-8")
                paths["monthly_html"] = str(p)
        res.update({"date": rep.date if rep else None,
                    "scores": rep.scores if rep else None,
                    "alerts": len(rep.alerts) if rep else 0,
                    "paths": paths})
        return res


# ------------------------------------------------------------------ helpers
def _f(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _guard(rec: Recommendation) -> Recommendation:
    d = ClinicalGuard.sanitize_recommendation(rec.to_dict())
    return Recommendation(
        id=d["id"], category=d["category"], title=d["title"],
        action=d["action"], why=d["why"], effort=d.get("effort", "small"),
        when=d.get("when", "today"), priority=d.get("priority", 1.0),
        evidence=d.get("evidence", {}))


def _score_block(row: pd.Series) -> Dict[str, Any]:
    out = {}
    for k in ("physical", "mental", "overall", "confidence", "anomaly", "risk",
              "forecast_mental"):
        if k in row.index and pd.notna(row[k]):
            out[k] = round(float(row[k]), 1 if k != "confidence" else 2)
    for k, v in row.items():
        if str(k).startswith("sub_") and pd.notna(v):
            out[str(k)] = round(float(v), 1)
    if "archetype" in row.index and isinstance(row["archetype"], str):
        out["archetype"] = row["archetype"]
    return out


def _direction(change: Optional[float]) -> str:
    if change is None:
        return "unknown"
    if change >= 3:
        return "improving"
    if change <= -3:
        return "declining"
    return "steady"
