"""
End-to-end checks.

These aren't unit tests for their own sake -- each one guards a claim the
system makes to the person using it. If the privacy test fails, the system is
lying about what leaves the machine. If the arc test fails, it can't see a
six-week decline, which is the only thing it exists to do.

Run:  python tests/test_pipeline.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from wellbeing_ai.config import Config
from wellbeing_ai.features import FeatureBuilder
from wellbeing_ai.genai import RecommendationEngine, TemplateCoach, build_payload
from wellbeing_ai.models import ModelSuite, Scorer, TrendAnalyzer
from wellbeing_ai.pipeline import WellbeingSystem
from wellbeing_ai.safety import ClinicalGuard, CrisisDetector
from wellbeing_ai.store import Store
from wellbeing_ai.utils import privacy as pv
from wellbeing_ai.utils import textanalysis as ta

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "ok  " if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
def test_privacy_primitives():
    print("\nPrivacy primitives")
    h1 = pv.hash_id("Amma", "salt")
    h2 = pv.hash_id("amma ", "salt")
    h3 = pv.hash_id("Amma", "other-salt")
    check("contact hashing is stable and case-insensitive", h1 == h2)
    check("hashing is salted", h1 != h3)
    check("hash is not reversible to the name", "amma" not in h1.lower())

    raw = ("Call me on +919876543210 or vijay@example.com, card "
           "4111 1111 1111 1111, see https://x.com/abc")
    red = pv.redact(raw)
    for leak in ("9876543210", "vijay@example.com", "4111", "x.com"):
        check(f"redaction removes {leak!r}", leak not in red)

    kws = pv.keywords_only("I am completely exhausted and overwhelmed by work")
    check("keywords_only drops stopwords", "and" not in kws and "exhausted" in kws)


def test_text_analysis():
    print("\nText analysis")
    check("positive sentiment", ta.sentiment("had a great day, feeling proud") > 0.2)
    check("negative sentiment",
          ta.sentiment("completely exhausted and overwhelmed") < -0.2)
    check("negation handled",
          ta.sentiment("not happy at all") < ta.sentiment("happy"))
    check("empty text is neutral", ta.sentiment("") == 0.0)
    check("health search categorised",
          ta.categorize("chest tightness symptoms") == "physical_health")
    check("mental-health search categorised",
          ta.categorize("how to stop overthinking") == "mental_health")
    check("pharmacy place categorised",
          ta.categorize_place("Apollo Pharmacy Koramangala") == "pharmacy")
    check("temple categorised as worship",
          ta.categorize_place("Sri Someshwara Temple") == "worship")


def test_clinical_guard():
    print("\nClinical guard")
    must_block = [
        "Take 2 tablets of melatonin before bed",
        "a dose of 300mg magnesium nightly",
        "Your doctor may prescribe an SSRI",
        "Start taking a supplement for this",
        "You have depression",
        "Consider sleeping pills",
    ]
    must_pass = [
        "Caffeine is around 320mg/day and sleep latency has stretched",
        "Protein averaged 45g/day",
        "Book a GP appointment and take a printout of this report",
        "Your mood scores have been below 45 for nine days",
        "Drink 2500 ml of water across the morning",
    ]
    check("blocks medication advice",
          all(ClinicalGuard.violates(t) for t in must_block))
    check("allows ordinary nutrition figures",
          not any(ClinicalGuard.violates(t) for t in must_pass))
    rec = {"id": "x", "category": "diet", "title": "Try melatonin",
           "action": "Take 5mg melatonin nightly", "why": "sleep is poor"}
    out = ClinicalGuard.sanitize_recommendation(rec)
    check("sanitiser reroutes to a clinician",
          out["category"] == "doctor_consult" and out.get("guarded") is True)


def test_crisis_detector():
    print("\nCrisis detection")
    d = CrisisDetector("IN")
    quiet = d.evaluate(texts=["busy day", "tired but ok"], mental_score=70)
    check("does not fire on ordinary text", not quiet.triggered)

    urgent = d.evaluate(texts=["i don't want to be here any more"],
                        mental_score=30)
    check("fires on serious distress language", urgent.triggered
          and urgent.level == "urgent")
    check("surfaces India helplines",
          any("14416" in h["contact"] for h in urgent.helplines))

    sustained = d.evaluate(texts=[], mental_score=33, days_below_40=6)
    check("fires on a sustained low run", sustained.triggered)

    msg = d.support_message("Vijay", urgent)
    check("support message names a helpline, not a tip", "14416" in msg
          and "exercise" not in msg.lower())
    check("support message has no score talk", "score" not in msg.lower())


def test_pipeline_arc(ws: WellbeingSystem):
    print("\nPipeline and the six-month arc")
    an = ws.analyze(force=True)
    check("features built", not an.features.empty,
          f"{an.features.shape[0]} days x {an.features.shape[1]} features")
    check("scores produced", not an.scores.empty)
    s = an.scores
    check("scores stay inside 0-100",
          bool(s[["physical", "mental", "overall"]].min().min() >= 0
               and s[["physical", "mental", "overall"]].max().max() <= 100))
    check("no NaN scores", not s[["physical", "mental"]].isna().any().any())

    # The synthetic person declines around days 95-138 and recovers after 150.
    baseline = s["mental"].iloc[20:90].mean()
    trough = s["mental"].iloc[125:138].mean()
    recovery = s["mental"].iloc[160:].mean()
    check("decline detected (trough well below baseline)",
          trough < baseline - 12, f"baseline {baseline:.0f} -> trough {trough:.0f}")
    check("recovery detected", recovery > trough + 12,
          f"trough {trough:.0f} -> recovery {recovery:.0f}")

    cp = TrendAnalyzer.changepoint(s["mental"].iloc[:140])
    check("changepoint lands inside the decline", cp is not None,
          f"changepoint {cp}")

    mk = TrendAnalyzer.mann_kendall(s["mental"].iloc[95:135])
    check("trend test calls the decline", mk["trend"] == "declining",
          f"{mk['trend']} p={mk['p']}")


def test_models(ws: WellbeingSystem):
    print("\nModels")
    an = ws.analyze()
    m = an.models
    check("anomaly scores produced", len(m.anomaly) > 0
          and m.anomaly.notna().any())
    check("anomaly is highest during the bad stretch",
          m.anomaly.iloc[120:140].mean() > m.anomaly.iloc[20:90].mean(),
          f"{m.anomaly.iloc[120:140].mean():.2f} vs "
          f"{m.anomaly.iloc[20:90].mean():.2f}")
    check("forecaster trained", len(m.forecast) > 0 and m.forecast.notna().any())
    check("risk model trained", len(m.risk) > 0 and m.risk.notna().any())
    check("risk rises before the trough",
          m.risk.iloc[100:130].mean() > m.risk.iloc[20:80].mean(),
          f"{m.risk.iloc[100:130].mean():.2f} vs "
          f"{m.risk.iloc[20:80].mean():.2f}")
    check("learned drivers named", len(m.drivers) >= 3,
          ", ".join(d["label"] for d in m.drivers[:3]))
    check("day archetypes named", len(m.archetype_names) >= 3,
          ", ".join(m.archetype_names.values()))


def test_bad_day_behaviour(ws: WellbeingSystem):
    """The whole point: on a genuinely bad day, does it say the right things?"""
    print("\nBehaviour on a bad day")
    an = ws.analyze()
    worst_i = int(an.scores["mental"].values.argmin())
    upto = an.scores.iloc[:worst_i + 1]
    feats = an.features.iloc[:worst_i + 1]
    row = feats.iloc[-1]

    alerts = ws.alerts.evaluate(upto, feats, an.models)
    check("alerts fire on the worst day", len(alerts) > 0,
          f"{len(alerts)}: {', '.join(a.kind for a in alerts)}")
    check("at least one is amber or red",
          any(a.level in ("amber", "red", "support") for a in alerts))

    recs = RecommendationEngine(ws.cfg).generate(
        row, {"days_mental_low": 8, "sustained_strain_days": 15}, limit=6)
    check("recommendations generated", len(recs) >= 3,
          ", ".join(r.category for r in recs))
    cats = {r.category for r in recs}
    check("advice spans more than one area", len(cats) >= 3)
    check("a clinician route is offered on a sustained low",
          "doctor_consult" in cats)
    check("no recommendation trips the guard",
          not any(ClinicalGuard.violates(
              f"{r.title} {r.action} {r.why}") for r in recs))
    check("every recommendation cites evidence",
          all(r.evidence for r in recs))
    check("no unfilled template placeholders survive",
          not any("{" in r.why for r in recs))


def test_payload_privacy(ws: WellbeingSystem):
    """Nothing identifying may reach the model. This one matters most."""
    print("\nWhat would be sent to the LLM")
    ws.cfg.llm.enabled = False
    rep = ws.daily_report(deliver=False)
    check("daily report produced", rep is not None)
    blob = json.dumps(rep.payload, default=str).lower()

    # Raw source text from the demo data that must never appear.
    forbidden = [
        "amma", "rohit", "priya",                     # contact names
        "apollo pharmacy", "manipal", "cubbon",       # place names
        "should i quit my job", "cant sleep",         # search queries
        "@", "9876",                                  # contact details
        "toit", "swiggy",                             # merchants
    ]
    leaked = [f for f in forbidden if f in blob]
    check("no contact names, places, queries or merchants in payload",
          not leaked, f"leaked: {leaked}" if leaked else "")
    check("payload carries scores", "scores" in rep.payload)
    check("payload carries named drivers", "drivers_today" in rep.payload)
    check("payload is small enough to eyeball", len(blob) < 12000,
          f"{len(blob)} chars")

    coach = TemplateCoach().write(rep.payload, rep.recommendations)
    check("offline coach writes a headline", bool(coach.headline))
    check("offline coach writes an assessment", len(coach.assessment) > 40)
    check("offline narrative passes the guard",
          not ClinicalGuard.violates(coach.as_text()))


def test_reports(ws: WellbeingSystem):
    print("\nReports")
    from wellbeing_ai.reports import render_dashboard, render_report, render_text
    an = ws.analyze()
    rep = ws.daily_report(deliver=False)
    txt = render_text(rep, an, ws.cfg)
    check("text report renders", len(txt) > 500)
    # The disclaimer is word-wrapped for the terminal, so compare on
    # normalised whitespace rather than the raw string.
    flat = " ".join(txt.split())
    check("text report carries the disclaimer",
          "not a medical device" in flat and "does not diagnose" in flat)

    html = render_report(ws.cfg, rep, an)
    check("HTML report renders", len(html) > 3000)
    check("HTML is self-contained (no external fetches)",
          "cdn" not in html.lower()
          and html.count("http://") == html.count("http://www.w3.org"))
    check("HTML has a chart", "<svg" in html)
    check("HTML has a table view for accessibility", "<table" in html)
    check("HTML supports dark mode", "prefers-color-scheme" in html)

    dash = render_dashboard(ws.cfg, an, rep)
    check("dashboard renders", len(dash) > 3000)

    wk = ws.period_report("weekly")
    mo = ws.period_report("monthly")
    check("weekly report produced", wk is not None and bool(wk.coach.headline))
    check("monthly report produced", mo is not None and bool(mo.coach.headline))
    check("weekly report has history", bool(wk.history.get("last_7d")))
    check("monthly report covers 30 days",
          mo.payload.get("period_summary", {}).get("days") == 30)


def test_history_and_persistence(ws: WellbeingSystem):
    print("\nPersistence")
    counts = ws.store.counts()
    check("scores persisted", counts.get("scores", 0) > 100,
          f"{counts.get('scores')} days")
    check("reports persisted", counts.get("reports", 0) > 0)
    saved = ws.store.read_df(
        "SELECT * FROM scores ORDER BY date DESC LIMIT 1")
    check("saved score row has subscores JSON",
          not saved.empty and bool(json.loads(saved["subscores"].iloc[0])))

    an = ws.analyze()
    hist = ws._history_block(an, "daily")
    check("history has 7/30 day windows",
          "last_7d" in hist and "last_30d" in hist)
    check("history has a monthly series", len(hist.get("monthly", [])) >= 5,
          f"{len(hist.get('monthly', []))} months")
    check("history states a direction",
          hist.get("direction") in ("improving", "declining", "steady"))


def test_missing_data_resilience():
    """A new user has three days of one source. It must not crash or lie."""
    print("\nResilience with almost no data")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config()
        cfg.db_path = str(Path(tmp) / "t.sqlite")
        cfg.raw_dir = str(Path(tmp) / "raw")
        cfg.out_dir = str(Path(tmp) / "out")
        cfg.llm.enabled = False
        cfg.ensure_dirs()
        ws = WellbeingSystem(cfg)
        base = datetime.now() - timedelta(days=4)
        ws.store.upsert_daily("sleep", [
            {"date": (base + timedelta(days=i)).strftime("%Y-%m-%d"),
             "duration_min": 420 + i * 10, "source": "t"} for i in range(4)])
        an = ws.analyze(force=True)
        check("analysis survives 4 days of one source", not an.scores.empty)
        check("confidence is honestly low",
              float(an.scores["confidence"].iloc[-1]) < 0.45,
              f"confidence {an.scores['confidence'].iloc[-1]:.2f}")
        check("models decline to train rather than guess",
              any("skipped" in n or "needs" in n for n in an.models.notes)
              or an.models.forecast.isna().all())
        rep = ws.daily_report(deliver=False)
        check("a report is still produced", rep is not None)
        check("no alerts invented from nothing",
              all(a.level != "red" for a in rep.alerts))

        empty = WellbeingSystem(Config(
            db_path=str(Path(tmp) / "empty.sqlite"),
            raw_dir=str(Path(tmp) / "raw"), out_dir=str(Path(tmp) / "out")))
        check("empty database returns None rather than crashing",
              empty.daily_report(deliver=False) is None)


def test_ingest_is_idempotent():
    """
    Regression: append-only tables used to double on every re-ingest, which
    silently doubled calorie and sugar intake and made the nutrition score
    meaningless for anyone who ran `ingest` twice.
    """
    print("\nIngest idempotency")
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        raw.mkdir()
        cfg = Config()
        cfg.db_path = str(Path(tmp) / "t.sqlite")
        cfg.raw_dir = str(raw)
        cfg.out_dir = str(Path(tmp) / "out")
        cfg.llm.enabled = False
        cfg.ensure_dirs()
        ws = WellbeingSystem(cfg)

        head = ("date,meal,item,calories,protein,carbs,fat,fiber,sugar,"
                "sodium,caffeine,alcohol,water\n")
        rows = "".join(
            f"2026-09-{d:02d},{meal},oats,400,20,50,10,8,6,300,50,0,600\n"
            for d in range(1, 6) for meal in ("breakfast", "lunch", "dinner"))
        food = raw / "food_log.csv"
        food.write_text(head + rows)

        ws.ingest()
        first = ws.store.counts()["food"]
        check("first ingest reads the file", first == 15, f"{first} rows")

        ws.ingest()
        second = ws.store.counts()["food"]
        check("re-running ingest adds nothing", second == first,
              f"{first} -> {second}")

        # A file that has genuinely grown must still be picked up.
        food.write_text(head + rows
                        + "2026-09-06,breakfast,idli,350,12,60,6,4,5,400,0,0,600\n")
        ws.ingest()
        third = ws.store.counts()["food"]
        check("appended rows are ingested", third == first + 1,
              f"{second} -> {third}")

        ws.ingest(force=True)
        forced = ws.store.counts()["food"]
        check("even --force doesn't duplicate (natural key holds)",
              forced == third, f"{third} -> {forced}")


def test_scoring_is_personal():
    """A consistent 5-hour sleeper should not be scored like someone who fell
    from 8 hours to 5. Baselines are personal or they are worthless."""
    print("\nScores are relative to the person")
    cfg = Config()
    cfg.llm.enabled = False
    idx = pd.date_range("2026-01-01", periods=40, freq="D")

    steady = pd.DataFrame({"sleep_hours": [6.2] * 40,
                           "steps": [7000] * 40}, index=idx)
    dropped = pd.DataFrame({"sleep_hours": [8.0] * 30 + [6.2] * 10,
                            "steps": [7000] * 40}, index=idx)
    from wellbeing_ai.features.builder import robust_z
    z_steady = robust_z(steady["sleep_hours"], 28).iloc[-1]
    z_drop = robust_z(dropped["sleep_hours"], 28).iloc[-1]
    check("a stable sleeper gets no baseline penalty",
          pd.isna(z_steady) or abs(z_steady) < 1.0, f"z={z_steady}")
    check("a real drop registers as a departure",
          pd.notna(z_drop) and z_drop < -1.5, f"z={z_drop}")


# --------------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("  wellbeing_ai -- end-to-end checks")
    print("=" * 72)

    cfg = Config.load()
    cfg.llm.enabled = False          # tests never hit the network
    ws = WellbeingSystem(cfg)
    if ws.store.counts().get("sleep", 0) == 0:
        print("\nNo data found -- generating the demo set first.")
        import generate_demo
        generate_demo.main()
        ws.ingest()

    test_privacy_primitives()
    test_text_analysis()
    test_clinical_guard()
    test_crisis_detector()
    test_pipeline_arc(ws)
    test_models(ws)
    test_bad_day_behaviour(ws)
    test_payload_privacy(ws)
    test_reports(ws)
    test_history_and_persistence(ws)
    test_missing_data_resilience()
    test_ingest_is_idempotent()
    test_scoring_is_personal()

    print("\n" + "=" * 72)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"    FAILED: {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
