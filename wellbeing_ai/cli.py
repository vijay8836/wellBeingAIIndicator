"""Command-line interface. `python -m wellbeing_ai --help`."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from .config import Config, PROJECT_ROOT
from .pipeline import WellbeingSystem
from .reports import render_dashboard, render_report, render_text
from .safety import ClinicalGuard


def _sys(args) -> WellbeingSystem:
    cfg = Config.load(args.config)
    if getattr(args, "no_llm", False):
        cfg.llm.enabled = False
    if getattr(args, "name", None):
        cfg.profile.name = args.name
    return WellbeingSystem(cfg)


# ------------------------------------------------------------------ commands
def cmd_init(args) -> int:
    path = Path(args.config or PROJECT_ROOT / "config" / "config.yaml")
    cfg = Config.load(path)
    cfg.ensure_dirs()
    cfg.save(path)
    print(f"Config written to {path}")
    print(f"Put your data exports in: {cfg.raw_dir}")
    print(f"Database:                 {cfg.db_path}")
    print(f"Reports:                  {cfg.out_dir}")
    print("\nEvery source is consent-gated in config.yaml under `consent:`.")
    print("Set ANTHROPIC_API_KEY to turn on the Claude-written narrative;")
    print("without it the system writes reports locally instead.")
    return 0


def cmd_ingest(args) -> int:
    ws = _sys(args)
    if args.reset:
        ws.store.forget_files()
        print("Ingest history cleared -- every file will be re-read.")
    results = ws.ingest(Path(args.path) if args.path else None,
                        only=args.only.split(",") if args.only else None,
                        force=args.force)
    for r in results:
        print(" ", r)
    start, end = ws.store.date_range()
    print(f"\nData spans {start} .. {end}")
    return 0


def cmd_analyze(args) -> int:
    ws = _sys(args)
    an = ws.analyze(force=True)
    if an.scores.empty:
        print("No data to analyse. Run `ingest` first.")
        return 1
    s = an.scores
    print(f"Days analysed : {len(s)}")
    print(f"Features      : {an.features.shape[1]}")
    print(f"Mental  (7d)  : {s['mental'].tail(7).mean():.1f}")
    print(f"Physical(7d)  : {s['physical'].tail(7).mean():.1f}")
    if an.models.archetype_names:
        print(f"Day types     : "
              f"{', '.join(an.models.archetype_names.values())}")
    for n in an.models.notes:
        print(f"  note: {n}")
    if an.models.drivers:
        print("\nStrongest predictors of your next few days:")
        for d in an.models.drivers[:8]:
            print(f"  {d['relationship']:<11} {d['label']}")
    low = sorted([(v, k) for k, v in an.coverage.items() if v < 0.4])[:8]
    if low:
        print("\nThin coverage (these features are mostly missing):")
        for v, k in low:
            print(f"  {k:<28} {v * 100:.0f}%")
    return 0


def cmd_report(args) -> int:
    ws = _sys(args)
    period = args.period
    rep = (ws.daily_report(deliver=args.notify) if period == "daily"
           else ws.period_report(period))
    if rep is None:
        print("No data to report on. Run `ingest` first.")
        return 1
    an = ws.analyze()
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2, default=str))
        return 0
    print(render_text(rep, an, ws.cfg))
    if args.html:
        out = Path(ws.cfg.out_dir) / f"{period}_{rep.period_key}.html"
        out.write_text(render_report(ws.cfg, rep, an), encoding="utf-8")
        print(f"\nHTML report: {out}")
    return 0


def cmd_dashboard(args) -> int:
    ws = _sys(args)
    an = ws.analyze(force=True)
    if an.scores.empty:
        print("No data. Run `ingest` first.")
        return 1
    rep = ws.daily_report(deliver=False)
    out = Path(args.out) if args.out else Path(ws.cfg.out_dir) / "dashboard.html"
    out.write_text(render_dashboard(ws.cfg, an, rep), encoding="utf-8")
    print(f"Dashboard: {out}")
    return 0


def cmd_alerts(args) -> int:
    ws = _sys(args)
    an = ws.analyze(force=True)
    if an.scores.empty:
        print("No data.")
        return 1
    alerts = ws.alerts.evaluate(an.scores, an.features, an.models,
                                ws._texts_for_safety())
    if not alerts:
        print("Nothing worth flagging today.")
        return 0
    for a in alerts:
        print(f"[{a.level.upper():<7}] {a.title}")
        print(f"          {a.body}\n")
    if args.notify:
        ws.alerts.persist(alerts)
        print(ws.alerts.deliver(alerts))
    return 0


def cmd_run(args) -> int:
    ws = _sys(args)
    res = ws.run_daily(deliver=not args.no_notify, html=True)
    if "error" in res:
        print(res["error"])
        return 1
    print(f"{res['date']}  mental {res['scores'].get('mental')}  "
          f"physical {res['scores'].get('physical')}  "
          f"alerts {res['alerts']}")
    for k, v in res.get("paths", {}).items():
        print(f"  {k:<14} {v}")
    return 0


def cmd_schedule(args) -> int:
    from .scheduler import install, run_forever
    if args.install:
        print(install(args.time))
        return 0
    print(f"Running scheduler in the foreground (daily at {args.time}). "
          "Ctrl-C to stop.")
    run_forever(args.time, Config.load(args.config))
    return 0


def cmd_demo(args) -> int:
    sys.path.insert(0, str(PROJECT_ROOT))
    from tests import generate_demo
    print("Generating synthetic data...")
    generate_demo.main()
    ws = _sys(args)
    print("\nIngesting...")
    for r in ws.ingest():
        print(" ", r)
    print("\nAnalysing...")
    an = ws.analyze(force=True)
    rep = ws.daily_report(deliver=False)
    print()
    print(render_text(rep, an, ws.cfg))
    out = Path(ws.cfg.out_dir)
    (out / "dashboard.html").write_text(render_dashboard(ws.cfg, an, rep),
                                        encoding="utf-8")
    (out / f"daily_{rep.date}.html").write_text(render_report(ws.cfg, rep, an),
                                                encoding="utf-8")
    wk = ws.period_report("weekly")
    if wk:
        (out / f"weekly_{wk.period_key}.html").write_text(
            render_report(ws.cfg, wk, an), encoding="utf-8")
    mo = ws.period_report("monthly")
    if mo:
        (out / f"monthly_{mo.period_key}.html").write_text(
            render_report(ws.cfg, mo, an), encoding="utf-8")
    print(f"\nReports written to {out}")
    return 0


def cmd_payload(args) -> int:
    """Print exactly what would be sent to the LLM, and send nothing."""
    ws = _sys(args)
    ws.cfg.llm.enabled = False
    an = ws.analyze(force=True)
    rep = ws.daily_report(deliver=False)
    if rep is None:
        print("No data.")
        return 1
    print(json.dumps(rep.payload, indent=2, default=str))
    return 0


def cmd_privacy(args) -> int:
    cfg = Config.load(args.config)
    ws = WellbeingSystem(cfg)
    print("Consent -- sources currently enabled:")
    for k in cfg.consent.enabled():
        print(f"  on   {k}")
    off = [k for k in vars(cfg.consent) if not cfg.consent.is_on(k)]
    for k in off:
        print(f"  off  {k}")
    p = cfg.privacy
    print(f"\nStore raw text        : {p.store_raw_text}")
    print(f"Hash contacts         : {p.hash_contacts}")
    print(f"Coarse location only  : {p.coarse_location_only}")
    print(f"Send raw text to LLM  : {p.llm_send_raw_text}")
    print(f"Detail retention      : {p.retention_days_detail} days")
    print(f"\nRows held locally: {json.dumps(ws.store.counts(), indent=2)}")
    if args.forget_before:
        removed = ws.store.prune(args.forget_before)
        print(f"\nDeleted rows before {args.forget_before}: {removed}")
    return 0


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="wellbeing_ai",
        description="A local-first wellbeing tracker. "
                    + ClinicalGuard.disclaimer())
    p.add_argument("--config", help="path to config.yaml")
    p.add_argument("--name", help="override the display name")
    p.add_argument("--no-llm", action="store_true",
                   help="never call the cloud model; write reports locally")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create config and folders")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("ingest", help="read export files into the local store")
    s.add_argument("--path", help="folder of exports (default: data/raw)")
    s.add_argument("--only", help="comma-separated connector names")
    s.add_argument("--force", action="store_true",
                   help="re-read files even if they haven't changed")
    s.add_argument("--reset", action="store_true",
                   help="forget ingest history entirely, then re-read")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("analyze", help="build features, scores and models")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("report", help="daily / weekly / monthly report")
    s.add_argument("period", nargs="?", default="daily",
                   choices=["daily", "weekly", "monthly"])
    s.add_argument("--html", action="store_true", help="also write HTML")
    s.add_argument("--json", action="store_true", help="print JSON instead")
    s.add_argument("--notify", action="store_true", help="send notifications")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("dashboard", help="write the full HTML dashboard")
    s.add_argument("--out", help="output path")
    s.set_defaults(fn=cmd_dashboard)

    s = sub.add_parser("alerts", help="evaluate alerts now")
    s.add_argument("--notify", action="store_true")
    s.set_defaults(fn=cmd_alerts)

    s = sub.add_parser("run", help="the full daily job: ingest, score, report")
    s.add_argument("--no-notify", action="store_true")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("schedule", help="run daily, or install a system job")
    s.add_argument("--time", default="08:00", help="HH:MM local time")
    s.add_argument("--install", action="store_true",
                   help="install a cron/launchd/Task Scheduler entry")
    s.set_defaults(fn=cmd_schedule)

    s = sub.add_parser("demo", help="generate synthetic data and run end to end")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("payload", help="show exactly what would be sent to the "
                                       "LLM (and send nothing)")
    s.set_defaults(fn=cmd_payload)

    s = sub.add_parser("privacy", help="what is stored, and forget things")
    s.add_argument("--forget-before", metavar="YYYY-MM-DD",
                   help="delete all detailed rows before this date")
    s.set_defaults(fn=cmd_privacy)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
