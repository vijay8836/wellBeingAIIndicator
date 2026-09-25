# wellbeing_ai

A local-first system that reads your own data exports — sleep, activity, food,
work hours, phone use, places, messages, searches, watch history, spending —
scores your physical and mental wellbeing each day **against your own
baseline**, tells you what changed and why, and speaks up when something has
actually shifted rather than just wobbled.

It runs on your machine. The database is a single SQLite file you own.

**It is not a medical device.** It does not diagnose anything and it does not
recommend medication or supplements. When the data suggests something clinical,
the only advice it gives is to see a qualified professional — enforced in code
by `safety.ClinicalGuard`, not just by prompt wording.

---

## What it does

| | |
|---|---|
| **Ingests** | 16 connectors reading the real export formats (Google Takeout, WhatsApp `_chat.txt`, Apple Health `export.xml`, SMS Backup & Restore XML, Chrome `Bookmarks`, fitness-app CSVs, `.ics` calendars) |
| **Engineers** | ~150 daily features: sleep regularity, circadian midpoint, social reciprocity, work overload, rumination markers, medical-seeking pressure, night-time digital load |
| **Scores** | Physical (sleep, activity, nutrition, recovery, medical load) and Mental (mood, stress, social, rumination, circadian, work balance), 0–100, personal baselines |
| **Models** | IsolationForest anomaly detection · gradient-boosted 3-day forecast · low-stretch risk classifier · KMeans day archetypes · Mann-Kendall trend + changepoint detection |
| **Writes** | Daily, weekly and monthly reports — narrative by Claude, or locally if there's no API key |
| **Recommends** | 24 evidence-triggered rules across diet, exercise, sleep, sunlight, yoga/breathwork, social, family, work boundaries, vacation, fun, digital hygiene, nature, and doctor consultation |
| **Alerts** | Persistence-gated, cooldown-limited, quiet-hours-aware. Desktop notification, HTML, JSON, or webhook (ntfy/Slack/Pushover) |
| **Encourages** | Praise alerts fire on genuine improvement, naming what changed |

---

## Quick start

```bash
pip install -r requirements.txt

# See the whole thing work on 6 months of synthetic data
python -m wellbeing_ai demo

# Then with your own data
python -m wellbeing_ai init          # writes config/config.yaml
#   ... drop your exports into data/raw/ ...
python -m wellbeing_ai ingest
python -m wellbeing_ai report daily --html
python -m wellbeing_ai dashboard
```

To have Claude write the narrative instead of the local template engine:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m wellbeing_ai report daily
```

Run it every morning:

```bash
python -m wellbeing_ai schedule --install --time 08:00
```

That installs a cron entry (Linux), launchd agent (macOS) or Task Scheduler
task (Windows). `python -m wellbeing_ai run` is the job it runs.

---

## Getting your data

Everything is consent-gated in `config.yaml` — each source is a separate
toggle, and nothing is read from a source you've turned off.

| Source | Where to get it | Drop into `data/raw/` as |
|---|---|---|
| Search history | [takeout.google.com](https://takeout.google.com) → My Activity → Search | `MyActivity.json` |
| YouTube history | Takeout → YouTube → history | `watch-history.json` |
| Places visited | Takeout → Location History (Timeline) | `*_2026_*.json` or `places.csv` |
| WhatsApp | In-chat → More → Export chat → **Without media** | `WhatsApp Chat with X.txt` |
| Instagram / Facebook | Account Centre → Download your information (JSON) | `posts_1.json`, `liked_posts.json` |
| Sleep & activity | Health Connect, Fitbit, Oura, Garmin, Samsung Health export | `sleep.csv`, `activity.csv` |
| Apple Health | Health app → profile → Export All Health Data | `export.xml` |
| Food log | MyFitnessPal / HealthifyMe export | `food.csv` |
| Spending SMS | SMS Backup & Restore (Android) | `sms.xml` |
| Screen time | Digital Wellbeing / Screen Time export | `screen.csv` |
| Work hours | ActivityWatch, RescueTime, or a calendar export | `work.csv`, `calendar.ics` |
| Bookmarks | Chrome profile folder → `Bookmarks` | `Bookmarks` |
| Notes & reminders | Any CSV with `date,text` | `notes.csv` |
| Mood check-ins | Your own `date,mood,energy,stress` file | `checkins.csv` |

Connectors match on filename globs and sniff columns, so exact naming and
column order don't matter much. Missing sources are simply absent — the
scorer lowers its confidence rather than inventing a number.

**The single highest-value file is `checkins.csv`.** Two minutes a day rating
mood, energy and stress 1–5 gives the models ground truth, and everything else
gets sharper.

---

## Privacy

This system reads unusually intimate data, so the defaults are strict and the
claims are testable.

- **Raw text is not stored by default.** Messages, searches and notes are
  reduced to sentiment, word counts and keyword bags at ingest.
  `privacy.store_raw_text: false`.
- **Contacts are hashed** with a local salt. `Amma` becomes `c_3f9a2b8e1d04`,
  one-way.
- **Locations become categories.** `Apollo Pharmacy Koramangala` is stored as
  `pharmacy`. No coordinates, ever.
- **PII is stripped** — emails, phone numbers, card and ID numbers, URLs — on
  the way in, by regex, before anything is written.
- **The LLM sees numbers only.** Inspect exactly what would be sent:

  ```bash
  python -m wellbeing_ai payload
  ```

  It prints the payload and sends nothing. The test suite asserts that no
  contact name, place name, search query or merchant appears in it.
- **Nothing leaves the machine at all** with `--no-llm` or
  `llm.enabled: false`. The local template engine writes the reports instead.
- **Retention**: detailed event rows older than `retention_days_detail` (400
  by default) are deleted on each run; daily aggregates are kept.

```bash
python -m wellbeing_ai privacy                        # what's stored
python -m wellbeing_ai privacy --forget-before 2026-01-01
```

---

## Safety

Two guards, both in `safety.py`:

**`ClinicalGuard`** filters every recommendation and every line of generated
narrative — including output from Claude itself. Medication names, dose
language, prescribing verbs and diagnostic claims are rejected; the
recommendation is rewritten into "take this to a clinician". The guard is
scoped to avoid false positives: `320mg of caffeine` is dietary information
and passes, `take 5mg melatonin nightly` does not.

**`CrisisDetector`** watches for sustained serious distress. When it fires,
the system stops coaching entirely — no scores, no tips, no charts — and
surfaces human help (Tele-MANAS 14416, KIRAN 1800-599-0019, AASRA, Vandrevala
for India; configurable by locale). It never stores or repeats the text that
triggered it.

---

## Design notes

**Personal baselines, not population norms.** A consistent six-hour sleeper is
not flagged; someone who drops from eight to six is. Everything comparative
uses rolling median/MAD z-scores over the person's own last 28 days.

**Every score is attributable.** Sub-scores are built from declared `Signal`
specs, so the system can always say *which behaviour* cost the points. An
alert reading "mood fell 14" is useless; "you slept 5h10m for four nights and
your bedtime moved 2h later" is something you can act on.

**Alerts must be earned.** One bad night is not an alert. Everything requires
persistence (`persistence_days`), a real baseline departure, or both — with
per-kind cooldowns and quiet hours. Score-scale z-values are floored so a
stretch of unusually consistent days can't manufacture a "3.4 SD" event.

**The forecast matters less than its feature importances.** The gradient
booster predicting your mood three days out is mainly a device for learning
*which of your own habits precede your dips* — that list appears in every
report, and it is derived from your history, not from general advice.

**It degrades honestly.** Four days of data yields a report with a stated
confidence of 0.22 and no trained models, not a confident-looking score.

---

## Layout

```
wellbeing_ai/
  config.py        consent toggles, privacy settings, weights, thresholds
  store.py         SQLite schema and access
  safety.py        ClinicalGuard, CrisisDetector, helplines
  pipeline.py      the orchestrator
  alerts.py        alert rules, cooldowns, delivery
  scheduler.py     cron / launchd / Task Scheduler installers
  cli.py           command-line interface
  connectors/      base.py, health.py, digital.py, context.py — 16 connectors
  features/        builder.py — ~150 daily features
  models/          scoring.py (sub-scores + attribution), ml.py (4 models)
  genai/           coach.py (Claude + local fallback), recommendations.py
  reports/         html.py (self-contained, dark mode), text.py (terminal)
tests/
  generate_demo.py  6 months of synthetic data in real export formats
  test_pipeline.py  82 end-to-end checks
```

## Tests

```bash
python tests/test_pipeline.py
```

82 checks covering privacy primitives, the clinical guard in both directions,
crisis detection, the full arc (the synthetic person's six-week decline must be
detected and the recovery too), model training, bad-day behaviour, payload
privacy, report rendering, persistence, and graceful degradation on four days
of data.

## Commands

```
init        create config and folders
ingest      read export files into the local store
analyze     build features, scores and models
report      daily | weekly | monthly   [--html --json --notify]
dashboard   write the full HTML dashboard
alerts      evaluate alerts now        [--notify]
run         the full daily job
schedule    run daily, or --install a system job
demo        generate synthetic data and run end to end
payload     print what would be sent to the LLM, and send nothing
privacy     what is stored  [--forget-before YYYY-MM-DD]
```

Global flags: `--config PATH`, `--name NAME`, `--no-llm`.
