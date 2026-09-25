"""SQLite store. One file, local to the machine, holding canonical events."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sleep (
  date TEXT PRIMARY KEY, bedtime TEXT, waketime TEXT,
  duration_min REAL, efficiency REAL, awakenings INTEGER,
  deep_min REAL, rem_min REAL, latency_min REAL, source TEXT
);

CREATE TABLE IF NOT EXISTS activity (
  date TEXT PRIMARY KEY, steps INTEGER, active_min REAL, workout_min REAL,
  workout_type TEXT, calories_out REAL, resting_hr REAL, hrv_ms REAL,
  spo2 REAL, weight_kg REAL, source TEXT
);

CREATE TABLE IF NOT EXISTS food (
  id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, meal TEXT, item TEXT,
  calories REAL, protein_g REAL, carbs_g REAL, fat_g REAL, fiber_g REAL,
  sugar_g REAL, sodium_mg REAL, caffeine_mg REAL, alcohol_units REAL,
  water_ml REAL, ultraprocessed INTEGER, source TEXT
);

CREATE TABLE IF NOT EXISTS work (
  date TEXT PRIMARY KEY, first_activity TEXT, last_activity TEXT,
  active_hours REAL, meetings_min REAL, after_hours_min REAL,
  weekend_min REAL, context_switches INTEGER, source TEXT
);

CREATE TABLE IF NOT EXISTS screen (
  date TEXT PRIMARY KEY, total_min REAL, night_min REAL, pickups INTEGER,
  social_min REAL, work_min REAL, entertainment_min REAL, source TEXT
);

CREATE TABLE IF NOT EXISTS sensors (
  date TEXT PRIMARY KEY, unlocks INTEGER, ambient_light_lux REAL,
  outdoor_min REAL, movement_entropy REAL, ambient_noise_db REAL,
  location_variance REAL, source TEXT
);

CREATE TABLE IF NOT EXISTS visits (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, date TEXT,
  place_category TEXT, place_label TEXT, dwell_min REAL, source TEXT
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, date TEXT, channel TEXT,
  direction TEXT, contact_hash TEXT, is_group INTEGER, word_count INTEGER,
  sentiment REAL, emotion TEXT, hour INTEGER, text TEXT, source TEXT
);

CREATE TABLE IF NOT EXISTS searches (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, date TEXT, hour INTEGER,
  category TEXT, sentiment REAL, query TEXT, keywords TEXT, source TEXT
);

CREATE TABLE IF NOT EXISTS media (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, date TEXT, hour INTEGER,
  platform TEXT, category TEXT, minutes REAL, title TEXT, source TEXT
);

CREATE TABLE IF NOT EXISTS social (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, date TEXT, platform TEXT,
  action TEXT, sentiment REAL, category TEXT, text TEXT, source TEXT
);

CREATE TABLE IF NOT EXISTS spending (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, date TEXT, amount REAL,
  currency TEXT, category TEXT, merchant_hash TEXT, source TEXT
);

CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, date TEXT, kind TEXT,
  sentiment REAL, text TEXT, keywords TEXT, source TEXT
);

CREATE TABLE IF NOT EXISTS bookmarks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, date TEXT, category TEXT,
  domain TEXT, title TEXT, source TEXT
);

CREATE TABLE IF NOT EXISTS checkins (
  date TEXT PRIMARY KEY, mood INTEGER, energy INTEGER, stress INTEGER,
  note TEXT, source TEXT
);

CREATE TABLE IF NOT EXISTS scores (
  date TEXT PRIMARY KEY, physical REAL, mental REAL, overall REAL,
  subscores TEXT, drivers TEXT, anomaly REAL, risk REAL,
  forecast_mental REAL, archetype TEXT, computed_at TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, date TEXT, level TEXT,
  kind TEXT, title TEXT, body TEXT, payload TEXT, delivered INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, period TEXT, period_key TEXT,
  headline TEXT, body TEXT, payload TEXT, path TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, rec_id TEXT,
  action TEXT, helpful INTEGER, note TEXT
);

-- Makes re-running `ingest` idempotent. Without this, append-only tables
-- (food especially, which has no timestamp to dedupe on) double on every
-- run and quietly inflate every nutrition figure.
CREATE TABLE IF NOT EXISTS ingested_files (
  path TEXT, connector TEXT, content_hash TEXT, rows INTEGER, ts TEXT,
  PRIMARY KEY (path, connector)
);

CREATE INDEX IF NOT EXISTS ix_msg_date ON messages(date);
CREATE INDEX IF NOT EXISTS ix_search_date ON searches(date);
CREATE INDEX IF NOT EXISTS ix_media_date ON media(date);
CREATE INDEX IF NOT EXISTS ix_visit_date ON visits(date);
CREATE INDEX IF NOT EXISTS ix_spend_date ON spending(date);
CREATE INDEX IF NOT EXISTS ix_food_date ON food(date);
"""

# Tables keyed by date (upsert replaces the day) vs. append-only event tables.
DAILY_TABLES = {"sleep", "activity", "work", "screen", "sensors", "checkins"}
EVENT_TABLES = {
    "food", "visits", "messages", "searches", "media", "social",
    "spending", "notes", "bookmarks",
}


class Store:
    def __init__(self, path: str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def conn(self):
        cx = sqlite3.connect(self.path, timeout=30)
        cx.row_factory = sqlite3.Row
        try:
            yield cx
            cx.commit()
        finally:
            cx.close()

    def init_schema(self) -> None:
        with self.conn() as cx:
            cx.executescript(SCHEMA)

    # ------------------------------------------------------------- writing
    def upsert_daily(self, table: str, rows: Iterable[Dict[str, Any]]) -> int:
        rows = [r for r in rows if r.get("date")]
        if not rows:
            return 0
        cols = sorted({k for r in rows for k in r})
        placeholders = ",".join("?" * len(cols))
        sql = (f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
               f"VALUES ({placeholders})")
        with self.conn() as cx:
            cx.executemany(sql, [[r.get(c) for c in cols] for r in rows])
        return len(rows)

    def insert_events(self, table: str, rows: Iterable[Dict[str, Any]],
                      dedupe_on: Optional[List[str]] = None) -> int:
        rows = list(rows)
        if not rows:
            return 0
        if dedupe_on:
            existing = self.read_df(f"SELECT {','.join(dedupe_on)} FROM {table}")
            if not existing.empty:
                seen = set(map(tuple, existing[dedupe_on].astype(str).values))
                rows = [r for r in rows
                        if tuple(str(r.get(k)) for k in dedupe_on) not in seen]
        if not rows:
            return 0
        cols = sorted({k for r in rows for k in r})
        sql = (f"INSERT INTO {table} ({','.join(cols)}) "
               f"VALUES ({','.join('?' * len(cols))})")
        with self.conn() as cx:
            cx.executemany(sql, [[r.get(c) for c in cols] for r in rows])
        return len(rows)

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.conn() as cx:
            cx.execute(sql, params)

    # -------------------------------------------------------- idempotency
    def file_seen(self, path: str, connector: str, content_hash: str) -> bool:
        """True if this exact file content has already been ingested."""
        df = self.read_df(
            "SELECT content_hash FROM ingested_files "
            "WHERE path = ? AND connector = ?", (str(path), connector))
        return (not df.empty) and df["content_hash"].iloc[0] == content_hash

    def mark_file(self, path: str, connector: str, content_hash: str,
                  rows: int) -> None:
        from datetime import datetime as _dt
        with self.conn() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO ingested_files "
                "(path, connector, content_hash, rows, ts) VALUES (?,?,?,?,?)",
                (str(path), connector, content_hash, rows,
                 _dt.now().isoformat(timespec="seconds")))

    def forget_files(self, connector: Optional[str] = None) -> None:
        """Forget ingest history so the next run re-reads everything."""
        with self.conn() as cx:
            if connector:
                cx.execute("DELETE FROM ingested_files WHERE connector = ?",
                           (connector,))
            else:
                cx.execute("DELETE FROM ingested_files")

    # ------------------------------------------------------------- reading
    def read_df(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self.conn() as cx:
            try:
                return pd.read_sql_query(sql, cx, params=params)
            except Exception:
                return pd.DataFrame()

    def table(self, name: str, since: Optional[str] = None) -> pd.DataFrame:
        if since:
            return self.read_df(f"SELECT * FROM {name} WHERE date >= ?", (since,))
        return self.read_df(f"SELECT * FROM {name}")

    def counts(self) -> Dict[str, int]:
        out = {}
        for t in sorted(DAILY_TABLES | EVENT_TABLES | {"scores", "alerts", "reports"}):
            df = self.read_df(f"SELECT COUNT(*) AS n FROM {t}")
            out[t] = int(df["n"].iloc[0]) if not df.empty else 0
        return out

    def date_range(self) -> tuple[Optional[str], Optional[str]]:
        df = self.read_df(
            "SELECT MIN(date) AS a, MAX(date) AS b FROM ("
            " SELECT date FROM sleep UNION SELECT date FROM activity"
            " UNION SELECT date FROM work UNION SELECT date FROM screen)"
        )
        if df.empty:
            return None, None
        return df["a"].iloc[0], df["b"].iloc[0]

    def prune(self, before_date: str) -> Dict[str, int]:
        """Retention: drop detailed event rows older than a cutoff."""
        removed = {}
        with self.conn() as cx:
            for t in EVENT_TABLES | {"messages", "searches"}:
                cur = cx.execute(f"DELETE FROM {t} WHERE date < ?", (before_date,))
                removed[t] = cur.rowcount
        return removed
