"""Connector base class and shared parsing helpers."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from ..config import Config
from ..store import Store


@dataclass
class IngestResult:
    connector: str
    table: str
    rows: int
    skipped: int = 0
    note: str = ""

    def __str__(self) -> str:
        s = f"{self.connector:<22} -> {self.table:<10} {self.rows:>6} rows"
        if self.skipped:
            s += f"  ({self.skipped} skipped)"
        if self.note:
            s += f"  [{self.note}]"
        return s


class Connector(ABC):
    """One data source. Reads files from disk, writes canonical rows to the DB."""

    name: str = "connector"
    table: str = ""
    consent_key: str = ""
    #: filename globs this connector knows how to read
    patterns: List[str] = []

    def __init__(self, cfg: Config, store: Store):
        self.cfg = cfg
        self.store = store

    # -------------------------------------------------------------- helpers
    def consented(self) -> bool:
        return self.cfg.consent.is_on(self.consent_key)

    def find_files(self, root: Optional[Path] = None) -> List[Path]:
        root = Path(root or self.cfg.raw_dir)
        if not root.exists():
            return []
        seen: Dict[str, Path] = {}
        for pat in self.patterns:
            for p in sorted(root.rglob(pat)):
                # One file can match several globs -- parse it only once.
                seen.setdefault(str(p.resolve()), p)
        return list(seen.values())

    def looks_right(self, df) -> bool:
        """Override to reject a file that matched the glob but isn't ours."""
        return True

    @abstractmethod
    def parse(self, path: Path) -> List[Dict[str, Any]]:
        """Turn one file into canonical rows."""

    def ingest(self, root: Optional[Path] = None,
               force: bool = False) -> IngestResult:
        if not self.consented():
            return IngestResult(self.name, self.table, 0, note="no consent")
        files = self.find_files(root)
        if not files:
            return IngestResult(self.name, self.table, 0, note="no files found")

        rows: List[Dict[str, Any]] = []
        fresh: List[tuple] = []
        unchanged = 0
        errors: List[str] = []
        for f in files:
            digest = file_hash(f)
            # Re-reading an unchanged file is how append-only tables end up
            # double-counted, so skip it unless explicitly forced.
            if not force and self.store.file_seen(str(f), self.name, digest):
                unchanged += 1
                continue
            try:
                parsed = [r for r in self.parse(f) if r.get("date")]
            except Exception as exc:     # one bad export shouldn't kill the run
                errors.append(f"{f.name}: {type(exc).__name__}")
                continue
            for r in parsed:
                r.setdefault("source", self.name)
            rows.extend(parsed)
            fresh.append((f, digest, len(parsed)))

        if not rows:
            note = "; ".join(errors) if errors else (
                f"{unchanged} file(s) unchanged" if unchanged else "nothing new")
            return IngestResult(self.name, self.table, 0, note=note)

        written = self.write(rows)
        for f, digest, n in fresh:
            self.store.mark_file(str(f), self.name, digest, n)
        note = "; ".join(errors)
        if unchanged:
            note = (note + "; " if note else "") + f"{unchanged} unchanged"
        return IngestResult(self.name, self.table, written,
                            skipped=len(rows) - written, note=note)

    def write(self, rows: List[Dict[str, Any]]) -> int:
        from ..store import DAILY_TABLES
        if self.table in DAILY_TABLES:
            return self.store.upsert_daily(self.table, rows)
        return self.store.insert_events(self.table, rows,
                                        dedupe_on=self.dedupe_on())

    #: Natural key per table, as a second line of defence behind file-level
    #: idempotency. Food has no timestamp, so it keys on the meal itself.
    DEDUPE_KEYS = {
        "food": ["date", "meal", "item", "calories"],
        "messages": ["ts", "date", "contact_hash", "word_count"],
        "searches": ["ts", "date", "keywords"],
        "media": ["ts", "date", "category"],
        "visits": ["ts", "date", "place_category"],
        "spending": ["ts", "date", "amount"],
        "social": ["ts", "date", "action"],
        "notes": ["ts", "date", "kind"],
        "bookmarks": ["ts", "date", "domain"],
    }

    def dedupe_on(self) -> Optional[List[str]]:
        keys = self.DEDUPE_KEYS.get(self.table)
        if not keys:
            return None
        return keys


# ------------------------------------------------------------------ parsing
def parse_ts(value: Any) -> Optional[datetime]:
    """Best-effort timestamp parsing across the formats exports actually use."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        v = float(value)
        # Heuristics: seconds / milliseconds / microseconds since epoch.
        if v > 1e17:
            v /= 1e6
        elif v > 1e14:
            v /= 1e3
        elif v > 1e12:
            v /= 1e3
        try:
            return datetime.fromtimestamp(v)
        except (ValueError, OSError, OverflowError):
            return None
    s = str(value).strip()
    if not s:
        return None
    # Epoch timestamps often arrive as digit strings ("1774555920000").
    if s.isdigit() and len(s) >= 10:
        return parse_ts(int(s))
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%d/%m/%Y, %H:%M", "%d/%m/%y, %H:%M", "%m/%d/%y, %H:%M",
                "%d-%m-%Y %H:%M", "%b %d, %Y, %I:%M:%S %p", "%d/%m/%Y",
                "%Y/%m/%d", "%d %b %Y %H:%M", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=None) if dt.tzinfo else dt
        except ValueError:
            continue
    try:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.notna(dt):
            return dt.to_pydatetime().replace(tzinfo=None)
    except Exception:
        pass
    return None


def day_of(ts: Optional[datetime]) -> Optional[str]:
    return ts.strftime("%Y-%m-%d") if ts else None


def sleep_day(ts: datetime) -> str:
    """A sleep session belongs to the morning you woke up on."""
    return (ts if ts.hour >= 12 else ts).strftime("%Y-%m-%d")


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    """Content hash, so an unchanged export is never ingested twice."""
    import hashlib
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                block = fh.read(chunk)
                if not block:
                    break
                h.update(block)
    except OSError:
        return f"stat:{path.stat().st_size}"
    return h.hexdigest()[:16]


def read_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # JSONL fallback
        out = []
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in "[]":
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out or None


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".csv", ".txt"):
        return pd.read_csv(path)
    if suf in (".tsv",):
        return pd.read_csv(path, sep="\t")
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".json":
        data = read_json(path)
        return pd.json_normalize(data) if data else pd.DataFrame()
    return pd.DataFrame()


def col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Find the first column matching any candidate, case/space insensitive."""
    norm = {str(c).lower().replace(" ", "").replace("_", ""): c
            for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in norm:
            return norm[key]
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        for k, orig in norm.items():
            if key in k:
                return orig
    return None


def num(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
