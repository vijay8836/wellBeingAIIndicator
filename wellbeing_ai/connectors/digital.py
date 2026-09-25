"""
Digital-trace connectors: search history, YouTube/OTT, WhatsApp, social
posts, bookmarks, notes.

All of these apply the privacy layer before anything is written: contacts are
hashed, free text is redacted, and raw text is only kept if
`privacy.store_raw_text` is explicitly turned on.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from ..utils import privacy as pv
from ..utils import textanalysis as ta
from .base import Connector, col, day_of, num, parse_ts, read_json, read_table


class _TextConnector(Connector):
    """Shared text handling: redact, score, and only keep raw text on request."""

    def keep_text(self, text: str) -> Optional[str]:
        if not self.cfg.privacy.store_raw_text:
            return None
        return pv.redact(text)[:500]

    def kw(self, text: str) -> str:
        return ",".join(pv.keywords_only(text, limit=8))


class GoogleSearchConnector(_TextConnector):
    """
    Google Takeout -> My Activity -> Search. Handles both the JSON export
    (`MyActivity.json`) and the HTML one, plus a plain `query,timestamp` CSV.
    """

    name = "google_search"
    table = "searches"
    consent_key = "search_history"
    patterns = ["*MyActivity*.json", "*search*history*.json",
                "*search*history*.csv", "*MyActivity*.html"]

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        if path.suffix.lower() == ".html":
            return self._parse_html(path)
        if path.suffix.lower() == ".csv":
            return self._parse_csv(path)
        return self._parse_json(path)

    def _row(self, query: str, ts) -> Optional[Dict[str, Any]]:
        d = day_of(ts)
        if not d or not query:
            return None
        clean = pv.redact(query)
        return {
            "ts": ts.isoformat(), "date": d, "hour": ts.hour,
            "category": ta.categorize(clean),
            "sentiment": ta.sentiment(clean),
            "query": self.keep_text(clean),
            "keywords": self.kw(clean),
        }

    def _parse_json(self, path: Path) -> List[Dict[str, Any]]:
        data = read_json(path)
        if not isinstance(data, list):
            return []
        rows = []
        for item in data:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "")
            if item.get("header") not in (None, "Search", "Google Search"):
                if "Search" not in str(item.get("header", "")):
                    continue
            q = re.sub(r"^(Searched for|Visited)\s+", "", title)
            r = self._row(q, parse_ts(item.get("time")))
            if r:
                rows.append(r)
        return rows

    def _parse_html(self, path: Path) -> List[Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        pat = re.compile(
            r"Searched for\s*</?[^>]*>?([^<]{2,200})</a>.*?"
            r"<br>([A-Z][a-z]{2} \d{1,2}, \d{4}, [^<]+)", re.S)
        rows = []
        for m in pat.finditer(text):
            r = self._row(m.group(1).strip(), parse_ts(m.group(2).strip()))
            if r:
                rows.append(r)
        return rows

    def _parse_csv(self, path: Path) -> List[Dict[str, Any]]:
        df = read_table(path)
        if df.empty:
            return []
        c_q = col(df, "query", "search", "term", "title")
        c_t = col(df, "timestamp", "time", "date")
        if not c_q or not c_t:
            return []
        rows = []
        for _, r in df.iterrows():
            row = self._row(str(r.get(c_q, "")), parse_ts(r.get(c_t)))
            if row:
                rows.append(row)
        return rows


class YouTubeConnector(_TextConnector):
    """Takeout -> YouTube -> history -> watch-history.json (or an OTT CSV)."""

    name = "youtube_history"
    table = "media"
    consent_key = "media_history"
    patterns = ["watch-history.json", "*watch*history*.json",
                "*youtube*.csv", "*ott*.csv", "*netflix*.csv", "*prime*.csv"]

    # Rough duration priors, since watch history has no runtime field.
    DEFAULT_MIN = 9.0

    CATS = [
        ("comfort", ["asmr", "relax", "calm", "sleep music", "rain sounds",
                     "lofi", "meditation", "guided"]),
        ("selfhelp", ["motivation", "productivity", "discipline", "self help",
                      "how to stop", "overcome", "habits", "therapy"]),
        ("doomscroll", ["shorts", "#shorts", "reaction", "drama", "exposed",
                        "rant", "fail compilation"]),
        ("news", ["news", "debate", "politics", "breaking", "headlines"]),
        ("fitness", ["workout", "yoga", "gym", "hiit", "stretch", "run"]),
        ("learning", ["tutorial", "course", "lecture", "explained", "crash course"]),
        ("entertainment", ["trailer", "episode", "movie", "song", "music video",
                           "comedy", "standup", "vlog"]),
    ]

    def _cat(self, title: str) -> str:
        low = (title or "").lower()
        for name, pats in self.CATS:
            if any(p in low for p in pats):
                return name
        return "other"

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if path.suffix.lower() == ".json":
            data = read_json(path)
            if not isinstance(data, list):
                return []
            for item in data:
                if not isinstance(item, dict):
                    continue
                ts = parse_ts(item.get("time"))
                d = day_of(ts)
                if not d:
                    continue
                title = re.sub(r"^Watched\s+", "", item.get("title", ""))
                rows.append({
                    "ts": ts.isoformat(), "date": d, "hour": ts.hour,
                    "platform": "youtube", "category": self._cat(title),
                    "minutes": self.DEFAULT_MIN,
                    "title": self.keep_text(title),
                })
            return rows

        df = read_table(path)
        if df.empty:
            return []
        c_t = col(df, "timestamp", "date", "time", "starttime")
        c_title = col(df, "title", "name", "video", "show")
        if not c_t:
            return []
        platform = "ott" if any(k in path.name.lower()
                                for k in ("netflix", "prime", "ott", "hotstar")) \
            else "youtube"
        c_min = col(df, "minutes", "duration", "watchtime")
        for _, r in df.iterrows():
            ts = parse_ts(r.get(c_t))
            d = day_of(ts)
            if not d:
                continue
            title = str(r.get(c_title, "") if c_title else "")
            mins = num(r.get(c_min)) if c_min else None
            rows.append({
                "ts": ts.isoformat(), "date": d, "hour": ts.hour,
                "platform": platform, "category": self._cat(title),
                "minutes": mins if mins is not None else (
                    45.0 if platform == "ott" else self.DEFAULT_MIN),
                "title": self.keep_text(title),
            })
        return rows


class WhatsAppConnector(_TextConnector):
    """
    WhatsApp "Export chat (without media)" -> `_chat.txt` / `WhatsApp Chat with X.txt`.

    Only derived signals are stored by default: who (hashed), when, how long,
    sentiment, rumination markers. The messages themselves stay on disk.
    """

    name = "whatsapp"
    table = "messages"
    consent_key = "messages"
    patterns = ["*_chat.txt", "WhatsApp Chat*.txt", "*whatsapp*.txt",
                "*whatsapp*.csv"]

    LINE = re.compile(
        r"^\[?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),?\s+"
        r"(\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap][Mm])?)\]?\s*[-–]?\s*"
        r"([^:]{1,60}?):\s?(.*)$"
    )
    SYSTEM = ("messages and calls are end-to-end encrypted",
              "created group", "added you", "changed the subject",
              "<media omitted>", "this message was deleted",
              "joined using this group's invite link", "image omitted",
              "security code changed")

    def _me(self) -> str:
        return (self.cfg.profile.name or "me").strip().lower()

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        if path.suffix.lower() == ".csv":
            return self._parse_csv(path)
        salt = self.cfg.privacy.salt
        is_group = "group" in path.name.lower()
        rows: List[Dict[str, Any]] = []
        buf_sender, buf_ts, buf_text = None, None, ""

        def flush():
            nonlocal buf_sender, buf_ts, buf_text
            if buf_ts and buf_text:
                low = buf_text.lower()
                if not any(s in low for s in self.SYSTEM):
                    clean = pv.redact(buf_text)
                    outgoing = (buf_sender or "").strip().lower() in (
                        self._me(), "you")
                    rows.append({
                        "ts": buf_ts.isoformat(), "date": day_of(buf_ts),
                        "hour": buf_ts.hour, "channel": "whatsapp",
                        "direction": "out" if outgoing else "in",
                        "contact_hash": pv.hash_id(buf_sender or "?", salt)
                        if self.cfg.privacy.hash_contacts else buf_sender,
                        "is_group": int(is_group),
                        "word_count": len(clean.split()),
                        "sentiment": ta.sentiment(clean),
                        "emotion": ta.emotion_label(ta.sentiment(clean)),
                        "text": self.keep_text(clean),
                    })
            buf_sender, buf_ts, buf_text = None, None, ""

        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.replace("‎", "").replace(" ", " ").strip()
            if not line:
                continue
            m = self.LINE.match(line)
            if m:
                flush()
                date_s, time_s, sender, text = m.groups()
                buf_ts = parse_ts(f"{date_s} {time_s}")
                buf_sender, buf_text = sender, text
            elif buf_ts:
                buf_text += " " + line
        flush()
        return rows

    def _parse_csv(self, path: Path) -> List[Dict[str, Any]]:
        df = read_table(path)
        if df.empty:
            return []
        c_t = col(df, "timestamp", "datetime", "date", "time")
        c_s = col(df, "sender", "from", "contact", "author")
        c_m = col(df, "message", "text", "body", "content")
        if not c_t:
            return []
        salt = self.cfg.privacy.salt
        rows = []
        for _, r in df.iterrows():
            ts = parse_ts(r.get(c_t))
            if not ts:
                continue
            text = pv.redact(str(r.get(c_m, "") if c_m else ""))
            sender = str(r.get(c_s, "?") if c_s else "?")
            s = ta.sentiment(text)
            rows.append({
                "ts": ts.isoformat(), "date": day_of(ts), "hour": ts.hour,
                "channel": "whatsapp",
                "direction": "out" if sender.strip().lower() in (self._me(), "you")
                else "in",
                "contact_hash": pv.hash_id(sender, salt),
                "is_group": 0, "word_count": len(text.split()),
                "sentiment": s, "emotion": ta.emotion_label(s),
                "text": self.keep_text(text),
            })
        return rows


class SocialConnector(_TextConnector):
    """Instagram / Facebook JSON exports: posts, comments, likes, saves."""

    name = "social_export"
    table = "social"
    consent_key = "social_posts"
    patterns = ["*posts_1.json", "*liked_posts.json", "*your_posts*.json",
                "*comments*.json", "*instagram*.json", "*facebook*.json",
                "*social*.csv"]

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        if path.suffix.lower() == ".csv":
            return self._parse_csv(path)
        data = read_json(path)
        if data is None:
            return []
        platform = "instagram" if "instagram" in str(path).lower() \
            else ("facebook" if "facebook" in str(path).lower() else "social")
        action = ("like" if "like" in path.name.lower()
                  else "comment" if "comment" in path.name.lower() else "post")
        rows: List[Dict[str, Any]] = []

        def walk(node):
            if isinstance(node, dict):
                ts = None
                for k in ("timestamp", "creation_timestamp", "taken_at",
                          "time", "date"):
                    if k in node:
                        ts = parse_ts(node[k])
                        break
                text = ""
                for k in ("title", "caption", "text", "value", "comment",
                          "description"):
                    v = node.get(k)
                    if isinstance(v, str) and len(v) > len(text):
                        text = v
                if ts:
                    clean = pv.redact(text)
                    s = ta.sentiment(clean)
                    rows.append({
                        "ts": ts.isoformat(), "date": day_of(ts),
                        "platform": platform, "action": action,
                        "sentiment": s, "category": ta.categorize(clean),
                        "text": self.keep_text(clean),
                    })
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(data)
        return rows

    def _parse_csv(self, path: Path) -> List[Dict[str, Any]]:
        df = read_table(path)
        if df.empty:
            return []
        c_t = col(df, "timestamp", "date", "time")
        c_x = col(df, "text", "caption", "content", "title")
        if not c_t:
            return []
        rows = []
        for _, r in df.iterrows():
            ts = parse_ts(r.get(c_t))
            if not ts:
                continue
            clean = pv.redact(str(r.get(c_x, "") if c_x else ""))
            rows.append({
                "ts": ts.isoformat(), "date": day_of(ts),
                "platform": str(r.get(col(df, "platform") or "", "social")),
                "action": str(r.get(col(df, "action", "type") or "", "post")),
                "sentiment": ta.sentiment(clean),
                "category": ta.categorize(clean),
                "text": self.keep_text(clean),
            })
        return rows


class BookmarkConnector(Connector):
    """Chrome/Edge `Bookmarks` JSON, or a Firefox/CSV export."""

    name = "bookmarks"
    table = "bookmarks"
    consent_key = "bookmarks"
    patterns = ["Bookmarks", "Bookmarks.json", "*bookmark*.json",
                "*bookmark*.csv", "*bookmark*.html"]

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if path.suffix.lower() == ".csv":
            df = read_table(path)
            c_t = col(df, "date", "added", "timestamp")
            c_u = col(df, "url", "link")
            c_n = col(df, "title", "name")
            for _, r in df.iterrows():
                ts = parse_ts(r.get(c_t)) if c_t else None
                if not ts:
                    continue
                url = str(r.get(c_u, "") if c_u else "")
                title = str(r.get(c_n, "") if c_n else "")
                rows.append(self._row(ts, url, title))
            return rows

        if path.suffix.lower() == ".html":
            text = path.read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(
                    r'<A HREF="([^"]+)"[^>]*ADD_DATE="(\d+)"[^>]*>([^<]*)</A>',
                    text, re.I):
                ts = parse_ts(int(m.group(2)))
                if ts:
                    rows.append(self._row(ts, m.group(1), m.group(3)))
            return rows

        data = read_json(path)
        if not isinstance(data, dict):
            return []

        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "url":
                    ts = parse_ts(node.get("date_added"))
                    # Chrome uses microseconds since 1601-01-01.
                    raw = node.get("date_added")
                    if raw and str(raw).isdigit() and int(raw) > 1e16:
                        import datetime as _dt
                        ts = (_dt.datetime(1601, 1, 1)
                              + _dt.timedelta(microseconds=int(raw)))
                    if ts:
                        rows.append(self._row(ts, node.get("url", ""),
                                              node.get("name", "")))
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(data.get("roots", data))
        return rows

    def _row(self, ts, url: str, title: str) -> Dict[str, Any]:
        domain = ""
        try:
            domain = urlparse(url).netloc.replace("www.", "")[:60]
        except Exception:
            pass
        return {"ts": ts.isoformat(), "date": day_of(ts),
                "category": ta.categorize(f"{title} {domain}"),
                "domain": domain, "title": title[:120]}


class NotesConnector(_TextConnector):
    """Phone/laptop notes and reminders: a .md/.txt folder or a CSV export."""

    name = "notes"
    table = "notes"
    consent_key = "notes"
    patterns = ["*note*.csv", "*reminder*.csv", "notes/*.txt", "notes/*.md",
                "*journal*.txt"]

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        if path.suffix.lower() in (".txt", ".md"):
            ts = parse_ts(path.stem) or parse_ts(
                __import__("datetime").datetime.fromtimestamp(
                    path.stat().st_mtime))
            if not ts:
                return []
            text = pv.redact(path.read_text(encoding="utf-8", errors="ignore"))
            return [{
                "ts": ts.isoformat(), "date": day_of(ts), "kind": "note",
                "sentiment": ta.sentiment(text),
                "text": self.keep_text(text),
                "keywords": self.kw(text),
            }]
        df = read_table(path)
        if df.empty:
            return []
        c_t = col(df, "date", "timestamp", "created", "time")
        c_x = col(df, "text", "note", "content", "body", "title")
        if not c_t:
            return []
        kind = "reminder" if "remind" in path.name.lower() else "note"
        rows = []
        for _, r in df.iterrows():
            ts = parse_ts(r.get(c_t))
            if not ts:
                continue
            text = pv.redact(str(r.get(c_x, "") if c_x else ""))
            k = kind
            if any(w in text.lower() for w in
                   ("doctor", "bp", "medicine", "appointment", "test", "scan",
                    "checkup", "dentist")):
                k = "health"
            rows.append({
                "ts": ts.isoformat(), "date": day_of(ts), "kind": k,
                "sentiment": ta.sentiment(text),
                "text": self.keep_text(text),
                "keywords": self.kw(text),
            })
        return rows
