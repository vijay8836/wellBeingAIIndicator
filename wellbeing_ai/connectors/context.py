"""
Context connectors: places visited (GPS), spending (bank SMS), work hours.

Location is reduced to a *category* -- hospital, pharmacy, gym, yoga, worship,
park, office, home -- before storage. No coordinates are kept when
`privacy.coarse_location_only` is on (the default).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils import privacy as pv
from ..utils import textanalysis as ta
from .base import Connector, col, day_of, num, parse_ts, read_json, read_table


class PlacesConnector(Connector):
    """
    Google Maps Timeline (`Semantic Location History/*.json`) or a simple
    `timestamp,place,minutes` CSV from any location logger.
    """

    name = "places"
    table = "visits"
    consent_key = "location_visits"
    patterns = ["*_*.json", "*semantic*location*.json", "*timeline*.json",
                "*places*.csv", "*location*.csv", "*visits*.csv"]

    def find_files(self, root: Optional[Path] = None) -> List[Path]:
        # The Takeout glob `*_*.json` is broad -- keep only plausible files.
        files = super().find_files(root)
        out = []
        for f in files:
            low = str(f).lower()
            if f.suffix.lower() == ".csv":
                out.append(f)
            elif ("location" in low or "timeline" in low or "semantic" in low
                  or "places" in low or re.search(r"\d{4}_[a-z]+\.json$", low)):
                out.append(f)
        return out

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        if path.suffix.lower() == ".csv":
            return self._parse_csv(path)
        return self._parse_timeline(path)

    def _row(self, ts, label: str, dwell: Optional[float]) -> Optional[Dict]:
        d = day_of(ts)
        if not d:
            return None
        cat = ta.categorize_place(label)
        keep_label = (label[:60] if not self.cfg.privacy.coarse_location_only
                      else None)
        return {"ts": ts.isoformat(), "date": d, "place_category": cat,
                "place_label": keep_label, "dwell_min": dwell}

    def _parse_timeline(self, path: Path) -> List[Dict[str, Any]]:
        data = read_json(path)
        if not isinstance(data, dict):
            return []
        objs = data.get("timelineObjects") or data.get("semanticSegments") or []
        rows = []
        for o in objs:
            if not isinstance(o, dict):
                continue
            pv_ = o.get("placeVisit") or o.get("visit")
            if not pv_:
                continue
            loc = (pv_.get("location") or {})
            label = (loc.get("name") or loc.get("address") or
                     (pv_.get("topCandidate") or {}).get("placeLocation", "") or "")
            dur = pv_.get("duration") or o.get("duration") or {}
            start = parse_ts(dur.get("startTimestamp") or dur.get("startTime"))
            end = parse_ts(dur.get("endTimestamp") or dur.get("endTime"))
            if not start:
                continue
            dwell = (end - start).total_seconds() / 60 if end else None
            r = self._row(start, str(label), dwell)
            if r:
                rows.append(r)
        return rows

    def _parse_csv(self, path: Path) -> List[Dict[str, Any]]:
        df = read_table(path)
        if df.empty:
            return []
        c_t = col(df, "timestamp", "date", "time", "arrival", "start")
        c_p = col(df, "place", "name", "location", "venue", "label", "category")
        c_d = col(df, "minutes", "dwell", "duration", "dwellmin")
        if not c_t:
            return []
        rows = []
        for _, r in df.iterrows():
            ts = parse_ts(r.get(c_t))
            if not ts:
                continue
            row = self._row(ts, str(r.get(c_p, "") if c_p else ""),
                            num(r.get(c_d)) if c_d else None)
            if row:
                rows.append(row)
        return rows


class SpendingConnector(Connector):
    """
    Transaction SMS. Reads:
      * `sms.xml` from SMS Backup & Restore
      * a `date,amount,merchant,category` CSV from any expense app

    Merchants are hashed; only the amount, category and time survive.
    """

    name = "spending_sms"
    table = "spending"
    consent_key = "spending"
    patterns = ["sms*.xml", "*transactions*.csv", "*spending*.csv",
                "*expense*.csv", "*bank*.csv"]

    AMOUNT = re.compile(
        r"(?:rs\.?|inr|₹|usd|\$|eur|€)\s*([\d,]+(?:\.\d{1,2})?)", re.I)
    DEBIT = re.compile(r"\b(debited|spent|paid|purchase|withdrawn|txn)\b", re.I)
    MERCHANT = re.compile(r"(?:at|to|towards|vpa)\s+([A-Za-z0-9&@._\- ]{3,40})", re.I)

    CATEGORIES = [
        ("medical", ["pharmacy", "apollo", "medplus", "hospital", "clinic",
                     "diagnost", "lab", "medical", "chemist", "practo",
                     "pharmeasy", "1mg"]),
        ("food_delivery", ["swiggy", "zomato", "ubereats", "dominos",
                           "deliver", "foodpanda", "eatsure"]),
        ("alcohol", ["wine", "liquor", "brew", "bar ", "pub ", "beer"]),
        ("fitness", ["gym", "cult", "decathlon", "fitness", "yoga", "sports"]),
        ("transport", ["uber", "ola", "rapido", "fuel", "petrol", "metro",
                       "irctc", "toll"]),
        ("travel", ["makemytrip", "goibibo", "airlines", "indigo", "hotel",
                    "oyo", "airbnb", "booking.com"]),
        ("groceries", ["bigbasket", "blinkit", "zepto", "dmart", "grocer",
                       "instamart", "more super"]),
        ("shopping", ["amazon", "flipkart", "myntra", "ajio", "mall", "nykaa"]),
        ("entertainment", ["netflix", "hotstar", "spotify", "bookmyshow",
                           "prime video", "pvr", "inox"]),
        ("bills", ["electricity", "recharge", "broadband", "insurance", "emi",
                   "rent", "gas ", "water bill"]),
    ]

    def _cat(self, text: str) -> str:
        low = (text or "").lower()
        for name, pats in self.CATEGORIES:
            if any(p in low for p in pats):
                return name
        return "other"

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        if path.suffix.lower() == ".xml":
            return self._parse_xml(path)
        return self._parse_csv(path)

    def _parse_xml(self, path: Path) -> List[Dict[str, Any]]:
        salt = self.cfg.privacy.salt
        rows = []
        for _, el in ET.iterparse(str(path), events=("end",)):
            if el.tag != "sms":
                continue
            body = el.get("body") or ""
            if not self.DEBIT.search(body):
                el.clear()
                continue
            m = self.AMOUNT.search(body)
            if not m:
                el.clear()
                continue
            ts = parse_ts(el.get("date"))
            if not ts:
                el.clear()
                continue
            mm = self.MERCHANT.search(body)
            merchant = (mm.group(1).strip() if mm else "unknown")
            rows.append({
                "ts": ts.isoformat(), "date": day_of(ts),
                "amount": float(m.group(1).replace(",", "")),
                "currency": "INR" if re.search(r"rs\.?|inr|₹", body, re.I) else "NA",
                "category": self._cat(body),
                "merchant_hash": pv.hash_id(merchant, salt),
            })
            el.clear()
        return rows

    def _parse_csv(self, path: Path) -> List[Dict[str, Any]]:
        df = read_table(path)
        if df.empty:
            return []
        c_t = col(df, "date", "timestamp", "time")
        c_a = col(df, "amount", "debit", "value", "spent")
        if not c_t or not c_a:
            return []
        c_m = col(df, "merchant", "payee", "description", "narration", "note")
        c_c = col(df, "category", "type")
        salt = self.cfg.privacy.salt
        rows = []
        for _, r in df.iterrows():
            ts = parse_ts(r.get(c_t))
            amt = num(r.get(c_a))
            if not ts or amt is None or amt <= 0:
                continue
            merchant = str(r.get(c_m, "unknown") if c_m else "unknown")
            cat = str(r.get(c_c)) if c_c and r.get(c_c) else self._cat(merchant)
            rows.append({
                "ts": ts.isoformat(), "date": day_of(ts), "amount": abs(amt),
                "currency": "INR", "category": cat.lower()[:30],
                "merchant_hash": pv.hash_id(merchant, salt),
            })
        return rows


class WorkHoursConnector(Connector):
    """
    Working hours from a laptop activity log (RescueTime/ActivityWatch CSV) or
    a calendar `.ics` file. Produces first/last activity, active hours,
    meeting load, after-hours and weekend minutes.
    """

    name = "work_hours"
    table = "work"
    consent_key = "work_hours"
    patterns = ["*work*.csv", "*activitywatch*.csv", "*rescuetime*.csv",
                "*laptop*.csv", "*calendar*.ics", "*.ics"]

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        if path.suffix.lower() == ".ics":
            return self._parse_ics(path)
        return self._parse_csv(path)

    def _parse_csv(self, path: Path) -> List[Dict[str, Any]]:
        df = read_table(path)
        if df.empty:
            return []
        c_d = col(df, "date", "day", "timestamp")
        if not c_d:
            return []
        rows = []
        for _, r in df.iterrows():
            d = day_of(parse_ts(r.get(c_d)))
            if not d:
                continue
            first = r.get(col(df, "firstactivity", "start", "login", "firstseen")
                          or "")
            last = r.get(col(df, "lastactivity", "end", "logout", "lastseen") or "")
            hours = num(r.get(col(df, "activehours", "hours", "productivehours")))
            if hours is None:
                mins = num(r.get(col(df, "activeminutes", "minutes", "duration")))
                hours = mins / 60 if mins is not None else None
            rows.append({
                "date": d,
                "first_activity": str(first) if first is not None else None,
                "last_activity": str(last) if last is not None else None,
                "active_hours": hours,
                "meetings_min": num(r.get(col(df, "meetings", "meetingmin",
                                              "callminutes"))),
                "after_hours_min": num(r.get(col(df, "afterhours",
                                                 "afterhoursmin"))),
                "weekend_min": num(r.get(col(df, "weekend", "weekendmin"))),
                "context_switches": num(r.get(col(df, "switches",
                                                  "contextswitches", "windows"))),
            })
        return rows

    def _parse_ics(self, path: Path) -> List[Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        events = re.findall(
            r"BEGIN:VEVENT(.*?)END:VEVENT", text, re.S)
        daily: Dict[str, Dict[str, Any]] = {}
        wk = self.cfg.profile
        for ev in events:
            s = re.search(r"DTSTART[^:]*:(\S+)", ev)
            e = re.search(r"DTEND[^:]*:(\S+)", ev)
            if not s:
                continue
            start = parse_ts(_ics_dt(s.group(1)))
            end = parse_ts(_ics_dt(e.group(1))) if e else None
            if not start:
                continue
            d = day_of(start)
            row = daily.setdefault(d, {"date": d, "meetings_min": 0.0,
                                       "after_hours_min": 0.0,
                                       "weekend_min": 0.0,
                                       "first_activity": None,
                                       "last_activity": None})
            mins = (end - start).total_seconds() / 60 if end else 30.0
            row["meetings_min"] += mins
            if start.hour >= wk.work_end_hour or start.hour < wk.work_start_hour:
                row["after_hours_min"] += mins
            if start.weekday() >= 5:
                row["weekend_min"] += mins
            hhmm = start.strftime("%H:%M")
            if row["first_activity"] is None or hhmm < row["first_activity"]:
                row["first_activity"] = hhmm
            end_hhmm = (end or start).strftime("%H:%M")
            if row["last_activity"] is None or end_hhmm > row["last_activity"]:
                row["last_activity"] = end_hhmm
        for row in daily.values():
            if row["first_activity"] and row["last_activity"]:
                h1, m1 = map(int, row["first_activity"].split(":"))
                h2, m2 = map(int, row["last_activity"].split(":"))
                row["active_hours"] = max(0.0, (h2 * 60 + m2 - h1 * 60 - m1) / 60)
        return list(daily.values())


def _ics_dt(v: str) -> str:
    v = v.strip().rstrip("Z")
    if "T" in v and len(v) >= 15:
        return f"{v[0:4]}-{v[4:6]}-{v[6:8]} {v[9:11]}:{v[11:13]}:{v[13:15]}"
    if len(v) == 8:
        return f"{v[0:4]}-{v[4:6]}-{v[6:8]}"
    return v
