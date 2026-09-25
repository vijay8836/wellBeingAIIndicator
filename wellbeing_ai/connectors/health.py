"""Body-facing connectors: sleep, activity, food, screen time, phone sensors."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .base import Connector, col, day_of, num, parse_ts, read_table


class SleepConnector(Connector):
    """
    Sleep from Health Connect / Fitbit / Oura / Samsung Health CSV exports.

    Expected-ish columns (any casing, extras ignored):
      date | start/bedtime | end/waketime | duration | efficiency |
      awakenings | deep | rem | latency
    """

    name = "sleep_csv"
    table = "sleep"
    consent_key = "sleep"
    patterns = ["*sleep*.csv", "*sleep*.json", "*sleep*.xlsx"]

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        df = read_table(path)
        if df.empty:
            return []
        c_start = col(df, "start", "bedtime", "sleepstart", "starttime", "date")
        c_end = col(df, "end", "waketime", "sleepend", "endtime")
        c_dur = col(df, "durationmin", "duration", "minutesasleep", "totalsleep",
                    "sleepduration", "hours")
        rows = []
        for _, r in df.iterrows():
            start = parse_ts(r.get(c_start)) if c_start else None
            end = parse_ts(r.get(c_end)) if c_end else None
            dur = num(r.get(c_dur)) if c_dur else None
            if dur is not None and dur < 24:       # given in hours
                dur *= 60
            if dur is None and start and end:
                dur = (end - start).total_seconds() / 60
            if dur is None:
                continue
            # The night belongs to the wake-up date.
            anchor = end or (start + timedelta(minutes=dur) if start else None)
            if anchor is None:
                continue
            rows.append({
                "date": day_of(anchor),
                "bedtime": start.isoformat() if start else None,
                "waketime": end.isoformat() if end else None,
                "duration_min": round(dur, 1),
                "efficiency": num(r.get(col(df, "efficiency", "sleepscore",
                                            "quality"))),
                "awakenings": num(r.get(col(df, "awakenings", "wakecount",
                                            "timesawake"))),
                "deep_min": num(r.get(col(df, "deep", "deepsleep",
                                          "deepminutes"))),
                "rem_min": num(r.get(col(df, "rem", "remsleep", "remminutes"))),
                "latency_min": num(r.get(col(df, "latency", "timetofallasleep",
                                             "minutestofallasleep"))),
            })
        return rows


class ActivityConnector(Connector):
    """Steps / workouts / heart metrics from a fitness-app CSV export."""

    name = "activity_csv"
    table = "activity"
    consent_key = "activity"
    patterns = ["*activity*.csv", "*workout*.csv", "*steps*.csv",
                "*activity*.json", "*fitness*.csv"]

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        df = read_table(path)
        if df.empty:
            return []
        c_date = col(df, "date", "day", "starttime", "timestamp")
        if not c_date:
            return []
        # A file can match "*activity*.csv" and still not be a fitness export
        # (social_activity.csv, for one). Require at least one real metric.
        if not any(col(df, *c) for c in
                   [("steps", "stepcount"), ("activeminutes", "activemin"),
                    ("calories", "caloriesburned"), ("restinghr", "rhr"),
                    ("hrv",), ("workoutminutes", "durationmin"),
                    ("workouttype", "activitytype")]):
            return []
        agg: Dict[str, Dict[str, Any]] = {}
        for _, r in df.iterrows():
            ts = parse_ts(r.get(c_date))
            d = day_of(ts)
            if not d:
                continue
            row = agg.setdefault(d, {"date": d, "workout_min": 0.0,
                                     "workout_type": None})
            for key, cands in [
                ("steps", ("steps", "stepcount", "totalsteps")),
                ("active_min", ("activeminutes", "activemin", "movemin",
                                "exerciseminutes", "veryactiveminutes")),
                ("calories_out", ("calories", "caloriesburned", "energyburned",
                                  "activecalories")),
                ("resting_hr", ("restinghr", "restingheartrate", "rhr")),
                ("hrv_ms", ("hrv", "heartratevariability", "rmssd")),
                ("spo2", ("spo2", "oxygensaturation")),
                ("weight_kg", ("weight", "weightkg", "bodyweight")),
            ]:
                c = col(df, *cands)
                if c is not None:
                    v = num(r.get(c))
                    if v is not None:
                        row[key] = v if key not in row else max(row[key] or 0, v) \
                            if key == "steps" else v
            c_wmin = col(df, "workoutminutes", "durationmin", "duration",
                         "exercisemin")
            if c_wmin is not None:
                v = num(r.get(c_wmin))
                if v is not None:
                    row["workout_min"] = (row.get("workout_min") or 0) + (
                        v * 60 if v < 12 else v)
            c_wtype = col(df, "workouttype", "activitytype", "type", "exercise")
            if c_wtype is not None and pd.notna(r.get(c_wtype)):
                row["workout_type"] = str(r.get(c_wtype))[:40]
        return list(agg.values())


class AppleHealthConnector(Connector):
    """Apple Health `export.xml` -- pulls sleep, steps, HR, HRV into both tables."""

    name = "apple_health_xml"
    table = "activity"
    consent_key = "activity"
    patterns = ["export.xml", "*apple*health*.xml"]

    TYPE_MAP = {
        "HKQuantityTypeIdentifierStepCount": "steps",
        "HKQuantityTypeIdentifierActiveEnergyBurned": "calories_out",
        "HKQuantityTypeIdentifierRestingHeartRate": "resting_hr",
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv_ms",
        "HKQuantityTypeIdentifierOxygenSaturation": "spo2",
        "HKQuantityTypeIdentifierBodyMass": "weight_kg",
        "HKQuantityTypeIdentifierAppleExerciseTime": "active_min",
    }

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        daily: Dict[str, Dict[str, Any]] = defaultdict(dict)
        sleep_rows: Dict[str, Dict[str, Any]] = {}
        sums = {"steps", "calories_out", "active_min"}
        for _, el in ET.iterparse(str(path), events=("end",)):
            if el.tag != "Record":
                el.clear()
                continue
            rtype = el.get("type", "")
            start = parse_ts(el.get("startDate"))
            end = parse_ts(el.get("endDate"))
            if rtype == "HKCategoryTypeIdentifierSleepAnalysis":
                if start and end and "Asleep" in (el.get("value") or ""):
                    d = day_of(end)
                    row = sleep_rows.setdefault(
                        d, {"date": d, "duration_min": 0.0, "bedtime": None,
                            "waketime": None})
                    row["duration_min"] += (end - start).total_seconds() / 60
                    if row["bedtime"] is None or start.isoformat() < row["bedtime"]:
                        row["bedtime"] = start.isoformat()
                    if row["waketime"] is None or end.isoformat() > row["waketime"]:
                        row["waketime"] = end.isoformat()
            elif rtype in self.TYPE_MAP:
                field = self.TYPE_MAP[rtype]
                d = day_of(start)
                v = num(el.get("value"))
                if d and v is not None:
                    row = daily[d]
                    row["date"] = d
                    row[field] = (row.get(field, 0) + v) if field in sums else v
            el.clear()
        if sleep_rows:
            self.store.upsert_daily("sleep", [
                {**r, "duration_min": round(r["duration_min"], 1),
                 "source": self.name} for r in sleep_rows.values()])
        return list(daily.values())


class FoodConnector(Connector):
    """Food log CSV (MyFitnessPal / HealthifyMe / manual)."""

    name = "food_csv"
    table = "food"
    consent_key = "food"
    patterns = ["*food*.csv", "*nutrition*.csv", "*meal*.csv", "*diet*.csv"]

    ULTRAPROCESSED = ("chips", "soda", "cola", "biscuit", "cookie", "candy",
                      "instant", "noodles", "fries", "burger", "pizza", "pastry",
                      "ice cream", "packaged", "energy drink", "samosa", "cake")

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        df = read_table(path)
        if df.empty:
            return []
        c_date = col(df, "date", "day", "timestamp")
        if not c_date:
            return []
        rows = []
        for _, r in df.iterrows():
            d = day_of(parse_ts(r.get(c_date)))
            if not d:
                continue
            item = str(r.get(col(df, "item", "food", "name", "description")
                              or "", "") or "")[:120]
            rows.append({
                "date": d,
                "meal": str(r.get(col(df, "meal", "mealtype", "category") or "",
                                  "") or "")[:20],
                "item": item,
                "calories": num(r.get(col(df, "calories", "energy", "kcal"))),
                "protein_g": num(r.get(col(df, "protein", "proteing"))),
                "carbs_g": num(r.get(col(df, "carbs", "carbohydrate"))),
                "fat_g": num(r.get(col(df, "fat", "fatg"))),
                "fiber_g": num(r.get(col(df, "fiber", "fibre"))),
                "sugar_g": num(r.get(col(df, "sugar", "sugars"))),
                "sodium_mg": num(r.get(col(df, "sodium", "salt"))),
                "caffeine_mg": num(r.get(col(df, "caffeine"))),
                "alcohol_units": num(r.get(col(df, "alcohol", "alcoholunits"))),
                "water_ml": num(r.get(col(df, "water", "waterml", "hydration"))),
                "ultraprocessed": int(any(u in item.lower()
                                          for u in self.ULTRAPROCESSED)),
            })
        return rows


class ScreenTimeConnector(Connector):
    """Digital Wellbeing / Screen Time daily export."""

    name = "screentime_csv"
    table = "screen"
    consent_key = "screen_time"
    patterns = ["*screen*.csv", "*wellbeing*.csv", "*usage*.csv",
                "*screentime*.json"]

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        df = read_table(path)
        if df.empty:
            return []
        c_date = col(df, "date", "day")
        if not c_date:
            return []
        rows = []
        for _, r in df.iterrows():
            d = day_of(parse_ts(r.get(c_date)))
            if not d:
                continue

            def g(*cands):
                c = col(df, *cands)
                v = num(r.get(c)) if c is not None else None
                return v * 60 if (v is not None and v < 25) else v

            rows.append({
                "date": d,
                "total_min": g("totalmin", "totalscreentime", "screentime",
                               "total"),
                "night_min": g("nightmin", "nightusage", "latenight"),
                "pickups": num(r.get(col(df, "pickups", "unlocks",
                                         "notifications") or "")),
                "social_min": g("social", "socialmedia"),
                "work_min": g("productivity", "work"),
                "entertainment_min": g("entertainment", "video", "streaming"),
            })
        return rows


class SensorConnector(Connector):
    """Phone sensor aggregates: light exposure, movement entropy, GPS variance."""

    name = "sensors_csv"
    table = "sensors"
    consent_key = "phone_sensors"
    patterns = ["*sensor*.csv", "*ambient*.csv", "*phone_metrics*.csv"]

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        df = read_table(path)
        if df.empty:
            return []
        c_date = col(df, "date", "day", "timestamp")
        if not c_date:
            return []
        rows = []
        for _, r in df.iterrows():
            d = day_of(parse_ts(r.get(c_date)))
            if not d:
                continue
            rows.append({
                "date": d,
                "unlocks": num(r.get(col(df, "unlocks", "pickups"))),
                "ambient_light_lux": num(r.get(col(df, "light", "lux",
                                                   "ambientlight"))),
                "outdoor_min": num(r.get(col(df, "outdoor", "outdoorminutes",
                                             "sunlight", "daylight"))),
                "movement_entropy": num(r.get(col(df, "entropy",
                                                  "movemententropy"))),
                "ambient_noise_db": num(r.get(col(df, "noise", "db",
                                                  "ambientnoise"))),
                "location_variance": num(r.get(col(df, "locationvariance",
                                                   "locvar", "gpsvariance"))),
            })
        return rows


class CheckinConnector(Connector):
    """Optional self-report: the ground truth that makes everything else honest."""

    name = "checkins_csv"
    table = "checkins"
    consent_key = "notes"
    patterns = ["*checkin*.csv", "*mood*.csv", "*journal*.csv"]

    def parse(self, path: Path) -> List[Dict[str, Any]]:
        df = read_table(path)
        if df.empty:
            return []
        c_date = col(df, "date", "day")
        if not c_date:
            return []
        rows = []
        for _, r in df.iterrows():
            d = day_of(parse_ts(r.get(c_date)))
            if not d:
                continue
            rows.append({
                "date": d,
                "mood": num(r.get(col(df, "mood", "feeling"))),
                "energy": num(r.get(col(df, "energy"))),
                "stress": num(r.get(col(df, "stress"))),
                "note": str(r.get(col(df, "note", "comment") or "", "") or "")[:500],
            })
        return rows
