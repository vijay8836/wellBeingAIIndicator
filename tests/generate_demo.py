"""
Generate ~6 months of realistic synthetic data in the *actual export formats*
the connectors expect, so running the demo exercises the real parsing path.

The person in this data has a story: a steady baseline, a project crunch that
starts around day 95 and grinds them down over six weeks (sleep slips, workouts
stop, messages get shorter and sourer, 3am searching starts, pharmacy visits
appear), a low point, then a deliberate recovery after they take leave.

That arc is what makes the demo worth looking at -- the system should find it
without being told.
"""
from __future__ import annotations

import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

RNG = random.Random(11)
NP = np.random.default_rng(11)

OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
DAYS = 180
START = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) \
    - timedelta(days=DAYS - 1)


# ------------------------------------------------------------------ the arc
def phase(i: int) -> str:
    if i < 95:
        return "baseline"
    if i < 125:
        return "crunch"
    if i < 138:
        return "trough"
    if i < 150:
        return "leave"
    return "recovery"


def strain(i: int) -> float:
    """0 = fine, 1 = worst. Drives every other signal."""
    p = phase(i)
    if p == "baseline":
        base = 0.18 + 0.10 * math.sin(i / 9.0)
    elif p == "crunch":
        base = 0.18 + 0.62 * ((i - 95) / 30.0)
    elif p == "trough":
        base = 0.82 + 0.05 * math.sin(i / 3.0)
    elif p == "leave":
        base = 0.80 - 0.60 * ((i - 138) / 12.0)
    else:
        base = 0.28 - 0.12 * min(1.0, (i - 150) / 25.0) + 0.06 * math.sin(i / 7.0)
    return float(np.clip(base + NP.normal(0, 0.05), 0, 1))


def dates():
    for i in range(DAYS):
        yield i, START + timedelta(days=i)


def jitter(v, sd, lo=None, hi=None):
    out = v + NP.normal(0, sd)
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


# --------------------------------------------------------------------- files
def write_sleep():
    rows = ["date,start,end,duration_min,efficiency,awakenings,deep_min,rem_min,latency_min"]
    for i, d in dates():
        s = strain(i)
        weekend = d.weekday() >= 5
        dur = jitter(450 - 95 * s + (35 if weekend else 0), 28, 220, 560)
        bed_h = 22.8 + 2.2 * s + (0.7 if weekend else 0) + NP.normal(0, 0.45)
        bed = (d - timedelta(days=1)).replace(
            hour=int(bed_h) % 24, minute=int((bed_h % 1) * 60))
        if bed_h >= 24:
            bed += timedelta(days=1)
        wake = bed + timedelta(minutes=dur)
        rows.append(",".join([
            d.strftime("%Y-%m-%d"), bed.strftime("%Y-%m-%d %H:%M"),
            wake.strftime("%Y-%m-%d %H:%M"), f"{dur:.0f}",
            f"{jitter(92 - 18 * s, 3, 55, 99):.0f}",
            f"{max(0, int(jitter(1 + 3.5 * s, 1))):d}",
            f"{jitter(95 - 38 * s, 12, 20, 160):.0f}",
            f"{jitter(105 - 30 * s, 15, 25, 170):.0f}",
            f"{jitter(12 + 34 * s, 7, 3, 95):.0f}",
        ]))
    (OUT / "sleep_export.csv").write_text("\n".join(rows))


def write_activity():
    rows = ["date,steps,active_minutes,workout_minutes,workout_type,calories,"
            "resting_hr,hrv,spo2,weight"]
    for i, d in dates():
        s = strain(i)
        weekend = d.weekday() >= 5
        trains = RNG.random() < max(0.05, 0.62 - 0.55 * s)
        steps = jitter(9200 - 4600 * s + (900 if weekend else 0), 1400, 700, 20000)
        wmin = jitter(48, 14, 20, 95) if trains else 0
        rows.append(",".join([
            d.strftime("%Y-%m-%d"), f"{steps:.0f}",
            f"{jitter(42 - 26 * s, 9, 2, 120):.0f}", f"{wmin:.0f}",
            (RNG.choice(["run", "gym", "cycling", "swim"]) if trains else ""),
            f"{jitter(2380 - 260 * s, 160, 1500, 3600):.0f}",
            f"{jitter(58 + 11 * s, 2.2, 48, 88):.1f}",
            f"{jitter(58 - 21 * s, 5.5, 14, 95):.1f}",
            f"{jitter(97 - 1.2 * s, 0.7, 92, 99):.1f}",
            f"{jitter(74 + 1.6 * s, 0.35, 66, 86):.1f}",
        ]))
    (OUT / "activity_export.csv").write_text("\n".join(rows))


def write_food():
    rows = ["date,meal,item,calories,protein,carbs,fat,fiber,sugar,sodium,"
            "caffeine,alcohol,water"]
    healthy = ["oats with fruit", "dal and rice", "grilled chicken salad",
               "idli sambar", "vegetable curry", "paneer bowl", "fish curry",
               "roti sabzi", "fruit bowl", "curd rice"]
    junk = ["instant noodles", "packaged chips", "cheese pizza", "samosa",
            "chocolate pastry", "burger and fries", "cola", "biscuits"]
    for i, d in dates():
        s = strain(i)
        n_meals = 3 if RNG.random() > 0.15 * s else 2
        water_total = jitter(2600 - 900 * s, 300, 600, 4000)
        for j, meal in enumerate(["breakfast", "lunch", "dinner", "snack"][:n_meals + 1]):
            is_junk = RNG.random() < (0.12 + 0.55 * s)
            item = RNG.choice(junk if is_junk else healthy)
            rows.append(",".join([
                d.strftime("%Y-%m-%d"), meal, item,
                f"{jitter(620 if is_junk else 480, 110, 150):.0f}",
                f"{jitter(9 if is_junk else 26, 6, 1):.0f}",
                f"{jitter(78 if is_junk else 52, 14, 5):.0f}",
                f"{jitter(28 if is_junk else 14, 6, 1):.0f}",
                f"{jitter(2 if is_junk else 9, 2.5, 0):.0f}",
                f"{jitter(26 if is_junk else 7, 7, 0):.0f}",
                f"{jitter(900 if is_junk else 420, 160, 60):.0f}",
                f"{jitter(95 + 140 * s, 45, 0):.0f}" if meal in ("breakfast", "snack") else "0",
                f"{max(0, jitter(2.4 * s, 0.9, 0)):.1f}" if (meal == "dinner" and RNG.random() < 0.15 + 0.4 * s) else "0",
                f"{water_total / (n_meals + 1):.0f}",
            ]))
    (OUT / "food_log.csv").write_text("\n".join(rows))


def write_work():
    rows = ["date,first_activity,last_activity,active_hours,meetings_min,"
            "after_hours_min,weekend_min,context_switches"]
    for i, d in dates():
        s = strain(i)
        weekend = d.weekday() >= 5
        if weekend and s < 0.4:
            rows.append(f"{d:%Y-%m-%d},,,0,0,0,0,0")
            continue
        start_h = jitter(9.2 - 0.6 * s, 0.4, 7.5, 11)
        hours = jitter(7.8 + 4.4 * s, 0.7, 2, 14.5)
        if weekend:
            hours *= 0.55
        end_h = min(23.7, start_h + hours + jitter(0.8, 0.3, 0))
        after = max(0.0, (end_h - 18.5) * 60)
        rows.append(",".join([
            d.strftime("%Y-%m-%d"),
            f"{int(start_h):02d}:{int((start_h % 1) * 60):02d}",
            f"{int(end_h):02d}:{int((end_h % 1) * 60):02d}",
            f"{hours:.2f}", f"{jitter(140 + 230 * s, 45, 0):.0f}",
            f"{after:.0f}", f"{(hours * 60 if weekend else 0):.0f}",
            f"{jitter(85 + 170 * s, 30, 20):.0f}",
        ]))
    (OUT / "work_laptop.csv").write_text("\n".join(rows))


def write_screen():
    rows = ["date,total_min,night_min,pickups,social_min,productivity_min,"
            "entertainment_min"]
    for i, d in dates():
        s = strain(i)
        total = jitter(230 + 180 * s, 35, 60, 620)
        rows.append(",".join([
            d.strftime("%Y-%m-%d"), f"{total:.0f}",
            f"{jitter(22 + 105 * s, 18, 0, 240):.0f}",
            f"{jitter(78 + 120 * s, 22, 20):.0f}",
            f"{total * jitter(0.24 + 0.16 * s, 0.05, 0.05, 0.7):.0f}",
            f"{total * 0.22:.0f}",
            f"{total * jitter(0.26 + 0.1 * s, 0.05, 0.05, 0.6):.0f}",
        ]))
    (OUT / "screen_wellbeing.csv").write_text("\n".join(rows))


def write_sensors():
    rows = ["date,unlocks,ambient_light,outdoor_minutes,movement_entropy,"
            "ambient_noise,location_variance"]
    for i, d in dates():
        s = strain(i)
        rows.append(",".join([
            d.strftime("%Y-%m-%d"), f"{jitter(80 + 115 * s, 20, 15):.0f}",
            f"{jitter(2400 - 1750 * s, 350, 90):.0f}",
            f"{max(0, jitter(62 - 52 * s, 14, 0)):.0f}",
            f"{jitter(0.72 - 0.34 * s, 0.07, 0.05, 1):.3f}",
            f"{jitter(54 + 6 * s, 3, 30):.0f}",
            f"{jitter(0.65 - 0.42 * s, 0.09, 0.02, 1):.3f}",
        ]))
    (OUT / "phone_sensors.csv").write_text("\n".join(rows))


def write_places():
    rows = ["timestamp,place,minutes"]
    for i, d in dates():
        s = strain(i)
        weekend = d.weekday() >= 5
        def add(hour, name, mins):
            ts = d.replace(hour=hour, minute=RNG.randint(0, 59))
            rows.append(f"{ts:%Y-%m-%d %H:%M},{name},{mins:.0f}")
        add(7, "Home", 600)
        if not weekend:
            add(10, "Prestige Tech Park Office", jitter(480 + 180 * s, 60, 240))
        if RNG.random() < max(0.05, 0.55 - 0.5 * s):
            add(18, "Cult.fit Gym Indiranagar", jitter(65, 12, 35))
        if RNG.random() < max(0.02, 0.28 - 0.26 * s):
            add(7, "Cubbon Park", jitter(45, 12, 20))
        if RNG.random() < (0.10 + 0.30 * (1 - s)):
            add(19, "Sri Someshwara Temple", 25)
        if RNG.random() < (0.05 + 0.34 * s):
            add(20, "Apollo Pharmacy Koramangala", 12)
        if RNG.random() < (0.01 + 0.10 * s):
            add(11, "Manipal Hospital", jitter(95, 30, 30))
        if RNG.random() < max(0.03, 0.42 - 0.38 * s):
            add(20, "Toit Brewpub / friends", jitter(120, 35, 45))
        if RNG.random() < (0.06 + 0.26 * s):
            add(21, "Yoga & Wellness Centre", 60) if RNG.random() < 0.4 else None
        if RNG.random() < (0.12 + 0.35 * s):
            add(13, "Truffles Restaurant", 55)
    (OUT / "places_visits.csv").write_text("\n".join(r for r in rows if r))


def write_spending_sms():
    """SMS Backup & Restore XML -- the real format."""
    root = ET.Element("smses", count="0")
    templates = [
        ("medical", "Rs.{amt} debited from A/c XX4412 on {dt} at APOLLO PHARMACY. Avl Bal Rs.48210"),
        ("food_delivery", "INR {amt} spent on your card at SWIGGY on {dt}"),
        ("groceries", "Rs.{amt} debited towards BLINKIT on {dt}"),
        ("transport", "Rs.{amt} paid to UBER INDIA on {dt}"),
        ("shopping", "Rs.{amt} debited at AMAZON PAY on {dt}"),
        ("alcohol", "Rs.{amt} spent at TOIT BREWPUB on {dt}"),
        ("fitness", "Rs.{amt} debited towards CULT FIT on {dt}"),
        ("entertainment", "INR {amt} spent at BOOKMYSHOW on {dt}"),
        ("bills", "Rs.{amt} debited towards BESCOM ELECTRICITY on {dt}"),
    ]
    for i, d in dates():
        s = strain(i)
        n = max(1, int(jitter(3.2 + 2.6 * s, 1.3, 0)))
        for _ in range(n):
            weights = {"medical": 0.04 + 0.22 * s,
                       "food_delivery": 0.12 + 0.26 * s,
                       "alcohol": 0.03 + 0.14 * s,
                       "fitness": 0.10 * (1 - s),
                       "groceries": 0.16, "transport": 0.14,
                       "shopping": 0.12 + 0.08 * s,
                       "entertainment": 0.08, "bills": 0.06}
            cats = list(weights)
            pick = RNG.choices(cats, weights=[weights[c] for c in cats])[0]
            body_t = dict(templates)[pick]
            amt = {"medical": 620, "food_delivery": 480, "alcohol": 1450,
                   "fitness": 1800, "groceries": 900, "transport": 260,
                   "shopping": 1900, "entertainment": 700,
                   "bills": 2600}[pick]
            hour = RNG.randint(23, 25) % 24 if (RNG.random() < 0.05 + 0.25 * s) \
                else RNG.randint(9, 21)
            ts = d.replace(hour=hour, minute=RNG.randint(0, 59))
            ET.SubElement(root, "sms", {
                "protocol": "0", "address": "HDFCBK", "type": "1",
                "date": str(int(ts.timestamp() * 1000)),
                "body": body_t.format(amt=f"{jitter(amt, amt * 0.3, 40):.0f}",
                                      dt=ts.strftime("%d-%m-%y")),
                "readable_date": ts.strftime("%d %b %Y %H:%M:%S"),
            })
    root.set("count", str(len(root)))
    ET.ElementTree(root).write(OUT / "sms-transactions.xml", encoding="utf-8",
                               xml_declaration=True)


def write_search_history():
    """Google Takeout MyActivity.json format."""
    pools = {
        "low": ["best trekking shoes", "python list comprehension",
                "weekend getaways near bangalore", "how to cook dal makhani",
                "ipl points table", "good books 2026", "hrv meaning",
                "half marathon training plan", "best coffee beans india"],
        "high": ["why am i so tired all the time", "chest tightness stress",
                 "can't sleep at night what to do", "burnout symptoms",
                 "is it anxiety or something else", "how to stop overthinking",
                 "should i quit my job", "manager toxic what to do",
                 "headache every evening", "acidity home remedy",
                 "therapist near koramangala", "how to calm down fast",
                 "notice period rules india", "vitamin d deficiency symptoms",
                 "feeling exhausted even after sleeping"],
    }
    items = []
    for i, d in dates():
        s = strain(i)
        n = max(2, int(jitter(9 + 12 * s, 3, 1)))
        for _ in range(n):
            q = RNG.choice(pools["high"] if RNG.random() < s * 0.85
                           else pools["low"])
            hour = (RNG.randint(23, 27) % 24 if RNG.random() < 0.04 + 0.30 * s
                    else RNG.randint(8, 22))
            ts = d.replace(hour=hour, minute=RNG.randint(0, 59))
            items.append({"header": "Search", "title": f"Searched for {q}",
                          "time": ts.isoformat() + "Z",
                          "products": ["Search"]})
    (OUT / "MyActivity.json").write_text(json.dumps(items, indent=1))


def write_youtube():
    titles = {
        "low": ["Bengaluru street food tour", "Kotlin in 100 seconds",
                "Highlights: India vs Australia", "Full body mobility routine",
                "Jazz for a Sunday morning"],
        "high": ["How to stop overthinking - ASMR guided",
                 "10 hours rain sounds for sleep", "Why you're always tired",
                 "Signs of burnout you're ignoring #shorts",
                 "Toxic workplace horror stories reaction",
                 "Guided meditation for anxiety relief",
                 "Discipline over motivation - stop wasting your life"],
    }
    items = []
    for i, d in dates():
        s = strain(i)
        n = max(2, int(jitter(7 + 14 * s, 3, 1)))
        for _ in range(n):
            t = RNG.choice(titles["high"] if RNG.random() < s * 0.8
                           else titles["low"])
            hour = (RNG.randint(23, 27) % 24 if RNG.random() < 0.05 + 0.35 * s
                    else RNG.randint(9, 22))
            ts = d.replace(hour=hour, minute=RNG.randint(0, 59))
            items.append({"header": "YouTube", "title": f"Watched {t}",
                          "time": ts.isoformat() + "Z"})
    (OUT / "watch-history.json").write_text(json.dumps(items, indent=1))


def write_whatsapp():
    """WhatsApp 'export chat' .txt format, one file per contact."""
    contacts = {
        "Amma": {"warm": True},
        "Rohit": {"warm": True},
        "Priya": {"warm": True},
        "Team Standup Group": {"warm": False, "group": True},
    }
    good_out = ["haha yes let's do it", "sounds great, see you saturday",
                "just finished a great run, feeling good",
                "thanks so much, that really helped",
                "love you, will call tonight", "had a lovely evening, thank you"]
    bad_out = ["sorry can't make it, work is crazy",
               "i'm so exhausted, this deadline is killing me",
               "not feeling great today, rain check?",
               "honestly i'm just drained, everything feels like too much",
               "sorry for the late reply, been a rough week",
               "can't sleep again, this is frustrating", "maybe next time"]
    neutral_out = ["ok", "sure", "will check", "got it", "on my way", "yes"]
    replies = ["ok take care", "no worries!", "call me when free",
               "hope you're okay", "let me know", "sure thing", "haha okay"]

    for name, meta in contacts.items():
        lines = []
        for i, d in dates():
            s = strain(i)
            base = 9 if meta.get("warm") else 5
            n = max(0, int(jitter(base * (1 - 0.65 * s), 2.2, 0)))
            for _ in range(n):
                hour = (RNG.randint(23, 26) % 24 if RNG.random() < 0.03 + 0.25 * s
                        else RNG.randint(8, 22))
                ts = d.replace(hour=hour, minute=RNG.randint(0, 59))
                if RNG.random() < 0.5:
                    if RNG.random() < s * 0.8:
                        msg = RNG.choice(bad_out)
                    elif RNG.random() < 0.4 + 0.3 * s:
                        msg = RNG.choice(neutral_out)
                    else:
                        msg = RNG.choice(good_out)
                    sender = "Vijay"
                else:
                    msg, sender = RNG.choice(replies), name.split()[0]
                lines.append(f"{ts:%d/%m/%Y, %H:%M} - {sender}: {msg}")
        fn = f"WhatsApp Chat with {name}.txt"
        (OUT / fn).write_text(
            "\n".join(["01/01/2026, 00:00 - Messages and calls are "
                       "end-to-end encrypted."] + lines), encoding="utf-8")


def write_notes():
    rows = ["date,text"]
    low = ["can't switch off. again.", "everything feels like too much lately",
           "why am i always tired", "should i just quit",
           "nothing i do is ever enough", "didn't sleep again",
           "always behind, never caught up", "skipped gym again"]
    high = ["good run this morning, felt strong", "call amma sunday",
            "pick up groceries", "plan trip to coorg",
            "finished the report, relieved", "book badminton court"]
    health = ["doctor appointment thursday 6pm", "bp check at clinic",
              "pharmacy: refill", "dentist next month",
              "blood test fasting saturday"]
    for i, d in dates():
        s = strain(i)
        if RNG.random() < 0.45 + 0.3 * s:
            pool = low if RNG.random() < s else high
            if RNG.random() < 0.06 + 0.22 * s:
                pool = health
            rows.append(f"{d:%Y-%m-%d},\"{RNG.choice(pool)}\"")
    (OUT / "notes_export.csv").write_text("\n".join(rows))


def write_checkins():
    rows = ["date,mood,energy,stress,note"]
    for i, d in dates():
        if RNG.random() < 0.55:
            s = strain(i)
            rows.append(",".join([
                d.strftime("%Y-%m-%d"),
                f"{int(np.clip(round(jitter(4.3 - 2.6 * s, 0.45)), 1, 5))}",
                f"{int(np.clip(round(jitter(4.1 - 2.5 * s, 0.5)), 1, 5))}",
                f"{int(np.clip(round(jitter(1.9 + 2.6 * s, 0.5)), 1, 5))}",
                "",
            ]))
    (OUT / "mood_checkins.csv").write_text("\n".join(rows))


def write_social():
    rows = ["timestamp,platform,action,text"]
    for i, d in dates():
        s = strain(i)
        if RNG.random() < max(0.05, 0.4 - 0.33 * s):
            ts = d.replace(hour=RNG.randint(10, 22))
            txt = RNG.choice([
                "Sunday trek done. Beautiful morning",
                "Great dinner with old friends",
                "Finally finished this one",
                "New personal best today",
            ])
            rows.append(f"{ts:%Y-%m-%d %H:%M},instagram,post,\"{txt}\"")
        for _ in range(int(jitter(6 + 9 * s, 3, 0))):
            ts = d.replace(hour=RNG.randint(9, 23))
            rows.append(f"{ts:%Y-%m-%d %H:%M},instagram,like,\"\"")
    (OUT / "social_activity.csv").write_text("\n".join(rows))


def write_bookmarks():
    rows = ["date,url,title"]
    for i, d in dates():
        s = strain(i)
        if RNG.random() < 0.2 + 0.4 * s:
            t, u = RNG.choice([
                ("Burnout: symptoms and recovery", "https://who.int/burnout"),
                ("Best trek in Coorg", "https://tripadvisor.in/coorg"),
                ("Sleep hygiene checklist", "https://sleepfoundation.org/x"),
                ("How to resign gracefully", "https://hbr.org/resign"),
                ("Beginner yoga for back pain", "https://youtube.com/yoga"),
            ])
            rows.append(f"{d:%Y-%m-%d},{u},\"{t}\"")
    (OUT / "bookmarks_export.csv").write_text("\n".join(rows))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*"):
        if f.is_file():
            f.unlink()
    steps = [write_sleep, write_activity, write_food, write_work, write_screen,
             write_sensors, write_places, write_spending_sms,
             write_search_history, write_youtube, write_whatsapp, write_notes,
             write_checkins, write_social, write_bookmarks]
    for fn in steps:
        fn()
        print(f"  wrote {fn.__name__.replace('write_', '')}")
    total = sum(f.stat().st_size for f in OUT.glob("*") if f.is_file())
    print(f"\n{len(list(OUT.glob('*')))} files, {total / 1024:.0f} KB in {OUT}")
    print(f"Date range: {START:%Y-%m-%d} .. "
          f"{(START + timedelta(days=DAYS - 1)):%Y-%m-%d}")


if __name__ == "__main__":
    main()
