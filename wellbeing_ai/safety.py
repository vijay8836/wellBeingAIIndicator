"""
Safety rails.

Two jobs:

1. `ClinicalGuard` -- keeps the system inside its lane. It never diagnoses a
   condition and never recommends, names or doses a medication or supplement
   regimen as treatment. Anything that looks clinical is converted into
   "here's what the data shows -- take it to a clinician".

2. `CrisisDetector` -- if signals suggest the person is in real distress, the
   system stops coaching and surfaces human help, warmly and without alarm.
   It is intentionally conservative: it flags for *support*, never for a
   verdict, and it does not store or repeat the text that triggered it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Distress markers. Deliberately phrased as help-seeking/hopelessness signals;
# the detector never records, echoes or catalogues specifics.
_DISTRESS_MARKERS = [
    r"\bno reason to (go on|live|continue)\b",
    r"\bcan'?t (go on|do this any ?more|take (it|this) any ?more)\b",
    r"\bbetter off without me\b",
    r"\bdon'?t want to (be here|wake up|exist)\b",
    r"\bend (it|my life)\b",
    r"\bwant to die\b",
    r"\bhurt myself\b",
    r"\bself[- ]harm\b",
    r"\bsuicid",
    r"\bhopeless\b.*\b(always|forever|never)\b",
    r"\bnobody would (notice|care|miss)\b",
    r"\bhelpline\b|\bcrisis line\b",
]
_DISTRESS_RE = [re.compile(p, re.I) for p in _DISTRESS_MARKERS]

# Secondary, softer signals -- only meaningful in combination + sustained.
_SOFT_MARKERS = [
    r"\bworthless\b", r"\bburden\b", r"\bnumb\b", r"\bempty inside\b",
    r"\bcan'?t feel anything\b", r"\bwhat'?s the point\b",
]
_SOFT_RE = [re.compile(p, re.I) for p in _SOFT_MARKERS]

HELPLINES: Dict[str, List[Dict[str, str]]] = {
    "IN": [
        {"name": "Tele-MANAS (Govt. of India, 24x7, multilingual)",
         "contact": "14416 or 1-800-891-4416"},
        {"name": "KIRAN Mental Health Helpline (24x7)",
         "contact": "1800-599-0019"},
        {"name": "AASRA (24x7)", "contact": "+91-9820466726"},
        {"name": "Vandrevala Foundation (24x7)", "contact": "+91-9999666555"},
    ],
    "US": [
        {"name": "988 Suicide & Crisis Lifeline", "contact": "call or text 988"},
        {"name": "Crisis Text Line", "contact": "text HOME to 741741"},
    ],
    "UK": [{"name": "Samaritans", "contact": "116 123"}],
    "DEFAULT": [
        {"name": "Find a local helpline",
         "contact": "findahelpline.com"},
        {"name": "International Association for Suicide Prevention",
         "contact": "iasp.info/resources/Crisis_Centres"},
    ],
}

# Advice the coach must never produce.
#
# These are deliberately scoped. An early version banned the bare string "mg",
# which quietly rewrote a perfectly good "your caffeine is around 320mg/day"
# tip into a doctor referral. A guard that fires on ordinary dietary units
# isn't protecting anyone -- it just makes the system useless and teaches the
# person to ignore it. So: drug names, prescribing verbs, and dose language
# tied to *taking something*, not every occurrence of a unit.
_BANNED_RECOMMENDATION_TERMS = [
    r"\bprescri\w*",
    r"\bdosage\b",
    r"\bdose\s+of\b",
    r"\b(take|start|try|use)\s+(?:\w+\s+){0,3}\d+\s*(?:mg|mcg|ml|iu)\b",
    r"\b(take|start|taking)\s+(?:a|an|some)?\s*(?:pill|tablet|capsule|"
    r"supplement|medication|medicine)s?\b",
    r"\b(ssri|snri|benzodiazepine|benzo|anxiolytic|antidepressant|"
    r"antipsychotic|beta.?blocker)s?\b",
    r"\bsleeping\s+(?:pill|tablet)s?\b",
    r"\b(zolpidem|alprazolam|sertraline|fluoxetine|escitalopram|clonazepam|"
    r"diazepam|lorazepam|quetiapine|bupropion|modafinil|ambien|xanax)\b",
    r"\b(melatonin|ashwagandha|valerian|st\.?\s*john'?s wort|5-?htp)\b"
    r"(?=.{0,40}\b(take|try|start|mg|supplement|nightly|before bed)\b)",
    r"\b(up|increase|lower|reduce)\s+your\s+dose\b",
    r"\bstack\s+of\s+supplements\b",
]
_BANNED_RE = [re.compile(p, re.I) for p in _BANNED_RECOMMENDATION_TERMS]

_DIAGNOSIS_TERMS = [
    "you have depression", "you are depressed", "you have anxiety disorder",
    "diagnosed with", "this is clinical", "you have bipolar", "you have adhd",
    "you are suffering from",
]


@dataclass
class CrisisFlag:
    triggered: bool = False
    level: str = "none"          # none | watch | urgent
    reasons: List[str] = field(default_factory=list)
    helplines: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"triggered": self.triggered, "level": self.level,
                "reasons": self.reasons, "helplines": self.helplines}


class CrisisDetector:
    """Scans text signals + score trajectory. Never stores the matched text."""

    def __init__(self, locale: str = "IN"):
        self.locale = locale

    def helplines(self) -> List[Dict[str, str]]:
        return HELPLINES.get(self.locale, HELPLINES["DEFAULT"])

    def scan_texts(self, texts: List[str]) -> tuple[int, int]:
        """Returns (hard_hits, soft_hits) -- counts only, never the content."""
        hard = soft = 0
        for t in texts:
            if not t:
                continue
            if any(r.search(t) for r in _DISTRESS_RE):
                hard += 1
            elif any(r.search(t) for r in _SOFT_RE):
                soft += 1
        return hard, soft

    def evaluate(self, texts: Optional[List[str]] = None,
                 mental_score: Optional[float] = None,
                 days_below_40: int = 0,
                 sleep_debt_nights: int = 0,
                 social_isolation_days: int = 0) -> CrisisFlag:
        hard, soft = self.scan_texts(texts or [])
        reasons: List[str] = []
        level = "none"

        if hard > 0:
            level = "urgent"
            reasons.append("language suggesting serious distress")
        elif soft >= 2 and (mental_score is not None and mental_score < 45):
            level = "watch"
            reasons.append("repeated low-mood language alongside a low mood score")

        if days_below_40 >= 5:
            reasons.append(f"mood score below 40 for {days_below_40} days running")
            level = "urgent" if level == "urgent" else "watch"
        if sleep_debt_nights >= 5 and social_isolation_days >= 5:
            reasons.append("sustained poor sleep together with little social contact")
            level = "urgent" if level == "urgent" else "watch"

        flag = CrisisFlag(triggered=level != "none", level=level, reasons=reasons)
        if flag.triggered:
            flag.helplines = self.helplines()
        return flag

    def support_message(self, name: str, flag: CrisisFlag) -> str:
        lines = [
            f"{name}, the last few days look genuinely heavy -- and this is one "
            "of those moments where a person will help more than an app can.",
            "",
            "Please consider reaching out to someone you trust today, or to one "
            "of these, any time of day or night:",
        ]
        for h in flag.helplines:
            lines.append(f"  - {h['name']}: {h['contact']}")
        lines += [
            "",
            "If you're in immediate danger, please contact local emergency "
            "services.",
            "",
            "I'm going to hold off on the usual tips and charts for now. They "
            "aren't what this moment needs.",
        ]
        return "\n".join(lines)


class ClinicalGuard:
    """Filters generated advice so the system stays a coach, not a clinician."""

    @staticmethod
    def violates(text: str) -> List[str]:
        issues = []
        low = (text or "").lower()
        for r in _BANNED_RE:
            if r.search(low):
                issues.append(f"medication-style advice: /{r.pattern}/")
        for d in _DIAGNOSIS_TERMS:
            if d in low:
                issues.append(f"diagnostic claim: '{d}'")
        return issues

    @staticmethod
    def sanitize_recommendation(rec: Dict) -> Dict:
        """Rewrite anything clinical into a 'see a professional' action."""
        blob = " ".join(str(rec.get(k, "")) for k in ("title", "action", "why"))
        if ClinicalGuard.violates(blob):
            return {
                **rec,
                "category": "doctor_consult",
                "title": "Talk this through with a clinician",
                "action": ("Book an appointment with your GP or a qualified "
                           "professional and take the last 2 weeks of this "
                           "report with you."),
                "why": ("Some of what the data suggests is outside what a "
                        "wellbeing tracker should weigh in on -- a clinician "
                        "can look at it properly."),
                "guarded": True,
            }
        return rec

    @staticmethod
    def disclaimer() -> str:
        return ("This is a wellbeing tracker built on your own behavioural data. "
                "It is not a medical device, it does not diagnose anything, and "
                "it does not recommend medication. Patterns it flags are worth "
                "discussing with a qualified clinician, not acting on alone.")
