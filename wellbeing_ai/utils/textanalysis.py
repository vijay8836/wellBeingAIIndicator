"""
Lightweight on-device NLP: sentiment, topic category, rumination markers.

Deliberately lexicon-based so it runs offline with zero model downloads and
no text ever leaves the machine. If `transformers` happens to be installed
and `use_transformer=True`, a small sentiment model is used instead.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- lexicons
POSITIVE = {
    "good": 1, "great": 2, "happy": 2, "love": 2, "excited": 2, "awesome": 2,
    "thanks": 1, "thank": 1, "glad": 1.5, "proud": 2, "win": 1.5, "won": 1.5,
    "enjoy": 1.5, "enjoyed": 1.5, "fun": 1.5, "relaxed": 1.5, "calm": 1.5,
    "rested": 1.5, "energised": 2, "energized": 2, "better": 1, "best": 1.5,
    "amazing": 2, "beautiful": 1.5, "hopeful": 2, "grateful": 2, "peaceful": 2,
    "celebrate": 2, "congrats": 2, "yay": 1.5, "nice": 1, "lovely": 1.5,
    "progress": 1.2, "achieved": 1.5, "confident": 1.5, "refreshed": 1.5,
}
NEGATIVE = {
    "tired": -1.5, "exhausted": -2, "stress": -2, "stressed": -2, "anxious": -2,
    "anxiety": -2, "worried": -1.8, "worry": -1.5, "sad": -2, "down": -1.2,
    "angry": -2, "annoyed": -1.5, "frustrated": -2, "hate": -2, "awful": -2,
    "terrible": -2, "bad": -1.2, "worse": -1.5, "worst": -2, "pain": -1.8,
    "sick": -1.8, "headache": -1.5, "burnt": -2, "burnout": -2.5, "drained": -2,
    "overwhelmed": -2.2, "lonely": -2.2, "alone": -1.2, "empty": -1.8,
    "hopeless": -2.8, "pointless": -2.2, "useless": -2.2, "failure": -2.2,
    "failed": -1.8, "guilty": -1.8, "ashamed": -2, "scared": -1.8, "afraid": -1.8,
    "panic": -2.5, "insomnia": -2, "cant": -0.8, "sleepless": -2, "sorry": -0.6,
    "problem": -1, "issue": -0.8, "deadline": -1.2, "pressure": -1.5,
    "argument": -1.8, "fight": -1.8, "crying": -2.2, "cried": -2.2, "numb": -2.2,
}
INTENSIFIERS = {"very": 1.5, "really": 1.4, "so": 1.3, "extremely": 1.8,
                "too": 1.3, "super": 1.4, "totally": 1.3, "absolutely": 1.6}
NEGATORS = {"not", "no", "never", "dont", "don't", "cant", "can't", "isnt",
            "isn't", "wasnt", "wasn't", "wouldnt", "aint", "hardly", "barely"}

# Rumination / cognitive-distortion markers (frequency matters, not content).
RUMINATION = {
    "always", "never", "everyone", "nobody", "everything", "nothing",
    "should", "shouldnt", "must", "why me", "what if", "cant stop",
    "keep thinking", "over and over", "my fault", "ruined", "again and again",
}

# Search/browse topic buckets.
CATEGORY_PATTERNS: List[Tuple[str, List[str]]] = [
    ("mental_health", ["anxiety", "depress", "panic attack", "therapist",
                       "counsell", "counsel", "psychiatr", "burnout",
                       "cant sleep", "insomnia", "overthinking", "stress relief",
                       "mental health", "mood swing", "meditation for anxiety"]),
    ("physical_health", ["symptom", "pain", "fever", "doctor near", "clinic",
                         "blood test", "bp ", "sugar level", "diabet",
                         "cholesterol", "medicine for", "side effect", "dosage",
                         "hospital", "pharmacy", "chemist", "rash", "cough",
                         "acidity", "migraine", "back pain", "physiotherap"]),
    ("fitness", ["workout", "gym", "exercise", "running", "yoga", "pilates",
                 "stretch", "protein", "cardio", "strength training", "10k"]),
    ("nutrition", ["recipe", "calorie", "diet", "meal plan", "nutrition",
                   "vitamin", "healthy food", "intermittent fasting"]),
    ("work", ["jira", "excel formula", "python error", "sql", "presentation",
              "deadline", "resign", "appraisal", "interview", "resume", "salary",
              "job change", "notice period", "layoff", "manager"]),
    ("finance", ["emi", "loan", "credit card", "tax", "mutual fund", "invest",
                 "insurance", "rent", "price of"]),
    ("relationships", ["girlfriend", "boyfriend", "wife", "husband", "partner",
                       "breakup", "argument with", "family", "parents",
                       "in laws", "marriage"]),
    ("travel", ["flight", "hotel", "trip", "vacation", "holiday", "itinerary",
                "things to do in", "booking", "resort", "weekend getaway"]),
    ("leisure", ["movie", "series", "song", "game", "netflix", "prime video",
                 "cricket", "football", "recipe", "meme"]),
    ("sleep", ["sleep", "melatonin", "nap", "circadian", "wake up early"]),
]

PLACE_CATEGORIES = {
    "hospital": ["hospital", "clinic", "medical cent", "diagnost", "lab",
                 "nursing home", "emergency"],
    "pharmacy": ["pharmac", "chemist", "medical store", "apollo pharmacy",
                 "medplus", "drug store"],
    "gym": ["gym", "fitness", "crossfit", "cult.fit", "sports complex"],
    "yoga": ["yoga", "meditation", "wellness cent", "ayurveda", "spa",
             "mindfulness"],
    "worship": ["temple", "mandir", "church", "mosque", "gurudwara",
                "dargah", "shrine"],
    "park": ["park", "lake", "garden", "trail", "beach", "forest", "hill"],
    "office": ["office", "tech park", "campus", "workplace", "coworking"],
    "home": ["home", "residence", "apartment", "house"],
    "restaurant": ["restaurant", "cafe", "coffee", "diner", "eatery", "hotel "],
    "bar": ["bar", "pub", "brewery", "liquor", "wine shop", "lounge"],
    "social": ["mall", "cinema", "theatre", "club", "stadium", "friend"],
    "transit": ["airport", "station", "metro", "bus stand", "terminal"],
}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", (text or "").lower())


def sentiment(text: Optional[str]) -> float:
    """Return a score in [-1, 1]. 0 means neutral / no signal."""
    toks = _tokens(text)
    if not toks:
        return 0.0
    total, hits = 0.0, 0
    for i, tok in enumerate(toks):
        val = POSITIVE.get(tok, 0.0) or NEGATIVE.get(tok, 0.0)
        if not val:
            continue
        mult = 1.0
        if i > 0:
            prev = toks[i - 1]
            mult *= INTENSIFIERS.get(prev, 1.0)
            if prev in NEGATORS:
                mult *= -0.8
        if i > 1 and toks[i - 2] in NEGATORS:
            mult *= -0.8
        total += val * mult
        hits += 1
    if hits == 0:
        return 0.0
    # Squash: long angry messages shouldn't blow past the range.
    return max(-1.0, min(1.0, total / (2.0 * math.sqrt(hits) + 1e-9) / 1.5))


def emotion_label(score: float) -> str:
    if score >= 0.35:
        return "positive"
    if score <= -0.45:
        return "distressed"
    if score <= -0.15:
        return "negative"
    return "neutral"


def rumination_score(text: Optional[str]) -> float:
    """0-1: density of absolutist / self-critical language."""
    low = (text or "").lower()
    toks = _tokens(low)
    if len(toks) < 4:
        return 0.0
    hits = sum(1 for m in RUMINATION if m in low)
    hits += sum(1 for t in toks if t in {"always", "never", "should", "must"})
    return min(1.0, hits / max(6.0, len(toks) / 6.0))


def categorize(text: Optional[str]) -> str:
    low = (text or "").lower()
    for name, pats in CATEGORY_PATTERNS:
        if any(p in low for p in pats):
            return name
    return "other"


def categorize_place(label: Optional[str]) -> str:
    low = (label or "").lower()
    for cat, pats in PLACE_CATEGORIES.items():
        if any(p in low for p in pats):
            return cat
    return "other"


@lru_cache(maxsize=1)
def _transformer():  # pragma: no cover -- optional dependency
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception:
        return None


def sentiment_batch(texts: List[str], use_transformer: bool = False) -> List[float]:
    if use_transformer:
        pipe = _transformer()
        if pipe is not None:  # pragma: no cover
            out = []
            for t in texts:
                try:
                    r = pipe(t[:512])[0]
                    s = r["score"] if r["label"] == "POSITIVE" else -r["score"]
                    out.append(float(s))
                except Exception:
                    out.append(sentiment(t))
            return out
    return [sentiment(t) for t in texts]
