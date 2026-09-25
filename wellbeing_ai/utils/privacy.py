"""Data minimisation helpers: hashing, redaction, coarse categorisation."""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Optional

# Patterns that should never make it into storage or a network request.
_PII_PATTERNS = [
    (re.compile(r"\b[\w.\-+]+@[\w\-]+\.[\w.\-]+\b"), "<email>"),
    (re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?\d{10}(?!\d)"), "<phone>"),
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}(?:[\s-]?\d{1,4})?\b"), "<card-or-id>"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "<gov-id>"),
    (re.compile(r"\bhttps?://\S+"), "<url>"),
    (re.compile(r"\b\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}\b"), "<coords>"),
]


def hash_id(value: str, salt: str) -> str:
    """Stable pseudonym for a contact/account. One-way; not reversible."""
    if value is None:
        return ""
    h = hashlib.sha256((salt + "|" + str(value).strip().lower()).encode("utf-8"))
    return "c_" + h.hexdigest()[:12]


def redact(text: Optional[str]) -> str:
    """Strip identifiers out of free text while keeping it readable."""
    if not text:
        return ""
    out = str(text)
    for pat, repl in _PII_PATTERNS:
        out = pat.sub(repl, out)
    return out.strip()


def keywords_only(text: Optional[str], stopwords: Optional[Iterable[str]] = None,
                  limit: int = 12) -> List[str]:
    """Reduce text to a bag of lowercase content words -- no sentences leave."""
    if not text:
        return []
    stop = set(stopwords or _STOP)
    words = re.findall(r"[a-zA-Z']{3,}", redact(text).lower())
    seen, out = set(), []
    for w in words:
        if w in stop or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= limit:
            break
    return out


_STOP = {
    "the", "and", "for", "you", "are", "was", "with", "that", "this", "have",
    "has", "not", "但", "from", "its", "but", "can", "all", "any", "how", "why",
    "what", "when", "who", "will", "would", "should", "could", "did", "does",
    "your", "mine", "our", "their", "there", "here", "just", "like", "get",
    "got", "about", "into", "than", "then", "them", "they", "she", "his", "her",
}
