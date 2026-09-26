"""Simple pair features shared by the fidelity proxy and the French pair builder.

All features compare a Source 1 record (name1, addr1) with a candidate record (name2, addr2).
They are descriptive statistics for checking the generator, not model features.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from synth_world import LEGAL_ALL, fold

LEGAL = set(LEGAL_ALL)
ABBR = {"st", "rd", "ave", "av", "dr", "ln", "ct", "blvd", "pl", "hwy", "pkwy",
        "r", "bd", "imp", "ch", "all", "rte"}
STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "en", "au", "aux", "a", "the", "of", "and"}
_DOT = re.compile(r"\b([a-z])\.([a-z])(?:\.([a-z]))?\.?")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")


def norm(text: str) -> list[str]:
    s = fold(text).lower()
    s = _DOT.sub(lambda m: "".join(g for g in m.groups() if g), s)
    return _NONALNUM.sub(" ", s).split()


def core_key(name: str) -> str:
    return " ".join(t for t in norm(name) if t not in LEGAL)


def digit_key(addr: str) -> str:
    return " ".join(sorted(_DIGITS.findall(addr)))


def grams(tokens: list[str]) -> Counter:
    s = " " + " ".join(tokens) + " "
    return Counter(s[i:i + 3] for i in range(len(s) - 2))


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def is_upper(s: str) -> bool:
    return s != s.lower() and s == s.upper()


def legal_relation(t1: list[str], t2: list[str]) -> str:
    l1, l2 = {t for t in t1 if t in LEGAL}, {t for t in t2 if t in LEGAL}
    if not l1 and not l2:
        return "none"
    if l1 == l2:
        return "same"
    if not l1:
        return "added"
    if not l2:
        return "dropped"
    return "changed"


def digit_relation(a1: str, a2: str) -> str:
    c1, c2 = Counter(_DIGITS.findall(a1)), Counter(_DIGITS.findall(a2))
    if not c1 and not c2:
        return "none"
    if c1 == c2:
        return "same"
    if not (c2 - c1):
        return "subset"
    if not (c1 - c2):
        return "superset"
    return "different"


def pair_features(n1: str, a1: str, n2: str, a2: str) -> dict:
    t1, t2 = norm(n1), norm(n2)
    s1, s2 = set(t1), set(t2)
    at1, at2 = norm(a1), norm(a2)
    return {
        "name_jaccard": len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 1.0,
        "name_cos3": cosine(grams(t1), grams(t2)),
        "addr_cos3": cosine(grams(at1), grams(at2)),
        "legal_rel": legal_relation(t1, t2),
        "digit_rel": digit_relation(a1, a2),
        "addr_empty": a2.strip() == "",
        "abbrev": any(t in ABBR for t in at2),
        "name_upper": is_upper(n2),
        "addr_upper": is_upper(a2),
        "name_exact": n1 == n2,
        "addr_exact": a1 == a2,
    }


NUMERIC = ["name_jaccard", "name_cos3", "addr_cos3"]
BOOLEAN = ["addr_empty", "abbrev", "name_upper", "addr_upper", "name_exact", "addr_exact"]
CATEGORICAL = {"legal_rel": ["none", "same", "added", "dropped", "changed"],
               "digit_rel": ["none", "same", "subset", "superset", "different"]}


def summarize(rows: list[dict]) -> dict:
    """Mean of numeric/boolean features and category shares, as a flat dict."""
    n = len(rows)
    out = {"n": n}
    if not n:
        return out
    for k in NUMERIC + BOOLEAN:
        out[k] = sum(float(r[k]) for r in rows) / n
    for k, cats in CATEGORICAL.items():
        c = Counter(r[k] for r in rows)
        for cat in cats:
            out[f"{k}={cat}"] = c.get(cat, 0) / n
    return out
