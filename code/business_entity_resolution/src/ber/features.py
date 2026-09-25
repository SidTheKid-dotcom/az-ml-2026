"""Label-free pair features for noisy business record matching.

The only information extracted from identifiers is the permitted source prefix.
Country is compared for agreement but is never encoded as a closed vocabulary.
Features are computed from positional pair indices, without using their values as
model inputs. Relative scores are within a complete candidate set for each S1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .normalization import normalize_text


_LEGAL = frozenset(
    "inc incorporated corp corporation co company llc llp lp ltd limited pvt "
    "private plc sarl sas sasu sa sci eurl gmbh bv pte".split()
)
_NAME_WORDS = {
    "intl": "international", "int": "international", "svcs": "services",
    "svc": "service", "associates": "associate", "assoc": "associate",
    "mfg": "manufacturing", "bros": "brothers", "technologies": "technology",
    "tech": "technology", "consultants": "consultant",
}
_ADDRESS_WORDS = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "ln": "lane", "dr": "drive", "ct": "court",
    "cir": "circle", "hwy": "highway", "pkwy": "parkway", "pl": "place",
    "sq": "square", "ter": "terrace", "trl": "trail", "ctr": "center",
    "ste": "unit", "suite": "unit", "apt": "unit", "apartment": "unit",
    "fl": "floor", "flr": "floor", "bldg": "building", "nr": "near",
    "opp": "opposite", "n": "north", "s": "south", "e": "east", "w": "west",
    "r": "rue", "rte": "route", "bd": "boulevard", "bat": "building",
}
# Equivalent written forms, not external business data. This deliberately leaves
# unseen countries and their address components intact.
_REGIONS = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct",
    "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as",
    "bihar": "br", "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj",
    "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "telangana": "ts", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb",
}
_REGION_PATTERN = re.compile(r"\b(?:" + "|".join(
    re.escape(x) for x in sorted(_REGIONS, key=len, reverse=True)
) + r")\b")
_NUMBERS = re.compile(r"\d+")
_NATIVE = re.compile(r"[\u0900-\u0dff]")
_PHONETIC = re.compile(r"ph|sh|ch|th|dh|kh|gh")
_PHONETIC_MAP = {"ph": "f", "sh": "s", "ch": "c", "th": "t",
                 "dh": "d", "kh": "k", "gh": "g"}


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _core_name(text: str) -> str:
    tokens = [_NAME_WORDS.get(t, t) for t in text.split() if t not in _LEGAL]
    return " ".join(tokens) or text


def _address(text: str, country: str) -> str:
    # Expand full region names to the same abbreviation already found in source
    # records, then apply identical token normalization to both representations.
    if country.casefold() in {"us", "usa", "united states", "india"}:
        text = _REGION_PATTERN.sub(lambda m: _REGIONS[m.group()], text)
    return " ".join(_ADDRESS_WORDS.get(t, t) for t in text.split())


def _phonetic(text: str) -> str:
    text = _PHONETIC.sub(lambda m: _PHONETIC_MAP[m.group()], text)
    text = text.translate(str.maketrans({"w": "v", "q": "k", "z": "s"}))
    text = re.sub(r"[aeiou]", "", text)
    return re.sub(r"(.)\1+", r"\1", text)


@dataclass(slots=True)
class _Record:
    name: str
    core: str
    phonetic: str
    raw_address: str
    address: str
    name_tokens: FrozenSet[str]
    address_tokens: FrozenSet[str]
    numbers: FrozenSet[str]
    long_numbers: FrozenSet[str]
    short_numbers: FrozenSet[str]
    country: str
    source3: bool
    native_name: bool
    native_address: bool


def _prepare(frame: pd.DataFrame, indices: np.ndarray) -> list[_Record]:
    records = []
    for entity_id, name, address, country in frame.iloc[indices][
        ["entity_id", "business_name", "business_address", "country"]
    ].itertuples(index=False, name=None):
        raw_name, raw_address, country = _text(name), _text(address), _text(country)
        name = normalize_text(raw_name)
        raw_normal_address = normalize_text(raw_address)
        core = _core_name(name)
        address = _address(raw_normal_address, country)
        numbers = frozenset(n.lstrip("0") or "0" for n in _NUMBERS.findall(address))
        long_numbers = frozenset(n for n in _NUMBERS.findall(address) if len(n) in (5, 6))
        short_numbers = frozenset(n.lstrip("0") or "0" for n in _NUMBERS.findall(address) if len(n) < 5)
        records.append(_Record(
            name, core, _phonetic(core), raw_normal_address, address,
            frozenset(core.split()), frozenset(address.split()), numbers,
            long_numbers, short_numbers, country.casefold(),
            _text(entity_id).startswith("S3-"), bool(_NATIVE.search(raw_name)),
            bool(_NATIVE.search(raw_address)),
        ))
    return records


def _fuzzy(left: str, right: str) -> tuple[float, ...]:
    if not left or not right:
        return (0.0,) * 6
    return (
        fuzz.ratio(left, right) / 100,
        fuzz.partial_ratio(left, right) / 100,
        fuzz.token_sort_ratio(left, right) / 100,
        fuzz.token_set_ratio(left, right) / 100,
        fuzz.WRatio(left, right) / 100,
        JaroWinkler.normalized_similarity(left, right),
    )


def _overlap(left: FrozenSet[str], right: FrozenSet[str]) -> tuple[float, ...]:
    common = len(left & right)
    return (
        common / max(len(left | right), 1), common / max(len(left), 1),
        common / max(len(right), 1), float(common),
    )


def _numbers(left: FrozenSet[str], right: FrozenSet[str]) -> tuple[float, ...]:
    both = bool(left and right)
    common = len(left & right)
    return (float(both), float(both and left == right),
            float(common > 0), float(both and common == 0),
            common / max(len(left | right), 1))


def _ratio(left: str, right: str) -> float:
    return min(len(left), len(right)) / max(len(left), len(right), 1)


_FUZZY_NAMES = ("ratio", "partial", "token_sort", "token_set", "wratio", "jaro")
BASE_FEATURE_NAMES = (
    ["name_" + x for x in _FUZZY_NAMES]
    + ["core_" + x for x in _FUZZY_NAMES]
    + ["address_" + x for x in _FUZZY_NAMES]
    + ["phonetic_ratio", "phonetic_token_sort", "raw_address_ratio"]
    + [prefix + "_" + suffix for prefix in ("name_token", "address_token")
       for suffix in ("jaccard", "left_coverage", "right_coverage", "common")]
    + [prefix + "_" + suffix for prefix in ("number", "long_number", "short_number")
       for suffix in ("both_present", "equal", "any_common", "disjoint", "jaccard")]
    + ["name_exact", "core_exact", "address_exact", "name_address_exact",
       "name_len_ratio", "address_len_ratio", "name_left_log_len", "name_right_log_len",
       "address_left_log_len", "address_right_log_len", "name_left_missing",
       "name_right_missing", "address_left_missing", "address_right_missing",
       "name_native_left", "name_native_right", "address_native_left", "address_native_right",
       "country_equal", "source3", "name_address_min", "name_address_product",
       "core_address_min", "core_address_product", "name_number_conflict",
       "address_name_conflict"]
)
CONTEXT_FEATURE_NAMES = [
    "retrieval_name", "retrieval_address", "retrieval_score", "retrieval_rank_reciprocal",
    "retrieval_both_product", "retrieval_both_min", "retrieval_channel_difference",
    "retrieval_name_gap", "retrieval_address_gap", "retrieval_score_gap",
    "retrieval_score_relative", "candidate_log_count", "best_text_gap",
]


def build_features(reference: pd.DataFrame, pool: pd.DataFrame,
                   pairs: pd.DataFrame) -> pd.DataFrame:
    """Return finite float32 features in exactly the order of ``pairs``.

    ``left_idx`` and ``right_idx`` must be positional indices into the supplied
    frames. Complete left-entity candidate groups should be kept together when
    calling this function in chunks, since relative features compare candidates.
    Labels, identifier numeric portions, and row positions are never features.
    """
    if pairs.empty:
        return pd.DataFrame(index=pairs.index, columns=BASE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES,
                            dtype=np.float32)
    li = pairs["left_idx"].to_numpy(dtype=np.int64)
    ri = pairs["right_idx"].to_numpy(dtype=np.int64)
    if li.min() < 0 or li.max() >= len(reference) or ri.min() < 0 or ri.max() >= len(pool):
        raise IndexError("Candidate pair position is outside reference/pool frames")
    lu, lm = np.unique(li, return_inverse=True)
    ru, rm = np.unique(ri, return_inverse=True)
    left_records, right_records = _prepare(reference, lu), _prepare(pool, ru)
    output = np.empty((len(pairs), len(BASE_FEATURE_NAMES)), dtype=np.float32)
    for i, (lpos, rpos) in enumerate(zip(lm, rm)):
        left, right = left_records[lpos], right_records[rpos]
        name, core, address = (_fuzzy(left.name, right.name),
                               _fuzzy(left.core, right.core),
                               _fuzzy(left.address, right.address))
        name_exact = bool(left.name and left.name == right.name)
        core_exact = bool(left.core and left.core == right.core)
        address_exact = bool(left.address and left.address == right.address)
        name_score, core_score, address_score = name[2], core[2], address[2]
        output[i] = (
            *name, *core, *address,
            fuzz.ratio(left.phonetic, right.phonetic) / 100 if left.phonetic and right.phonetic else 0,
            fuzz.token_sort_ratio(left.phonetic, right.phonetic) / 100 if left.phonetic and right.phonetic else 0,
            fuzz.ratio(left.raw_address, right.raw_address) / 100 if left.raw_address and right.raw_address else 0,
            *_overlap(left.name_tokens, right.name_tokens),
            *_overlap(left.address_tokens, right.address_tokens),
            *_numbers(left.numbers, right.numbers),
            *_numbers(left.long_numbers, right.long_numbers),
            *_numbers(left.short_numbers, right.short_numbers),
            name_exact, core_exact, address_exact, name_exact and address_exact,
            _ratio(left.name, right.name), _ratio(left.address, right.address),
            np.log1p(len(left.name)), np.log1p(len(right.name)),
            np.log1p(len(left.address)), np.log1p(len(right.address)),
            not left.name, not right.name, not left.address, not right.address,
            left.native_name, right.native_name, left.native_address, right.native_address,
            bool(left.country and left.country == right.country), right.source3,
            min(name_score, address_score), name_score * address_score,
            min(core_score, address_score), core_score * address_score,
            name_score * float(bool(left.numbers and right.numbers and not left.numbers & right.numbers)),
            address_score * (1 - core_score),
        )
    result = pd.DataFrame(output, index=pairs.index, columns=BASE_FEATURE_NAMES)
    def channel(name: str, default: float = 0) -> np.ndarray:
        if name not in pairs:
            return np.full(len(pairs), default, dtype=np.float32)
        return np.nan_to_num(pairs[name].to_numpy(dtype=np.float32), nan=0, posinf=0, neginf=0)

    name, address, score, rank = (channel("name_score"), channel("address_score"),
                                 channel("retrieval_score"), channel("retrieval_rank", 1))
    result["retrieval_name"], result["retrieval_address"], result["retrieval_score"] = name, address, score
    # Ranks emitted by our retriever start at one; tolerate zero-based rank data.
    result["retrieval_rank_reciprocal"] = 1 / np.maximum(rank, 1)
    result["retrieval_both_product"] = name * address
    result["retrieval_both_min"] = np.minimum(name, address)
    result["retrieval_channel_difference"] = np.abs(name - address)
    contextual = pd.DataFrame({"left_idx": li, "name": name, "address": address,
                               "score": score, "text": result["core_address_product"].to_numpy()})
    groups = contextual.groupby("left_idx", sort=False)
    for col in ("name", "address", "score"):
        maximum = groups[col].transform("max").to_numpy()
        result["retrieval_" + col + "_gap"] = contextual[col].to_numpy() - maximum
        if col == "score":
            result["retrieval_score_relative"] = score / np.maximum(maximum, 1e-6)
    result["candidate_log_count"] = np.log1p(groups["score"].transform("size").to_numpy())
    result["best_text_gap"] = contextual["text"].to_numpy() - groups["text"].transform("max").to_numpy()
    return result[BASE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES].astype(np.float32, copy=False)
