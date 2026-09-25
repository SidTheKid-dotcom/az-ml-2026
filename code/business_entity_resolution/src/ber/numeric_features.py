"""Additional address-number evidence for distinguishing close lookalikes."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

_DIGITS = re.compile(r'\d{1,9}')
_PREFIX = re.compile(r'\b([a-z]{1,3})\s*[-/#]\s*(\d{1,6})\b')
_SUFFIX = re.compile(r'\b(\d{1,6})\s*[-/]?\s*([a-z])\b')
_UNIT = re.compile(r'\b(?:unit|suite|ste|apt|apartment|flat|room|rm)\s*[-:#]?\s*([a-z0-9-]+)')
NAMES = [
    'num_short_min_absdiff_log', 'num_short_small_offset', 'num_short_equal_length',
    'num_short_best_digit_similarity', 'num_short_prefix_related', 'num_short_same_digits_reordered',
    'num_short_exact_any', 'num_short_exact_all', 'num_shared_only_long',
    'num_postcode_conflict', 'num_postcode_exact', 'num_postcode_best_digit_similarity',
    'num_alpha_prefix_exact', 'num_alpha_prefix_conflict_same_digits',
    'num_alpha_suffix_exact', 'num_alpha_suffix_conflict_same_digits',
    'num_unit_exact', 'num_unit_conflict', 'num_name_exact', 'num_name_conflict',
]


def _parse(frame, positions):
    records = []
    for row in frame.iloc[positions].itertuples(index=False):
        address = unidecode(str(row.business_address or '')).lower()
        name = unidecode(str(row.business_name or '')).lower()
        numbers = set(_DIGITS.findall(address))
        short = tuple(sorted(x for x in numbers if len(x) <= 4))
        long = tuple(sorted(x for x in numbers if len(x) >= 5))
        prefixes = {(digits, letters) for letters, digits in _PREFIX.findall(address)}
        suffixes = set(_SUFFIX.findall(address))
        records.append((short, long, numbers, prefixes, suffixes,
                        set(_UNIT.findall(address)), set(_DIGITS.findall(name))))
    return records


def _alpha(left, right):
    exact = bool(left & right)
    conflict = any(n1 == n2 and c1 != c2 for n1, c1 in left for n2, c2 in right)
    return exact, conflict


def build_numeric_features(reference, pool, pairs):
    if pairs.empty:
        return pd.DataFrame(index=pairs.index, columns=NAMES, dtype=np.float32)
    lu, lm = np.unique(pairs.left_idx.to_numpy(), return_inverse=True)
    ru, rm = np.unique(pairs.right_idx.to_numpy(), return_inverse=True)
    left_records, right_records = _parse(reference, lu), _parse(pool, ru)
    result = np.zeros((len(pairs), len(NAMES)), dtype=np.float32)
    for i, (l, r) in enumerate(zip(lm, rm)):
        ls, ll, ln, lp, lx, lu_, lname = left_records[l]
        rs, rl, rn, rp, rx, ru_, rname = right_records[r]
        left_unmatched, right_unmatched = set(ls) - set(rs), set(rs) - set(ls)
        combinations = [(a, b) for a in left_unmatched for b in right_unmatched]
        if not combinations:
            combinations = [(a, b) for a in ls for b in rs]
        minimum = min((abs(int(a)-int(b)) for a,b in combinations), default=0)
        short_overlap = set(ls) & set(rs)
        long_overlap = set(ll) & set(rl)
        result[i] = (
            np.log1p(minimum), bool(combinations and 0 < minimum <= 10),
            any(len(a) == len(b) for a,b in combinations),
            max((fuzz.ratio(a,b)/100 for a,b in combinations), default=0),
            any(a != b and (a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)) for a,b in combinations),
            any(a != b and sorted(a) == sorted(b) for a,b in combinations),
            bool(short_overlap), bool(ls and rs and set(ls) == set(rs)),
            bool(long_overlap and ls and rs and not short_overlap),
            bool(ll and rl and not long_overlap), bool(long_overlap),
            max((fuzz.ratio(a,b)/100 for a in ll for b in rl), default=0),
            *_alpha(lp,rp), *_alpha(lx,rx),
            bool(lu_ & ru_), bool(lu_ and ru_ and not lu_ & ru_),
            bool(lname and rname and lname == rname), bool(lname and rname and not lname & rname),
        )
    return pd.DataFrame(result, index=pairs.index, columns=NAMES)
