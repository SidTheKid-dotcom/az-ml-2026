"""Label-free forensic pair features: digits, names, exact keys, street tokens, scripts.

Batch API over aligned pairs ``(left_name, left_address, right_name, right_address,
country)``; the left side is a Source 1 reference and the right side an S2/S3
record. Text folding is vectorized with pyarrow; the per-pair set/alignment work is
a single Python pass over integer-coded unique strings, so repeated S1 texts are
normalized once per batch.

Vocabularies (administrative phrases, generic street words, abbreviations) are
derived from the supplied data only, by ``ber.derive_forensic_vocab``, and stored
in ``resources/forensic_vocab.json``. No external place or business list is used.

Conventions
-----------
* Names: NFKD fold to ASCII, lowercase, every character outside ``[a-z0-9 ]``
  becomes a space, runs of >= 2 single-letter tokens are joined ("s a s" -> "sas",
  so dotted suffixes such as "S.A.S" and "E.U.R.L." are recognized), then the
  legal-suffix tokens in ``LEGAL_SUFFIXES`` are dropped. An empty normalized name
  never counts as equal to anything.
* Digit groups: ``[0-9]+`` runs of the folded address (native-script digits do not
  occur in the released addresses).
* Exact key: normalized name + "|" + all address digits concatenated in order.
* Street tokens: comma components of the folded address; a component that is an
  administrative phrase of the country (city, region, departement, state; exact or
  fuzzy >= 88 for components of >= 5 characters) is dropped; abbreviations are
  expanded; tokens containing a digit and generic street words are dropped.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

LEGAL_SUFFIXES = frozenset(
    "the inc incorporated corp corporation co company llc ltd limited pvt private "
    "llp plc sarl sas sasu sa sci eurl gmbh".split())

ADDRESS_KINDS = ("empty", "no_digits", "has_digits")
EMPTY, NO_DIGITS, HAS_DIGITS = range(3)

DIGIT_RELATIONS = ("no_digits", "cand_no_digits", "same_sequence", "same_set_reordered",
                   "cand_dropped_some", "cand_added_some", "same_count_substituted",
                   "other_mismatch")
(REL_NO_DIGITS, REL_CAND_NO_DIGITS, REL_SAME_SEQUENCE, REL_SAME_SET_REORDERED,
 REL_CAND_DROPPED_SOME, REL_CAND_ADDED_SOME, REL_SAME_COUNT_SUBSTITUTED,
 REL_OTHER_MISMATCH) = range(8)

DEFAULT_VOCAB_PATH = Path(__file__).resolve().parents[2] / "resources" / "forensic_vocab.json"

FEATURE_COLUMNS = (
    "address_kind", "left_address_kind", "digit_relation",
    "n_same_length_substitutions", "n_length_changing_edits", "n_zero_pad_edits",
    "n_deletions", "n_insertions", "max_group_edit_distance", "digit_transposition",
    "norm_name_equal", "name_token_jaccard", "exact_key_equal",
    "street_token_jaccard", "street_tokens_equal", "street_token_ratio",
    "n_left_street_tokens", "n_right_street_tokens",
    "left_name_native", "left_address_native", "right_name_native", "right_address_native",
)

_NULL_PATTERN = r"(?i)<\s*null\s*>"
_NATIVE_PATTERN = r"[\x{0900}-\x{0DFF}]"
# Letters that NFKD leaves undecomposed; plain orthographic folding, not a place list.
_LIGATURES = (("œ", "oe"), ("Œ", "oe"), ("æ", "ae"), ("Æ", "ae"), ("ß", "ss"),
              ("ø", "o"), ("Ø", "o"), ("ł", "l"), ("Ł", "l"), ("đ", "d"), ("Đ", "d"))
_LIGATURE_TABLE = str.maketrans({a: b for a, b in _LIGATURES})
_LIGATURE_PATTERN = "[" + "".join(a for a, _ in _LIGATURES) + "]"
_HAS_DIGIT = re.compile("[0-9]").search
FUZZY_ADMIN_CUTOFF = 88.0
FUZZY_ADMIN_MIN_CHARS = 5


# ---------------------------------------------------------------------------
# Vectorized text folding
# ---------------------------------------------------------------------------

def as_string_array(values) -> pa.Array:
    """Any sequence of str/None/NaN (or arrow array) -> non-null ``large_string`` array."""
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    if isinstance(values, pa.Array):
        array = values.cast(pa.large_string())
    else:
        if isinstance(values, pd.Series):
            values = values.to_numpy(dtype=object)
        array = pa.array(values, type=pa.large_string(), from_pandas=True)
    return pc.fill_null(array, "")


def fold_ascii(values, *, keep_commas: bool = False) -> pa.Array:
    """NFKD fold to ASCII, lowercase, non ``[a-z0-9 ]`` -> space, single spaces.

    ``<NULL>`` placeholders become spaces. With ``keep_commas`` the comma component
    separators survive (surrounding spaces are removed). Work is done once per
    distinct string.
    """
    array = as_string_array(values)
    if len(array) > 64:
        encoded = pc.dictionary_encode(array)
        if len(encoded.dictionary) < 0.9 * len(array):
            return pc.take(_fold_unique(encoded.dictionary, keep_commas), encoded.indices)
    return _fold_unique(array, keep_commas)


def _on_subset(array: pa.Array, mask: pa.Array, transform) -> pa.Array:
    """Apply ``transform`` only to the rows where ``mask`` is true."""
    if not pc.any(mask).as_py():
        return array
    rows = pa.array(np.flatnonzero(mask.to_numpy(zero_copy_only=False)))
    return pc.replace_with_mask(array, mask, transform(array.take(rows)).cast(array.type))


def _fold_unique(array: pa.Array, keep_commas: bool) -> pa.Array:
    array = _on_subset(array, pc.match_substring(array, "<"),
                       lambda a: pc.replace_substring_regex(a, _NULL_PATTERN, " "))
    array = _on_subset(array, pc.invert(pc.string_is_ascii(array)), _fold_non_ascii)
    array = pc.ascii_lower(array)
    # One pass: every run of characters outside [a-z0-9] (or [a-z0-9,]) incl. spaces -> " ".
    array = pc.replace_substring_regex(array, "[^a-z0-9,]+" if keep_commas else "[^a-z0-9]+", " ")
    if keep_commas:
        array = pc.replace_substring(array, " , ", ",")
        array = pc.replace_substring(array, ", ", ",")
        array = pc.replace_substring(array, " ,", ",")
    return pc.ascii_trim_whitespace(array)


def _fold_non_ascii(array: pa.Array) -> pa.Array:
    special = pc.match_substring_regex(array, _LIGATURE_PATTERN)
    if pc.any(special).as_py():
        rows = np.flatnonzero(special.to_numpy(zero_copy_only=False))
        fixed = [text.translate(_LIGATURE_TABLE) for text in array.take(rows).to_pylist()]
        array = pc.replace_with_mask(array, special, pa.array(fixed, type=array.type))
    array = pc.utf8_normalize(array, form="NFKD")
    array = pc.replace_substring_regex(array, r"\p{Mn}+", "")
    return pc.utf8_lower(array)


def _encode(array: pa.Array) -> tuple[np.ndarray, list[str]]:
    encoded = pc.dictionary_encode(array)
    return (encoded.indices.to_numpy(zero_copy_only=False).astype(np.int64, copy=False),
            encoded.dictionary.to_pylist())


_SINGLE_LETTER_RUN = re.compile(r"(?:^| )[a-z] [a-z](?: |$)").search


def _strip_legal(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return ""
    if _SINGLE_LETTER_RUN(text) is None:
        if LEGAL_SUFFIXES.isdisjoint(tokens):
            return " ".join(tokens)
        return " ".join(t for t in tokens if t not in LEGAL_SUFFIXES)
    joined: list[str] = []
    run: list[str] = []
    for token in tokens + [""]:
        if len(token) == 1 and token.isalpha():
            run.append(token)
            continue
        if len(run) >= 2:
            joined.append("".join(run))
        elif run:
            joined.append(run[0])
        run = []
        if token:
            joined.append(token)
    return " ".join(t for t in joined if t not in LEGAL_SUFFIXES)


def normalize_names(values) -> np.ndarray:
    """Normalized names (object array of str); see module conventions."""
    codes, uniques = _encode(fold_ascii(values))
    return np.asarray([_strip_legal(u) for u in uniques] + [""], dtype=object)[codes]


def digit_strings(folded: pa.Array) -> pa.Array:
    """Digit groups of already folded text, joined by single spaces ("5 118")."""
    return pc.utf8_trim_whitespace(pc.replace_substring_regex(folded, "[^0-9]+", " "))


def digit_groups(addresses) -> list[tuple[str, ...]]:
    """``re.findall(r"[0-9]+")`` over each folded address, as tuples."""
    return [tuple(s.split()) for s in digit_strings(fold_ascii(addresses)).to_pylist()]


def address_kinds(addresses) -> np.ndarray:
    """int8 codes of ``ADDRESS_KINDS``: empty (missing, blank, <NULL>, ## ...), no_digits, has_digits."""
    raw = as_string_array(addresses)
    return _kinds(raw, digit_strings(fold_ascii(raw)))


def _kinds(raw: pa.Array, digits: pa.Array) -> np.ndarray:
    raw = pc.replace_substring_regex(raw, _NULL_PATTERN, " ")
    has_content = pc.match_substring_regex(raw, r"[\p{L}\p{N}]").to_numpy(zero_copy_only=False)
    has_digit = (pc.utf8_length(digits).to_numpy(zero_copy_only=False) > 0)
    return np.where(~has_content, EMPTY, np.where(has_digit, HAS_DIGITS, NO_DIGITS)).astype(np.int8)


def native_script(values) -> np.ndarray:
    """int8 flag: any character in U+0900-U+0DFF (Indic scripts)."""
    return pc.match_substring_regex(as_string_array(values), _NATIVE_PATTERN).to_numpy(
        zero_copy_only=False).astype(np.int8)


def exact_keys(names, addresses, *, normalized_names: np.ndarray | None = None) -> np.ndarray:
    """Exact key per row: normalized name + "|" + address digits in order; "" if name empty."""
    name_array = as_string_array(names)
    address_array = as_string_array(addresses)
    if len(name_array) != len(address_array):
        raise ValueError("name and address lengths differ")
    names_n = normalize_names(name_array) if normalized_names is None else np.asarray(normalized_names)
    if len(names_n) != len(name_array):
        raise ValueError("normalized_names length differs from names")
    digits = pc.replace_substring(digit_strings(fold_ascii(address_array)), " ", "")
    keys = pc.binary_join_element_wise(pa.array(names_n, type=pa.large_string()), digits,
                                      pa.scalar("|", pa.large_string()))
    out = np.asarray(keys.to_pylist(), dtype=object)
    out[names_n == ""] = ""
    return out


def normalize_full_addresses(addresses) -> np.ndarray:
    """Order-free full address: folded comma components, sorted, joined by ", "."""
    codes, uniques = _encode(fold_ascii(addresses, keep_commas=True))
    table = [", ".join(sorted(c for c in u.split(",") if c)) for u in uniques] + [""]
    return np.asarray(table, dtype=object)[codes]


# ---------------------------------------------------------------------------
# Street vocabulary
# ---------------------------------------------------------------------------

@dataclass
class CountryVocabulary:
    admin: frozenset = frozenset()
    generic: frozenset = frozenset()
    abbreviations: Mapping[str, str] = field(default_factory=dict)
    admin_level: Mapping[str, str] = field(default_factory=dict)
    _admin_by_initial: dict = field(default_factory=dict, repr=False)
    _fuzzy_cache: dict = field(default_factory=dict, repr=False)
    _parse_cache: dict = field(default_factory=dict, repr=False)
    _parse_cache_exact: dict = field(default_factory=dict, repr=False)
    _abbreviation_keys: frozenset = field(default_factory=frozenset, repr=False)
    _component_cache: dict = field(default_factory=dict, repr=False)
    _component_cache_exact: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.admin = frozenset(self.admin)
        self.generic = frozenset(self.generic)
        self._abbreviation_keys = frozenset(self.abbreviations)
        buckets: dict = {}
        for phrase in sorted(self.admin):
            if len(phrase) >= FUZZY_ADMIN_MIN_CHARS - 1:
                buckets.setdefault((phrase[0], len(phrase)), []).append(phrase)
        self._admin_by_initial = buckets

    def admin_match(self, phrase: str, fuzzy: bool = True) -> str | None:
        """The admin phrase a digit-free component denotes, or None.

        Fuzzy matching (rapidfuzz ratio >= 88) only compares phrases with the same
        first letter, a compatible length and at least 5 characters, which keeps
        India's large admin list fast and avoids matching short state codes.
        """
        if phrase in self.admin:
            return phrase
        if not fuzzy or len(phrase) < FUZZY_ADMIN_MIN_CHARS:
            return None
        if phrase in self._fuzzy_cache:
            return self._fuzzy_cache[phrase]
        # ratio >= 88 needs 2*min/(a+b) >= 0.88, i.e. the other length within [0.785 L, 1.28 L].
        size = len(phrase)
        index = self._admin_by_initial
        bucket = [p for n in range(int(size * 0.78), int(size * 1.28) + 1)
                  for p in index.get((phrase[0], n), ())]
        if not bucket:
            self._fuzzy_cache[phrase] = None
            return None
        if phrase not in self._fuzzy_cache:
            if len(self._fuzzy_cache) > 2_000_000:
                self._fuzzy_cache.clear()
            hit = process.extractOne(phrase, bucket, scorer=fuzz.ratio,
                                     score_cutoff=FUZZY_ADMIN_CUTOFF)
            self._fuzzy_cache[phrase] = hit[0] if hit else None
        return self._fuzzy_cache[phrase]

    def parse(self, folded: str, fuzzy: bool = True) -> tuple[frozenset, frozenset]:
        """(street tokens, admin phrases) of one folded, comma-kept address."""
        cache = self._parse_cache if fuzzy else self._parse_cache_exact
        cached = cache.get(folded)
        if cached is not None:
            return cached
        components = self._component_cache if fuzzy else self._component_cache_exact
        street: list[str] = []
        admins: list[str] = []
        for component in folded.split(","):
            parsed = components.get(component)
            if parsed is None:
                parsed = components[component] = self._parse_component(component, fuzzy)
            if parsed.__class__ is str:
                admins.append(parsed)
            else:
                street += parsed
        result = (frozenset(street), frozenset(admins))
        if len(cache) > 2_000_000:
            cache.clear()
        if len(components) > 2_000_000:
            components.clear()
        cache[folded] = result
        return result

    def _parse_component(self, component: str, fuzzy: bool):
        """Admin phrase (str) or the tuple of street tokens of one component."""
        words = component.split()
        if not words:
            return ()
        if self.abbreviations and not self._abbreviation_keys.isdisjoint(words):
            words = [self.abbreviations.get(w, w) for w in words]
        if not _HAS_DIGIT(component):
            # Expanded spelling first, so "st nazaire" reports "saint nazaire".
            expanded = " ".join(words)
            if expanded in self.admin:
                return expanded
            plain = " ".join(component.split())
            if plain in self.admin:
                return plain
            hit = self.admin_match(expanded, fuzzy)
            if hit is not None:
                return hit
        generic = self.generic
        return tuple([w for w in words if w not in generic and w.isalpha()])

    def cities(self, admins: Iterable[str]) -> frozenset:
        return frozenset(a for a in admins if self.admin_level.get(a, "city") == "city")


class StreetVocabulary:
    """Per-country admin phrases, generic street words and abbreviations."""

    def __init__(self, countries: Mapping[str, CountryVocabulary] | None = None):
        self.countries = dict(countries or {})
        self._empty = CountryVocabulary()

    @classmethod
    def from_dict(cls, payload: Mapping) -> "StreetVocabulary":
        countries = {}
        for name, spec in payload.get("countries", {}).items():
            countries[name] = CountryVocabulary(
                admin=frozenset(spec.get("admin_phrases", ())),
                generic=frozenset(spec.get("generic_tokens", ())),
                abbreviations=dict(spec.get("abbreviations", {})),
                admin_level=dict(spec.get("admin_level", {})))
        return cls(countries)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_VOCAB_PATH) -> "StreetVocabulary":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __getitem__(self, country: str) -> CountryVocabulary:
        return self.countries.get(country, self._empty)


_DEFAULT_VOCAB: StreetVocabulary | None = None


def default_vocabulary() -> StreetVocabulary:
    global _DEFAULT_VOCAB
    if _DEFAULT_VOCAB is None:
        _DEFAULT_VOCAB = (StreetVocabulary.load() if DEFAULT_VOCAB_PATH.exists()
                          else StreetVocabulary())
    return _DEFAULT_VOCAB


def parse_addresses(addresses, countries, vocab: StreetVocabulary | None = None, *,
                    fuzzy_admin: bool = True) -> tuple[list[frozenset], list[frozenset]]:
    """Per row (street token set, admin phrase set)."""
    vocab = vocab or default_vocabulary()
    folded = fold_ascii(addresses, keep_commas=True).to_pylist()
    country_list = _country_list(countries, len(folded))
    streets, admins = [], []
    for text, country in zip(folded, country_list):
        s, a = vocab[country].parse(text, fuzzy_admin)
        streets.append(s)
        admins.append(a)
    return streets, admins


def _country_list(countries, n: int) -> list[str]:
    if isinstance(countries, str):
        return [countries] * n
    out = as_string_array(countries).to_pylist()
    if len(out) != n:
        raise ValueError("country length differs from the pair batch")
    return out


def template_similar(a: str, b: str) -> bool:
    """Whether normalized names are equal or differ at exactly one token position."""
    if not a or not b:
        return False
    if a == b:
        return True
    left, right = a.split(), b.split()
    return (len(left) == len(right) and len(left) >= 2
            and sum(x != y for x, y in zip(left, right)) == 1)


# ---------------------------------------------------------------------------
# Digit alignment
# ---------------------------------------------------------------------------

def _is_adjacent_transposition(a: str, b: str) -> bool:
    if len(a) != len(b) or a == b:
        return False
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]


def align_digit_groups(left: Sequence[str], right: Sequence[str]) -> tuple[int, int, int, int, int, int, int, int]:
    """Relation class and edit counts between two digit-group sequences.

    Returns ``(relation, n_same_length_substitutions, n_length_changing_edits,
    n_zero_pad_edits, n_deletions, n_insertions, max_group_edit_distance,
    transposition)``. Identical groups are matched first as a multiset (order-free,
    because vendors reorder components); groups equal as integers ("0029"/"29") are
    zero-pad edits; the rest are paired greedily by smallest Levenshtein distance.
    Unpaired left groups are deletions, unpaired right groups insertions.
    """
    left, right = tuple(left), tuple(right)
    if left == right:
        return (REL_SAME_SEQUENCE if left else REL_NO_DIGITS), 0, 0, 0, 0, 0, 0, 0
    if len(left) == 1 and len(right) == 1:  # fast path: one house number each
        a, b = left[0], right[0]
        if a.lstrip("0") == b.lstrip("0"):
            return REL_SAME_COUNT_SUBSTITUTED, 0, 0, 1, 0, 0, 0, 0
        dist = Levenshtein.distance(a, b)
        if len(a) == len(b):
            return REL_SAME_COUNT_SUBSTITUTED, 1, 0, 0, 0, 0, dist, int(_is_adjacent_transposition(a, b))
        return REL_SAME_COUNT_SUBSTITUTED, 0, 1, 0, 0, 0, dist, 0
    remaining = list(right)
    rest_left = []
    for group in left:
        if group in remaining:
            remaining.remove(group)
        else:
            rest_left.append(group)
    if not left:
        relation = REL_CAND_ADDED_SOME
    elif not right:
        relation = REL_CAND_NO_DIGITS
    elif not rest_left and not remaining:
        relation = REL_SAME_SET_REORDERED
    elif not remaining:
        relation = REL_CAND_DROPPED_SOME
    elif not rest_left:
        relation = REL_CAND_ADDED_SOME
    elif len(left) == len(right):
        relation = REL_SAME_COUNT_SUBSTITUTED
    else:
        relation = REL_OTHER_MISMATCH
    zero_pad = 0
    if rest_left and remaining:
        still = []
        for group in rest_left:
            stripped = group.lstrip("0")
            for j, other in enumerate(remaining):
                if other.lstrip("0") == stripped:
                    remaining.pop(j)
                    zero_pad += 1
                    break
            else:
                still.append(group)
        rest_left = still
    same_len = len_change = max_edit = transposition = 0
    if rest_left and remaining:
        if len(rest_left) == 1 and len(remaining) == 1:
            pairs = [(Levenshtein.distance(rest_left[0], remaining[0]), 0, 0)]
        else:
            candidates = sorted((Levenshtein.distance(a, b), i, j)
                                for i, a in enumerate(rest_left) for j, b in enumerate(remaining))
            used_l, used_r, pairs = set(), set(), []
            for dist, i, j in candidates:
                if i not in used_l and j not in used_r:
                    used_l.add(i)
                    used_r.add(j)
                    pairs.append((dist, i, j))
        for dist, i, j in pairs:
            a, b = rest_left[i], remaining[j]
            if len(a) == len(b):
                same_len += 1
                transposition |= _is_adjacent_transposition(a, b)
            else:
                len_change += 1
            max_edit = max(max_edit, dist)
        matched = len(pairs)
    else:
        matched = 0
    return (relation, same_len, len_change, zero_pad, len(rest_left) - matched,
            len(remaining) - matched, max_edit, int(transposition))


# ---------------------------------------------------------------------------
# Pair features
# ---------------------------------------------------------------------------

def _factorize(values: list) -> tuple[np.ndarray, list]:
    codes, uniques = pd.factorize(np.asarray(values, dtype=object), use_na_sentinel=False)
    return codes.astype(np.int64, copy=False), list(uniques)


def forensic_features(left_name, left_address, right_name, right_address, country,
                      vocab: StreetVocabulary | None = None, *,
                      fuzzy_admin: bool = True) -> pd.DataFrame:
    """Forensic features for aligned pairs; one row per pair, columns ``FEATURE_COLUMNS``.

    ``country`` is a sequence aligned with the pairs or one string for all pairs.
    Dtypes are int8 for flags/codes/counts (counts clipped to 127) and float32 for
    ratios; Jaccard/ratio values are NaN when either side is empty.
    """
    vocab = vocab or default_vocabulary()
    ln, la, rn, ra = (as_string_array(v) for v in (left_name, left_address, right_name, right_address))
    n = len(ln)
    if not (len(la) == len(rn) == len(ra) == n):
        raise ValueError("pair columns have different lengths")
    countries = _country_list(country, n)
    nan = np.float32(np.nan)

    # Names: every computation once per distinct raw name.
    name_enc = pc.dictionary_encode(pa.concat_arrays([ln, rn]))
    name_codes = name_enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    name_native = native_script(name_enc.dictionary)
    norm_codes, norm_uniques = _factorize([_strip_legal(t) for t in fold_ascii(name_enc.dictionary).to_pylist()])
    norm_sets = [frozenset(u.split()) for u in norm_uniques]
    norm_size = np.fromiter((len(x) for x in norm_sets), dtype=np.int64, count=len(norm_sets))
    lnorm, rnorm = norm_codes[name_codes[:n]], norm_codes[name_codes[n:]]
    name_equal = (lnorm == rnorm) & (norm_size[lnorm] > 0)

    # Addresses: fold, digits, kinds, script once per distinct raw address.
    addr_enc = pc.dictionary_encode(pa.concat_arrays([la, ra]))
    addr_codes = addr_enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    folded = fold_ascii(addr_enc.dictionary, keep_commas=True)
    digits = digit_strings(folded)
    addr_kind = _kinds(addr_enc.dictionary, digits)
    addr_native = native_script(addr_enc.dictionary)
    digit_codes, digit_uniques = _factorize(digits.to_pylist())
    digit_tuples = [tuple(u.split()) for u in digit_uniques]
    cat_codes, _ = _factorize(pc.replace_substring(digits, " ", "").to_pylist())
    lac, rac = addr_codes[:n], addr_codes[n:]
    ldig, rdig = digit_codes[lac], digit_codes[rac]
    key_equal = name_equal & (cat_codes[lac] == cat_codes[rac])

    # Street tokens once per distinct (country, address).
    country_codes, country_names = _factorize(countries)
    combos, combo_inverse = np.unique(addr_codes * max(len(country_names), 1) + np.tile(country_codes, 2),
                                      return_inverse=True)
    folded_list = folded.to_pylist()
    street_ids: dict = {}
    street_sets: list = []
    street_strs: list = []
    combo_street = np.empty(len(combos), dtype=np.int64)
    for j, combo in enumerate(combos.tolist()):
        a, c = divmod(combo, max(len(country_names), 1))
        street, _ = vocab[country_names[c]].parse(folded_list[a], fuzzy_admin)
        sid = street_ids.get(street)
        if sid is None:
            sid = street_ids[street] = len(street_sets)
            street_sets.append(street)
            street_strs.append(" ".join(sorted(street)))
        combo_street[j] = sid
    combo_inverse = combo_inverse.reshape(-1)
    lst, rst = combo_street[combo_inverse[:n]], combo_street[combo_inverse[n:]]
    street_size = np.fromiter((len(x) for x in street_sets), dtype=np.int64, count=len(street_sets))
    n_lstreet, n_rstreet = street_size[lst], street_size[rst]

    street_jac = np.full(n, nan, dtype=np.float32)
    street_ratio = np.full(n, nan, dtype=np.float32)
    street_eq = np.full(n, -1, dtype=np.int8)
    both = (n_lstreet > 0) & (n_rstreet > 0)
    same = both & (lst == rst)
    street_jac[same] = street_ratio[same] = 1.0
    street_eq[same] = 1
    todo = np.flatnonzero(both & ~same)
    street_eq[todo] = 0
    ratio = fuzz.ratio
    # Candidate lists commonly repeat the same street pair. Compute each pair once.
    pair_codes = lst[todo] * max(len(street_sets), 1) + rst[todo]
    unique_pairs, inverse = np.unique(pair_codes, return_inverse=True)
    unique_jac = np.empty(len(unique_pairs), dtype=np.float32)
    unique_ratio = np.empty(len(unique_pairs), dtype=np.float32)
    for i, code in enumerate(unique_pairs.tolist()):
        a, b = divmod(code, max(len(street_sets), 1))
        sa, sb = street_sets[a], street_sets[b]
        inter = len(sa & sb)
        unique_jac[i] = inter / (len(sa) + len(sb) - inter)
        unique_ratio[i] = ratio(street_strs[a], street_strs[b]) / 100.0
    street_jac[todo] = unique_jac[inverse]
    street_ratio[todo] = unique_ratio[inverse]

    name_jac = np.full(n, nan, dtype=np.float32)
    both = (norm_size[lnorm] > 0) & (norm_size[rnorm] > 0)
    name_jac[both & (lnorm == rnorm)] = 1.0
    todo = np.flatnonzero(both & (lnorm != rnorm))
    for i, a, b in zip(todo.tolist(), lnorm[todo].tolist(), rnorm[todo].tolist()):
        sa, sb = norm_sets[a], norm_sets[b]
        inter = len(sa & sb)
        name_jac[i] = inter / (len(sa) + len(sb) - inter)

    digit_empty = np.fromiter((not t for t in digit_tuples), dtype=bool, count=len(digit_tuples))
    relation = np.where(digit_empty[ldig], REL_NO_DIGITS, REL_SAME_SEQUENCE).astype(np.int8)
    counts = np.zeros((n, 7), dtype=np.int16)
    todo = np.flatnonzero(ldig != rdig)
    memo: dict = {}
    for i, a, b in zip(todo.tolist(), ldig[todo].tolist(), rdig[todo].tolist()):
        out = memo.get((a, b))
        if out is None:
            out = memo[(a, b)] = align_digit_groups(digit_tuples[a], digit_tuples[b])
        relation[i] = out[0]
        counts[i] = out[1:]

    def clip8(values):
        return np.clip(values, -128, 127).astype(np.int8)

    frame = pd.DataFrame({
        "address_kind": addr_kind[rac],
        "left_address_kind": addr_kind[lac],
        "digit_relation": relation,
        "n_same_length_substitutions": clip8(counts[:, 0]),
        "n_length_changing_edits": clip8(counts[:, 1]),
        "n_zero_pad_edits": clip8(counts[:, 2]),
        "n_deletions": clip8(counts[:, 3]),
        "n_insertions": clip8(counts[:, 4]),
        "max_group_edit_distance": clip8(counts[:, 5]),
        "digit_transposition": clip8(counts[:, 6]),
        "norm_name_equal": name_equal.astype(np.int8),
        "name_token_jaccard": name_jac,
        "exact_key_equal": key_equal.astype(np.int8),
        "street_token_jaccard": street_jac,
        "street_tokens_equal": street_eq,
        "street_token_ratio": street_ratio,
        "n_left_street_tokens": clip8(n_lstreet),
        "n_right_street_tokens": clip8(n_rstreet),
        "left_name_native": name_native[name_codes[:n]],
        "left_address_native": addr_native[lac],
        "right_name_native": name_native[name_codes[n:]],
        "right_address_native": addr_native[rac],
    })
    return frame[list(FEATURE_COLUMNS)]


# ---------------------------------------------------------------------------
# Source 1 multiplicities
# ---------------------------------------------------------------------------

def s1_multiplicities(names, addresses, countries) -> pd.DataFrame:
    """Per Source 1 row: normalized name/address/key and their counts within country.

    Counts are 0 where the normalized value is empty (an empty name or address is
    never a shared identity). Returns columns ``norm_name, norm_address, exact_key,
    name_count, address_count, key_count`` aligned with the input rows.
    """
    name_array = as_string_array(names)
    address_array = as_string_array(addresses)
    if len(name_array) != len(address_array):
        raise ValueError("name and address lengths differ")
    country_list = _country_list(countries, len(name_array))
    norm_name = normalize_names(name_array)
    frame = pd.DataFrame({
        "country": country_list,
        "norm_name": norm_name,
        "norm_address": normalize_full_addresses(address_array),
        "exact_key": exact_keys(name_array, address_array, normalized_names=norm_name),
    })
    for column, target in (("norm_name", "name_count"), ("norm_address", "address_count"),
                           ("exact_key", "key_count")):
        counts = frame.groupby(["country", column], sort=False)[column].transform("size")
        frame[target] = np.where(frame[column].to_numpy() != "", counts.to_numpy(), 0).astype(np.int32)
    return frame.drop(columns="country")
