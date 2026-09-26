"""Collective ("generator logic") pair features computed over the whole record pool.

A pair (S1 ``s``, record ``r``) is judged by what else the pool says about ``r``'s name:

* **Core name** = folded name without legal forms, corporate suffix words and fillers, as a
  sorted token set. Name siblings of ``r`` are the other pool records with the same core.
* **Address key** = sorted set of the address's digit groups (leading zeros dropped); ``dmain``
  is its longest digit group. An empty-address record has no key; its address-bearing name
  siblings say where the entity sits (at this S1's key, elsewhere, or at a key no S1 with that
  core has, i.e. an orphan entity).
* **Alias recurrence**: pool records with the same folded full name, and how many keys they span.
* **Bridges**: pool records ``<alias> dba|d/b/a|aka|nee|formerly|fka|t/a <name>`` link two cores.

No labels are used. The same tables work for the arena and for test (or any new pair set).
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.compute as pc

from ber.forensic_features import LEGAL_SUFFIXES, _encode, _strip_legal, digit_strings, fold_ascii

EXTRA_LEGAL = frozenset("ei snc opc lp pllc pc llp scop selarl sarlu".split())
SUFFIX = frozenset("group groupe holdings holding fils sons france india usa international cie associes "
                   "developpement distribution".split())
FILLER = frozenset("et and the".split())
DROP = LEGAL_SUFFIXES | EXTRA_LEGAL | SUFFIX | FILLER
BRIDGE = re.compile(r"^(.+?) (?:dba|d b a|aka|a k a|nee|formerly|fka|f k a|t a|trading as) (.+)$")

FEATURES = ["core_eq", "core_jaccard", "core_subset", "core_superset", "core_empty", "log1p_pool_core", "log1p_pool_core_addr",
            "pool_core_keys", "log1p_s1_core", "s1_core_other", "sib_at_s1_key", "sib_at_s1_main",
            "sib_off_s1_key", "orphan_keys", "orphan_recs", "s1_same_core_same_key", "log1p_pool_name",
            "pool_name_keys", "key_eq", "main_eq", "rec_key_empty", "rec_is_bridge", "rec_bridge_s1",
            "alias_bridged_to_s1", "log1p_core_bridges"]


def _core(folded_legal_stripped: str) -> str:
    return " ".join(sorted({t for t in folded_legal_stripped.split() if t not in DROP}))


def _names(values):
    """(folded full name codes' strings, core strings, bridge core pairs) per row."""
    codes, uniques = _encode(fold_ascii(values))
    stripped = [_strip_legal(u) for u in uniques]
    core_u = np.asarray([_core(s) for s in stripped] + [""], dtype=object)
    full_u = np.asarray(uniques + [""], dtype=object)
    bridge_a = np.full(len(uniques) + 1, "", dtype=object)
    bridge_b = np.full(len(uniques) + 1, "", dtype=object)
    for i, u in enumerate(uniques):
        m = BRIDGE.match(u)
        if m:
            bridge_a[i], bridge_b[i] = _core(_strip_legal(m.group(1))), _core(_strip_legal(m.group(2)))
    return full_u[codes], core_u[codes], bridge_a[codes], bridge_b[codes]


def _keys(addresses):
    digits = digit_strings(fold_ascii(addresses))
    codes, uniques = _encode(digits)
    key_u, main_u = [], []
    for u in uniques:
        groups = sorted({g.lstrip("0") or "0" for g in u.split()})
        key_u.append(" ".join(groups))
        main_u.append(max(groups, key=len) if groups else "")
    return np.asarray(key_u + [""], dtype=object)[codes], np.asarray(main_u + [""], dtype=object)[codes]


def _count_pairs(a, b, n_b):
    """Counts of the (a, b) combinations (both >= 0) as a sorted-key lookup."""
    ok = (a >= 0) & (b >= 0)
    keys, counts = np.unique(a[ok].astype(np.int64) * n_b + b[ok], return_counts=True)
    return keys, counts


def _lookup(keys, counts, query):
    out = np.zeros(len(query), dtype=np.int64)
    if len(keys) == 0:
        return out
    pos = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
    hit = keys[pos] == query
    out[hit] = counts[pos[hit]]
    return out


def build_tables(ref_names, ref_addresses, pool_names, pool_addresses, log=print):
    """Per-S1 / per-record codes and pool-wide counts. Inputs are array-likes of strings."""
    log("collective: names")
    s1_full, s1_core, _, _ = _names(ref_names)
    rec_full, rec_core, br_a, br_b = _names(pool_names)
    log("collective: address keys")
    s1_key, s1_main = _keys(ref_addresses)
    rec_key, rec_main = _keys(pool_addresses)
    n_s1, n_rec = len(s1_core), len(rec_core)
    core_codes, core_u = pd.factorize(np.concatenate([s1_core, rec_core, br_a, br_b]), sort=False)
    empty_core = np.flatnonzero(core_u == "")
    core_codes = core_codes.astype(np.int64)
    if len(empty_core):
        core_codes[core_codes == empty_core[0]] = -1
    c_s1, c_rec = core_codes[:n_s1], core_codes[n_s1:n_s1 + n_rec]
    c_ba, c_bb = core_codes[n_s1 + n_rec:n_s1 + 2 * n_rec], core_codes[n_s1 + 2 * n_rec:]
    n_core = int(core_codes.max()) + 1
    key_codes, key_u = pd.factorize(np.concatenate([s1_key, rec_key]), sort=False)
    key_codes = key_codes.astype(np.int64)
    empty_key = np.flatnonzero(key_u == "")
    if len(empty_key):
        key_codes[key_codes == empty_key[0]] = -1
    k_s1, k_rec = key_codes[:n_s1], key_codes[n_s1:]
    n_key = int(key_codes.max()) + 1
    main_codes, main_u = pd.factorize(np.concatenate([s1_main, rec_main]), sort=False)
    main_codes = main_codes.astype(np.int64)
    empty_main = np.flatnonzero(main_u == "")
    if len(empty_main):
        main_codes[main_codes == empty_main[0]] = -1
    m_s1, m_rec = main_codes[:n_s1], main_codes[n_s1:]
    n_main = int(main_codes.max()) + 1
    full_codes, _ = pd.factorize(rec_full, sort=False)
    full_codes = full_codes.astype(np.int64)
    full_codes[rec_full == ""] = -1
    n_full = int(full_codes.max()) + 1

    log("collective: counts")
    ok_core = c_rec >= 0
    pool_core = np.bincount(c_rec[ok_core], minlength=n_core)
    pool_core_addr = np.bincount(c_rec[ok_core & (k_rec >= 0)], minlength=n_core)
    pcd_keys, pcd_counts = _count_pairs(c_rec, k_rec, n_key)
    pool_core_keys = np.bincount(pcd_keys // n_key, minlength=n_core)
    pcm_keys, pcm_counts = _count_pairs(c_rec, m_rec, n_main)
    s1_core_count = np.bincount(c_s1[c_s1 >= 0], minlength=n_core)
    scd_keys, scd_counts = _count_pairs(c_s1, k_s1, n_key)
    orphan = ~np.isin(pcd_keys, scd_keys)
    orphan_keys = np.bincount(pcd_keys[orphan] // n_key, minlength=n_core)
    orphan_recs = np.bincount(pcd_keys[orphan] // n_key, weights=pcd_counts[orphan], minlength=n_core)
    ok_full = full_codes >= 0
    pool_name = np.bincount(full_codes[ok_full], minlength=n_full)
    pnk_keys, _ = _count_pairs(full_codes, k_rec, n_key)
    pool_name_keys = np.bincount(pnk_keys // n_key, minlength=n_full)
    is_bridge = (c_ba >= 0) & (c_bb >= 0)
    ba, bb = c_ba[is_bridge], c_bb[is_bridge]
    bridge_pairs = np.unique(np.concatenate([ba * n_core + bb, bb * n_core + ba]))
    core_bridges = np.bincount(np.concatenate([ba, bb]), minlength=n_core)
    log(f"collective: {n_s1} S1s, {n_rec} records, {n_core} cores, {int(is_bridge.sum())} bridge records")
    return SimpleNamespace(
        c_s1=c_s1, c_rec=c_rec, c_ba=c_ba, c_bb=c_bb, k_s1=k_s1, k_rec=k_rec, m_s1=m_s1, m_rec=m_rec,
        full=full_codes, n_core=n_core, n_key=n_key, n_main=n_main,
        core_tokens=np.asarray([frozenset(str(c).split()) for c in core_u] + [frozenset()], dtype=object),
        pool_core=pool_core, pool_core_addr=pool_core_addr, pcd=(pcd_keys, pcd_counts), pcm=(pcm_keys, pcm_counts),
        pool_core_keys=pool_core_keys, s1_core=s1_core_count, scd=(scd_keys, scd_counts), orphan_keys=orphan_keys,
        orphan_recs=orphan_recs, pool_name=pool_name, pool_name_keys=pool_name_keys, bridge_pairs=bridge_pairs,
        core_bridges=core_bridges)


def pair_features(t, left, right, chunk=5_000_000) -> np.ndarray:
    """Float32 matrix (len(left), len(FEATURES)) for S1 indices ``left`` and record indices ``right``."""
    out = np.empty((len(left), len(FEATURES)), dtype=np.float32)
    for start in range(0, len(left), chunk):
        out[start:start + chunk] = _pair_chunk(t, np.asarray(left[start:start + chunk], dtype=np.int64),
                                               np.asarray(right[start:start + chunk], dtype=np.int64))
    return out


def _pair_chunk(t, s, r):
    cs, cr = t.c_s1[s], t.c_rec[r]
    ks, kr, ms, mr = t.k_s1[s], t.k_rec[r], t.m_s1[s], t.m_rec[r]
    valid = cr >= 0
    crc = np.where(valid, cr, 0)
    core_eq = valid & (cs == cr)
    ts, tr = t.core_tokens[cs], t.core_tokens[cr]  # code -1 -> the trailing empty set
    jaccard = np.fromiter((len(a & b) / len(a | b) if (a and b) else 0.0 for a, b in zip(ts, tr)), dtype=np.float32,
                          count=len(s))
    subset = np.fromiter((bool(b) and b < a for a, b in zip(ts, tr)), dtype=bool, count=len(s))
    superset = np.fromiter((bool(a) and a < b for a, b in zip(ts, tr)), dtype=bool, count=len(s))
    pool_core = np.where(valid, t.pool_core[crc], 0)
    pool_core_addr = np.where(valid, t.pool_core_addr[crc], 0)
    at_key = np.where(valid & (ks >= 0), _lookup(*t.pcd, crc * t.n_key + np.maximum(ks, 0)), 0)
    self_at = (valid & (ks >= 0) & (kr == ks)).astype(np.int64)
    sib_at = at_key - self_at
    at_main = np.where(valid & (ms >= 0), _lookup(*t.pcm, crc * t.n_main + np.maximum(ms, 0)), 0)
    sib_at_main = at_main - (valid & (ms >= 0) & (mr == ms)).astype(np.int64)
    self_off = (valid & (kr >= 0) & (kr != ks)).astype(np.int64)
    sib_off = pool_core_addr - at_key - self_off
    s1_core_r = np.where(valid, t.s1_core[crc], 0)
    same_core_same_key = np.where((cs >= 0) & (ks >= 0),
                                  _lookup(*t.scd, np.maximum(cs, 0) * t.n_key + np.maximum(ks, 0)), 1) - 1
    fr = t.full[r]
    pool_name = np.where(fr >= 0, t.pool_name[np.maximum(fr, 0)], 0)
    pool_name_keys = np.where(fr >= 0, t.pool_name_keys[np.maximum(fr, 0)], 0)
    ba, bb = t.c_ba[r], t.c_bb[r]
    is_bridge = (ba >= 0) & (bb >= 0)
    bridge_s1 = is_bridge & ((ba == cs) | (bb == cs))
    alias_bridged = valid & (cs >= 0) & np.isin(crc * t.n_core + np.maximum(cs, 0), t.bridge_pairs)
    cols = [core_eq, jaccard, subset, superset, ~valid, np.log1p(pool_core), np.log1p(pool_core_addr),
            np.minimum(np.where(valid, t.pool_core_keys[crc], 0), 50), np.log1p(s1_core_r),
            s1_core_r - core_eq.astype(np.int64), sib_at, sib_at_main, sib_off,
            np.where(valid, t.orphan_keys[crc], 0), np.where(valid, t.orphan_recs[crc], 0), same_core_same_key,
            np.log1p(pool_name), np.minimum(pool_name_keys, 50), (ks >= 0) & (kr == ks), (ms >= 0) & (mr == ms),
            kr < 0, is_bridge, bridge_s1, alias_bridged,
            np.log1p(np.where(valid, t.core_bridges[crc], 0))]
    return np.column_stack([np.asarray(c, dtype=np.float32) for c in cols])
