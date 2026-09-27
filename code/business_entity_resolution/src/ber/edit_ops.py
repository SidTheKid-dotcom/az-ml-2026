"""K3: generator-likelihood (edit-operation) features for (S1, record) pairs.

Each pair is aligned S1 -> record and described ONLY by the edit operations that
turn one into the other: legal-form relation, added/dropped suffix words (copy
suffix / decoy word / marker), house-number relation (equal, truncated, zero
padded, letter added, signed shift, delta-set, even shift), name word operations
(exact / typo / abbreviation / initials / join matches, substitutions,
insertions, deletions, reorder), address operations (street-token matches, typo,
abbreviation, missing components, last-component equality) and case/format
flags. Text is accent-folded and lower-cased; no raw token, token identity or
country enters the feature vector (the word classes are fixed closed lists that
describe generator operations, e.g. legal forms or decoy words).

    python -m ber.edit_ops build --left L.npy --right R.npy --ref REF.parquet --pool POOL.parquet --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

# ----------------------------------------------------------------------------- closed word classes

LEGAL_CLASS = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "llc": "llc", "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "llp": "llp", "plc": "plc",
    "lp": "lp", "pllc": "pllc", "pc": "pc", "opc": "opc", "gmbh": "gmbh",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci", "eurl": "eurl", "ei": "ei",
    "snc": "snc", "selarl": "selarl", "scop": "scop", "sca": "sca", "sccv": "sccv", "scm": "scm", "gie": "gie",
    "earl": "earl", "gaec": "gaec", "scea": "scea", "selas": "selas", "scp": "scp",
}
DECOY = frozenset(
    "group groupe groups holdings holding overseas industries ventures exports infratech public "
    "developpement development participations distribution international france india usa global "
    "worldwide".split())
COPY_SUFFIX = frozenset("sons fils associes associates cie services service brothers bros freres partners".split())
MARKER = frozenset("dba aka nee formerly fka".split())
FILLER = frozenset("the and et of de la le les des du d l".split())
DOMAIN = frozenset("www com net org biz info".split())

DELTA_SET = frozenset((1, 3, 5, 7, 9, 11, 13, 21))
LEGAL_REL = ("none", "same", "same_class_other_surface", "added", "dropped", "swapped", "partial")
HN_REL = ("none", "s1_only", "rec_only", "equal", "zero_pad", "truncated", "extended", "other")

FEATURES = [
    # emptiness / format
    "s1_addr_empty", "rec_addr_empty", "s1_addr_has_digits", "rec_addr_has_digits",
    "rec_name_native", "rec_addr_native", "s1_name_native",
    "name_case_same", "rec_name_upper", "rec_name_lower", "addr_case_same", "rec_addr_upper",
    "rec_name_double_space", "rec_bracket", "rec_domain_added", "name_folded_equal", "addr_folded_equal",
    # legal form
    "legal_rel", "legal_n_s1", "legal_n_rec", "legal_position_changed",
    # suffix / decoy / marker words
    "decoy_added", "decoy_dropped", "copysfx_added", "copysfx_dropped", "marker_rec", "marker_s1",
    # name core alignment
    "core_n_s1", "core_n_rec", "core_equal", "core_set_equal", "core_exact", "core_typo", "core_typo_ed_sum",
    "core_typo_ed_max", "core_abbrev", "core_initials", "core_join", "core_sub", "core_del", "core_ins",
    "core_reorder", "core_sub_best_ratio", "core_ratio", "core_token_sort", "core_partial", "core_rest_ratio",
    "name_ratio_full", "core_doubled", "name_digits_rel",
    # house number
    "hn_rel", "hn_letter_added", "hn_letter_dropped", "hn_diff", "hn_abs_diff", "hn_in_delta", "hn_even_shift",
    "hn_s1_in_rec_any", "dg_n_s1", "dg_n_rec", "dg_common", "dg_seq_equal", "postal_eq",
    # address alignment
    "at_n_s1", "at_n_rec", "at_exact", "at_typo", "at_typo_ed_sum", "at_abbrev", "at_del", "at_ins", "at_sub",
    "at_reorder", "addr_ratio", "addr_token_set", "comp_n_s1", "comp_n_rec", "comp_match_frac", "comp_last_eq",
    "comp_first_eq", "comp_reordered",
]
NF = len(FEATURES)

# ----------------------------------------------------------------------------- normalization

_LIG = str.maketrans({"œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae", "ß": "ss", "ø": "o", "Ø": "o", "ł": "l",
                      "Ł": "l", "đ": "d", "Đ": "d"})
_NULL = re.compile(r"<\s*null\s*>", re.I)
_TOKEN = re.compile(r"[a-z0-9]+")
_DIGITS = re.compile(r"[0-9]+")
_NATIVE = re.compile(r"[ऀ-෿]")
_BRACKET = re.compile(r"[\(\[][^\)\]]*[\)\]]")


def clean(text):
    if text is None:
        return ""
    text = _NULL.sub(" ", str(text))
    return text.strip()


def fold(text):
    if text.isascii():
        return text.lower()
    text = unicodedata.normalize("NFKD", text.translate(_LIG))
    return text.encode("ascii", "ignore").decode().lower()


def tokens(folded):
    return _TOKEN.findall(folded.replace("'", "").replace("’", ""))


def case_kind(raw):
    letters = [c for c in raw if c.isalpha()]
    if not letters:
        return 0
    if all(c.isupper() for c in letters):
        return 1
    if all(c.islower() for c in letters):
        return 2
    return 3


@lru_cache(maxsize=200_000)
def parse_name(raw):
    raw = clean(raw)
    folded = fold(raw)
    bracket = bool(_BRACKET.search(folded))
    toks = tokens(folded)
    legal, legal_pos, decoy, copysfx, marker, core, domain, digits = [], [], set(), set(), False, [], False, []
    n = len(toks)
    for i, t in enumerate(toks):
        if t in LEGAL_CLASS:
            legal.append(t)
            legal_pos.append(0 if i < n / 2 else 1)
        elif t in DECOY:
            decoy.add(t)
        elif t in COPY_SUFFIX:
            copysfx.add(t)
        elif t in MARKER:
            marker = True
        elif t in DOMAIN:
            domain = True
        elif t in FILLER:
            continue
        else:
            if t.isdigit():
                digits.append(t)
            core.append(t)
    return {"raw": raw, "folded": " ".join(toks), "case": case_kind(raw), "native": bool(_NATIVE.search(raw)),
            "double_space": "  " in raw, "bracket": bracket, "legal": legal, "legal_pos": legal_pos,
            "legal_cls": frozenset(LEGAL_CLASS[t] for t in legal), "decoy": decoy, "copysfx": copysfx,
            "marker": marker, "domain": domain, "core": core, "digits": digits}


@lru_cache(maxsize=200_000)
def parse_addr(raw):
    raw = clean(raw)
    folded = fold(raw)
    toks = tokens(folded)
    groups = _DIGITS.findall(folded)
    alpha = [t for t in toks if not any(c.isdigit() for c in t)]
    # house-number token: first token containing a digit
    hn_tok = next((t for t in toks if any(c.isdigit() for c in t)), None)
    comps = [" ".join(tokens(c)) for c in folded.split(",")]
    comps = [c for c in comps if c]
    return {"raw": raw, "folded": " ".join(toks), "empty": not toks, "case": case_kind(raw),
            "native": bool(_NATIVE.search(raw)), "groups": groups, "alpha": alpha, "hn_tok": hn_tok,
            "comps": comps}


# ----------------------------------------------------------------------------- alignment

def is_abbrev(short, long):
    """``short`` abbreviates ``long``: a strict prefix, or an in-order subsequence sharing the first letter."""
    if len(short) >= len(long) or not short or short[0] != long[0]:
        return False
    if long.startswith(short):
        return True
    it = iter(long)
    return all(c in it for c in short)


def align(a, b):
    """Greedy S1 -> record token alignment. Returns counts of each operation and the unmatched remainders."""
    a_left = list(a)
    b_left = list(b)
    matched_pairs = []  # (i in a, j in b)
    a_idx = list(range(len(a)))
    b_idx = list(range(len(b)))
    # exact
    exact = 0
    for ia in list(a_idx):
        t = a[ia]
        for jb in b_idx:
            if b[jb] == t:
                matched_pairs.append((ia, jb))
                a_idx.remove(ia)
                b_idx.remove(jb)
                exact += 1
                break
    # typo (edit distance <= max(1, len/4)), best first
    typo, typo_sum, typo_max = 0, 0, 0
    if a_idx and b_idx:
        cands = []
        for ia in a_idx:
            for jb in b_idx:
                x, y = a[ia], b[jb]
                if x.isdigit() or y.isdigit() or min(len(x), len(y)) < 3:
                    continue
                d = Levenshtein.distance(x, y, score_cutoff=max(1, max(len(x), len(y)) // 4))
                if d <= max(1, max(len(x), len(y)) // 4):
                    cands.append((d, ia, jb))
        cands.sort()
        for d, ia, jb in cands:
            if ia in a_idx and jb in b_idx:
                matched_pairs.append((ia, jb))
                a_idx.remove(ia)
                b_idx.remove(jb)
                typo += 1
                typo_sum += d
                typo_max = max(typo_max, d)
    # abbreviation (either direction)
    abbrev = 0
    for ia in list(a_idx):
        for jb in b_idx:
            if is_abbrev(b[jb], a[ia]) or is_abbrev(a[ia], b[jb]):
                matched_pairs.append((ia, jb))
                a_idx.remove(ia)
                b_idx.remove(jb)
                abbrev += 1
                break
    # initials: one record token = initials of >= 2 consecutive unmatched S1 tokens (or reverse)
    initials = 0
    for side_short, side_long, idx_short, idx_long in ((b, a, b_idx, a_idx), (a, b, a_idx, b_idx)):
        for js in list(idx_short):
            t = side_short[js]
            if len(t) < 2 or len(t) > 6 or not t.isalpha():
                continue
            longs = sorted(idx_long)
            for start in range(len(longs) - len(t) + 1):
                run = longs[start:start + len(t)]
                if run[-1] - run[0] == len(t) - 1 and "".join(side_long[k][0] for k in run) == t:
                    for k in run:
                        idx_long.remove(k)
                    idx_short.remove(js)
                    initials += 1
                    break
    # join/split: one token equals the concatenation of two consecutive tokens on the other side
    join = 0
    for side_one, side_two, idx_one, idx_two in ((b, a, b_idx, a_idx), (a, b, a_idx, b_idx)):
        for j1 in list(idx_one):
            t = side_one[j1]
            two = sorted(idx_two)
            for k in range(len(two) - 1):
                if two[k + 1] == two[k] + 1 and side_two[two[k]] + side_two[two[k + 1]] == t:
                    idx_two.remove(two[k])
                    idx_two.remove(two[k + 1])
                    idx_one.remove(j1)
                    join += 1
                    break
    n_del, n_ins = len(a_idx), len(b_idx)
    sub = min(n_del, n_ins)
    best_sub = 0.0
    if a_idx and b_idx:
        best_sub = max(fuzz.ratio(a[i], b[j]) for i in a_idx for j in b_idx)
    matched_pairs.sort(key=lambda p: p[1])
    order = [p[0] for p in matched_pairs]
    reorder = int(any(order[k] > order[k + 1] for k in range(len(order) - 1)))
    rest_a = " ".join(a[i] for i in a_idx)
    rest_b = " ".join(b[j] for j in b_idx)
    return (exact, typo, typo_sum, typo_max, abbrev, initials, join, sub, n_del - sub, n_ins - sub, reorder,
            best_sub, rest_a, rest_b)


def _int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def number_features(sa, ra):
    """House-number relation between S1 address ``sa`` and record address ``ra`` (parsed)."""
    sg, rg = sa["groups"], ra["groups"]
    nan = float("nan")
    if not sg and not rg:
        rel, diff = 0, nan
    elif not rg:
        rel, diff = 1, nan
    elif not sg:
        rel, diff = 2, nan
    else:
        s0, r0 = sg[0], rg[0]
        si, ri = _int(s0), _int(r0)
        diff = float(max(-100000, min(100000, ri - si))) if si is not None and ri is not None else nan
        if s0 == r0:
            rel = 3
        elif si == ri:
            rel = 4
        elif s0.startswith(r0):
            rel = 5
        elif r0.startswith(s0):
            rel = 6
        else:
            rel = 7
    letter_added = letter_dropped = 0
    st, rt = sa["hn_tok"], ra["hn_tok"]
    if st is not None and rt is not None:
        s_alpha, r_alpha = any(c.isalpha() for c in st), any(c.isalpha() for c in rt)
        letter_added = int(r_alpha and not s_alpha)
        letter_dropped = int(s_alpha and not r_alpha)
    ad = abs(diff) if diff == diff else nan
    in_delta = float(diff in DELTA_SET) if diff == diff else nan
    even = float(diff != 0 and diff % 2 == 0 and ad <= 20) if diff == diff else nan
    s_in_any = float(sg[0] in rg) if sg and rg else nan
    common = 0
    if sg and rg:
        pool = list(rg)
        for g in sg:
            if g in pool:
                pool.remove(g)
                common += 1
    seq_eq = float(sg == rg) if sg or rg else nan
    s_post = max(sg, key=len) if sg else ""
    r_post = max(rg, key=len) if rg else ""
    postal = float(s_post == r_post) if len(s_post) >= 5 and len(r_post) >= 5 else nan
    return [rel, letter_added, letter_dropped, diff, ad, in_delta, even, s_in_any, len(sg), len(rg), common,
            seq_eq, postal]


def pair_features(s_name, s_addr, r_name, r_addr):
    sn, rn = parse_name(s_name), parse_name(r_name)
    sa, ra = parse_addr(s_addr), parse_addr(r_addr)
    nan = float("nan")
    f = []
    # emptiness / format
    f += [float(sa["empty"]), float(ra["empty"]), float(bool(sa["groups"])), float(bool(ra["groups"])),
          float(rn["native"]), float(ra["native"]), float(sn["native"]),
          float(sn["case"] == rn["case"]), float(rn["case"] == 1), float(rn["case"] == 2),
          float(sa["case"] == ra["case"]) if not (sa["empty"] or ra["empty"]) else nan,
          float(ra["case"] == 1) if not ra["empty"] else nan,
          float(rn["double_space"]), float(rn["bracket"] and not sn["bracket"]),
          float(rn["domain"] and not sn["domain"]), float(sn["folded"] == rn["folded"]),
          float(sa["folded"] == ra["folded"]) if not (sa["empty"] or ra["empty"]) else nan]
    # legal form relation
    ls, lr = sn["legal_cls"], rn["legal_cls"]
    if not ls and not lr:
        rel = 0
    elif ls == lr:
        rel = 1 if sorted(sn["legal"]) == sorted(rn["legal"]) else 2
    elif not ls:
        rel = 3
    elif not lr:
        rel = 4
    elif ls.isdisjoint(lr):
        rel = 5
    else:
        rel = 6
    pos_changed = nan
    if sn["legal_pos"] and rn["legal_pos"]:
        pos_changed = float(sn["legal_pos"][0] != rn["legal_pos"][0])
    f += [rel, len(sn["legal"]), len(rn["legal"]), pos_changed]
    f += [len(rn["decoy"] - sn["decoy"]), len(sn["decoy"] - rn["decoy"]), len(rn["copysfx"] - sn["copysfx"]),
          len(sn["copysfx"] - rn["copysfx"]), float(rn["marker"]), float(sn["marker"])]
    # name core alignment
    a, b = sn["core"], rn["core"]
    (exact, typo, typo_sum, typo_max, abbrev, initials, join, sub, dele, ins, reorder, best_sub, rest_a,
     rest_b) = align(a, b)
    ja, jb = " ".join(a), " ".join(b)
    doubled = float(len(b) != len(set(b)) and len(a) == len(set(a)))
    sd, rd = sn["digits"], rn["digits"]
    digits_rel = 0 if not sd and not rd else (1 if sd == rd else (2 if not rd else (3 if not sd else 4)))
    f += [len(a), len(b), float(a == b), float(set(a) == set(b)), exact, typo, typo_sum, typo_max, abbrev,
          initials, join, sub, dele, ins, reorder, best_sub, fuzz.ratio(ja, jb), fuzz.token_sort_ratio(ja, jb),
          fuzz.partial_ratio(ja, jb) if ja and jb else 0.0,
          fuzz.ratio(rest_a, rest_b) if (rest_a or rest_b) else 100.0,
          fuzz.ratio(sn["folded"], rn["folded"]), doubled, digits_rel]
    # house number
    f += number_features(sa, ra)
    # address alignment
    if sa["empty"] or ra["empty"]:
        f += [len(sa["alpha"]), len(ra["alpha"])] + [nan] * 16
    else:
        aa, ab = sa["alpha"], ra["alpha"]
        (x_exact, x_typo, x_typo_sum, _, x_abbrev, x_init, x_join, x_sub, x_del, x_ins, x_reorder, _, _,
         _) = align(aa, ab)
        sc, rc = sa["comps"], ra["comps"]
        match = 0
        for c in sc:
            if any(c == d or fuzz.token_set_ratio(c, d) >= 85 for d in rc):
                match += 1
        comp_frac = match / len(sc) if sc else nan
        last_eq = float(fuzz.token_set_ratio(sc[-1], rc[-1]) >= 85) if sc and rc else nan
        first_eq = float(fuzz.token_set_ratio(sc[0], rc[0]) >= 85) if sc and rc else nan
        reordered = float(len(sc) > 1 and len(rc) > 1 and sorted(sc) == sorted(rc) and sc != rc)
        f += [len(aa), len(ab), x_exact, x_typo, x_typo_sum, x_abbrev + x_init + x_join, x_del, x_ins, x_sub,
              x_reorder, fuzz.ratio(sa["folded"], ra["folded"]), fuzz.token_set_ratio(sa["folded"], ra["folded"]),
              len(sc), len(rc), comp_frac, last_eq, first_eq, reordered]
    return f


def _check_len():
    v = pair_features("Acme Holdings LLC", "12 Main St, Springfield, IL 62704", "ACME llc", "13 Main Street, Springfield")
    assert len(v) == NF, (len(v), NF)


def chunk_features(args):
    """Worker: rows of strings -> float32 feature matrix."""
    s_names, s_addrs, r_names, r_addrs = args
    out = np.empty((len(s_names), NF), dtype=np.float32)
    for i in range(len(s_names)):
        out[i] = pair_features(s_names[i], s_addrs[i], r_names[i], r_addrs[i])
    return out


# ----------------------------------------------------------------------------- build

def load_text(path):
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["business_name", "business_address"])
    names = np.asarray(t.column("business_name").to_pylist(), dtype=object)
    addrs = np.asarray(t.column("business_address").to_pylist(), dtype=object)
    return names, addrs


def build(left, right, ref_path, pool_path, out_dir, workers, chunk=100_000, tag="feat"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s_names, s_addrs = load_text(ref_path)
    r_names, r_addrs = load_text(pool_path)
    print(f"[{time.time() - t0:.0f}s] text loaded", flush=True)
    n = len(left)
    starts = list(range(0, n, chunk))
    todo = [s for s in starts if not (out_dir / f"{tag}_{s:010d}.npy").exists()]

    def jobs():
        for s in todo:
            l, r = left[s:s + chunk], right[s:s + chunk]
            yield s, (s_names[l].tolist(), s_addrs[l].tolist(), r_names[r].tolist(), r_addrs[r].tolist())

    ctx = get_context("fork")
    with ctx.Pool(workers, maxtasksperchild=50) as pool:
        it = jobs()
        pending = []
        done = 0
        for s, payload in it:
            pending.append((s, pool.apply_async(chunk_features, (payload,))))
            while len(pending) >= workers * 2:
                s0, res = pending.pop(0)
                np.save(out_dir / f"{tag}_{s0:010d}.npy", res.get())
                done += 1
                if done % 20 == 0:
                    print(f"[{time.time() - t0:.0f}s] {done}/{len(todo)} chunks", flush=True)
        for s0, res in pending:
            np.save(out_dir / f"{tag}_{s0:010d}.npy", res.get())
    x = np.concatenate([np.load(out_dir / f"{tag}_{s:010d}.npy") for s in starts]) if starts else \
        np.zeros((0, NF), np.float32)
    np.save(out_dir / f"{tag}.npy", x)
    for s in starts:
        (out_dir / f"{tag}_{s:010d}.npy").unlink()
    (out_dir / f"{tag}.json").write_text(json.dumps({"features": FEATURES, "rows": int(n),
                                                     "seconds": round(time.time() - t0, 1)}) + "\n")
    print(f"[{time.time() - t0:.0f}s] wrote {out_dir / (tag + '.npy')} {x.shape}", flush=True)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build", "selftest"])
    p.add_argument("--left")
    p.add_argument("--right")
    p.add_argument("--rows", help="optional .npy of row indices into left/right to restrict to")
    p.add_argument("--ref")
    p.add_argument("--pool")
    p.add_argument("--out")
    p.add_argument("--tag", default="feat")
    p.add_argument("--workers", type=int, default=os.cpu_count() - 2)
    a = p.parse_args()
    _check_len()
    if a.command == "selftest":
        for x in [("Acme Holdings LLC", "12 Main St, Springfield, IL 62704", "ACME llc", "13 Main Street, Springfield"),
                  ("Boulangerie Dupont SARL", "12 rue de la Paix, 75002 Paris", "Boulangerie Dupont SAS",
                   "14 Rue de la Paix, 75002 Paris"),
                  ("International Business Machines Corp", "", "IBM Corporation", "1 New Orchard Rd, Armonk")]:
            print(dict(zip(FEATURES, pair_features(*x))))
        return
    left = np.load(a.left).astype(np.int64)
    right = np.load(a.right).astype(np.int64)
    if a.rows:
        rows = np.load(a.rows)
        left, right = left[rows], right[rows]
    build(left, right, a.ref, a.pool, a.out, a.workers, tag=a.tag)


if __name__ == "__main__":
    main()
