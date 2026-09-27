"""Language-agnostic string-feature pair features (Agent G, run27).

No cross-encoder, LLM or dense-embedding score is used as a feature. Text is
folded (anyascii for non-Latin/accents, lowercase, punctuation stripped);
legal-form tokens, trailing copy suffixes and dba/aka alias parts are split
into their own fields. Pair features are computed per (S1, record) pair from
per-unique-string preprocessing, in forked workers writing into a memmap.

Country is never a stage-1 feature (France must be handled without it).
"""
from __future__ import annotations

import os
import re
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- vocabularies (label-free)
LEGAL = frozenset(("inc incorporated corp corporation llc ltd limited pvt llp plc pllc lp "
                   "sarl sas sasu sci eurl snc selarl scop sca gie gmbh opc sa ei co company").split())
LEGAL_CANON = {"incorporated": "inc", "corporation": "corp", "limited": "ltd", "company": "co"}
PRIVPUB = frozenset({"private", "public"})
COPY_SUFFIX = frozenset({"cie", "services", "service", "associes", "associates", "fils", "sons", "freres"})
CONNECT = frozenset({"and", "et"})
ALIAS_MARKERS = frozenset({"dba", "aka", "nee", "formerly", "fka"})
DECOY = frozenset(("group groupe holdings holding overseas industries ventures exports infratech "
                   "developpement development participations distribution international france global").split())
GEO = frozenset(("north south east west central national american america usa united pacific atlantic midwest "
                 "northern southern eastern western valley harbor downtown metro summit greater coastal northside "
                 "highland westgate midtown lakeside southside eastgate riverside uptown").split())
COPYWORD = frozenset(("center centers partners partner associates associes associe services service cie fils "
                      "sons ctr freres").split())
HONOR = frozenset("mr mrs ms dr sri shri smt mme mlle".split())
ABBREV = {"st": "street", "str": "street", "ave": "avenue", "av": "avenue", "rd": "road", "ln": "lane",
          "dr": "drive", "blvd": "boulevard", "bd": "boulevard", "ct": "court", "pl": "place", "cir": "circle",
          "hwy": "highway", "pkwy": "parkway", "r": "rue", "all": "allee", "q": "quai", "imp": "impasse",
          "crs": "cours", "ter": "terrace", "trl": "trail", "sq": "square", "flr": "floor", "fl": "floor",
          "chem": "chemin", "rte": "route", "pte": "porte", "fbg": "faubourg"}
ADDR_DROP = frozenset({"null", "no", "nr", "unit", "apt", "apartment", "suite", "ste", "bis", "na"})
TRANSLIT = {"praivet": "private", "praivhet": "private", "prayvet": "private", "elelpi": "llp", "limited": "limited",
            "limiteda": "limited", "kampani": "company", "inkarporeted": "incorporated"}
DECOY_DELTAS = frozenset({1, 3, 5, 7, 9, 11, 13, 21})
_TOK = re.compile(r"[a-z0-9]+")
_NULL = re.compile(r"<\s*(?:null|nan|na|none)\s*>", re.IGNORECASE)
_DIG = re.compile(r"[0-9]+")


def fold(value):
    if value is None or (isinstance(value, float) and value != value):
        return "", False
    s = _NULL.sub(" ", str(value))
    native = False
    if not s.isascii():
        native = any("ऀ" <= c <= "෿" for c in s)
        s = anyascii(s)
    s = s.lower().replace("'", "").replace("’", "").replace("&", " and ")
    return s, native


# ----------------------------------------------------------------------------- per-unique names

def _join_letters(toks):
    """Join runs of >= 2 single-letter tokens ("s a r l" -> "sarl", from S.A.R.L.)."""
    out, run = [], []
    for t in toks + [""]:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        elif run:
            out.append(run[0])
        run = []
        if t:
            out.append(t)
    return out


def name_fields(raw):
    """(full, core, alias_core, legal_key, copy_key, native, core_tokens)."""
    s, native = fold(raw)
    toks = ["and" if t == "et" else TRANSLIT.get(t, t) if native else t for t in _join_letters(_TOK.findall(s))]
    full = " ".join(toks)
    alias = []
    for i, t in enumerate(toks):
        if t in ALIAS_MARKERS and 0 < i < len(toks) - 1:
            alias = toks[i + 1:]
            toks = toks[:i]
            break
    core_list, legal = _split_legal(toks)
    core_list, copy = _split_copy(core_list)
    alias_core, _ = _split_legal(alias)
    alias_core, _ = _split_copy(alias_core)
    if core_list and core_list[0] == "the" and len(core_list) > 1:
        core_list = core_list[1:]
    return (full, " ".join(core_list), " ".join(alias_core), " ".join(sorted(set(legal))),
            " ".join(copy), native, tuple(core_list))


def _split_legal(toks):
    core, legal = [], []
    n = len(toks)
    for i, t in enumerate(toks):
        if t in LEGAL:
            legal.append(LEGAL_CANON.get(t, t))
        elif t in PRIVPUB and i + 1 < n and toks[i + 1] in ("ltd", "limited", "co", "company"):
            legal.append("pvt" if t == "private" else t)
        else:
            core.append(t)
    if not core and toks:  # name made only of legal tokens: keep it
        return list(toks), legal
    return core, legal


def _split_copy(toks):
    toks = list(toks)
    copy = []
    while len(toks) > 1 and (toks[-1] in COPY_SUFFIX or toks[-1] in CONNECT):
        copy.append(toks.pop())
    return toks, [t for t in reversed(copy) if t not in CONNECT]


# ----------------------------------------------------------------------------- per-unique addresses

def addr_fields(raw):
    """(full, alpha, street, h_str, h_tok, digits, loc_comps, kind, alpha_set, postcodes)."""
    s, _ = fold(raw)
    comps = []
    for c in s.split(","):
        toks = [ABBREV.get(t, t) for t in _TOK.findall(c)]
        toks = [t for t in toks if t not in ADDR_DROP]
        if toks:
            comps.append(toks)
    all_toks = [t for c in comps for t in c]
    if not all_toks:
        return ("", "", "", "", "", (), frozenset(), 0, frozenset(), frozenset())
    digits = tuple(d for t in all_toks for d in _DIG.findall(t))
    alpha = [t for t in all_toks if not any(ch.isdigit() for ch in t)]
    h_str, h_tok, street_idx = "", "", -1
    for ci, c in enumerate(comps):
        for t in c:
            m = _DIG.search(t)
            if m:
                h_str, h_tok, street_idx = m.group(0), t, ci
                break
        if street_idx >= 0:
            break
    if street_idx < 0:
        street_idx = 0
    street_comps = {street_idx}
    street = [t for t in comps[street_idx] if not any(ch.isdigit() for ch in t)]
    if not street and street_idx + 1 < len(comps):
        street_comps.add(street_idx + 1)
        street = [t for t in comps[street_idx + 1] if not any(ch.isdigit() for ch in t)]
    loc = frozenset(" ".join(c) for i, c in enumerate(comps)
                    if i not in street_comps and any(not ch.isdigit() for t in c for ch in t))
    post = frozenset(d for d in digits if len(d) >= 5)
    kind = 2 if digits else 1
    return (" ".join(all_toks), " ".join(alpha), " ".join(street), h_str, h_tok, digits, loc, kind,
            frozenset(alpha), post)


def _map_unique(func, values, workers):
    ctx = get_context("fork")
    chunks = [values[i:i + 20000] for i in range(0, len(values), 20000)]
    with ctx.Pool(workers) as pool:
        parts = pool.map(_apply_chunk, [(func, c) for c in chunks], chunksize=1)
    return [x for p in parts for x in p]


def _apply_chunk(args):
    func, chunk = args
    return [func(v) for v in chunk]


class Table:
    """Unique-value tables for names and addresses of S1s and records."""

    def __init__(self, s1_names, s1_addrs, rec_names, rec_addrs, workers):
        n_codes, n_uniq = pd.factorize(pd.Series(np.concatenate([np.asarray(s1_names, object),
                                                                 np.asarray(rec_names, object)])).fillna(""))
        a_codes, a_uniq = pd.factorize(pd.Series(np.concatenate([np.asarray(s1_addrs, object),
                                                                 np.asarray(rec_addrs, object)])).fillna(""))
        n1 = len(s1_names)
        self.s1_n, self.rec_n = n_codes[:n1].astype(np.int32), n_codes[n1:].astype(np.int32)
        self.s1_a, self.rec_a = a_codes[:n1].astype(np.int32), a_codes[n1:].astype(np.int32)
        log(f"unique names {len(n_uniq)}, addresses {len(a_uniq)}")
        nf = _map_unique(name_fields, list(n_uniq), workers)
        af = _map_unique(addr_fields, list(a_uniq), workers)
        log("per-unique fields done")
        self.N = list(zip(*nf))  # field-major lists
        self.A = list(zip(*af))
        self.n_core = np.asarray(self.N[1], dtype=object)
        self.a_kind = np.asarray(self.A[7], dtype=np.int8)
        counts = np.bincount(np.concatenate([self.s1_n, self.rec_n]), minlength=len(n_uniq))
        freq = {}
        for toks, c in zip(self.N[6], counts):
            for t in set(toks):
                freq[t] = freq.get(t, 0) + int(c)
        total = float(len(self.s1_n) + len(self.rec_n))
        self.logfreq = {t: float(np.log10(c / total)) for t, c in freq.items()}
        log(f"token frequencies: {len(freq)} tokens")


# ----------------------------------------------------------------------------- pair features

FEATURES = [
    "n_ratio", "n_partial", "n_tset", "n_tsort", "n_jw", "nf_ratio", "nf_tset", "n_core_eq", "n_full_eq",
    "n_jacc", "n_added", "n_removed", "n_added_decoy", "n_added_loc", "n_removed_decoy", "n_wordsub",
    "n_first_eq", "legal_rel", "copy_rel", "alias_any", "alias_best", "len_s1", "len_rec", "ntok_s1",
    "ntok_rec", "native_s1", "native_rec", "log_s1_mult",
    "kind_s1", "kind_rec", "a_tset", "a_tsort", "a_ratio", "a_alpha_tset", "st_ratio", "st_tset", "st_jw",
    "a_jacc", "loc_s1_cov", "loc_rec_cov", "h_rel", "h_delta", "dig_eq", "dig_jacc", "post_rel", "a_full_eq",
    "all_tset", "rec_s3", "n_added_chars", "n_removed_chars", "h1_in_rec_digits", "h2_in_s1_digits",
    "n_added_copy", "n_added_honor", "n_added_other", "add_freq_max", "add_freq_min", "rem_freq_max", "sub_sim",
    "add_has_digit",
]
CATEGORICAL = ["legal_rel", "copy_rel", "h_rel", "post_rel", "kind_s1", "kind_rec"]

_G = {}


def _legal_rel(a, b):
    if not a and not b:
        return 0
    if a == b:
        return 1
    if not a:
        return 2
    if not b:
        return 3
    return 4


def _h_rel(h1, t1, h2, t2):
    if not h1 and not h2:
        return 0, np.nan
    if not h1:
        return 1, np.nan
    if not h2:
        return 2, np.nan
    d = float(int(h2[:12]) - int(h1[:12]))
    if h1 == h2:
        if t1 != t2 and (t2.startswith(t1) or t1.startswith(t2)):
            return 6, d
        return 3, d
    if h2.lstrip("0") == h1.lstrip("0"):
        return 5, d
    if h1.startswith(h2) or (h2 and h1.endswith(h2)):
        return 4, d
    if d > 0 and int(d) in DECOY_DELTAS:
        return 7, d  # generator +delta set
    if d > 0:
        return 8, d  # other positive shift (incl. even shifts)
    return 9, d  # negative shift


def _chunk(args):
    lo, hi = args
    g = _G
    left, right = g["left"][lo:hi], g["right"][lo:hi]
    T = g["table"]
    ln, rn = T.s1_n[left], T.rec_n[right]
    la, ra = T.s1_a[left], T.rec_a[right]
    N, A = T.N, T.A
    n = hi - lo
    out = np.lib.format.open_memmap(g["path"], mode="r+")
    X = np.zeros((n, len(FEATURES)), dtype=np.float32)
    core1 = [N[1][i] for i in ln]
    core2 = [N[1][i] for i in rn]
    full1 = [N[0][i] for i in ln]
    full2 = [N[0][i] for i in rn]
    kw = dict(workers=1, dtype=np.float32)
    X[:, 0] = process.cpdist(core1, core2, scorer=fuzz.ratio, **kw)
    X[:, 1] = process.cpdist(core1, core2, scorer=fuzz.partial_ratio, **kw)
    X[:, 2] = process.cpdist(core1, core2, scorer=fuzz.token_set_ratio, **kw)
    X[:, 3] = process.cpdist(core1, core2, scorer=fuzz.token_sort_ratio, **kw)
    X[:, 4] = process.cpdist(core1, core2, scorer=JaroWinkler.normalized_similarity, **kw)
    X[:, 5] = process.cpdist(full1, full2, scorer=fuzz.ratio, **kw)
    X[:, 6] = process.cpdist(full1, full2, scorer=fuzz.token_set_ratio, **kw)
    af1 = [A[0][i] for i in la]
    af2 = [A[0][i] for i in ra]
    X[:, 30] = process.cpdist(af1, af2, scorer=fuzz.token_set_ratio, **kw)
    X[:, 31] = process.cpdist(af1, af2, scorer=fuzz.token_sort_ratio, **kw)
    X[:, 32] = process.cpdist(af1, af2, scorer=fuzz.ratio, **kw)
    al1 = [A[1][i] for i in la]
    al2 = [A[1][i] for i in ra]
    X[:, 33] = process.cpdist(al1, al2, scorer=fuzz.token_set_ratio, **kw)
    st1 = [A[2][i] for i in la]
    st2 = [A[2][i] for i in ra]
    X[:, 34] = process.cpdist(st1, st2, scorer=fuzz.ratio, **kw)
    X[:, 35] = process.cpdist(st1, st2, scorer=fuzz.token_set_ratio, **kw)
    X[:, 36] = process.cpdist(st1, st2, scorer=JaroWinkler.normalized_similarity, **kw)
    X[:, 46] = process.cpdist([c + " " + a for c, a in zip(core1, al1)], [c + " " + a for c, a in zip(core2, al2)],
                              scorer=fuzz.token_set_ratio, **kw)
    # alias: best ratio across available alias combinations, -1 if neither side has an alias
    alias1 = [N[2][i] for i in ln]
    alias2 = [N[2][i] for i in rn]
    has = np.fromiter(((a != "") or (b != "") for a, b in zip(alias1, alias2)), dtype=bool, count=n)
    X[:, 19] = has
    X[:, 20] = -1
    idx = np.flatnonzero(has)
    if idx.size:
        q1 = [alias1[i] or core1[i] for i in idx]
        q2 = [alias2[i] or core2[i] for i in idx]
        best = np.maximum(process.cpdist(q1, [core2[i] for i in idx], scorer=fuzz.ratio, **kw),
                          process.cpdist([core1[i] for i in idx], q2, scorer=fuzz.ratio, **kw))
        X[idx, 20] = best
    s1_mult = g["s1_mult"][left]
    rec_s3 = g["rec_s3"][right]
    geo = GEO
    decoy = DECOY
    lf = T.logfreq
    ratio = fuzz.ratio
    for j in range(n):
        i1, i2, a1, a2 = ln[j], rn[j], la[j], ra[j]
        t1, t2 = N[6][i1], N[6][i2]
        s1set, s2set = set(t1), set(t2)
        added = s2set - s1set
        removed = s1set - s2set
        inter = len(s1set & s2set)
        union = len(s1set | s2set)
        c1, c2 = core1[j], core2[j]
        row = X[j]
        row[7] = c1 == c2 and c1 != ""
        row[8] = full1[j] == full2[j] and full1[j] != ""
        row[9] = inter / union if union else 0.0
        row[10] = len(added)
        row[11] = len(removed)
        if added:
            row[12] = sum(1 for t in added if t in decoy)
            s1_alpha = A[8][a1]
            row[13] = sum(1 for t in added if t in geo or t in s1_alpha)
            row[48] = sum(len(t) for t in added)
            nd, nl = row[12], row[13]
            nc = sum(1 for t in added if t in COPYWORD)
            nh = sum(1 for t in added if t in HONOR)
            row[52] = nc
            row[53] = nh
            row[54] = max(len(added) - nd - nl - nc - nh, 0)
            fs = [lf.get(t, -8.0) for t in added]
            row[55] = max(fs)
            row[56] = min(fs)
            row[59] = any(ch.isdigit() for t in added for ch in t)
            if removed:
                row[58] = max(ratio(a, b) for a in added for b in removed)
            else:
                row[58] = -1
        else:
            row[55] = 1
            row[56] = 1
            row[58] = -1
        if removed:
            row[14] = sum(1 for t in removed if t in decoy)
            row[49] = sum(len(t) for t in removed)
            row[57] = max(lf.get(t, -8.0) for t in removed)
        else:
            row[57] = 1
        if len(t1) == len(t2) and len(t1) > 0:
            row[15] = sum(1 for x, y in zip(t1, t2) if x != y) == 1
        row[16] = bool(t1) and bool(t2) and t1[0] == t2[0]
        row[17] = _legal_rel(N[3][i1], N[3][i2])
        row[18] = _legal_rel(N[4][i1], N[4][i2])
        row[21] = len(c1)
        row[22] = len(c2)
        row[23] = len(t1)
        row[24] = len(t2)
        row[25] = N[5][i1]
        row[26] = N[5][i2]
        row[27] = np.log1p(s1_mult[j])
        k1, k2 = A[7][a1], A[7][a2]
        row[28] = k1
        row[29] = k2
        aset1, aset2 = A[8][a1], A[8][a2]
        d1, d2 = A[5][a1], A[5][a2]
        full_u = len(aset1 | aset2) + len(set(d1) | set(d2))
        row[37] = (len(aset1 & aset2) + len(set(d1) & set(d2))) / full_u if full_u else 0.0
        loc1, loc2 = A[6][a1], A[6][a2]
        li = len(loc1 & loc2)
        row[38] = li / len(loc1) if loc1 else -1
        row[39] = li / len(loc2) if loc2 else -1
        rel, delta = _h_rel(A[3][a1], A[4][a1], A[3][a2], A[4][a2])
        row[40] = rel
        row[41] = delta
        if d1 or d2:
            row[42] = sorted(d1) == sorted(d2)
            s1d, s2d = set(d1), set(d2)
            row[43] = len(s1d & s2d) / len(s1d | s2d)
        else:
            row[42] = -1
            row[43] = -1
        h1, h2 = A[3][a1], A[3][a2]
        row[50] = (h1 in d2) if h1 else -1
        row[51] = (h2 in d1) if h2 else -1
        p1, p2 = A[9][a1], A[9][a2]
        row[44] = 0 if not p1 and not p2 else 1 if not p2 else 2 if not p1 else 3 if p1 & p2 else 4
        row[45] = A[0][a1] == A[0][a2] and A[0][a1] != ""
        row[47] = rec_s3[j]
    out[lo:hi] = X
    out.flush()
    return hi - lo


def pair_features(table, left, right, s1_mult, rec_s3, path, workers, chunk=250_000, resume=False):
    """Write the float32 feature matrix (len(left), len(FEATURES)) to ``path`` (.npy memmap)."""
    import gc
    jobs = [(lo, min(lo + chunk, len(left))) for lo in range(0, len(left), chunk)]
    if resume and os.path.exists(path):
        out = np.load(path, mmap_mode="r")
        assert out.shape == (len(left), len(FEATURES))
        col = FEATURES.index("all_tset")
        jobs = [(lo, hi) for lo, hi in jobs if not (out[lo:hi, col] > 0).any() or not (out[hi - 1000:hi, col] > 0).any()]
        log(f"resume: {len(jobs)} chunks to compute")
        del out
    else:
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(left), len(FEATURES)))
        del out
    _G.update(table=table, left=left, right=right, s1_mult=s1_mult, rec_s3=rec_s3, path=str(path))
    done = 0
    gc.collect()
    gc.freeze()  # keep forked workers from copying every object page via GC
    with get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        for k in pool.imap_unordered(_chunk, jobs):
            done += k
            if done // (chunk * 40) != (done - k) // (chunk * 40):
                log(f"pair features {done}/{len(left)}")
    log(f"pair features done {len(left)}")
    return np.load(path, mmap_mode="r")


def s1_multiplicity(table, countries):
    core = table.n_core[table.s1_n]
    frame = pd.DataFrame({"c": np.asarray(countries, dtype=object), "n": core})
    return frame.groupby(["c", "n"], sort=False)["n"].transform("size").to_numpy(dtype=np.float32)


def source_s3(entity_ids):
    return pd.Series(entity_ids).str.startswith("S3").to_numpy(dtype=np.float32)
