"""Agent F tasks 4, 5, 7: rebuild French rows with F1 C2 scores (#5 rule unchanged), label-free diagnostics,
assemble upload #1 (US/India rows from the Qwen-tree file, French rows new) and the fallback. No labels read.
Usage: rebuild.py [--tiebreak]"""
import json, re, sys, time, shutil
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "20260927_reset"))
from fcommon import OUT, S1_SHA, S1_FILE, QWEN_FILE, T, load, plain_of, key
from ber.pair_tree_c2 import write_tsv, validate_tsv, sha256, log, c2_blend, decode
import sibling_check as sib

TIEBREAK = "--tiebreak" in sys.argv
import os
G2DIR = os.environ.get("F_G2", "scores_step040000")
TAG = os.environ.get("F_TAG", "")
t0 = time.time()
d = load()
left, right, base, c2, france = d["left"], d["right"], d["base"], d["c2"], d["france"]
s1_ids, rec_ids, country = d["s1_ids"], d["rec_ids"], d["country"]
n_s1, n_rec = len(s1_ids), len(rec_ids)
st = np.load(OUT / "state_old.npz")
pp_old, pl_old, accept, prob = st["plain_prob"], st["plain"], st["accept"], st["prob"]
R = {}

# ---------- new C2 on changed rows
g2 = OUT / "g2"; rows = np.load(g2 / "rows.npy"); n_ch = int(np.load(g2 / "n_changed.npy")[0])
sc = np.load(g2 / G2DIR / "scores.npy")
assert (g2 / G2DIR / "complete.json").exists() and len(sc) == len(rows)
ctrl_new, ctrl_old = sc[n_ch:], c2[rows[n_ch:]]
blend_new_c = 0.9 * ctrl_new + 0.1 * base[rows[n_ch:]]; blend_old_c = 0.9 * ctrl_old + 0.1 * base[rows[n_ch:]]
R["control_unchanged_rows"] = {"n": int(len(ctrl_new)), "max_abs_diff": float(np.abs(ctrl_new - ctrl_old).max()),
                               "mean_abs_diff": float(np.abs(ctrl_new - ctrl_old).mean()),
                               "threshold_0.1_crossings": int(((blend_new_c >= 0.1) != (blend_old_c >= 0.1)).sum())}
log(f"controls {R['control_unchanged_rows']}")
c2n = c2.copy(); ch_rows = rows[:n_ch]; c2n[ch_rows] = sc[:n_ch]
if os.environ.get("F_GUARD") == "1":
    # F1-guarded (post-hoc, label-free): keep the OLD C2 score on changed rows that carry the F3 decoy house-number
    # signature (record first number - S1 first number in {1,3,5,7,9,11,13,21}).
    _ra = d["ref_table"].column("business_address").to_pylist(); _pa = d["pool_table"].column("business_address").to_pylist()
    def _h(a):
        m = re.search(r"\d+", a or ""); return int(m.group()) if m else None
    _sh = np.array([(lambda hi, hj: hi is not None and hj is not None and (hj - hi) in {1, 3, 5, 7, 9, 11, 13, 21})(_h(_ra[i]), _h(_pa[j]))
                    for i, j in zip(left[ch_rows], right[ch_rows])], bool)
    c2n[ch_rows[_sh]] = c2[ch_rows[_sh]]
    R["guard_reverted_rows"] = int(_sh.sum())
    log(f"guard reverted {int(_sh.sum())} of {n_ch} changed rows")
if os.environ.get("F_GUARD") == "2":
    # F1-strict (post-hoc, label-free): new C2 only on the targeted class -- normalized names equal as token keys
    # (case/accent/punctuation-insensitive) AND same house number (first 1-4 digit number) on both sides; else old C2.
    from fcommon import TOK
    _rn = pq.read_table(OUT / "inputs/reference.parquet", columns=["business_name"]).column(0).to_pylist()
    _pn = pq.read_table(OUT / "inputs/pool.parquet", columns=["business_name"]).column(0).to_pylist()
    _ra = d["ref_table"].column("business_address").to_pylist(); _pa = d["pool_table"].column("business_address").to_pylist()
    def _h(a):
        m = re.search(r"(?<!\d)\d{1,4}(?!\d)", a or ""); return int(m.group()) if m else None
    def _k(s):
        return tuple(key(t) for t in TOK.findall(s or ""))
    _ok = np.array([_k(_rn[i]) == _k(_pn[j]) and _h(_ra[i]) is not None and _h(_ra[i]) == _h(_pa[j])
                    for i, j in zip(left[ch_rows], right[ch_rows])], bool)
    c2n[ch_rows[~_ok]] = c2[ch_rows[~_ok]]
    R["strict_kept_new_rows"] = int(_ok.sum())
    log(f"strict: new C2 kept on {int(_ok.sum())} of {n_ch} changed rows")
MODE = os.environ.get("F_MODE")
if MODE:
    # Follow-up: C2 step_010000 on all gated French candidate rows (original inputs). 10k | mean | strict10k.
    g4 = OUT / "g4"; r4 = np.load(g4 / "rows.npy")
    assert (g4 / "scores_step010000/complete.json").exists()
    s10 = np.load(g4 / "scores_step010000/scores.npy"); m4 = np.load(g4 / "scores_step010000/scored_mask.npy")
    r4, s10 = r4[m4], s10[m4]
    assert np.isfinite(c2[r4]).all()
    c2n = c2.copy()
    if MODE == "10k":
        c2n[r4] = s10
    elif MODE == "mean":
        c2n[r4] = (c2[r4] + s10) / 2
    elif MODE == "strict10k":
        from fcommon import TOK
        _rn = pq.read_table(OUT / "inputs/reference.parquet", columns=["business_name"]).column(0).to_pylist()
        _pn = pq.read_table(OUT / "inputs/pool.parquet", columns=["business_name"]).column(0).to_pylist()
        _ra = d["ref_table"].column("business_address").to_pylist(); _pa = d["pool_table"].column("business_address").to_pylist()
        def _h(a):
            m = re.search(r"(?<!\d)\d{1,4}(?!\d)", a or ""); return int(m.group()) if m else None
        def _k(s):
            return tuple(key(t) for t in TOK.findall(s or ""))
        _ok = np.array([_k(_rn[i]) == _k(_pn[j]) and _h(_ra[i]) is not None and _h(_ra[i]) == _h(_pa[j])
                        for i, j in zip(left[r4], right[r4])], bool)
        c2n[r4[_ok]] = np.maximum(c2[r4[_ok]], s10[_ok])
        R["strict10k_rows"] = int(_ok.sum())
    else:
        raise SystemExit(f"unknown F_MODE {MODE}")
    R["mode"] = MODE; R["g4_scored_rows"] = int(len(r4))
    R["10k_vs_40k_on_french_gated"] = {"mean_10k": float(s10.mean()), "mean_40k": float(c2[r4].mean()),
                                       "share_ge_0.5_10k": float((s10 >= .5).mean()), "share_ge_0.5_40k": float((c2[r4] >= .5).mean())}
    ch_rows = r4
    log(f"mode {MODE}: {R.get('strict10k_rows', len(r4))} rows use 10k")
old_ch, new_ch = c2[rows[:n_ch]], sc[:n_ch]
R["changed_rows_c2"] = {"n": n_ch, "mean_old": float(np.nanmean(old_ch)), "mean_new": float(new_ch.mean()),
                        "share_ge_0.5_old": float((old_ch >= .5).mean()), "share_ge_0.5_new": float((new_ch >= .5).mean()),
                        "up_by_0.5": int((new_ch - old_ch > .5).sum()), "down_by_0.5": int((old_ch - new_ch > .5).sum())}
pp_new, pl_new = plain_of(d, c2n)
R["non_french_accept_changed"] = int(((pl_new != pl_old) & ~france).sum())
assert R["non_french_accept_changed"] == 0

# ---------- sibling groups (loose key, France)
ref_df = d["ref_table"].to_pandas()
fr = sib.frame(ref_df)
gid, gsz = sib.groups(fr, "loose")
fr_s1 = country == "France"
gid_fr = np.where(fr_s1 & (gsz >= 2), gid, -1)
R["french_sibling_groups"] = int(len(np.unique(gid_fr[gid_fr >= 0]))); R["french_s1_in_sibling_groups"] = int((gid_fr >= 0).sum())

def owners(acc):
    o = np.full(n_rec, -1, np.int64); r = np.flatnonzero(acc & france); o[right[r]] = left[r]; return o

def tiebreak(pp, pl):
    """When two sibling S1s (same loose group) both have a candidate row for a record with blends >= threshold
    within 0.05 of each other, give the record to the S1 whose raw legal-form set equals the record's."""
    legal_s1 = fr["legal"].values
    pool_df = d["pool_table"].select(["business_name"]).to_pandas()
    fr_rows = np.flatnonzero(france & (pp >= d["pt"]) & (gid_fr[left] >= 0))
    df = pd.DataFrame({"row": fr_rows, "rec": right[fr_rows], "s1": left[fr_rows], "g": gid_fr[left[fr_rows]], "p": pp[fr_rows]})
    df = df[df.groupby(["rec", "g"]).s1.transform("size") >= 2]
    pp2 = pp.copy(); moved = 0
    for rec, grp in df.groupby("rec"):
        top = grp.p.max(); close = grp[grp.p >= top - 0.05]
        if len(close) < 2:
            continue
        t = sib.norm(pool_df.business_name.iat[rec])
        rl = " ".join(sorted(set(sib.SYN.get(x, x) for x in t if x in sib.LEGAL)))
        match = close[[legal_s1[s] == rl for s in close.s1]]
        if len(match) == 1:
            w = int(match.row.iat[0])
            if pp2[w] < top:
                moved += 1
            pp2[w] = top + 1e-4
    pl2 = decode(pp2, d["pt"], "model_then_name_only", left, right, d["cache"]["rec_empty"], d["cache"]["name_exact"], d["mult"])
    return pp2, pl2, moved

def owner_changes(pl_a, pl_b):
    oa, ob = owners(pl_a), owners(pl_b)
    both = (oa >= 0) & (ob >= 0) & (oa != ob)
    same_group = both & (gid_fr[np.maximum(oa, 0)] >= 0) & (gid_fr[np.maximum(oa, 0)] == gid_fr[np.maximum(ob, 0)])
    return {"records_owner_changed": int(both.sum()), "owner_changed_within_sibling_group": int(same_group.sum()),
            "links_added_or_removed_touching_sibling_s1": int((((pl_a != pl_b) & france) & (gid_fr[left] >= 0)).sum())}

# ---------- diagnostics
DECOY = {"groupe", "holding", "developpement", "participations", "distribution", "international", "france"}
DELTAS = {1, 3, 5, 7, 9, 11, 13, 21}
ref_name = ref_df.business_name.fillna("").values; ref_addr = ref_df.business_address.fillna("").values
pool_na = d["pool_table"].select(["business_name", "business_address"]).to_pandas().fillna("")
pool_name, pool_addr = pool_na.business_name.values, pool_na.business_address.values
def house(a):
    m = re.search(r"\d+", a); return int(m.group()) if m else None
def falsifier(sel_rows):
    hits_shift = hits_decoy = 0
    for r in sel_rows:
        i, j = left[r], right[r]
        hi, hj = house(ref_addr[i]), house(pool_addr[j])
        if hi is not None and hj is not None and (hj - hi) in DELTAS:
            hits_shift += 1
        ti, tj = set(sib.norm(ref_name[i])), set(sib.norm(pool_name[j]))
        if (tj - ti) & DECOY:
            hits_decoy += 1
    n = max(len(sel_rows), 1)
    return {"n": int(len(sel_rows)), "shift_rate": hits_shift / n, "decoy_word_rate": hits_decoy / n,
            "either_rate": None if not len(sel_rows) else None}
def falsifier_full(sel_rows):
    out = falsifier(sel_rows); either = 0
    for r in sel_rows:
        i, j = left[r], right[r]; hi, hj = house(ref_addr[i]), house(pool_addr[j])
        s = hi is not None and hj is not None and (hj - hi) in DELTAS
        if s or ((set(sib.norm(pool_name[j])) - set(sib.norm(ref_name[i]))) & DECOY):
            either += 1
    out["either_rate"] = either / max(len(sel_rows), 1); return out

n_fr = int(fr_s1.sum())
def signatures(c2v, pp, pl):
    fa = pl & france
    return {"uncertain_band_per_s1_blend_0.05_0.5": float((france & (pp >= .05) & (pp < .5)).sum() / n_fr),
            "tree_high_c2_low_per_s1": float((france & (base >= .9) & (np.nan_to_num(c2v, nan=0) < .05)).sum() / n_fr),
            "mean_list_len": float(fa.sum() / n_fr), "links": int(fa.sum()),
            "empty_french_s1": int(n_fr - len(np.unique(left[fa])))}

rng = np.random.default_rng(0)
def sample(r, k=20000):
    return r if len(r) <= k else np.sort(rng.choice(r, k, replace=False))
g1rows = np.load(OUT / "g1/rows.npy"); g1set = np.load(OUT / "g1/set.npy")

def diag(name, c2v, pp, pl):
    add = np.flatnonzero(pl & ~pl_old & france); rem = np.flatnonzero(pl_old & ~pl & france)
    conf_old = np.flatnonzero(pl_old & france & (pp_old >= .7))
    res = {"signatures_new": signatures(c2v, pp, pl), "links_added": int(len(add)), "links_removed": int(len(rem)),
           "s1_rows_changed": int(len(np.unique(left[np.concatenate([add, rem])]))),
           "added_on_changed_text_rows": int(np.isin(add, ch_rows).sum()),
           "falsifier_added": falsifier_full(sample(add)), "falsifier_removed": falsifier_full(sample(rem)),
           "falsifier_old_confident_ge_0.7": falsifier_full(sample(conf_old)),
           "sibling": owner_changes(pl_old, pl),
           "probe_anchor": {nm: {"mean_c2_old": float(np.nanmean(c2[g1rows[g1set == k]])), "mean_c2_new": float(np.nanmean(c2v[g1rows[g1set == k]])),
                                 "share_ge_0.5_old": float((c2[g1rows[g1set == k]] >= .5).mean()), "share_ge_0.5_new": float((c2v[g1rows[g1set == k]] >= .5).mean()),
                                 "still_accepted_new": float(pl[g1rows[g1set == k]].mean())}
                            for nm, k in (("probe_A_T2r", 0), ("probe_Cprime", 1))}}
    return res

R["signatures_old"] = signatures(c2, pp_old, pl_old)
R["f1"] = diag("f1", c2n, pp_new, pl_new)
log(json.dumps(R["f1"]["sibling"]))
HOLD = R["f1"]["sibling"]["owner_changed_within_sibling_group"] > 2000
R["hold_triggered"] = bool(HOLD)
variants = {"f1": (pp_new, pl_new)}
if HOLD or TIEBREAK:
    pp_t, pl_t, moved = tiebreak(pp_new, pl_new)
    R["f1_tiebreak"] = diag("f1_tiebreak", c2n, pp_t, pl_t); R["f1_tiebreak"]["records_moved_by_tiebreak"] = moved
    variants["f1_tiebreak"] = (pp_t, pl_t)

# ---------- files
la = np.asarray(np.load(T / "index/left.npy", mmap_mode="r")).astype(np.int64); ra = np.asarray(np.load(T / "index/right.npy", mmap_mode="r")).astype(np.int64)
pair_keys = np.sort(la * n_rec + ra); del la, ra
rec_index = {v: i for i, v in enumerate(rec_ids)}
s1_lines = S1_FILE.read_text(encoding="utf-8").split("\n"); q_lines = QWEN_FILE.read_text(encoding="utf-8").split("\n")
assert s1_lines[-1] == "" and q_lines[-1] == "" and len(s1_lines) == len(q_lines) == n_s1 + 2
fr_line = np.array([False] + list(fr_s1) + [False])
assert all(a.split("\t")[0] == b.split("\t")[0] for a, b in zip(s1_lines, q_lines)), "S1 order differs between S1 and Qwen files"
R["qwen_file"] = {"sha256": sha256(QWEN_FILE),
                  "french_rows_equal_s1_file": bool(all(s1_lines[i] == q_lines[i] for i in np.flatnonzero(fr_line))),
                  "us_india_rows_differing_from_s1_file": int(sum(s1_lines[i] != q_lines[i] for i in range(1, n_s1 + 1) if not fr_line[i]))}
log(f"qwen file {R['qwen_file']}")

def extra_checks(path):
    raw = path.read_bytes(); lines = raw.decode("utf-8").split("\n")
    ids = [l.split("\t")[0] for l in lines[1:-1]]
    recs = [r for l in lines[1:-1] for r in (l.split("\t")[1].split(",") if l.split("\t")[1] else [])]
    return {"header_ok": lines[0] == "source1_entity_id\tmatched_entity_ids", "data_rows": len(lines) - 2,
            "ends_with_lf": raw.endswith(b"\n"), "no_cr": b"\r" not in raw, "unique_s1": len(set(ids)) == len(ids),
            "order_equals_s1_file": ids == [l.split("\t")[0] for l in s1_lines[1:-1]],
            "only_s2_s3_ids": all(r.startswith(("S2-", "S3-")) for r in recs), "records_unique": len(set(recs)) == len(recs)}

for vname, (pp, pl) in variants.items():
    final = np.where(france, pl, accept); fprob = np.where(france, pp, prob)
    full = OUT / (TAG + vname) / "s1based/matching_results.tsv"
    write_tsv(full, s1_ids, rec_ids, left, right, final, fprob)
    new_lines = full.read_text(encoding="utf-8").split("\n")
    assert all(new_lines[i] == s1_lines[i] for i in range(len(s1_lines)) if not fr_line[i]), "non-French rows differ from S1 file"
    out_lines = [new_lines[i] if fr_line[i] else q_lines[i] for i in range(len(s1_lines))]
    up = OUT / (TAG + ("upload1" if vname == "f1" else "upload1_" + vname)) / "matching_results.tsv"
    up.parent.mkdir(parents=True, exist_ok=True)
    with open(up, "w", encoding="utf-8", newline="\n") as h:
        h.write("\n".join(out_lines))
    chk = validate_tsv(up, s1_ids, rec_index, pair_keys, n_rec); chk.update(extra_checks(up))
    R[f"file_{vname}"] = {"path": str(up), "sha256": sha256(up), "validation": chk,
                          "french_rows_changed_vs_s1_file": int(sum(out_lines[i] != s1_lines[i] for i in np.flatnonzero(fr_line))),
                          "us_india_rows_equal_qwen_file": bool(all(out_lines[i] == q_lines[i] for i in range(len(s1_lines)) if not fr_line[i]))}
    log(json.dumps(R[f"file_{vname}"]))
if not MODE:  # the fallback copy and the F1 score dump are not needed for the 10k variants
    fb = OUT / (TAG + "upload1_fallback") / "matching_results.tsv"; fb.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(QWEN_FILE, fb)
    chk = validate_tsv(fb, s1_ids, rec_index, pair_keys, n_rec); chk.update(extra_checks(fb))
    R["file_fallback"] = {"path": str(fb), "sha256": sha256(fb), "validation": chk}
    np.save(OUT / (TAG + "c2_new_cand_rows_french_changed.npy"), sc[:n_ch])
R["seconds"] = round(time.time() - t0, 1); R["labels_read"] = False
(OUT / (TAG + "rebuild.json")).write_text(json.dumps(R, indent=1, default=float) + "\n")
print(json.dumps(R, indent=1, default=float))
