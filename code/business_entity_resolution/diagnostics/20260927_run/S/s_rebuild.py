"""Agent S task 6 (CPU, no labels): rebuild French rows with C2-FR scores under the unchanged #5 rule
(0.9*C2 + 0.1*tree >= 0.10, empty-address unique exact-name rule, one owner, cap 11) and compare with
the S1 file's French rows (F's state_old.npz = the scored S1 file, parity-checked by Agent F).

Usage: s_rebuild.py SCORES[,SCORES...] [TAG]
  SCORES: score_external output dir over the FULL test pair index (French rows scored), or "c2_40k"
          (stored C2 step-40k scores); several comma-separated sources are averaged (e.g. DIR_10k,c2_40k).
Writes $S_OUT (default /root/S)/{TAG}french_rows.tsv, {TAG}diagnostics.json. Never touches /workspace.
"""
import json, re, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "F")); sys.path.insert(0, str(HERE.parent.parent / "20260927_reset"))
from fcommon import OUT as FOUT, load, plain_of
from ber.pair_tree_c2 import write_tsv, sha256, log
import sibling_check as sib

SC = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else ""
import os
S = Path(os.environ.get("S_OUT", "/root/S")); S.mkdir(parents=True, exist_ok=True)
S1_SHA = "559f37b9b66f4e18fc28c0b25ae60d08543c386e0bdb32c619990b4c9cbd1a78"
t0 = time.time()
d = load()
OLD_PT = d["pt"]
left, right, base, c2, france = d["left"], d["right"], d["base"], d["c2"], d["france"]
s1_ids, rec_ids, country = d["s1_ids"], d["rec_ids"], d["country"]
n_s1, n_rec = len(s1_ids), len(rec_ids)
# old French rows = the scored S1 file (#5 rule on stored C2 40k); parity is checked on the written file below
pp_old, pl_old = plain_of(d, c2)
accept, prob = d["accept"], d["prob"]
cand = d["cache"]["cand_rows"].astype(np.int64)
rows = np.flatnonzero(france & (base >= .01))  # the French rescoring set (pre-count: 3,121,523 rows)
parts = []
for item in str(sys.argv[1]).split(","):
    if item == "c2_40k":
        v = c2[rows]; assert not np.isnan(v).any(); parts.append(v.astype(np.float32)); continue
    p = Path(item)
    if not (p / "complete.json").exists():
        raise SystemExit(f"scores incomplete: {p}")
    v = np.load(p / "scores.npy", mmap_mode="r"); m = np.load(p / "scored_mask.npy", mmap_mode="r")
    assert len(v) == d["n_pairs"], f"expected full test-index scores: {p}"
    v, m = np.asarray(v[cand[rows]]), np.asarray(m[cand[rows]])
    assert m.all(), f"scores miss French rescoring rows: {p}"
    parts.append(v.astype(np.float32))
sc = np.mean(parts, axis=0).astype(np.float32)
c2n = c2.copy(); c2n[rows] = sc
if os.environ.get("S_THRESHOLD"):
    d["pt"] = float(os.environ["S_THRESHOLD"])  # variant only: the old rows above used the unchanged threshold
pp_new, pl_new = plain_of(d, c2n)
if os.environ.get("S_THRESHOLD"):  # the override is a French-only variant: non-French rows keep the unchanged rule
    pl_new = np.where(france, pl_new, pl_old); pp_new = np.where(france, pp_new, pp_old)
R = {"scores": str(SC), "rows_rescored": int(len(rows)), "rule": {"weight": d["pw"], "threshold": d["pt"],
     "decode": "model_then_name_only (empty-address unique exact-name), one owner, cap 11"}}
R["non_french_accept_changed"] = int(((pl_new != pl_old) & ~france).sum())
assert R["non_french_accept_changed"] == 0
old_r = np.nan_to_num(c2[rows], nan=0.0)
R["rescored_rows_c2"] = {"mean_c2_40k": float(old_r.mean()), "mean_c2fr": float(sc.mean()),
                         "share_ge_0.5_c2_40k": float((old_r >= .5).mean()), "share_ge_0.5_c2fr": float((sc >= .5).mean()),
                         "up_by_0.5": int((sc - old_r > .5).sum()), "down_by_0.5": int((old_r - sc > .5).sum()),
                         "pearson": float(np.corrcoef(old_r, sc)[0, 1])}

ref_df = d["ref_table"].to_pandas()
fr = sib.frame(ref_df)
gid, gsz = sib.groups(fr, "loose")
fr_s1 = country == "France"
gid_fr = np.where(fr_s1 & (gsz >= 2), gid, -1)
n_fr = int(fr_s1.sum())

def owners(acc):
    o = np.full(n_rec, -1, np.int64); r = np.flatnonzero(acc & france); o[right[r]] = left[r]; return o

def owner_changes(pl_a, pl_b):
    oa, ob = owners(pl_a), owners(pl_b)
    both = (oa >= 0) & (ob >= 0) & (oa != ob)
    same_group = both & (gid_fr[np.maximum(oa, 0)] >= 0) & (gid_fr[np.maximum(oa, 0)] == gid_fr[np.maximum(ob, 0)])
    return {"records_owner_changed": int(both.sum()), "owner_changed_within_sibling_group": int(same_group.sum()),
            "links_added_or_removed_touching_sibling_s1": int((((pl_a != pl_b) & france) & (gid_fr[left] >= 0)).sum())}

DECOY = {"groupe", "holding", "developpement", "participations", "distribution", "international", "france"}
DELTAS = {1, 3, 5, 7, 9, 11, 13, 21}
ref_name = ref_df.business_name.fillna("").values; ref_addr = ref_df.business_address.fillna("").values
pool_na = d["pool_table"].select(["business_name", "business_address"]).to_pandas().fillna("")
pool_name, pool_addr = pool_na.business_name.values, pool_na.business_address.values
def house(a):
    m = re.search(r"\d+", a); return int(m.group()) if m else None
EVEN = {2, 4, 6, 8, 10, 12}
def falsifier(sel_rows):
    shift = decoy = either = even = 0
    for r in sel_rows:
        i, j = left[r], right[r]; hi, hj = house(ref_addr[i]), house(pool_addr[j])
        s = hi is not None and hj is not None and (hj - hi) in DELTAS
        e = hi is not None and hj is not None and (hj - hi) in EVEN
        w = bool((set(sib.norm(pool_name[j])) - set(sib.norm(ref_name[i]))) & DECOY)
        shift += s; decoy += w; either += (s or w); even += e
    n = max(len(sel_rows), 1)
    return {"n": int(len(sel_rows)), "shift_rate": shift / n, "decoy_word_rate": decoy / n, "either_rate": either / n,
            "even_shift_2_12_rate": even / n}
rng = np.random.default_rng(0)
def sample(r, k=20000):
    return r if len(r) <= k else np.sort(rng.choice(r, k, replace=False))
def signatures(c2v, pp, pl):
    fa = pl & france
    return {"uncertain_band_per_s1_blend_0.05_0.5": float((france & (pp >= .05) & (pp < .5)).sum() / n_fr),
            "tree_high_c2_low_per_s1": float((france & (base >= .9) & (np.nan_to_num(c2v, nan=0) < .05)).sum() / n_fr),
            "mean_list_len": float(fa.sum() / n_fr), "links": int(fa.sum()),
            "empty_french_s1": int(n_fr - len(np.unique(left[fa])))}
G1 = Path(os.environ.get("S_G1", "/root/S/g1"))
have_probe = (G1 / "rows.npy").exists()
if have_probe:
    g1rows = np.load(G1 / "rows.npy"); g1set = np.load(G1 / "set.npy")
add = np.flatnonzero(pl_new & ~pl_old & france); rem = np.flatnonzero(pl_old & ~pl_new & france)
R["signatures_old_s1_file"] = signatures(c2, pp_old, pl_old)
R["signatures_new"] = signatures(c2n, pp_new, pl_new)
R["links_added"] = int(len(add)); R["links_removed"] = int(len(rem))
R["s1_rows_changed"] = int(len(np.unique(left[np.concatenate([add, rem])])))
R["falsifier_added"] = falsifier(sample(add)); R["falsifier_removed"] = falsifier(sample(rem))
R["falsifier_old_confident_ge_0.7"] = falsifier(sample(np.flatnonzero(pl_old & france & (pp_old >= .7))))
R["falsifier_kept"] = falsifier(sample(np.flatnonzero(pl_old & pl_new & france)))
R["sibling"] = owner_changes(pl_old, pl_new)
R["probe_anchor"] = "unavailable (probe files are on the hung /workspace volume)" if not have_probe else {nm: {"n": int((g1set == k).sum()), "mean_c2_40k": float(np.nanmean(c2[g1rows[g1set == k]])),
                          "mean_c2fr": float(np.nanmean(c2n[g1rows[g1set == k]])),
                          "share_ge_0.5_c2_40k": float((np.nan_to_num(c2[g1rows[g1set == k]]) >= .5).mean()),
                          "share_ge_0.5_c2fr": float((c2n[g1rows[g1set == k]] >= .5).mean()),
                          "still_accepted_new": float(pl_new[g1rows[g1set == k]].mean()),
                          "rescored_share": float(np.isin(g1rows[g1set == k], rows).mean())}
                     for nm, k in (("probe_A_T2r_64-69pct_true", 0), ("probe_Cprime_92-96pct_true", 1))}
# files: French rows only (matching_results format)
final = np.where(france, pl_new, accept); fprob = np.where(france, pp_new, prob)
full = S / f"{TAG}full_tmp/matching_results.tsv"
write_tsv(full, s1_ids, rec_ids, left, right, final, fprob)
lines = full.read_text(encoding="utf-8").split("\n")
assert lines[-1] == "" and len(lines) == n_s1 + 2
fr_lines = [lines[0]] + [lines[i + 1] for i in np.flatnonzero(fr_s1)]
out = S / f"{TAG}french_rows.tsv"
with open(out, "w", encoding="utf-8", newline="\n") as h:
    h.write("\n".join(fr_lines) + "\n")
recs = [r for l in fr_lines[1:] for r in (l.split("\t")[1].split(",") if l.split("\t")[1] else [])]
lens = [len(l.split("\t")[1].split(",")) if l.split("\t")[1] else 0 for l in fr_lines[1:]]
R["file"] = {"path": str(out), "sha256": sha256(out), "french_rows": len(fr_lines) - 1, "max_list": max(lens),
             "records_unique": len(set(recs)) == len(recs), "only_s2_s3": all(r.startswith(("S2-", "S3-")) for r in recs),
             "links": len(recs), "full_file_tmp": str(full), "full_file_sha256": sha256(full)}
R["file"]["full_file_equals_s1_file"] = R["file"]["full_file_sha256"] == S1_SHA
R["seconds"] = round(time.time() - t0, 1); R["labels_read"] = False
(S / f"{TAG}diagnostics.json").write_text(json.dumps(R, indent=1, default=float) + "\n")
print(json.dumps(R, indent=1, default=float))
