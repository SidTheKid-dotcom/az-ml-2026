"""U4 output (27 Sep, Agent U): US/India matching rows from the mean of the 5 fold models + label-free checks.

    python u4_output.py VARIANT [THRESHOLD]

Decode: US/India test candidate rows, mean fold probability >= threshold (default: the variant's full-OOF
threshold), one owner, cap 11. Writes u4/<variant>/us_india_rows.tsv (header + US/India S1 rows in test
reference order) and output_report.json: validation, per-country link stats, OOF vs test-average
distributions, per-fold-model acceptance counts (averaging effect). No labels read.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.pair_tree_arena import decide

ROOT = Path("/home/ubuntu/run27/U")
PREP = ROOT / "prep"
variant = sys.argv[1]
TAG, TEST, BREST = os.environ.get("U_TAG", ""), os.environ.get("U_TEST", "test"), os.environ.get("U_BREST", "brest")
work = ROOT / "u4" / (((TAG + "__") if TAG else "") + variant.replace("+", "_"))
rep = json.loads((work / "report.json").read_text())
thr = float(sys.argv[2]) if len(sys.argv) > 2 else float(rep["thresholds"]["full_oof_threshold"])
HEADER = "source1_entity_id\tmatched_entity_ids"

tm = json.loads((PREP / TEST / "meta.json").read_text())
countries = np.asarray(tm["countries"])
s1_country = countries[np.load(PREP / TEST / "country_code.npy")]
left, right = np.load(PREP / TEST / "left.npy"), np.load(PREP / TEST / "right.npy")
rows = np.load(work / "test_rows.npy")
prob = np.load(work / "test_prob_mean.npy")
folds = np.load(work / "test_pred_folds.npy").astype(np.float32)
L, Rr = left[rows], right[rows]
acc = decide(prob, L, Rr, thr, None)
s1_ids = np.asarray(pq.read_table(ROOT / "data/test/dense_test/reference.parquet", columns=["entity_id"])
                    .column("entity_id").to_pylist(), dtype=object)
rec_ids = np.asarray(pq.read_table(ROOT / "data/test/dense_test/pool.parquet", columns=["entity_id"])
                     .column("entity_id").to_pylist(), dtype=object)
if (PREP / TEST / "extra_rec_ids.json").exists():
    rec_ids = np.concatenate([rec_ids, np.asarray(json.loads((PREP / TEST / "extra_rec_ids.json").read_text()), dtype=object)])
ui = np.flatnonzero(np.isin(s1_country, ["US", "India"]))
sel = np.flatnonzero(acc)
sel = sel[np.lexsort((-prob[sel], L[sel]))]
starts = np.searchsorted(L[sel], ui, side="left")
stops = np.searchsorted(L[sel], ui, side="right")
path = work / "us_india_rows.tsv"
with open(path, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(HEADER + "\n")
    for i, s in enumerate(ui):
        fh.write(f"{s1_ids[s]}\t{','.join(rec_ids[Rr[sel[starts[i]:stops[i]]]])}\n")
lengths = stops - starts
# validation
recs = Rr[acc]
problems = []
if len(recs) and np.bincount(recs).max() > 1:
    problems.append("contested record")
if lengths.max() > 11:
    problems.append("list > 11")
pairs = L[acc] * (len(rec_ids) + 1) + Rr[acc]
if len(np.unique(pairs)) != len(pairs):
    problems.append("duplicate pair")
raw = path.read_bytes()
if b"\r" in raw:
    problems.append("CR")
n_lines = raw.count(b"\n")


def stats(p, lft, cc, t):
    out = {}
    for c in ("US", "India"):
        m = cc[lft] == c
        out[c] = {"rows": int(m.sum()), "mean_p": round(float(p[m].mean()), 5),
                  "share_ge_t": round(float((p[m] >= t).mean()), 5),
                  "band_0.3_0.9": round(float(((p[m] >= 0.3) & (p[m] < 0.9)).mean()), 5)}
    return out


bm = json.loads((PREP / BREST / "meta.json").read_text())
b_country = np.asarray(bm["countries"])[np.load(PREP / BREST / "country_code.npy")]
b_left = np.load(PREP / BREST / "left.npy")
oof = np.load(work / "oof.npy")
per_fold_links = {}
for k in range(folds.shape[0]):
    a = decide(folds[k], L, Rr, thr, None)
    per_fold_links[str(k)] = {c: int((a & (s1_country[L] == c)).sum()) for c in ("US", "India")}
by_country = {c: {"s1": int((s1_country[ui] == c).sum()),
                  "links": int(lengths[s1_country[ui] == c].sum()),
                  "mean_list": round(float(lengths[s1_country[ui] == c].mean()), 4),
                  "empty_rows": round(float((lengths[s1_country[ui] == c] == 0).mean()), 5)} for c in ("US", "India")}
b_acc = decide(oof, b_left, np.load(PREP / BREST / "right.npy"), thr, None)
pop = np.load(PREP / BREST / "population.npy")
b_k = np.bincount(b_left[b_acc], minlength=len(pop))
brest_lists = {c: {"mean_list": round(float(b_k[pop & (b_country == c)].mean()), 4),
                   "empty_rows": round(float((b_k[pop & (b_country == c)] == 0).mean()), 5)} for c in ("US", "India")}
spread = folds.std(0)
out = {"variant": variant, "threshold": thr, "file": str(path),
       "sha256": hashlib.sha256(raw).hexdigest(), "lines_incl_header": n_lines, "us_india_s1": int(len(ui)),
       "validation": {"passed": not problems, "problems": problems, "longest_list": int(lengths.max()),
                      "links": int(acc.sum())},
       "test_by_country": by_country, "brest_oof_lists_at_threshold": brest_lists,
       "distribution": {"brest_oof": stats(oof, b_left, b_country, thr), "test_mean": stats(prob, L, s1_country, thr)},
       "test_per_fold_model_links": per_fold_links,
       "fold_spread": {"mean_std": round(float(spread.mean()), 5),
                       "share_rows_std_gt_0.05": round(float((spread > 0.05).mean()), 5),
                       "rows_near_threshold_flipping": int(((folds >= thr).any(0) & (folds < thr).any(0)).sum())},
       "labels_read": False}
(work / "output_report.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
