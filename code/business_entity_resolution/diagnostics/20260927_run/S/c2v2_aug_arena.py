"""Agent S: C2 v2 continuation arena = arena v1 (dense top-50, TRAIN labels) + Agent R's partition-A reverse pairs.
All reverse positives are kept; a seeded --neg-keep share of reverse negatives is kept (tree proxy 0.5, blend 0: they
join fast_trainer's random tree-gated negatives, never the blend-typed ones). Orphan records (right_idx == -1) are
appended to the pool from the supplied raw train TSV rows (orphans_train.parquet). No SLICE/HOLDOUT-2 label is read.
Usage: c2v2_aug_arena.py IN_DIR REV_DIR OUT_DIR [--neg-keep 0.3]"""
import argparse, json, shutil, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from ber.v2_ce_train import file_sha256

p = argparse.ArgumentParser(); p.add_argument("inp"); p.add_argument("rev"); p.add_argument("out")
p.add_argument("--neg-keep", type=float, default=.3); p.add_argument("--seed", type=int, default=20260927)
a = p.parse_args()
I, V, O = Path(a.inp), Path(a.rev), Path(a.out)
(O / "arena").mkdir(parents=True, exist_ok=True); (O / "labels").mkdir(parents=True, exist_ok=True)
t0 = time.time()
part = np.load(I / "s1_partition.npy")
pairs = pq.read_table(I / "arena/pairs.parquet", columns=["left_idx", "right_idx"])
left = pairs.column(0).to_numpy().astype(np.int64); right = pairs.column(1).to_numpy().astype(np.int64); del pairs
label = np.load(I / "labels/label.npy"); owner = np.load(I / "labels/rec_owner.npy").astype(np.int64)
tc = np.load(I / "labels/true_count.npy")
tree = np.load(I / "arena/base_probabilities.npy"); blend = np.load(I / "arena/probabilities.npy")
n_pool = len(owner)
rv = pq.read_table(V / "r_reverse_pairs_a.parquet", columns=["left_idx", "right_idx", "raw_row_idx", "rec_id", "label"]).to_pandas()
assert (part[rv.left_idx.to_numpy()] == 0).all(), "reverse A pairs must be partition A"
orph = pq.read_table(V / "orphans_train.parquet").to_pandas()
pos_of = pd.Series(np.arange(len(orph)), index=orph.raw_row_idx.to_numpy())
rng = np.random.default_rng(a.seed)
keep = rv.label.to_numpy().astype(bool) | (rng.random(len(rv)) < a.neg_keep)
rv = rv[keep].reset_index(drop=True)
is_orph = rv.right_idx.to_numpy() < 0
used_orph = np.unique(pos_of[rv.raw_row_idx.to_numpy()[is_orph]].to_numpy())
new_pos = np.full(len(orph), -1, np.int64); new_pos[used_orph] = n_pool + np.arange(len(used_orph))
r_right = rv.right_idx.to_numpy().astype(np.int64)
r_right[is_orph] = new_pos[pos_of[rv.raw_row_idx.to_numpy()[is_orph]].to_numpy()]
r_left = rv.left_idx.to_numpy().astype(np.int64); r_label = rv.label.to_numpy().astype(bool)
# dedupe against existing arena pairs
key_old = left * (n_pool + len(used_orph)) + right
key_new = r_left * (n_pool + len(used_orph)) + r_right
dup = np.isin(key_new, key_old)
r_left, r_right, r_label = r_left[~dup], r_right[~dup], r_label[~dup]
owner_new = np.concatenate([owner, np.full(len(used_orph), -1, np.int64)])
orph_true = r_right >= n_pool
owner_new[r_right[orph_true & r_label]] = r_left[orph_true & r_label]
bad_in_pool = (~orph_true) & (r_label != (owner_new[r_right] == r_left))
R = {"reverse_a_rows": int(len(keep)), "kept_after_neg_sampling": int(keep.sum()), "duplicates_of_arena_pairs": int(dup.sum()),
     "added_pairs": int(len(r_left)), "added_positives": int(r_label.sum()), "orphan_records_added": int(len(used_orph)),
     "label_owner_mismatch_in_pool": int(bad_in_pool.sum())}
assert R["label_owner_mismatch_in_pool"] == 0, R
L = np.concatenate([left, r_left]); Rr = np.concatenate([right, r_right])
lab = np.concatenate([label, r_label]); tr = np.concatenate([tree, np.full(len(r_left), .5, np.float32)]).astype(np.float32)
bl = np.concatenate([blend, np.zeros(len(r_left), np.float32)]).astype(np.float32)
src = np.concatenate([np.zeros(len(left), np.uint8), np.ones(len(r_left), np.uint8)])
o = np.lexsort((Rr, L)); L, Rr, lab, tr, bl, src = L[o], Rr[o], lab[o], tr[o], bl[o], src[o]
assert not np.any((L[1:] == L[:-1]) & (Rr[1:] <= Rr[:-1])), "duplicate pairs"
assert np.array_equal(lab, owner_new[Rr] == L), "labels != owners"
assert not np.any(np.bincount(L[lab], minlength=len(tc)) > tc), "retrieved > true_count"
pq.write_table(pa.table({"left_idx": L.astype(np.int32), "right_idx": Rr.astype(np.int32), "source": src}),
               O / "arena/pairs.parquet", row_group_size=4_000_000)
np.save(O / "arena/v1_tree.npy", tr); np.save(O / "arena/v1_blend.npy", bl)
shutil.copyfile(I / "arena/reference.parquet", O / "arena/reference.parquet")
cols = ["entity_id", "business_name", "business_address", "country"]
pool = pq.read_table(I / "arena/pool.parquet", columns=cols)
pool = pool.cast(pa.schema([(c, pa.large_string()) for c in cols]))
add = pa.table({c: pa.array(orph[c].to_numpy()[used_orph].tolist(), type=pa.large_string()) for c in cols})
pq.write_table(pa.concat_tables([pool, add]), O / "arena/pool.parquet", row_group_size=2_000_000)
np.save(O / "labels/label.npy", lab); np.save(O / "labels/true_count.npy", tc); np.save(O / "labels/rec_owner.npy", owner_new.astype(np.int32))
summ = {"pairs": int(len(lab)), "positive_pairs": int(lab.sum()), "true_links": int(tc.sum()), "singletons": int((tc == 0).sum()),
        "holdout2_rows_read": 0, "reverse_pairs": R}
(O / "labels/labels_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
shutil.copyfile(I / "s1_partition.npy", O / "s1_partition.npy")
m = {"schema": 1, "source": "arena v1 + Agent R partition-A reverse pairs (c2v2_aug_arena.py)",
     "references": pq.read_metadata(O / "arena/reference.parquet").num_rows, "pool_records": pq.read_metadata(O / "arena/pool.parquet").num_rows,
     "pairs": int(len(L)), "labels_read": False, "holdout2_excluded": True, "pair_order": "left_then_right",
     "sha256": {n: file_sha256(O / "arena" / n) for n in ("reference.parquet", "pool.parquet", "pairs.parquet", "v1_tree.npy", "v1_blend.npy")}}
(O / "arena/manifest.json").write_text(json.dumps(m, indent=1) + "\n")
R["seconds"] = round(time.time() - t0, 1); print(json.dumps(R))
