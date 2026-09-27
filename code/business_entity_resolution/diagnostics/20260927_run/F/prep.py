"""Agent F task 1 (+ parity of task 4, + pair sets for G1/G2/G3a). CPU only, no labels read.

Writes under /workspace/az-ml-new/run27/F/:
  inputs/reference.parquet, inputs/pool.parquet  normalized copies (French names only changed)
  g2/{left,right,tree,rows}.npy                  changed French gated candidate rows (+ 5,000 unchanged controls)
  g1/{left,right,tree,rows,set}.npy              probe A (drop_T2r) and probe C' links
  prep.json                                      counts, 20 examples, parity result
  state.npz                                      old French plain_prob / plain on cache rows
"""
import json, random, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from fcommon import OUT, S1_SHA, S1_FILE, load, normalize_name, plain_of, T
from ber.pair_tree_c2 import write_tsv, sha256, log

t0 = time.time()
OUT.mkdir(parents=True, exist_ok=True)
d = load()
left, right, base, c2, france = d["left"], d["right"], d["base"], d["c2"], d["france"]
s1_ids, rec_ids, country, pool_country = d["s1_ids"], d["rec_ids"], d["country"], d["pool_country"]
n_rec = len(rec_ids)
log(f"loaded; cand rows {len(left)}; French cand rows {int(france.sum())}; pool countries {Counter(pool_country.tolist())}")

# ---------- parity (task 4 first half)
plain_prob, plain = plain_of(d, c2)
final = np.where(france, plain, d["accept"]); final_prob = np.where(france, plain_prob, d["prob"])
par = OUT / "parity_old/matching_results.tsv"
write_tsv(par, s1_ids, rec_ids, left, right, final, final_prob)
par_sha = sha256(par)
parity = {"rebuilt_sha256": par_sha, "s1_file_sha256": S1_SHA, "ok": par_sha == S1_SHA}
log(f"parity {parity}")
if not parity["ok"]:
    raise SystemExit("PARITY FAILED")
np.savez(OUT / "state_old.npz", plain_prob=plain_prob, plain=plain, accept=d["accept"], prob=d["prob"])

# ---------- normalization (task 1)
def norm_col(names, mask):
    new = names.copy(); dropped = Counter(); changed = np.zeros(len(names), bool)
    for i in np.flatnonzero(mask):
        nm = names[i]
        if nm is None:
            continue
        nn, dk = normalize_name(nm)
        if dk:
            new[i] = nn; changed[i] = True; dropped.update(dk)
    return new, changed, dropped

ref_names = np.asarray(d["ref_table"].column("business_name").to_pylist(), dtype=object)
pool_names = np.asarray(d["pool_table"].column("business_name").to_pylist(), dtype=object)
ref_fr = country == "France"; pool_fr = pool_country == "France"
ref_new, ref_ch, ref_drop = norm_col(ref_names, ref_fr)
pool_new, pool_ch, pool_drop = norm_col(pool_names, pool_fr)
log(f"normalized: ref changed {int(ref_ch.sum())}/{int(ref_fr.sum())}, pool changed {int(pool_ch.sum())}/{int(pool_fr.sum())}")
assert not (ref_ch & ~ref_fr).any() and not (pool_ch & ~pool_fr).any()

def write_norm(table, new_names, path):
    cols = {}
    for c in table.column_names:
        col = table.column(c)
        if c == "business_name":
            col = pa.array(new_names.tolist(), type=pa.large_string())
        elif pa.types.is_string(col.type):
            col = col.cast(pa.large_string())
        cols[c] = col
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)

write_norm(d["ref_table"], ref_new, OUT / "inputs/reference.parquet")
write_norm(d["pool_table"], pool_new, OUT / "inputs/pool.parquet")
np.save(OUT / "inputs/ref_changed.npy", ref_ch); np.save(OUT / "inputs/pool_changed.npy", pool_ch)

rng = random.Random(20260927)
ex_ref = [(str(ref_names[i]), str(ref_new[i])) for i in rng.sample(list(np.flatnonzero(ref_ch)), 10)]
ex_pool = [(str(pool_names[i]), str(pool_new[i])) for i in rng.sample(list(np.flatnonzero(pool_ch)), 10)]

# changed candidate pairs
pair_ch = france & (ref_ch[left] | pool_ch[right])
gated = base >= 0.01
g2_rows = np.flatnonzero(pair_ch & gated)
ctrl_pool = np.flatnonzero(france & gated & ~pair_ch)
ctrl = np.sort(np.random.default_rng(0).choice(ctrl_pool, 5000, replace=False))
rows = np.concatenate([g2_rows, ctrl])
(OUT / "g2").mkdir(parents=True, exist_ok=True)
np.save(OUT / "g2/left.npy", left[rows]); np.save(OUT / "g2/right.npy", right[rows])
np.save(OUT / "g2/tree.npy", np.maximum(base[rows], 0.01).astype(np.float32)); np.save(OUT / "g2/rows.npy", rows)
np.save(OUT / "g2/n_changed.npy", np.array([len(g2_rows)]))

# ---------- probe sets (G1)
def parse(path, fr_only=True):
    out = set()
    with open(path, encoding="utf-8") as h:
        next(h)
        for line in h:
            s, ids = line.rstrip("\n").split("\t")
            if ids:
                for r in ids.split(","):
                    out.add((s, r))
    return out
fr_s1 = set(s1_ids[ref_fr].tolist())
s1_links = {p for p in parse(S1_FILE) if p[0] in fr_s1}
t2r_links = {p for p in parse(Path("/workspace/az-ml-new/eureka/probes/drop_T2r/matching_results.tsv")) if p[0] in fr_s1}
probe_a = sorted(s1_links - t2r_links)
assert not (t2r_links - s1_links), "drop_T2r has links not in the S1 file"
probe_c = [tuple(x) for x in json.loads(Path("/workspace/az-ml-new/eureka/probes/drop_Cprime/dropped.json").read_text())]
s1_pos = {v: i for i, v in enumerate(s1_ids)}; rec_pos = {v: i for i, v in enumerate(rec_ids)}
keys = left * n_rec + right; o = np.argsort(keys); ks = keys[o]
def rows_of(pairs):
    q = np.array([s1_pos[s] * n_rec + rec_pos[r] for s, r in pairs], dtype=np.int64)
    pos = np.minimum(np.searchsorted(ks, q), len(ks) - 1)
    assert (ks[pos] == q).all(), "probe link not in candidate rows"
    return o[pos]
ra, rc = rows_of(probe_a), rows_of(probe_c)
prow = np.concatenate([ra, rc]); pset = np.concatenate([np.zeros(len(ra), np.int8), np.ones(len(rc), np.int8)])
(OUT / "g1").mkdir(parents=True, exist_ok=True)
np.save(OUT / "g1/left.npy", left[prow]); np.save(OUT / "g1/right.npy", right[prow])
np.save(OUT / "g1/tree.npy", np.ones(len(prow), np.float32)); np.save(OUT / "g1/rows.npy", prow); np.save(OUT / "g1/set.npy", pset)

report = {
    "parity": parity,
    "french_s1": int(ref_fr.sum()), "french_pool_records": int(pool_fr.sum()),
    "ref_names_changed": int(ref_ch.sum()), "pool_names_changed": int(pool_ch.sum()),
    "dropped_token_counts_ref": dict(ref_drop.most_common()), "dropped_token_counts_pool": dict(pool_drop.most_common()),
    "cand_rows": int(len(left)), "french_cand_rows": int(france.sum()),
    "french_cand_rows_text_changed": int(pair_ch.sum()),
    "french_cand_rows_text_changed_gated_tree_ge_0.01": int(len(g2_rows)),
    "french_cand_rows_text_changed_below_gate": int((pair_ch & ~gated).sum()),
    "changed_by_side": {"s1_only": int((france & ref_ch[left] & ~pool_ch[right] & gated).sum()),
                        "record_only": int((france & ~ref_ch[left] & pool_ch[right] & gated).sum()),
                        "both": int((france & ref_ch[left] & pool_ch[right] & gated).sum())},
    "changed_gated_old_accepted": int(plain[g2_rows].sum()),
    "control_rows": int(len(ctrl)),
    "probe_a_links": len(probe_a), "probe_c_links": len(probe_c),
    "examples_s1": ex_ref, "examples_record": ex_pool,
    "seconds": round(time.time() - t0, 1),
}
(OUT / "prep.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
print(json.dumps({k: v for k, v in report.items() if not k.startswith("examples")}, indent=1, ensure_ascii=False))
log("prep done")
