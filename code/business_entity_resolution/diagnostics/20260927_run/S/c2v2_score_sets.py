"""Agent S: C2 v2 scoring inputs for route (Agent U) and reverse (Agent R) pairs on B_REST and test.
Extended pools = the compact pool + missing records' text from the supplied raw TSVs (orphans / route records).
Writes /dev/shm/c2v2/score/{brest_ext,test_ext}/pool.parquet and per set {left,right,tree}.npy + keys.parquet."""
import json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
import os
V = Path(os.environ.get("SETS_REV", "/dev/shm/c2v2/rev")); S = Path(os.environ.get("SETS_OUT", "/dev/shm/c2v2/score")); S.mkdir(parents=True, exist_ok=True)
ARENA_POOL = Path(os.environ.get("SETS_ARENA_POOL", "/dev/shm/c2v2/in/arena/pool.parquet"))
TEST_POOL = Path(os.environ.get("SETS_TEST_POOL", "/opt/az-ml-new/test_inputs/dense_test/pool.parquet"))
COLS = ["entity_id", "business_name", "business_address", "country"]
R = {}

def ext_pool(base_path, extra: pd.DataFrame, out: Path):
    base = pq.read_table(base_path, columns=COLS).cast(pa.schema([(c, pa.large_string()) for c in COLS]))
    add = pa.table({c: pa.array(extra[c].tolist(), type=pa.large_string()) for c in COLS})
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables([base, add]), out / "pool.parquet", row_group_size=2_000_000)
    return base.num_rows, np.asarray(base.column("entity_id").to_pylist(), dtype=object)

def save(name, left, right, keys):
    d = S / name; d.mkdir(parents=True, exist_ok=True)
    np.save(d / "left.npy", left.astype(np.int64)); np.save(d / "right.npy", right.astype(np.int64))
    np.save(d / "tree.npy", np.ones(len(left), np.float32))
    keys.to_parquet(d / "keys.parquet", index=False)
    R[name] = int(len(left))

# ---- B_REST: routes (arena indices) + reverse (arena pool or train orphans)
ub = pq.read_table(V / "u_route_pairs_brest.parquet", columns=["left_idx", "right_idx", "s1_id", "rec_id"]).to_pandas()
rb = pq.read_table(V / "r_reverse_pairs_b_rest.parquet", columns=["left_idx", "right_idx", "raw_row_idx", "s1_id", "rec_id"]).to_pandas()
orph = pq.read_table(V / "orphans_train.parquet").to_pandas()
need = np.unique(rb.raw_row_idx.to_numpy()[rb.right_idx.to_numpy() < 0])
extra = orph.set_index("raw_row_idx").loc[need].reset_index()
n0, _ = ext_pool(ARENA_POOL, extra, S / "brest_ext")
pos = pd.Series(n0 + np.arange(len(extra)), index=extra.raw_row_idx.to_numpy())
save("brest_routes", ub.left_idx.to_numpy(), ub.right_idx.to_numpy(), ub)
rr = rb.right_idx.to_numpy().astype(np.int64); o = rr < 0; rr[o] = pos[rb.raw_row_idx.to_numpy()[o]].to_numpy()
save("brest_reverse", rb.left_idx.to_numpy(), rr, rb)
# ---- test: routes (by rec_id) + reverse (dense pool or test orphans)
ut = pq.read_table(V / "u_route_pairs_test.parquet", columns=["left_idx", "right_idx", "s1_id", "rec_id"]).to_pandas()
rt = pq.read_table(V / "r_reverse_pairs_test.parquet", columns=["left_idx", "right_idx", "raw_row_idx", "s1_id", "rec_id"]).to_pandas()
torph = pq.read_table(V / "orphans_test.parquet").to_pandas(); rrec = pq.read_table(V / "route_recs_test.parquet").to_pandas()
base_ids = np.asarray(pq.read_table(TEST_POOL, columns=["entity_id"]).column(0).to_pylist(), dtype=object)
in_dense = pd.Series(np.arange(len(base_ids)), index=base_ids)
extra = pd.concat([torph, rrec[~rrec.entity_id.isin(in_dense.index)]]).drop_duplicates("entity_id").reset_index(drop=True)
n0, _ = ext_pool(TEST_POOL, extra, S / "test_ext")
idx = pd.concat([in_dense, pd.Series(n0 + np.arange(len(extra)), index=extra.entity_id.to_numpy())])
save("test_routes", ut.left_idx.to_numpy(), idx[ut.rec_id.to_numpy()].to_numpy(), ut)
save("test_reverse", rt.left_idx.to_numpy(), idx[rt.rec_id.to_numpy()].to_numpy(), rt)
chk = rt.right_idx.to_numpy() >= 0
assert (idx[rt.rec_id.to_numpy()[chk]].to_numpy() == rt.right_idx.to_numpy()[chk]).all(), "reverse right_idx != dense pool row"
R["test_extra_records"] = int(len(extra)); R["route_rows_outside_dense"] = int((~ut.rec_id.isin(in_dense.index)).sum())
(S / "sets.json").write_text(json.dumps(R, indent=1) + "\n"); print(json.dumps(R))
