"""Collective features on the TEST pool for US/India candidate rows (27 Sep, Agent U).

Same code as the arena (ber.collective_features.build_tables / pair_features). Tables are built over the
US/India test S1s and US/India test records only (the arena is US/India only). France rows get NaN.
Writes prep/test/x_coll.npy aligned with prep/test/left.npy. Label-free.
"""
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber.collective_features import build_tables, pair_features

ROOT = Path("/home/ubuntu/run27/U")
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


d = ROOT / "prep" / "test"
ref = pq.read_table(ROOT / "data/test/dense_test/reference.parquet", columns=["business_name", "business_address", "country"])
pool = pq.read_table(ROOT / "data/test/dense_test/pool.parquet", columns=["business_name", "business_address", "country"])
meta = json.loads((d / "meta.json").read_text())
if len(ref) != meta["n_s1"] or len(pool) != meta["n_rec"]:
    raise SystemExit("test parquet sizes differ from the prep index")
keep = ("US", "India")
s1c = np.asarray(ref.column("country").to_pylist(), dtype=object)
rc = np.asarray(pool.column("country").to_pylist(), dtype=object)
s1_sel = np.flatnonzero(np.isin(s1c, keep))
rec_sel = np.flatnonzero(np.isin(rc, keep))
s1_map = np.full(len(s1c), -1, np.int64)
s1_map[s1_sel] = np.arange(len(s1_sel))
rec_map = np.full(len(rc), -1, np.int64)
rec_map[rec_sel] = np.arange(len(rec_sel))
left = np.load(d / "left.npy")
right = np.load(d / "right.npy")
rows = np.flatnonzero(s1_map[left] >= 0)
ok = rec_map[right[rows]] >= 0
log(f"US/India S1 {len(s1_sel)}, records {len(rec_sel)}; US/India candidate rows {len(rows)}, "
    f"record outside US/India {int((~ok).sum())}")
tk = lambda t, c, idx: t.column(c).take(pa.array(idx))  # noqa: E731
tables = build_tables(tk(ref, "business_name", s1_sel), tk(ref, "business_address", s1_sel),
                      tk(pool, "business_name", rec_sel), tk(pool, "business_address", rec_sel), log=log)
x = np.full((len(left), 25), np.nan, np.float32)
r_ok = rows[ok]
x[r_ok] = pair_features(tables, s1_map[left[r_ok]], rec_map[right[r_ok]])
np.save(d / "x_coll.npy", x)
json.dump({"rows_computed": int(len(r_ok)), "rows_us_india": int(len(rows)), "record_outside": int((~ok).sum()),
           "tables_scope": "US/India test S1s + US/India test records"}, open(d / "x_coll_meta.json", "w"))
log("saved")
