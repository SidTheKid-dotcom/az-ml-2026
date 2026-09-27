"""Density-shift check (27 Sep, Agent U): rebuild the pool-wide blocks on B_REST at a test-like S1 density and
re-score the already-fitted 5-fold models (no refit).

Test has 0.53x the arena's US S1s (663k vs 1.24M) and about the same number of India S1s. Keep every B_REST S1
and every India S1, and a seeded random 53% of the other US S1s. The record pool stays whole ("drop rival S1s,
keep owners"). Recompute the collective block (and optionally u1) for B_REST rows, predict each row with its
fold's model, decode at the fold's saved threshold, and compare with the arena-density OOF (paired bootstrap).

    python density_check.py VARIANT [--u1]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from ber.arena_decision import component_bootstrap  # noqa: E402
from ber.collective_features import build_tables, pair_features  # noqa: E402
from ber.pair_tree_c2 import accept_per_row, proba, slim  # noqa: E402
from u4_tree import PREP, ROOT, blocks, load_arena  # noqa: E402

KEEP_US = 663106 / 1243156
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


variant = sys.argv[1]
use_u1 = "--u1" in sys.argv
work = ROOT / "u4" / variant.replace("+", "_")
rep = json.loads((work / "report.json").read_text())
arena, component = load_arena("brest")
part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
ref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=["business_name", "business_address", "country"])
pool = pq.read_table(ROOT / "data/arena/pool.parquet", columns=["business_name", "business_address"])
country = np.asarray(ref.column("country").to_pylist(), dtype=object)
rng = np.random.default_rng(20260927)
keep = (part == 1) | (country == "India") | (rng.random(len(part)) < KEEP_US)
sel = np.flatnonzero(keep)
s1_map = np.full(len(part), -1, np.int64)
s1_map[sel] = np.arange(len(sel))
log(f"kept S1s {len(sel)} of {len(part)} (US {int((keep & (country == 'US')).sum())})")
cache = work / "density_x_coll.npy"
if cache.exists():
    xc = np.load(cache)
else:
    tables = build_tables(ref.column("business_name").take(pa.array(sel)),
                          ref.column("business_address").take(pa.array(sel)),
                          pool.column("business_name"), pool.column("business_address"), log=log)
    xc = pair_features(tables, s1_map[arena.left], arena.right)
    np.save(cache, xc)
names = variant.split("+")
xu1 = None
if "u1" in names:
    cache_u1 = ROOT / "u2" / "density_u1_search.npz"
    if cache_u1.exists():
        z = dict(np.load(cache_u1))
    else:
        from ber.forensic_features import EMPTY, address_kinds
        from u2_ea_name import norm, search
        rec_c = np.asarray(pq.read_table(ROOT / "data/arena/pool.parquet", columns=["country"]).column("country")
                           .to_pylist(), dtype=object)
        rec_ea = address_kinds(pool.column("business_address")) == EMPTY
        s1n = norm(ref.column("business_name"))
        ea_idx = np.flatnonzero(rec_ea)
        rn = norm(pool.column("business_name").take(pa.array(ea_idx)))
        z = {"best": np.full(len(rec_ea), -1, np.int64), "bsim": np.zeros(len(rec_ea), np.float32),
             "ssim": np.zeros(len(rec_ea), np.float32), "nties": np.zeros(len(rec_ea), np.int16)}
        for c in ("India", "US"):
            si = np.flatnonzero((country == c) & keep)
            qi = np.flatnonzero(rec_c[ea_idx] == c)
            bb, bs, ss, nt = search(s1n[si], rn[qi])
            r = ea_idx[qi]
            z["best"][r] = np.where(bb >= 0, si[np.maximum(bb, 0)], -1)
            z["bsim"][r], z["ssim"][r], z["nties"][r] = bs, ss, nt
        np.savez(cache_u1, **z)
    xu1 = blocks("brest", "u1").copy()
    rows = np.flatnonzero(~np.isnan(xu1[:, 0]))
    r = arena.right[rows]
    this = xu1[rows, 4]
    xu1[rows, 0], xu1[rows, 1] = z["bsim"][r], z["ssim"][r]
    xu1[rows, 2] = z["bsim"][r] - z["ssim"][r]
    xu1[rows, 3] = z["nties"][r]
    xu1[rows, 5] = (z["best"][r] == arena.left[rows]).astype(np.float32)
    xu1[rows, 6] = z["bsim"][r] - this
    log(f"u1 rebuilt at test-like density for {len(rows)} EA rows")
parts = []
for b in names:
    if b == "coll":
        parts.append(xc)
    elif b == "u1":
        parts.append(xu1)
    else:
        parts.append(blocks("brest", b))
x = np.hstack(parts).astype(np.float32)
from catboost import CatBoostClassifier  # noqa: E402

fold_s1 = np.load(ROOT / "folds" / "brest_fold5_u4-20260927.npy")
fold_row = fold_s1[arena.left]
oof_d = np.zeros(len(x), np.float32)
for k in range(5):
    m = CatBoostClassifier()
    m.load_model(str(work / f"model_fold{k}.cbm"))
    rows = fold_row == k
    oof_d[rows] = proba(m, x[rows])
picks = rep["thresholds"]["per_fold_threshold"]
thr = np.asarray([picks[str(k)] for k in range(5)], np.float32)[fold_row]
res = arena.score(accept_per_row(arena, oof_d, thr, "model"), full=True)
per_d = res.pop("per_s1")
per = np.load(work / "per_s1.npy")
boot = component_bootstrap(per, per_d, component, n=1000, seed=20260927, population=arena.population)
base_per = np.load(ROOT / "u4" / (sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "base") / "per_s1.npy")
boot_vs_base = component_bootstrap(base_per, per_d, component, n=1000, seed=20260927, population=arena.population)
out = {"variant": variant, "keep_us_fraction": KEEP_US, "kept_s1": int(len(sel)), "arena_density": rep["result"],
       "test_like_density": slim(res), "density_minus_arena": boot, "density_minus_base_tree": boot_vs_base,
       "mean_abs_prob_change": float(np.abs(oof_d - np.load(work / "oof.npy")).mean())}
(work / "density_check.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
log(json.dumps({"arena": rep["result"]["macro_f05"], "density": res["macro_f05"],
                "delta": (boot["delta"], boot["ci_low"], boot["ci_high"]),
                "vs_base": (boot_vs_base["delta"], boot_vs_base["ci_low"], boot_vs_base["ci_high"])}))
