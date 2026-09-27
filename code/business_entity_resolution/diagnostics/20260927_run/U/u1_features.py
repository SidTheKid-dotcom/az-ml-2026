"""U1-min feature family (27 Sep, Agent U): typo-tolerant name search of each empty-address (EA) record over
all same-country S1s, as pair features on candidate rows (no rule, no new pairs).

    python u1_features.py arena   # reuses u2/u2_name_search.npz (arena search over all arena S1s)
    python u1_features.py test    # runs the same search over US/India test S1s / EA test records

Features (NaN for non-EA records): u1_best, u1_second, u1_margin, u1_nties, u1_this, u1_is_best, u1_gap,
u1_this_rank (rank of this S1 among the record's top-10 by ratio; 10 if absent).
Writes prep/{brest,test}/x_u1.npy and prep/brest/names_u1.json. Label-free.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from u2_ea_name import norm, search  # noqa: E402
from ber.forensic_features import EMPTY, address_kinds  # noqa: E402

ROOT = Path("/home/ubuntu/run27/U")
PREP = ROOT / "prep"
NAMES = ["u1_best", "u1_second", "u1_margin", "u1_nties", "u1_this", "u1_is_best", "u1_gap"]
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def features(split, ref_path, pool_path, best, bsim, ssim, nties, rec_ea, s1_keep=None):
    d = PREP / split
    left, right = np.load(d / "left.npy"), np.load(d / "right.npy")
    rows = np.flatnonzero(rec_ea[right] & (np.ones(len(left), bool) if s1_keep is None else s1_keep[left]))
    ref = pq.read_table(ref_path, columns=["business_name"]).column("business_name")
    pool = pq.read_table(pool_path, columns=["business_name"]).column("business_name")
    s_idx, r_idx = left[rows], right[rows]
    us, ui = np.unique(s_idx, return_inverse=True)
    ur, ri = np.unique(r_idx, return_inverse=True)
    sn = norm(ref.take(us))
    rn = norm(pool.take(ur))
    this = np.fromiter((fuzz.ratio(sn[a], rn[b]) / 100.0 for a, b in zip(ui, ri)), np.float32, count=len(rows))
    x = np.full((len(left), len(NAMES)), np.nan, np.float32)
    x[rows, 0] = bsim[r_idx]
    x[rows, 1] = ssim[r_idx]
    x[rows, 2] = bsim[r_idx] - ssim[r_idx]
    x[rows, 3] = nties[r_idx]
    x[rows, 4] = this
    x[rows, 5] = (best[r_idx] == s_idx).astype(np.float32)
    x[rows, 6] = bsim[r_idx] - this
    np.save(d / "x_u1.npy", x)
    (PREP / "brest" / "names_u1.json").write_text(json.dumps(NAMES))
    log(f"{split}: {len(rows)} EA candidate rows featurized of {len(left)}")


def main():
    split = sys.argv[1]
    if split == "arena":
        z = np.load(ROOT / "u2" / "u2_name_search.npz")
        features("brest", ROOT / "data/arena/reference.parquet", ROOT / "data/arena/pool.parquet", z["best"],
                 z["bsim"], z["ssim"], z["nties"], z["rec_ea"])
        return
    ref_path, pool_path = ROOT / "data/test/dense_test/reference.parquet", ROOT / "data/test/dense_test/pool.parquet"
    ref = pq.read_table(ref_path, columns=["business_name", "country"])
    pool = pq.read_table(pool_path, columns=["business_name", "business_address", "country"])
    s1c = np.asarray(ref.column("country").to_pylist(), dtype=object)
    rc = np.asarray(pool.column("country").to_pylist(), dtype=object)
    rec_ea = (address_kinds(pool.column("business_address")) == EMPTY) & np.isin(rc, ["US", "India"])
    s1_norm = norm(ref.column("business_name"))
    ea_idx = np.flatnonzero(rec_ea)
    rec_norm = norm(pool.column("business_name").take(ea_idx))
    n = len(rc)
    best = np.full(n, -1, np.int64)
    bsim, ssim = np.zeros(n, np.float32), np.zeros(n, np.float32)
    nties = np.zeros(n, np.int16)
    for c in ("India", "US"):
        si = np.flatnonzero(s1c == c)
        qi = np.flatnonzero(rc[ea_idx] == c)
        log(f"test {c}: {len(si)} S1s, {len(qi)} EA records")
        b, bs, ss, nt = search(s1_norm[si], rec_norm[qi])
        r = ea_idx[qi]
        best[r] = np.where(b >= 0, si[np.maximum(b, 0)], -1)
        bsim[r], ssim[r], nties[r] = bs, ss, nt
    np.savez(ROOT / "u2" / "u1_test_name_search.npz", best=best, bsim=bsim, ssim=ssim, nties=nties, rec_ea=rec_ea)
    features("test", ref_path, pool_path, best, bsim, ssim, nties, rec_ea, np.isin(s1c, ["US", "India"]))


if __name__ == "__main__":
    main()
