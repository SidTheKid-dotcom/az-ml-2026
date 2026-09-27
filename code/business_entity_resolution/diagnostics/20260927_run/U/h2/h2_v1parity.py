"""v1-scoring parity sample (label-free): 2,000 arena S1s (fixed stride) with their stored arena pairs, written in the
arena layout to H2/v1parity/, plus the stored blend/tree for those rows (H2/v1parity/stored_{blend,tree}.npy).
Re-scoring this directory with the frozen v1 streaming predictor must reproduce the stored arrays.

    python h2_v1parity.py            # build
    python h2_v1parity.py compare OUTDIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2_pairs import write_layout  # noqa: E402

ROOT = Path("/home/ubuntu/run27/U")
OUT = ROOT / "h2" / "v1parity"
COLS = ["entity_id", "business_name", "business_address", "country"]


def build():
    n_arena = 2_072_821
    par = np.arange(0, n_arena, n_arena // 2000)[:2000]
    t = pq.read_table(ROOT / "data/arena/pairs.parquet", columns=["left_idx", "right_raw_idx", "dense_score", "dense_rank"])
    al = t.column("left_idx").to_numpy()
    psel = np.flatnonzero(np.isin(al, par))
    ar = t.column("right_raw_idx").to_numpy()[psel].astype(np.int64)
    sub_left = np.searchsorted(par, al[psel])
    aref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=COLS).to_pandas()
    rep = write_layout(OUT, aref.iloc[par].reset_index(drop=True), sub_left, ar,
                       t.column("dense_score").to_numpy()[psel], t.column("dense_rank").to_numpy()[psel])
    order = np.lexsort((ar, sub_left))
    rows = psel[order]
    np.save(OUT / "arena_pair_rows.npy", rows)
    blend = np.load(ROOT / "data/arena/probabilities.npy", mmap_mode="r")
    base = np.load(ROOT / "data/arena/base_probabilities.npy", mmap_mode="r")
    np.save(OUT / "stored_blend.npy", np.asarray(blend[rows], np.float32))
    np.save(OUT / "stored_tree.npy", np.asarray(base[rows], np.float32))
    print(json.dumps(rep))


def compare(out):
    out = Path(out)
    got_b = np.load(out / "probabilities.npy")
    got_t = np.load(out / "base_probabilities.npy") if (out / "base_probabilities.npy").exists() else None
    sb, st = np.load(OUT / "stored_blend.npy"), np.load(OUT / "stored_tree.npy")
    rep = {"pairs": int(len(sb))}
    if got_t is not None:
        rep["blend_max_abs_diff"] = float(np.nanmax(np.abs(got_b - sb)))
        rep["blend_accept_disagreements_at_0.775"] = int(((got_b >= 0.775) != (sb >= 0.775)).sum())
        rep["tree_max_abs_diff"] = float(np.nanmax(np.abs(got_t - st)))
    else:  # tree-only run: probabilities = tree
        rep["tree_max_abs_diff"] = float(np.nanmax(np.abs(got_b - st)))
        rep["tree_gate_disagreements_at_0.01"] = int(((got_b >= 0.01) != (st >= 0.01)).sum())
    (out / "parity.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep))


if __name__ == "__main__":
    compare(sys.argv[2]) if len(sys.argv) > 1 and sys.argv[1] == "compare" else build()
