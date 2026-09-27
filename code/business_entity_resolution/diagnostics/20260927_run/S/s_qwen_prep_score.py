"""Agent S: tokenized Qwen score inputs (ber.qwen_train.score layout) for a custom row selection.

  s_qwen_prep_score.py eval  ARENA_DIR MODEL OUT   # arena_fr_v3: partition != A (B_REST monitor + French tuning) & tree >= 0.01
  s_qwen_prep_score.py france TEST_DIR  MODEL OUT  # test: French S1s & v1 tree >= 0.01 (the 3,121,523-pair rescoring set)
No labels are read.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from ber.qwen_train import (MODEL_ID, _raw, _texts, json_dump, load_tokenizer, save_npy, sha256_file, token_meta,
                            tokenize_pairs, log)

kind, src, model, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
if kind == "eval":
    t = pq.read_table(src / "arena/pairs.parquet", columns=["left_idx", "right_idx"])
    left = t.column(0).to_numpy().astype(np.int64); right = t.column(1).to_numpy().astype(np.int64)
    tree = np.load(src / "arena/v1_tree.npy")
    part = np.load(src / "s1_partition.npy")
    sel = (tree >= .01) & (part[left] != 0)
    ref_path, pool_path = src / "arena/reference.parquet", src / "arena/pool.parquet"
    mkind = "arena"
elif kind == "pairs":  # SRC = set dir with left/right.npy; REF/POOL from env QREF/QPOOL; every pair scored
    import os
    left = np.load(src / "left.npy").astype(np.int64); right = np.load(src / "right.npy").astype(np.int64)
    sel = np.ones(len(left), bool)
    ref_path, pool_path = Path(os.environ["QREF"]), Path(os.environ["QPOOL"])
    mkind = "test"
else:
    left = np.load(src / "index/left.npy").astype(np.int64); right = np.load(src / "index/right.npy").astype(np.int64)
    tree = np.load(src / "base_probabilities.npy", mmap_mode="r")
    country = pq.read_table(src / "dense_test/reference.parquet", columns=["country"]).column(0).to_numpy(zero_copy_only=False)
    sel = (np.asarray(tree) >= .01) & (country[left] == "France")
    ref_path, pool_path = src / "dense_test/reference.parquet", src / "dense_test/pool.parquet"
    mkind = "test"
rows = np.flatnonzero(sel)
li, ri = left[rows], right[rows]
reference = pd.read_parquet(ref_path, columns=["business_name", "business_address"])
pool = pd.read_parquet(pool_path, columns=["business_name", "business_address"])
tok = load_tokenizer(model)
tokens, offsets = tokenize_pairs(tok, _texts(reference), _texts(pool), li, ri)
save_npy(out / "tokens.npy", tokens); save_npy(out / "offsets.npy", offsets); save_npy(out / "pair_index.npy", rows.astype(np.int64))
lengths = np.diff(offsets)
manifest = {"kind": mkind, "state": "complete", "model_id": MODEL_ID, **token_meta(tok), "gate": .01,
            "total_pairs": int(len(left)), "scored_pairs": int(len(rows)), "selection": kind,
            "reference": str(ref_path), "pool": str(pool_path),
            "alignment": "row k scores pair pair_index[k] of the full pair arrays",
            "token_length": {"mean": float(lengths.mean()), "p99": float(np.percentile(lengths, 99)), "max": int(lengths.max())},
            "files": {n: sha256_file(out / n) for n in ("tokens.npy", "offsets.npy", "pair_index.npy")}, "labels_read": False}
json_dump(out / "manifest.json", manifest)
log(json.dumps({k: manifest[k] for k in ("kind", "total_pairs", "scored_pairs", "token_length")}))
