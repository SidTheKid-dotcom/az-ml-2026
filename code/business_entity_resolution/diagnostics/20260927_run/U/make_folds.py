"""Frozen 5-fold B_REST fold file (27 Sep, Agent U).

fold = int(sha256(f"{salt}:{component}")[:16], 16) % 5 over arena claim-graph components
(finalize_v1/component_ids.npy: S1s sharing a record with v1 blend >= 0.05, as P3 specifies).
Per-S1 int8 array over all arena S1s; -1 outside B_REST. Label-free.

    python make_folds.py FINALIZE_DIR OUT_DIR [SALT]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

fin, out = Path(sys.argv[1]), Path(sys.argv[2])
salt = sys.argv[3] if len(sys.argv) > 3 else "u4-20260927"
part = np.load(fin / "s1_partition.npy")
comp = np.load(fin / "component_ids.npy").astype(np.int64)
brest = part == 1
ucomp, inv = np.unique(comp[brest], return_inverse=True)
cf = np.asarray([int(hashlib.sha256(f"{salt}:{int(c)}".encode()).hexdigest()[:16], 16) % 5 for c in ucomp], np.int8)
fold = np.full(len(part), -1, np.int8)
fold[brest] = cf[inv]
sizes = np.bincount(inv)
out.mkdir(parents=True, exist_ok=True)
path = out / f"brest_fold5_{salt}.npy"
np.save(path, fold)
manifest = {
    "salt": salt, "rule": "fold = int(sha256(f'{salt}:{component_id}').hexdigest()[:16], 16) % 5",
    "components_source": str(fin / "component_ids.npy"),
    "component_sha256": hashlib.sha256((fin / "component_ids.npy").read_bytes()).hexdigest(),
    "partition_sha256": hashlib.sha256((fin / "s1_partition.npy").read_bytes()).hexdigest(),
    "component_rule": "claim graph, v1 blend >= 0.05 (finalize_v1 manifest)",
    "b_rest_s1": int(brest.sum()), "b_rest_components": int(len(ucomp)),
    "fold_s1_counts": np.bincount(fold[brest], minlength=5).tolist(),
    "fold_component_counts": np.bincount(cf, minlength=5).tolist(),
    "largest_component_s1": int(sizes.max()), "largest_component_fraction": float(sizes.max() / brest.sum()),
    "largest_ok_lt_0.5pct": bool(sizes.max() / brest.sum() < 0.005),
    "file": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
}
(out / f"brest_fold5_{salt}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(json.dumps(manifest, indent=1))
