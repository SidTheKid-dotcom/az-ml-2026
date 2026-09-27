"""Agent S: manifest.json for the arena v1 copy used by C2 v2 (same fields fast_trainer.load_score_data checks)."""
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
from ber.v2_ce_train import file_sha256
A = Path(sys.argv[1])
m = {"schema": 1, "source": "arena v1 (S3 inputs/artifacts/arena/v1) for C2 v2; C2's original c2_input used the same files",
     "references": pq.read_metadata(A / "reference.parquet").num_rows, "pool_records": pq.read_metadata(A / "pool.parquet").num_rows,
     "pairs": pq.read_metadata(A / "pairs.parquet").num_rows, "labels_read": False, "holdout2_excluded": True,
     "pair_order": "left_then_right",
     "sha256": {n: file_sha256(A / n) for n in ("reference.parquet", "pool.parquet", "pairs.parquet", "base_probabilities.npy", "probabilities.npy")}}
(A / "manifest.json").write_text(json.dumps(m, indent=1) + "\n"); print(json.dumps(m))
