"""Per-S1 boolean mask of French test S1s (test reference order). Usage: s_france_mask.py TEST_DIR OUT.npy"""
import sys
import numpy as np, pyarrow.parquet as pq
c = pq.read_table(sys.argv[1] + "/dense_test/reference.parquet", columns=["country"]).column(0).to_numpy(zero_copy_only=False)
m = c == "France"; np.save(sys.argv[2], m); print(int(m.sum()))
