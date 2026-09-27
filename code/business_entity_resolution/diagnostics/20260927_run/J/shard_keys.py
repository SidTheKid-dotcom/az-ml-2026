"""Agent J: split key files into scoring shards (priority order: brest, test, fr tier1, fr tier2)."""
import json, sys
import numpy as np, pandas as pd
K = sys.argv[1]
plan = [("brest", "keys_brest_unc.parquet", None, 2, "train"), ("test", "keys_test_unc.parquet", None, 4, "test"),
        ("fr1", "keys_fr.parquet", 1, 4, "test"), ("fr2", "keys_fr.parquet", 2, 4, "test")]
out = []
for name, f, tier, n, side in plan:
    k = pd.read_parquet(f"{K}/{f}")
    if tier is not None:
        k = k[k.tier == tier].reset_index(drop=True)
    for i, part in enumerate(np.array_split(np.arange(len(k)), n)):
        path = f"{K}/shard_{name}_{i}.parquet"
        k.iloc[part].reset_index(drop=True).to_parquet(path, index=False)
        out.append({"shard": f"{name}_{i}", "rows": int(len(part)), "side": side, "path": path})
json.dump(out, open(f"{K}/shards.json", "w"), indent=1)
print(json.dumps(out))
