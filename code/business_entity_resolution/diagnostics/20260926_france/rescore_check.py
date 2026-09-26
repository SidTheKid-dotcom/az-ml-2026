import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
from pathlib import Path
from ber.pair_tree_arena import load_scores
from ber.cross_encoder_experiment import load_reranker
from ber.fast_trainer import verified_checkpoint
from ber.text_views import record_texts
T = Path("/opt/az-ml-new/test_inputs"); R = Path("/opt/az-ml-new/runs/c2_step040000")
import json; cp = json.load(open(R / "test_scores/complete.json"))["checkpoint"]; print("checkpoint", cp)
ck = Path(cp) if Path(str(cp)).exists() else None
ref = pq.read_table(T / "dense_test/reference.parquet", columns=["business_name", "business_address", "country"]).to_pandas()
pool = pq.read_table(T / "dense_test/pool.parquet", columns=["business_name", "business_address"]).to_pandas()
left = np.asarray(np.load(T / "index/left.npy", mmap_mode="r")).astype(np.int64); right = np.asarray(np.load(T / "index/right.npy", mmap_mode="r")).astype(np.int64)
c2 = load_scores(R / "test_scores/scores.npy"); tree = load_scores(T / "base_probabilities.npy")
fr = ref.country.to_numpy()[left]
rng = np.random.default_rng(3)
pick = np.r_[rng.choice(np.flatnonzero((fr == "France") & (tree >= .9) & (c2 < .05)), 24, replace=False),
             rng.choice(np.flatnonzero((fr == "US") & (tree >= .9) & (c2 < .05)), 8, replace=False),
             rng.choice(np.flatnonzero((fr == "France") & (c2 > .9)), 6, replace=False)]
print(json.dumps(cp) if isinstance(cp, dict) else cp)
path = "/workspace/az-ml-new/c2_plain/ckpt/step_040000"
config, _ = verified_checkpoint(Path(path)); prof = config["profile"]
lt = record_texts(ref.iloc[np.unique(left[pick])].fillna(""), canonical=prof["canonical_text"], field_tagged=prof["field_tagged"])
model, tok = load_reranker(path, "cuda"); model.eval()
L = {i: t for i, t in zip(np.unique(left[pick]), lt)}
rt = record_texts(pool.iloc[np.unique(right[pick])].fillna(""), canonical=prof["canonical_text"], field_tagged=prof["field_tagged"])
Rt = {i: t for i, t in zip(np.unique(right[pick]), rt)}
a = [L[i] for i in left[pick]]; b = [Rt[j] for j in right[pick]]
enc = tok(a, b, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt").to("cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    fresh = torch.sigmoid(model(**enc).logits[:, 0].float()).cpu().numpy()
for k, i in enumerate(pick):
    print(f"{fr[i]:6s} stored={c2[i]:.4f} fresh={fresh[k]:.4f} tree={tree[i]:.3f} || {a[k]}  <>  {b[k]}")
