"""Diagnose the native-script links the baseline misses (B_REST owners only): transliteration similarity to the owner,
owner in the retrieval top-K or not, and 15 examples. B_REST labels only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nat_dict import N, ROOT, existing_keys_arena, is_native, lat_tokens, texts, translit, truth  # noqa: E402

d = json.load(open(N / "dictionary.json"))["dictionary"]
ref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=["entity_id", "business_name", "business_address",
                                                                     "country"]).to_pandas()
part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
gt = truth(set(ref.entity_id[part == 1]))
ids, names, addrs, cc = texts("train_pool")
pos = {v: i for i, v in enumerate(ids)}
s1_idx = {v: i for i, v in enumerate(ref.entity_id)}
base = existing_keys_arena()
cand = pd.read_parquet(N / "arena_candidates.parquet", columns=["s1", "raw", "rank", "score"])
ck = dict(zip(cand.s1.to_numpy().astype(np.int64) * 20_000_000 + cand.raw.to_numpy(), cand["rank"].to_numpy()))
pairs = [(s1_idx[s1], pos[r]) for s1, recs in gt.items() for r in recs if r in pos and is_native(names[pos[r]])]
keys = np.asarray([a * 20_000_000 + b for a, b in pairs], np.int64)
missing = ~np.isin(keys, base)
rows = []
for (si, ri), key in zip(np.asarray(pairs)[missing], keys[missing]):
    if True:
        t = translit(names[ri], d)
        sn = " ".join(lat_tokens(ref.business_name[si]))
        rows.append({"ratio": fuzz.ratio(t, sn), "tsr": fuzz.token_set_ratio(t, sn), "rank": ck.get(key, -1),
                     "s1": ref.business_name[si], "s1_addr": ref.business_address[si], "rec": names[ri],
                     "translit": t, "rec_addr": addrs[ri]})
df = pd.DataFrame(rows)
out = {"missed_native_links": len(df),
       "owner_in_topk": int((df["rank"] >= 0).sum()),
       "ratio_quantiles": df.ratio.quantile([.1, .25, .5, .75, .9]).round(1).tolist(),
       "tsr_ge_90": int((df.tsr >= 90).sum()), "tsr_lt_50": int((df.tsr < 50).sum())}
print(json.dumps(out, indent=1))
for _, x in df.sample(15, random_state=1).iterrows():
    print(f"[{x['rank']}] S1: {x.s1} | {x.s1_addr}\n    REC: {x.rec} | {x.rec_addr}\n    TR : {x.translit}  (tsr {x.tsr})")
