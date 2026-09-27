"""Agent J: join scored shards with their keys -> llm_brest / llm_test / llm_fr (+ brestconf) parquets + sha256 report.

Usage: assemble.py KEYS_DIR OUT_DIR RAW_DIR [RAW_DIR ...]
  Each RAW_DIR/<shard>/logits_gated.npy (+ complete.json) holds one scoring pass (NaN = row not scored in that
  pass, e.g. outside that pass's probability-band mask). Passes are combined per row (first finite value);
  only rows scored in some pass are written. Column ``pass`` names the RAW_DIR that scored the row.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

K, OUT, RAWS = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
OUT.mkdir(parents=True, exist_ok=True)
shards = json.load(open(K / "shards.json"))
groups = {}
report = {"passes": [str(p) for p in RAWS], "shards": {}}
for sh in shards:
    name = sh["shard"]
    keys = pd.read_parquet(K / f"shard_{name}.parquet")
    z = np.full(len(keys), np.nan, dtype=np.float32)
    src = np.full(len(keys), "", dtype=object)
    info = {"rows": len(keys)}
    for raw in RAWS:
        f, c = raw / name / "logits_gated.npy", raw / name / "complete.json"
        if not (f.exists() and c.exists()):
            continue
        v = np.load(f).astype(np.float32)
        if len(v) != len(keys):
            raise SystemExit(f"{raw}/{name}: {len(v)} logits for {len(keys)} keys")
        take = np.isfinite(v) & ~np.isfinite(z)
        z[take] = v[take]
        src[take] = raw.name
        comp = json.load(open(c))
        info[raw.name] = {"scored": int(np.isfinite(v).sum()), "checkpoint": comp["checkpoint"],
                          "seconds": comp["runtime"]["seconds"]}
    ok = np.isfinite(z)
    info["scored_rows"] = int(ok.sum())
    report["shards"][name] = info
    if not ok.any():
        continue
    part = keys[ok].copy()
    part["llm_logit"] = z[ok]
    part["llm_p"] = (1.0 / (1.0 + np.exp(-z[ok].astype(np.float64)))).astype(np.float32)
    part["pass"] = src[ok]
    groups.setdefault("fr" if name.startswith("fr") else name.split("_")[0], []).append(part)
for g, parts in groups.items():
    df = pd.concat(parts, ignore_index=True)
    if df.duplicated(["s1_id", "rec_id"]).any():
        raise SystemExit(f"{g}: duplicate keys")
    path = OUT / f"llm_{g}.parquet"
    df.to_parquet(path, index=False)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    report[g] = {"rows": len(df), "file": path.name, "sha256": h, "by_pass": df["pass"].value_counts().to_dict(),
                 "llm_p_ge_0.5": int((df.llm_p >= .5).sum()), "columns": list(df.columns)}
json.dump(report, open(OUT / "assemble_report.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in report.items() if k not in ("shards",)}, indent=1, default=str))
print({k: v["scored_rows"] for k, v in report["shards"].items()})
