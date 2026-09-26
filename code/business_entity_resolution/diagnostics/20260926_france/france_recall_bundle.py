"""France recall bundle: base file + (a) French legal-form rule adds from F1 whose record is unused in the base,
+ (b) Track R French new pairs (top-3 per S1) with C2 >= THRESH whose record is unused, highest C2 first,
lists capped at 11. French rows only; US/India rows are copied byte-for-byte. No labels are read."""
import json, sys, hashlib
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
base_path, out_dir, thresh = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
use_f1 = len(sys.argv) > 4 and sys.argv[4] == "with_f1"
E = Path("/workspace/az-ml-new/eureka"); D = E / "retrieval/test"
ref = pq.read_table(D / "reference.parquet", columns=["entity_id", "country"]).to_pandas()
pool_ids = np.asarray(pq.read_table(D / "pool.parquet", columns=["entity_id"]).column("entity_id").to_pylist(), dtype=object)
fr = set(ref.entity_id[ref.country == "France"])
def load(p):
    lines = Path(p).read_text().split("\n"); head = lines[0]
    rows = [l.split("\t") for l in lines[1:] if l]
    return head, rows
head, rows = load(base_path)
lists = {s: (ids.split(",") if ids else []) for s, ids in rows}
used = {r for ids in lists.values() for r in ids}
added = {"f1": 0, "new": 0}
if use_f1:
    _, f1 = load("/workspace/az-ml-new/france_legal/f1/matching_results.tsv")
    s1f = dict((s, ids.split(",") if ids else []) for s, ids in load("/opt/az-ml-new/runs/s1_pair_tree_40k/model_us_india_c2_france/matching_results.tsv")[1])
    for s, ids in f1:
        if s not in fr: continue
        for r in (ids.split(",") if ids else []):
            if r not in set(s1f[s]) and r not in used and len(lists[s]) < 11:
                lists[s].append(r); used.add(r); added["f1"] += 1
l = np.load(E / "fr_new/left.npy"); r = np.load(E / "fr_new/right.npy"); sc = np.load(E / "fr_new/c2/scores.npy")
order = np.argsort(-sc)
for i in order:
    if sc[i] < thresh: break
    s, rid = ref.entity_id[l[i]], pool_ids[r[i]]
    if s not in fr or rid in used or len(lists[s]) >= 11: continue
    lists[s].append(rid); used.add(rid); added["new"] += 1
out_dir.mkdir(parents=True, exist_ok=True); out = out_dir / "matching_results.tsv"
with open(out, "w", encoding="utf-8", newline="\n") as h:
    h.write(head + "\n")
    for s, _ in rows: h.write(f"{s}\t{','.join(lists[s])}\n")
allr = [x for ids in lists.values() for x in ids]
rep = {"base": str(base_path), "threshold": thresh, "with_f1": use_f1, "added": added, "rows": len(rows),
       "contested": len(allr) - len(set(allr)), "longest": max(map(len, lists.values())),
       "french_links": sum(len(lists[s]) for s in lists if s in fr), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
(out_dir / "report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep))
