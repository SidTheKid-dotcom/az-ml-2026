"""Upload-2 probe: the S1 public file (0.978287) minus French accepted links in the template-swap class
(same address digits, one core name word replaced by a different common French template word; see template_swap.py).
French rows only; US/India rows are byte-identical. No labels are read."""
import hashlib, json
from pathlib import Path
E = Path("/workspace/az-ml-new/eureka")
swaps = set(map(tuple, json.loads((E / "template_swap_fr.json").read_text())))
src = Path("/opt/az-ml-new/runs/s1_pair_tree_40k/model_us_india_c2_france/matching_results.tsv").read_text(encoding="utf-8").split("\n")
out, dropped, rows_changed = [src[0]], 0, 0
for line in src[1:]:
    if not line:
        continue
    s, ids = line.split("\t")
    lst = ids.split(",") if ids else []
    keep = [r for r in lst if (s, r) not in swaps]
    dropped += len(lst) - len(keep)
    rows_changed += len(keep) != len(lst)
    out.append(s + "\t" + ",".join(keep))
p = E / "drop_template_swap" / "matching_results.tsv"
p.parent.mkdir(parents=True, exist_ok=True)
with open(p, "w", encoding="utf-8", newline="\n") as h:
    h.write("\n".join(out) + "\n")
print(json.dumps({"swap_pairs": len(swaps), "dropped": dropped, "rows_changed": rows_changed,
                  "rows": len(out) - 1, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}))
