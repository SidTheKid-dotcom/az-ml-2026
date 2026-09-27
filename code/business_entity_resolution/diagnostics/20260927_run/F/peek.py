"""Agent F: print sample added links (F1 vs old) with names/addresses/scores, split by shift signature. No labels."""
import re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from fcommon import OUT, load, plain_of
d = load(need_accept=False)
left, right, c2 = d["left"], d["right"], d["c2"]
st = np.load(OUT / "state_old.npz"); pl_old = st["plain"]
rows = np.load(OUT / "g2/rows.npy"); n = int(np.load(OUT / "g2/n_changed.npy")[0]); sc = np.load(OUT / "g2/scores_step040000/scores.npy")
c2n = c2.copy(); c2n[rows[:n]] = sc[:n]
pp, pl = plain_of(d, c2n)
add = np.flatnonzero(pl & ~pl_old & d["france"])
rn = d["ref_table"].column("business_name").to_pylist(); ra = d["ref_table"].column("business_address").to_pylist()
pn = d["pool_table"].column("business_name").to_pylist(); pa = d["pool_table"].column("business_address").to_pylist()
def h(a):
    m = re.search(r"\d+", a or ""); return int(m.group()) if m else None
rng = np.random.default_rng(1)
sh = np.array([(h(pa[right[r]]) or -99) - (h(ra[left[r]]) or -999) in {1,3,5,7,9,11,13,21} for r in add])
same = np.array([h(pa[right[r]]) is not None and h(pa[right[r]]) == h(ra[left[r]]) for r in add])
print("added", len(add), "shift", int(sh.sum()), "same_first_number", int(same.sum()))
for lab, m in (("SHIFT", sh), ("NO-SHIFT", ~sh)):
    for r in rng.choice(add[m], 8, replace=False):
        i, j = left[r], right[r]
        print(lab, f"c2 {c2[r]:.3f}->{c2n[r]:.3f} tree {d['base'][r]:.2f} | {rn[i]} | {ra[i]} || {pn[j]} | {pa[j]}")
