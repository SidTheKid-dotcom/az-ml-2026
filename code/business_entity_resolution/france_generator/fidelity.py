"""Fidelity proxy: synthetic US/India worlds vs real train pairs (arena partition A only).

For each country: sample N partition-A S1s (seed 7), build a synthetic world from them with
synth_world, and compare on simple pair features (pairfeat.py):
  * copies:      real true pairs (arena rec_owner == S1) vs synthetic copies;
  * look-alikes: records NOT owned by the S1 and anchored to it by the same rule on both sides:
      - core:    identical core name (legal forms removed) to an S1 whose core name is unique
                 among all arena S1s of the country;
      - address: identical sorted address digit runs to an S1 unique by that key, address char-3gram
                 cosine >= 0.5, different core name.
    Real side: arena pool records. Synthetic side: the synthetic pool (copies + decoys).
Labels: only arena labels (labels_summary.holdout2_rows_read == 0) for partition-A S1s are used.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_world as sw  # noqa: E402
from pairfeat import core_key, digit_key, pair_features, summarize, cosine, grams, norm  # noqa: E402


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def anchored(s1_core_unique, s1_digit_unique, s1_core_of, names, addrs, owners, ids_country_mask=None):
    """Return list of (s1_pos, rec_pos, anchor_type) for records anchored to a unique S1 and not owned by it."""
    out = []
    for i, (n, a, o) in enumerate(zip(names, addrs, owners)):
        ck = core_key(n)
        s = s1_core_unique.get(ck) if ck else None
        if s is not None:
            if o != s:
                out.append((s, i, "core"))
            continue
        dk = digit_key(a)
        s = s1_digit_unique.get(dk) if dk else None
        if s is not None and o != s and ck != s1_core_of[s]:
            out.append((s, i, "address"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", default="/workspace/az-ml-new/arena")
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--countries", default="US,India")
    a = ap.parse_args()
    arena, out = Path(a.arena), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((arena / "labels/labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise SystemExit("labels_summary must have holdout2_rows_read == 0")
    params = json.loads(Path(a.params).read_text())
    ref = pd.read_parquet(arena / "v1/reference.parquet",
                          columns=["entity_id", "business_name", "business_address", "country"])
    part = np.load(arena / "finalize_v1/s1_partition.npy")
    assert len(part) == len(ref)
    owner = np.load(arena / "labels/rec_owner.npy")
    pool = pd.read_parquet(arena / "v1/pool.parquet",
                           columns=["entity_id", "business_name", "business_address", "country"])
    assert len(owner) == len(pool)
    log(f"loaded ref {len(ref):,} pool {len(pool):,}")
    report = {"seed": a.seed, "n_per_country": a.n, "partition": "A (0) only", "countries": {}}
    for country in a.countries.split(","):
        rng = np.random.default_rng(a.seed)
        cmask = (ref.country == country).to_numpy()
        cand = np.flatnonzero(cmask & (part == 0))
        samp = np.sort(rng.choice(cand, a.n, replace=False))
        s1 = ref.iloc[samp].reset_index(drop=True)
        pos_of = {int(g): i for i, g in enumerate(samp)}
        # Uniqueness of anchor keys among ALL arena S1s of the country (labels not needed).
        cidx = np.flatnonzero(cmask)
        core_all = [core_key(x) for x in ref.business_name.to_numpy()[cidx]]
        dig_all = [digit_key(x) for x in ref.business_address.to_numpy()[cidx]]
        core_cnt, dig_cnt = Counter(core_all), Counter(dig_all)
        s1_core = [core_key(x) for x in s1.business_name]
        s1_dig = [digit_key(x) for x in s1.business_address]
        core_u = {k: i for i, k in enumerate(s1_core) if k and core_cnt[k] == 1}
        dig_u = {k: i for i, k in enumerate(s1_dig) if k and dig_cnt[k] == 1}
        log(f"{country}: sample {len(s1):,}; unique-core {len(core_u):,} unique-digits {len(dig_u):,}")

        # ---- real side
        pmask = (pool.country == country).to_numpy()
        pidx = np.flatnonzero(pmask)
        own_local = np.array([pos_of.get(int(o), -1) for o in owner[pidx]], dtype=np.int64)
        real_copy = [(int(s), int(p)) for s, p in zip(own_local, pidx) if s >= 0]
        pn = pool.business_name.to_numpy()
        pa_ = pool.business_address.to_numpy()
        pe = pool.entity_id.to_numpy()
        real_look = anchored(core_u, dig_u, s1_core, pn[pidx], pa_[pidx], own_local)
        real_look = [(s, int(pidx[i]), t) for s, i, t in real_look]
        log(f"{country}: real copies {len(real_copy):,}, real anchored look-alikes {len(real_look):,}")

        # ---- synthetic side
        pools, gt = sw.build(s1, params, a.seed)
        syn = pd.concat([pools["S2"], pools["S3"]], ignore_index=True)
        anchor_pos = syn._anchor.map({e: i for i, e in enumerate(s1.entity_id)}).to_numpy()
        is_copy = (syn._kind == "copy").to_numpy()
        syn_owner = np.where(is_copy, anchor_pos, -1)
        syn_look = anchored(core_u, dig_u, s1_core, syn.business_name.to_numpy(),
                            syn.business_address.to_numpy(), syn_owner)
        log(f"{country}: synthetic records {len(syn):,}, anchored look-alikes {len(syn_look):,}")

        sn, sa_, se = s1.business_name.to_numpy(), s1.business_address.to_numpy(), s1.entity_id.to_numpy()

        def feats(pairs, names, addrs, ids):
            rows = []
            for s, r in pairs:
                f = pair_features(sn[s], sa_[s], names[r], addrs[r])
                f["source"] = str(ids[r])[:2]
                rows.append(f)
            return rows

        res = {}
        rc = feats(real_copy, pn, pa_, pe)
        sc = feats([(int(syn_owner[i]), i) for i in np.flatnonzero(is_copy)],
                   syn.business_name.to_numpy(), syn.business_address.to_numpy(), syn.entity_id.to_numpy())
        res["copies"] = {"real": summarize(rc), "synthetic": summarize(sc)}
        for src in ("S2", "S3"):
            res[f"copies_{src}"] = {"real": summarize([r for r in rc if r["source"] == src]),
                                    "synthetic": summarize([r for r in sc if r["source"] == src])}
        kinds = syn._kind.to_numpy()
        for t in ("core", "address"):
            rl = feats([(s, r) for s, r, tt in real_look if tt == t], pn, pa_, pe)
            sl_pairs = [(s, r) for s, r, tt in syn_look if tt == t]
            sl = feats(sl_pairs, syn.business_name.to_numpy(), syn.business_address.to_numpy(),
                       syn.entity_id.to_numpy())
            res[f"lookalike_{t}"] = {"real": summarize(rl), "synthetic": summarize(sl),
                                     "synthetic_kind_mix": dict(Counter(kinds[r] for _, r in sl_pairs)),
                                     "real_per_anchor_s1": len(rl) / max(1, len(core_u if t == "core" else dig_u)),
                                     "synthetic_per_anchor_s1": len(sl) / max(1, len(core_u if t == "core" else dig_u))}
        # all synthetic decoys by kind, with their known anchor (no selection)
        for k in ("decoy_A", "decoy_B", "decoy_C"):
            idx = np.flatnonzero(kinds == k)
            dk = feats([(int(anchor_pos[i]), i) for i in idx], syn.business_name.to_numpy(),
                       syn.business_address.to_numpy(), syn.entity_id.to_numpy())
            res[f"synthetic_{k}_all"] = {"synthetic": summarize(dk)}
        res["counts"] = {"real_true_per_s1": len(real_copy) / len(s1),
                         "synthetic_true_per_s1": int(is_copy.sum()) / len(s1),
                         "synthetic_decoys_per_s1": int((~is_copy).sum()) / len(s1)}
        report["countries"][country] = res
    (out / "fidelity.json").write_text(json.dumps(report, indent=1))
    # markdown table
    lines = []
    for country, res in report["countries"].items():
        for block in ("copies", "copies_S2", "copies_S3", "lookalike_core", "lookalike_address"):
            r, s = res[block]["real"], res[block]["synthetic"]
            lines.append(f"\n### {country} {block}  (real n={r.get('n')}, synthetic n={s.get('n')})\n")
            lines.append("| feature | real | synthetic | gap (syn-real) |\n|---|---:|---:|---:|")
            for k in r:
                if k == "n" or k not in s:
                    continue
                lines.append(f"| {k} | {r[k]:.3f} | {s[k]:.3f} | {s[k] - r[k]:+.3f} |")
            if block.startswith("lookalike"):
                lines.append(f"\nper anchor S1: real {res[block]['real_per_anchor_s1']:.3f}, synthetic "
                             f"{res[block]['synthetic_per_anchor_s1']:.3f}; synthetic kinds "
                             f"{res[block]['synthetic_kind_mix']}")
        lines.append(f"\ncounts: {res['counts']}")
    (out / "fidelity.md").write_text("\n".join(lines) + "\n")
    log("done")


if __name__ == "__main__":
    main()
