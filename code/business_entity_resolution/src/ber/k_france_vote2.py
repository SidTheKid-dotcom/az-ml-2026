"""France score vote (label-free) over G's French pair universe, with probe-based Platt calibration.

Universe: every French pair G scored (``run27/G/<v>/pair_scores.parquet``: dense top-50 + routes). For each pair:
K3 edit-op features + K3 probability (computed here), S1-file membership (#5 rule), probe membership. Families:
G (already calibrated), optional C2-FR / Qwen-FR scores (parquet ``s1_id, rec_id|record_id, <score>``), each
recalibrated with a 2-parameter Platt fit on logit(score) so that probe A (T2r) averages 0.665 and probe C' 0.94
(pricing-model inferences; group labels, not pair labels). K3 is reported and usable as a veto.

Rule (per variant): a family accepts at calibrated p >= tau; keep links with >= 2 accepting families (of those
available; with 2 families both must accept); 1-vote links fall back to the S1 file; G override: G >= 0.9 with
equal core name and equal house number is kept; additions (not in the S1 file) are blocked when both addresses
carry a house number and they disagree (hn relation "other"/"extended", incl. delta and even shifts: twins/decoys);
optional acronym additions (record name = initials of the S1 core or vice versa, same house number or postcode,
no decoy/legal-swap/shift); optional K3 veto; one owner by mean calibrated probability; cap 11.

    python -m ber.k_france_vote2 prep --g DIR/pair_scores.parquet
    python -m ber.k_france_vote2 vote --out DIR [--c2fr P --c2fr-col score] [--qfr P --qfr-col score] ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import edit_ops as eo
from ber.k3_test import DATA, sha256
from ber.k_routes import k3_features, k3_prob

T0 = time.time()
CACHE = Path("/home/ubuntu/k/cache/fr_universe.parquet")
HEADER = "source1_entity_id\tmatched_entity_ids"
OPS = ["legal_rel", "hn_rel", "hn_diff", "hn_in_delta", "hn_even_shift", "decoy_added", "rec_addr_empty",
       "core_equal", "core_initials", "postal_eq", "comp_last_eq", "hn_letter_added", "s1_addr_empty",
       "core_n_s1", "core_n_rec"]


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def prep(a):
    g = pq.read_table(a.g, columns=["s1_id", "rec_id", "route", "prob"]).to_pandas()
    ref = pq.read_table(DATA / "test/reference.parquet", columns=["entity_id", "country"]).to_pandas()
    fr = set(ref.loc[ref.country == "France", "entity_id"])
    g = g[g.s1_id.isin(fr)].reset_index(drop=True)
    log(f"G French pairs: {len(g)}")
    s_names, s_addrs = eo.load_text(DATA / "test/reference.parquet")
    sidx = pd.Series(np.arange(len(ref)), index=ref.entity_id)
    li = sidx.reindex(g.s1_id).to_numpy()
    pool = pq.read_table(DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
    p_names, p_addrs = eo.load_text(DATA / "test/pool.parquet")
    pidx = pd.Series(np.arange(len(pool)), index=pool)
    ri = pidx.reindex(g.rec_id).to_numpy()
    rn = np.empty(len(g), dtype=object)
    ra_ = np.empty(len(g), dtype=object)
    ok = ~np.isnan(ri)
    rn[ok] = p_names[ri[ok].astype(np.int64)]
    ra_[ok] = p_addrs[ri[ok].astype(np.int64)]
    if (~ok).any():
        extra = pq.read_table(DATA / "u8src/test_pool.parquet", columns=["entity_id", "business_name",
                                                                          "business_address"]).to_pandas()
        extra = extra.drop_duplicates("entity_id").set_index("entity_id")
        miss = g.rec_id[~ok]
        e = extra.reindex(miss)
        rn[~ok] = e.business_name.to_numpy()
        ra_[~ok] = e.business_address.to_numpy()
        log(f"{int((~ok).sum())} pairs outside the dense pool; text missing for {int(e.business_name.isna().sum())}")
    rn = np.where(pd.isna(rn), "", rn)
    ra_ = np.where(pd.isna(ra_), "", ra_)
    li = li.astype(np.int64)
    feat = k3_features(s_names[li].tolist(), s_addrs[li].tolist(), rn.tolist(), ra_.tolist())
    rec_s3 = g.rec_id.str.startswith("S3").to_numpy()
    g["k3"] = k3_prob(feat, rec_s3)
    for n in OPS:
        g[n] = feat[:, eo.FEATURES.index(n)]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(CACHE)
    log(f"saved {CACHE}")


def isin(keys, values):
    """Hash-based membership for large string arrays (np.isin on object arrays is too slow)."""
    return pd.Series(keys).isin(set(values) if not isinstance(values, set) else values).to_numpy()


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(score, in_a, in_c, target_a=0.665, target_c=0.94):
    """(a, b) with mean sigmoid(a*logit(s)+b) = target on probe A and probe C'."""
    from scipy.optimize import least_squares

    za, zc = logit(score[in_a]), logit(score[in_c])
    f = lambda v: [np.mean(1 / (1 + np.exp(-(v[0] * za + v[1])))) - target_a,  # noqa: E731
                   np.mean(1 / (1 + np.exp(-(v[0] * zc + v[1])))) - target_c]
    sol = least_squares(f, x0=[1.0, 0.0], bounds=([0.01, -20], [20, 20]))
    return float(sol.x[0]), float(sol.x[1]), [float(x) for x in f(sol.x)]


def load_family(path, col, keys):
    t = pq.read_table(path).to_pandas()
    rc = "rec_id" if "rec_id" in t.columns else "record_id"
    s = pd.Series(t[col].to_numpy(dtype=np.float64), index=t["s1_id"] + "|" + t[rc])
    s = s[~s.index.duplicated()]
    return s.reindex(keys).to_numpy()


def vote(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(CACHE)
    keys = (u.s1_id + "|" + u.rec_id).to_numpy()
    ref = pq.read_table(DATA / "test/reference.parquet", columns=["entity_id", "country"]).to_pandas()
    fr_ids = ref.loc[ref.country == "France", "entity_id"].tolist()
    fr_set = set(fr_ids)
    base = []
    with open(DATA / "s1file/matching_results.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s, ids = line.rstrip("\n").split("\t")
            if ids and s in fr_set:
                base += [s + "|" + r for r in ids.split(",")]
    base = set(base)
    u["in_s1"] = isin(keys, list(base))
    probe = {}
    for name, fn in (("A", "probeA_T2r.tsv"), ("C", "probeCprime.tsv")):
        pr = pd.read_csv(DATA / "probes" / fn, sep="\t", dtype=str)
        probe[name] = isin(keys, (pr.iloc[:, 0] + "|" + pr.iloc[:, 1]).to_numpy())
    fams = {"G": u["prob"].to_numpy(dtype=np.float64)}
    calib = {"G": {"probe_A_mean": float(fams["G"][probe["A"]].mean()), "probe_C_mean": float(fams["G"][probe["C"]].mean()),
                   "note": "used as is (already calibrated)"}}
    for name, path, col in (("C2FR", a.c2fr, a.c2fr_col), ("QFR", a.qfr, a.qfr_col)):
        if not path:
            continue
        raw = load_family(path, col, keys)
        have = ~np.isnan(raw)
        ca, cb, resid = platt(raw, probe["A"] & have, probe["C"] & have)
        cal = np.where(have, 1 / (1 + np.exp(-(ca * logit(np.nan_to_num(raw, nan=0.5)) + cb))), np.nan)
        fams[name] = cal
        calib[name] = {"a": ca, "b": cb, "residuals": resid, "coverage": float(have.mean()),
                       "raw_probe_A_mean": float(np.nanmean(raw[probe["A"]])),
                       "raw_probe_C_mean": float(np.nanmean(raw[probe["C"]]))}
        log(f"{name} Platt a={ca:.3f} b={cb:.3f} coverage {have.mean():.3f}")
    for spec in a.rows_family or []:
        name, path = spec.split("=", 1)
        acc_keys = []
        with open(path, encoding="utf-8") as f:
            next(f)
            for line in f:
                s, ids = line.rstrip("\n").split("\t")
                if ids and s in fr_set:
                    acc_keys += [s + "|" + r for r in ids.split(",")]
        acc_keys = np.asarray(acc_keys, dtype=object)
        hit = isin(keys, acc_keys)
        fams[name] = hit.astype(np.float64)
        calib[name] = {"binary_rows_family": path, "links": int(len(acc_keys)),
                       "links_outside_universe": int(len(acc_keys) - hit.sum()),
                       "probe_A_accept": float(hit[probe["A"]].mean()), "probe_C_accept": float(hit[probe["C"]].mean())}
        log(f"rows family {name}: {len(acc_keys)} links, {int(hit.sum())} in universe")
    if a.kfr:  # C2-free K-France probability, .npy aligned to the French rows of U's test_prep order
        d = DATA / "test_prep"
        tl = np.load(d / "left.npy").astype(np.int64)
        tr_ = np.load(d / "right.npy").astype(np.int64)
        code = np.load(d / "country_code.npy")
        rows = np.flatnonzero(code[tl] == 0)
        kp = np.load(a.kfr)
        if len(kp) != len(rows):
            raise SystemExit("K-France probabilities do not match the French test_prep rows")
        s1_all = np.asarray(ref["entity_id"].tolist(), dtype=object)
        pool_ids = np.asarray(pq.read_table(DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id")
                              .to_pylist(), dtype=object)
        kk = pd.Series(kp.astype(np.float64), index=s1_all[tl[rows]] + "|" + pool_ids[tr_[rows]])
        kk = kk[~kk.index.duplicated()]
        fams["KFR"] = kk.reindex(keys).to_numpy()
        have = ~np.isnan(fams["KFR"])
        calib["KFR"] = {"coverage_of_universe": float(have.mean()),
                        "probe_A_mean": float(np.nanmean(fams["KFR"][probe["A"]])),
                        "probe_C_mean": float(np.nanmean(fams["KFR"][probe["C"]]))}
        log(f"KFR scores: coverage {have.mean():.3f}")
    if a.only:
        fams = {k: v for k, v in fams.items() if k in a.only.split(",")}
    k3 = u["k3"].to_numpy(dtype=np.float64)
    calib["K3"] = {"probe_A_mean": float(k3[probe["A"]].mean()), "probe_C_mean": float(k3[probe["C"]].mean())}
    if a.k3_family:
        fams["K3"] = k3
    names = list(fams)
    P = np.column_stack([fams[n] for n in names])
    mean_p = np.nanmean(P, axis=1)
    hn_rel = u["hn_rel"].to_numpy()
    hn_agree = np.isin(hn_rel, (3, 4, 5)) | np.isin(hn_rel, (0, 1, 2))  # equal/zero-pad/truncated or a side missing
    twin_block = ~hn_agree  # both numbers present and different (incl. delta/even shifts)
    clean = (u["decoy_added"].to_numpy() == 0) & (u["legal_rel"].to_numpy() != 5) & ~twin_block
    acronym = (u["core_initials"].to_numpy() >= 1) & ((hn_rel == 3) | (u["postal_eq"].to_numpy() == 1)) & clean
    gp = fams["G"] if "G" in fams else u["prob"].to_numpy(dtype=np.float64)
    g_override = (gp >= 0.9) & (u["core_equal"].to_numpy() == 1) & (hn_rel == 3) & (not a.no_g_override)
    estimates = {"sum_G_prob_all_pairs": float(np.nansum(gp)),
                 "sum_mean_family_prob": float(np.nansum(mean_p)),
                 "sum_K3_prob_dense": float(np.nansum(k3)), "s1_file_links": len(base)}
    in_s1 = u["in_s1"].to_numpy()
    variants = {}
    for tau in [float(t) for t in a.taus.split(",")]:
        acc = np.where(np.isnan(P), 0, P >= tau).sum(axis=1)
        avail = (~np.isnan(P)).sum(axis=1)
        need = np.where(avail >= 3, 2, avail) if a.need is None else np.minimum(a.need, avail)
        keep = (acc >= need) & (avail > 0)
        disputed = (acc >= 1) & ~keep
        if a.g_resolve:  # disagreements: G decides when confident, else the S1 file (#5 rule)
            g = gp
            keep |= disputed & (g >= a.g_hi)
            keep |= disputed & (g > a.g_lo) & (g < a.g_hi) & in_s1
        else:
            keep |= disputed & in_s1
        keep |= g_override
        keep &= ~(~in_s1 & twin_block)  # no additions across twins / shifted numbers
        variants[f"vote_t{tau}"] = keep.copy()
        variants[f"vote_t{tau}_acr"] = keep | (acronym & ~in_s1 & (mean_p >= 0.5))
        if a.veto is not None:
            variants[f"vote_t{tau}_veto"] = keep & ~(k3 < a.veto)
            variants[f"vote_t{tau}_acr_veto"] = variants[f"vote_t{tau}_acr"] & ~(k3 < a.veto)
    variants["s1_file_veto"] = in_s1 & ~(k3 < (a.veto if a.veto is not None else 0.05))
    s1_of = u.s1_id.to_numpy()
    rec_of = u.rec_id.to_numpy()

    def finalize(keep):
        idx = np.flatnonzero(keep)
        d = pd.DataFrame({"s1": s1_of[idx], "rec": rec_of[idx], "p": mean_p[idx], "in_s1": in_s1[idx]})
        d = d.sort_values(["rec", "p"], ascending=[True, False]).drop_duplicates("rec")
        d = d.sort_values(["s1", "in_s1", "p"], ascending=[True, False, False])
        d = d[d.groupby("s1").cumcount() < 11]
        return d

    def rates(mask):
        f = u[mask]
        if not len(f):
            return {"links": 0}
        return {"links": int(len(f)), "hn_delta_shift": float((f.hn_in_delta == 1).mean()),
                "hn_even_shift": float((f.hn_even_shift == 1).mean()), "decoy_word_added": float((f.decoy_added > 0).mean()),
                "legal_swapped": float((f.legal_rel == 5).mean()), "legal_added_or_dropped": float(f.legal_rel.isin([3, 4]).mean()),
                "record_empty_address": float((f.rec_addr_empty == 1).mean()), "route": float(f.route.mean()),
                "mean_G": float(f.prob.mean()), "mean_K3": float(f.k3.mean())}

    report = {"families": names, "calibration": calib, "estimates": estimates,
              "acronym_candidates_not_in_s1": int((acronym & ~in_s1).sum()),
              "acronym_candidates_not_in_s1_mean_G": float(gp[acronym & ~in_s1].mean()) if (acronym & ~in_s1).any() else None,
              "g_override_not_in_s1": int((g_override & ~in_s1).sum()),
              "twin_blocked_additions_any_family_ge_0.5": int((twin_block & ~in_s1 & (np.nanmax(P, axis=1) >= 0.5)).sum()),
              "variants": {}}
    finals = {}
    for name, keep in variants.items():
        d = finalize(keep)
        finals[name] = d
        kk = isin(keys, (d.s1 + "|" + d.rec).to_numpy())
        lens = d.groupby("s1").size().reindex(fr_ids, fill_value=0).to_numpy()
        report["variants"][name] = {"links": int(len(d)), "added": rates(kk & ~in_s1), "removed": rates(in_s1 & ~kk),
                                    "probe_A_kept": float(kk[probe["A"]].mean()), "probe_C_kept": float(kk[probe["C"]].mean()),
                                    "mean_list": float(lens.mean()), "empty_rows": int((lens == 0).sum()),
                                    "s1_file_links_outside_universe": int(len(base - set(keys)))}
        log(f"{name}: {json.dumps(report['variants'][name], default=float)[:400]}")
    files = {}
    for name in [x for x in a.write.split(",") if x]:
        d = finals[name]
        lists = d.groupby("s1")["rec"].apply(list).to_dict()
        path = out / f"france_rows_{name}.tsv"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(HEADER + "\n")
            for s in fr_ids:
                f.write(f"{s}\t{','.join(lists.get(s, []))}\n")
        files[name] = {"path": str(path), "sha256": sha256(path)}
    report["files"] = files
    report["labels_read"] = False
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    log("done")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prep", "vote"])
    p.add_argument("--g", default="/home/ubuntu/k/data/gfr/g_v1_pair_scores.parquet")
    p.add_argument("--out")
    p.add_argument("--c2fr")
    p.add_argument("--c2fr-col", default="score")
    p.add_argument("--qfr")
    p.add_argument("--qfr-col", default="score")
    p.add_argument("--taus", default="0.7,0.8,0.9")
    p.add_argument("--rows-family", action="append", help="name=rows.tsv (binary accept family)")
    p.add_argument("--k3-family", action="store_true", help="use K3 probability as a voting family")
    p.add_argument("--g-resolve", action="store_true", help="disputed links: keep if G>=g_hi, drop if G<=g_lo, else S1 file")
    p.add_argument("--g-hi", type=float, default=0.9)
    p.add_argument("--kfr", help="K-France probabilities .npy (French test_prep rows)")
    p.add_argument("--only", help="comma list of families to keep (e.g. KFR)")
    p.add_argument("--need", type=int, help="votes needed (default: 2 of >=3, else all)")
    p.add_argument("--no-g-override", action="store_true")
    p.add_argument("--g-lo", type=float, default=0.1)
    p.add_argument("--veto", type=float, default=0.05)
    p.add_argument("--write", default="")
    a = p.parse_args()
    (prep if a.command == "prep" else vote)(a)


if __name__ == "__main__":
    main()
