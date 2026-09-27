"""Apply the arena-trained string GBDT to test pairs; write French rows, a France-only upload and diagnostics.

    python -m ber.string_gbdt_apply --feat O/test_France --work W --base S1.tsv --probes P --out DIR
        [--threshold T --variant V]

Only French S1 rows are replaced in ``--base``; every other line is copied byte-for-byte.
No labels exist for test; all diagnostics are label-free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ber.string_gbdt import FEATURES, log
from ber.string_gbdt_train import GATE, decode, listwise, load_feat, veto_mask

HEADER = "source1_entity_id\tmatched_entity_ids"
TEST_ROWS = 1_732_544


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def score(feat, work, stage1_only=False):
    import lightgbm as lgb
    work = Path(work)
    pairs, X, s1, rec = load_feat(feat)
    left, right = pairs["left"].astype(np.int64), pairs["right"].astype(np.int64)
    m1 = [lgb.Booster(model_file=str(work / f"stage1_fold{f}.txt")) for f in (0, 1)]
    m2 = lgb.Booster(model_file=str(work / "stage2.txt"))
    p1 = np.empty(len(left), dtype=np.float32)
    for i in range(0, len(left), 4_000_000):
        x = X[i:i + 4_000_000]
        p1[i:i + 4_000_000] = np.mean([m.predict(x, num_threads=16) for m in m1], axis=0)
    core_eq = np.asarray(X[:, FEATURES.index("n_core_eq")])
    if stage1_only:
        log("stage-1 only: prob = p1")
        return pairs, X, s1, rec, left, right, p1, p1.copy(), core_eq
    Z = listwise(left, right, p1, core_eq, len(s1), len(rec))
    cand = np.flatnonzero((p1 >= GATE) | (Z[:, 1] < 3))
    prob = np.zeros(len(left), dtype=np.float32)
    for i in range(0, len(cand), 4_000_000):
        rr = cand[i:i + 4_000_000]
        prob[rr] = m2.predict(np.hstack([X[rr], Z[rr]]), num_threads=16)
    log(f"scored {len(left)} pairs, stage2 rows {len(cand)}")
    return pairs, X, s1, rec, left, right, p1, prob, core_eq


def read_links(path, keep_ids=None):
    """Set of (s1, rec) links from a matching file (optionally only S1s in keep_ids)."""
    out = set()
    with open(path) as f:
        next(f)
        for line in f:
            sid, _, recs = line.rstrip("\n").partition("\t")
            if keep_ids is not None and sid not in keep_ids:
                continue
            if recs:
                out.update((sid, r) for r in recs.split(","))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--probes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--country", default="France")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--variant")
    ap.add_argument("--tag", default="")
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--stage1-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((Path(args.work) / "report.json").read_text())
    best = report["stage2"]["a_tune_best"]
    thr = args.threshold if args.threshold is not None else best["threshold"]
    variant = args.variant or best["variant"]
    pairs, X, s1, rec, left, right, p1, prob, core_eq = score(args.feat, args.work, args.stage1_only)
    rec_empty = rec["empty"].to_numpy()[right]
    mult = s1["mult"].to_numpy()[left]
    acc = decode(prob, thr, variant, left, right, rec_empty, core_eq, mult, veto_mask(X))
    s1_ids = s1.entity_id.to_numpy()
    rec_ids = rec.entity_id.to_numpy()
    fr = s1.country.to_numpy() == args.country
    fr_ids = set(s1_ids[fr])
    guard_info = None
    if args.guard:
        # France-only guard on ADDITIONS vs the base file: keep the base decision (no link) when the new link
        # carries a decoy signature (house shift with S1 number absent, legal-form swap, or decoy word added).
        base_pairs = read_links(args.base, fr_ids)
        rows_acc = np.flatnonzero(acc & fr[left])
        is_new = np.fromiter(((s1_ids[l], rec_ids[r]) not in base_pairs for l, r in zip(left[rows_acc], right[rows_acc])),
                             dtype=bool, count=len(rows_acc))
        sig = (veto_mask(X, rows_acc) | (X[rows_acc, FEATURES.index("legal_rel")] == 4)
               | (X[rows_acc, FEATURES.index("n_added_decoy")] > 0))
        drop = rows_acc[is_new & sig]
        acc[drop] = False
        guard_info = {"added_candidates": int(is_new.sum()), "guarded_out": int(len(drop))}
        log(f"guard: {guard_info}")
    # pair table for others (Agent U)
    pd.DataFrame({"s1_id": s1_ids[left], "rec_id": rec_ids[right], "route": pairs["route"], "p1": p1, "prob": prob,
                  "accept": acc}).to_parquet(out / "pair_scores.parquet", index=False)
    # rows
    rows = np.flatnonzero(acc)
    order = rows[np.lexsort((-prob[rows], left[rows]))]
    lists = {}
    for l, r in zip(left[order], right[order]):
        lists.setdefault(l, []).append(rec_ids[r])
    fr_idx = np.flatnonzero(fr)
    fr_lines = {s1_ids[i]: s1_ids[i] + "\t" + ",".join(lists.get(i, [])) + "\n" for i in fr_idx}
    with open(out / "france_rows.tsv", "w", newline="\n") as f:
        f.write(HEADER + "\n")
        for i in fr_idx:
            f.write(fr_lines[s1_ids[i]])
    up = out / "upload_france_only"
    up.mkdir(exist_ok=True)
    seen, replaced, n = set(), 0, 0
    with open(args.base) as fin, open(up / "matching_results.tsv", "w", newline="\n") as fout:
        head = next(fin)
        assert head.rstrip("\n") == HEADER
        fout.write(head)
        for line in fin:
            sid = line.split("\t", 1)[0]
            assert sid not in seen
            seen.add(sid)
            n += 1
            if sid in fr_lines:
                fout.write(fr_lines[sid])
                replaced += 1
            else:
                fout.write(line)
    # validation
    owners, bad = {}, []
    with open(up / "matching_results.tsv", "rb") as f:
        data = f.read()
    assert b"\r" not in data
    lines = data.decode().split("\n")
    assert lines[0] == HEADER and lines[-1] == ""
    body = lines[1:-1]
    assert len(body) == TEST_ROWS, len(body)
    ids = [ln.split("\t")[0] for ln in body]
    assert len(set(ids)) == TEST_ROWS
    for ln in body:
        sid, recs = ln.split("\t")
        rs = recs.split(",") if recs else []
        if len(rs) > 11 or len(set(rs)) != len(rs):
            bad.append(sid)
        for r in rs:
            if r in owners:
                bad.append(sid)
            owners[r] = sid
    assert not bad, bad[:5]
    assert replaced == len(fr_idx), (replaced, len(fr_idx))
    log(f"upload validated: {TEST_ROWS} rows, {replaced} French rows replaced")
    # diagnostics
    base_links = read_links(args.base, fr_ids)
    key = pd.Series(np.arange(len(left)), index=pd.MultiIndex.from_arrays([s1_ids[left], rec_ids[right]]))
    new_links = {(s1_ids[l], rec_ids[r]) for l, r in zip(left[rows], right[rows]) if fr[l]}
    added, removed, kept = new_links - base_links, base_links - new_links, new_links & base_links
    hcol, dcol, lcol = FEATURES.index("h_rel"), FEATURES.index("n_added_decoy"), FEATURES.index("legal_rel")

    def sig(links):
        idx = key.reindex(list(links)).dropna().astype(np.int64).to_numpy()
        if not len(idx):
            return {"n": len(links), "in_candidates": 0}
        h = X[np.sort(idx)][:, [hcol, dcol, lcol]]
        shift = np.isin(h[:, 0], (7, 8, 9))
        return {"n": len(links), "in_candidates": int(len(idx)), "mean_prob": float(prob[idx].mean()),
                "share_prob_ge_0_5": float((prob[idx] >= 0.5).mean()),
                "house_shift_rate": float(shift.mean()), "gen_plus_delta_rate": float((h[:, 0] == 7).mean()),
                "decoy_word_rate": float((h[:, 1] > 0).mean()), "legal_swap_rate": float((h[:, 2] == 4).mean()),
                "shift_or_decoy_rate": float((shift | (h[:, 1] > 0)).mean())}

    k_new = np.bincount(left[rows], minlength=len(s1))[fr_idx]
    diag = {"threshold": thr, "variant": variant, "french_s1": int(len(fr_idx)), "guard": guard_info,
            "links_new": len(new_links), "links_base": len(base_links),
            "links_per_s1_new": float(k_new.mean()), "empty_rows_new": int((k_new == 0).sum()),
            "added": sig(added), "removed": sig(removed), "kept": sig(kept), "base_all": sig(base_links)}
    if args.probes:
        pr = Path(args.probes)
        probes = {}
        cp = json.loads((pr / "drop_Cprime/dropped.json").read_text())
        probes["drop_Cprime"] = {tuple(x) for x in cp if x[0] in fr_ids}
        for name in ("drop_T2r", "drop_T1r"):
            if (pr / name / "matching_results.tsv").exists():
                probes[name] = base_links - read_links(pr / name / "matching_results.tsv", fr_ids)
        diag["probes"] = {k: sig(v) for k, v in probes.items()}
    files = {"france_rows": str(out / "france_rows.tsv"), "upload": str(up / "matching_results.tsv")}
    diag["sha256"] = {k: sha256(v) for k, v in files.items()}
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=1) + "\n")
    log(json.dumps(diag, indent=1))


if __name__ == "__main__":
    main()
