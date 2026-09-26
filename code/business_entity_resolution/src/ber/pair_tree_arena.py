"""Partition-A CatBoost on v1 scores plus name, address-kind and exact-key features.

Question: can a pair model beat the empty-address name-only rule on B_REST by
at least +0.0005? The model is fit on 80% of partition-A claim components.
Threshold and variant (model alone, or model followed by name-only) are chosen
on the other 20% of A. B_REST is scored once against name-only. SLICE is not
read. HOLDOUT-2 is refused. ``ber.pair_tree_test`` applies the saved model to
test with the same :func:`build_pairs`.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.arena_decision import component_bootstrap, evaluate, name_only_rule, one_owner
from ber.forensic_features import EMPTY, address_kinds, exact_keys, s1_multiplicities
from ber.name_only import name_multiplicity, norm_name

ART = Path(os.environ.get("ARENA", "/opt/az-ml-new/artifacts/arena"))
INDEX = Path(os.environ.get("WAVE_INDEX", "/opt/az-ml-new/runs/cpu_wave"))
OUT = Path(os.environ.get("TREE_OUT", "/opt/az-ml-new/runs/pair_tree_arena"))
THREADS = int(os.environ.get("TREE_THREADS", "28"))
FIT_ROWS = int(os.environ.get("TREE_FIT_ROWS", "3000000"))
THRESHOLDS = tuple(np.round(np.arange(0.30, 0.951, 0.025), 3))
SEED = 0
FEATURES = ["blend", "base", "blend_rank_in_s1", "blend_gap_to_s1_top", "s1_candidates_ge_0_5",
            "rival_blend", "record_claimants_ge_0_5", "name_exact", "log1p_name_mult", "record_kind",
            "key_self", "key_other"]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def slim(result):
    return {key: result[key] for key in ("macro_f05", "precision", "recall", "false_links", "s1", "singleton_accuracy")}


def rank_within(group, score):
    """0-based rank of score (descending) inside each group; ties by row order."""
    order = np.lexsort((np.arange(len(score)), -score, group))
    g = group[order]
    start = np.ones(len(g), dtype=bool)
    start[1:] = g[1:] != g[:-1]
    idx = np.arange(len(g))
    first = np.maximum.accumulate(np.where(start, idx, 0))
    rank = np.empty(len(g), dtype=np.int32)
    rank[order] = (idx - first).astype(np.int32)
    return rank


def top_two(group, score, n_groups):
    order = np.lexsort((-score, group))
    g, s = group[order], score[order]
    start = np.ones(len(g), dtype=bool)
    start[1:] = g[1:] != g[:-1]
    top1 = np.full(n_groups, -1.0, dtype=np.float32)
    top2 = np.full(n_groups, -1.0, dtype=np.float32)
    top1[g[start]] = s[start]
    second = np.zeros(len(g), dtype=bool)
    second[1:] = start[:-1] & ~start[1:]
    top2[g[second]] = s[second]
    return top1, top2


def decide(prob, left, right, threshold, n_s1):
    """Threshold, one owner per record, then at most 11 links per S1 by probability."""
    accept = one_owner(prob >= threshold, left, right, prob)
    rows = np.flatnonzero(accept)
    if rows.size:
        rank = rank_within(left[rows], prob[rows])
        accept[rows[rank >= 11]] = False
    return accept


def false_merge(accept, left, population, true_count):
    no_match = population & (true_count == 0)
    k = np.bincount(left[accept], minlength=len(true_count))
    hit = no_match & (k > 0)
    return {"no_match_s1": int(no_match.sum()), "false_merges": int(hit.sum()),
            "rate": float(hit.sum() / no_match.sum()) if no_match.any() else None}


def build_pairs(ref, pool, left, right, blend, base):
    """Candidate rows and the feature builder. ``ref``/``pool`` need name, address, country."""
    n_s1, n_rec = len(ref), len(pool)
    log(f"{n_s1} S1s, {n_rec} records; exact keys")
    mult = s1_multiplicities(ref.business_name, ref.business_address, ref.country)
    s1_key = np.array([f"{c}|{k}" if k else "" for c, k in zip(ref.country.to_numpy(), mult.exact_key.to_numpy())],
                      dtype=object)
    unique_key = (mult.key_count.to_numpy() == 1) & (s1_key != "")
    rec_key = np.empty(n_rec, dtype=object)
    for code, sub in pool.groupby("country", sort=False):
        keys = exact_keys(sub.business_name, sub.business_address).astype(object)
        rec_key[sub.index.to_numpy()] = np.array([f"{code}|{k}" if k else "" for k in keys], dtype=object)
    codes, _ = pd.factorize(np.concatenate([s1_key, rec_key]), sort=False)
    lookup = np.full(int(codes.max()) + 1, -1, np.int32)
    lookup[codes[:n_s1][unique_key]] = np.flatnonzero(unique_key).astype(np.int32)
    key_owner = lookup[codes[n_s1:]]
    key_owner[rec_key == ""] = -1
    del s1_key, rec_key, codes, lookup, mult

    log("normalized names and address kinds")
    s1_name = ref.business_name.map(norm_name).to_numpy()
    rec_name = pool.business_name.map(norm_name).to_numpy()
    s1_mult = name_multiplicity(ref.country, s1_name)
    both, _ = pd.factorize(np.concatenate([s1_name, rec_name]), sort=False)
    both = both.astype(np.int32, copy=False)
    name_s1, name_rec = both[:n_s1], both[n_s1:]
    name_s1[s1_name == ""] = -1
    name_rec[rec_name == ""] = -1
    kind = address_kinds(pool.business_address)
    del s1_name, rec_name, both

    log("pair features")
    name_exact = (name_s1[left] == name_rec[right]) & (name_s1[left] >= 0)
    rec_kind = kind[right]
    key_self = key_owner[right] == left
    key_other = (key_owner[right] >= 0) & ~key_self
    s1_top, _ = top_two(left, blend, n_s1)
    rec_top1, rec_top2 = top_two(right, blend, n_rec)
    rival = np.where(blend >= rec_top1[right], rec_top2[right], rec_top1[right]).astype(np.float32)
    strong = blend >= 0.5
    s1_strong = np.bincount(left[strong], minlength=n_s1).astype(np.float32)
    rec_strong = np.bincount(right[strong], minlength=n_rec).astype(np.float32)

    candidate = (base >= 0.01) | name_exact | key_self
    log(f"candidate pairs {int(candidate.sum())} of {len(left)}")
    cand_rows = np.flatnonzero(candidate)
    rank = np.full(len(left), 99, dtype=np.int16)
    rank[cand_rows] = np.minimum(rank_within(left[cand_rows], blend[cand_rows]), 99).astype(np.int16)

    def features(rows):
        l, r = left[rows], right[rows]
        return np.column_stack([
            blend[rows], base[rows], rank[rows].astype(np.float32), blend[rows] - s1_top[l], s1_strong[l],
            rival[rows], rec_strong[r], name_exact[rows].astype(np.float32),
            np.log1p(s1_mult[l]).astype(np.float32), rec_kind[rows].astype(np.float32),
            key_self[rows].astype(np.float32), key_other[rows].astype(np.float32),
        ]).astype(np.float32)

    return SimpleNamespace(cand_rows=cand_rows, features=features, name_exact=name_exact,
                           rec_kind=rec_kind, s1_mult=s1_mult)


def predict(model, pairs, n_pairs):
    prob = np.zeros(n_pairs, dtype=np.float32)
    for start in range(0, len(pairs.cand_rows), 5_000_000):
        chunk = pairs.cand_rows[start:start + 5_000_000]
        prob[chunk] = model.predict_proba(pairs.features(chunk))[:, 1].astype(np.float32)
    return prob


def load_scores(path):
    return np.nan_to_num(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32), nan=0.0)


def main() -> None:
    summary = json.loads((ART / "labels" / "labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise SystemExit("refusing HOLDOUT-2 labels")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "report.json").exists():
        raise SystemExit(f"{OUT} already has a report")

    columns = ["business_name", "business_address", "country"]
    ref = pq.read_table(ART / "v1" / "reference.parquet", columns=columns).to_pandas()
    pool = pq.read_table(ART / "v1" / "pool.parquet", columns=columns).to_pandas()
    n_s1, n_rec = len(ref), len(pool)
    left = np.asarray(np.load(INDEX / "left.npy", mmap_mode="r")).astype(np.int64)
    right = np.asarray(np.load(INDEX / "right.npy", mmap_mode="r")).astype(np.int64)
    blend = load_scores(ART / "v1" / "probabilities.npy")
    base = load_scores(ART / "v1" / "base_probabilities.npy")
    pairs = build_pairs(ref, pool, left, right, blend, base)
    del ref, pool
    cand_rows, name_exact, rec_kind, s1_mult = pairs.cand_rows, pairs.name_exact, pairs.rec_kind, pairs.s1_mult
    label = np.asarray(np.load(ART / "labels" / "label.npy", mmap_mode="r"))
    true_count = np.load(ART / "labels" / "true_count.npy")
    partition = np.load(ART / "finalize_v1" / "s1_partition.npy")
    component = np.load(ART / "finalize_v1" / "component_ids.npy")

    _, comp = np.unique(component, return_inverse=True)
    rng = np.random.default_rng(SEED)
    comp_is_a = np.zeros(int(comp.max()) + 1, dtype=bool)
    comp_is_a[comp[partition == 0]] = True
    a_comps = np.flatnonzero(comp_is_a)
    tune_comp = np.zeros_like(comp_is_a)
    tune_comp[rng.choice(a_comps, size=int(round(0.2 * len(a_comps))), replace=False)] = True
    s1_fit = (partition == 0) & ~tune_comp[comp]
    s1_tune = (partition == 0) & tune_comp[comp]
    s1_brest = partition == 1
    fit_rows = cand_rows[s1_fit[left[cand_rows]]]
    if len(fit_rows) > FIT_ROWS:
        fit_rows = np.sort(rng.choice(fit_rows, size=FIT_ROWS, replace=False))
    log(f"A fit S1 {int(s1_fit.sum())}, A tune S1 {int(s1_tune.sum())}, fit rows {len(fit_rows)}")

    from catboost import CatBoostClassifier

    model = CatBoostClassifier(iterations=800, depth=8, learning_rate=0.08, loss_function="Logloss",
                               thread_count=THREADS, random_seed=SEED, verbose=100, allow_writing_files=False)
    model.fit(pairs.features(fit_rows), label[fit_rows].astype(np.int8))
    model.save_model(str(OUT / "model.cbm"))
    importance = dict(zip(FEATURES, map(float, model.get_feature_importance())))
    log(f"importance {importance}")
    prob = predict(model, pairs, len(left))
    log("predicted all candidate pairs")

    base_accept = one_owner(blend >= 0.775, left, right, blend)
    name_only, _ = name_only_rule(base_accept, left, right, rec_kind == EMPTY, name_exact, s1_mult[left])
    tune_base = evaluate(name_only, left, label, true_count, population=s1_tune)["macro_f05"]
    sweep = []
    best = None
    for threshold in THRESHOLDS:
        accept = decide(prob, left, right, float(threshold), n_s1)
        stacked, _ = name_only_rule(accept, left, right, rec_kind == EMPTY, name_exact, s1_mult[left])
        for variant, mask in (("model", accept), ("model_then_name_only", stacked)):
            score = evaluate(mask, left, label, true_count, population=s1_tune)["macro_f05"]
            sweep.append({"threshold": float(threshold), "variant": variant, "a_tune_f05": score})
            if best is None or score > best["a_tune_f05"]:
                best = {"threshold": float(threshold), "variant": variant, "a_tune_f05": score}
    log(f"A-tune name-only {tune_base:.6f}; chosen {best}")

    chosen = decide(prob, left, right, best["threshold"], n_s1)
    if best["variant"] == "model_then_name_only":
        chosen, _ = name_only_rule(chosen, left, right, rec_kind == EMPTY, name_exact, s1_mult[left])
    before = evaluate(name_only, left, label, true_count, population=s1_brest, return_per_s1=True)
    after = evaluate(chosen, left, label, true_count, population=s1_brest, return_per_s1=True)
    paired = component_bootstrap(before["per_s1"], after["per_s1"], component, n=300, seed=0, population=s1_brest)
    fm_before = false_merge(name_only, left, s1_brest, true_count)
    fm_after = false_merge(chosen, left, s1_brest, true_count)
    gate = (paired["delta"] >= 0.0005 and paired["ci_low"] > 0
            and fm_after["rate"] - fm_before["rate"] <= 0.002)
    lengths = np.bincount(left[chosen], minlength=n_s1)
    report = {
        "question": "partition-A pair CatBoost vs empty-address name-only on B_REST",
        "features": FEATURES,
        "feature_importance": importance,
        "fit": {"rows": int(len(fit_rows)), "positives": int(label[fit_rows].sum()), "a_fit_s1": int(s1_fit.sum()),
                "a_tune_s1": int(s1_tune.sum()), "iterations": 800, "depth": 8, "learning_rate": 0.08,
                "candidate_rule": "base >= 0.01 or exact normalized name or exact-key self", "seed": SEED},
        "a_tune_name_only_f05": tune_base,
        "a_tune_sweep": sweep,
        "chosen_on_a_tune": best,
        "b_rest_name_only": slim(before),
        "b_rest_chosen": slim(after),
        "paired_bootstrap_chosen_minus_name_only": paired,
        "no_match_false_merges": {"name_only": fm_before, "chosen": fm_after},
        "longest_list": int(lengths.max()),
        "contested_records": int((np.bincount(right[chosen], minlength=n_rec) > 1).sum()),
        "passes_promotion_gate": bool(gate),
        "holdout2_rows_read": 0,
        "slice_read": False,
        "elapsed_seconds": round(time.time() - T0, 1),
    }
    np.save(OUT / "probabilities.npy", prob)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    log(json.dumps({k: report[k] for k in ("chosen_on_a_tune", "b_rest_name_only", "b_rest_chosen",
                                             "paired_bootstrap_chosen_minus_name_only", "no_match_false_merges",
                                             "passes_promotion_gate")}, indent=2))


if __name__ == "__main__":
    main()
