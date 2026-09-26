"""Pair CatBoost plus graded digit and street differences versus the rival claimant.

The saved pair tree sees the rival's blend and an exact-key bit. It does not see
whether this claimant and the rival differ by a substituted unit number or by
street tokens. Those comparisons already failed as hard rules. This run keeps
the pair tree's candidates, fit size, component split and decision rule, and
adds only those graded text features. Fuzzy admin matching is off.

Fit on 80% of partition-A components. Choose the threshold on the other 20%.
Read B_REST once, against the saved pair tree at its chosen threshold 0.70.
SLICE is not read. HOLDOUT-2 is refused.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.arena_decision import component_bootstrap, evaluate
from ber.forensic_features import default_vocabulary, forensic_features
from ber.pair_tree_arena import (
    FEATURES as SCORE_FEATURES,
    build_pairs,
    decide,
    false_merge,
    load_scores,
    log,
    predict,
    slim,
)

ART = Path(os.environ.get("ARENA", "/opt/az-ml-new/artifacts/arena"))
INDEX = Path(os.environ.get("WAVE_INDEX", "/opt/az-ml-new/runs/cpu_wave"))
OUT = Path(os.environ.get("RIVAL_OUT", "/opt/az-ml-new/runs/rival_text_arena"))
PRIOR = Path(os.environ.get("PAIR_TREE", "/opt/az-ml-new/runs/pair_tree_arena"))
THREADS = int(os.environ.get("TREE_THREADS", "24"))
FIT_ROWS = int(os.environ.get("TREE_FIT_ROWS", "3000000"))
BATCH = int(os.environ.get("FORENSIC_BATCH", "250000"))
SEED = 0
PRIOR_THRESHOLD = 0.70
EXPECTED_PRIOR_ATUNE = 0.9765805561040934
THRESHOLDS = tuple(np.round(np.arange(0.30, 0.951, 0.025), 3))
TEXT = ("digit_relation", "n_same_length_substitutions", "n_insertions", "n_deletions",
        "max_group_edit_distance", "name_token_jaccard", "street_token_jaccard")
FEATURES = SCORE_FEATURES + [
    "digit_relation", "n_same_length_substitutions", "n_insertions", "n_deletions",
    "max_group_edit_distance", "name_token_jaccard", "street_token_jaccard",
    "rival_digit_relation", "rival_n_same_length_substitutions", "rival_name_token_jaccard",
    "rival_street_token_jaccard", "street_jaccard_minus_rival",
]
CAT = ["digit_relation", "rival_digit_relation"]
STARTED = time.time()


def rival_owners(left, right, blend, n_rec):
    """Best and second S1 per record, plus the top blend.

    Sort is record, then blend descending, then the earlier row. Owners are
    int32 and use -1 when a record has no such claimant. Top blend is -1 when
    the record has no pair.
    """
    order = np.lexsort((np.arange(len(blend)), -blend, right))
    rec = right[order]
    owner = left[order]
    score = blend[order]
    start = np.empty(len(rec), dtype=bool)
    start[0] = True
    start[1:] = rec[1:] != rec[:-1]
    best = np.full(n_rec, -1, np.int32)
    second = np.full(n_rec, -1, np.int32)
    top1 = np.full(n_rec, -1.0, dtype=np.float32)
    best[rec[start]] = owner[start].astype(np.int32)
    top1[rec[start]] = score[start]
    follow = np.zeros(len(rec), dtype=bool)
    follow[1:] = start[:-1] & ~start[1:]
    second[rec[follow]] = owner[follow].astype(np.int32)
    return best, second, top1


def rival_of(rows, left, right, blend, best, second, top1):
    """Runner-up when this row ties the record's top blend; otherwise the top S1.

    A tied runner-up is compared with the other tied claimant, not with itself.
    """
    rec = right[rows]
    rival = np.where(blend[rows] >= top1[rec], second[rec], best[rec]).astype(np.int32)
    same = rival == left[rows]
    if same.any():
        rec_same = rec[same]
        other = np.where(second[rec_same] == left[rows][same], best[rec_same], second[rec_same])
        rival[same] = other
    rival[rival == left[rows]] = -1
    return rival


def _text_block(s1_idx, rec_idx, names_s1, addr_s1, country_s1, names_rec, addr_rec, vocab):
    n = len(s1_idx)
    columns = {name: np.full(n, np.nan, dtype=np.float32) for name in TEXT}
    for start in range(0, n, BATCH):
        stop = min(start + BATCH, n)
        rows = slice(start, stop)
        frame = forensic_features(
            names_s1[s1_idx[rows]], addr_s1[s1_idx[rows]],
            names_rec[rec_idx[rows]], addr_rec[rec_idx[rows]],
            country_s1[s1_idx[rows]], vocab, fuzzy_admin=False)
        for name in TEXT:
            columns[name][rows] = frame[name].to_numpy(dtype=np.float32, na_value=np.nan)
        log(f"forensic {stop}/{n}")
    return columns


def _scatter_unique(s1_idx, rec_idx, n_rec, names_s1, addr_s1, country_s1, names_rec, addr_rec, vocab):
    """Text columns for possibly repeated (S1, record) pairs, computed once each."""
    key = s1_idx.astype(np.int64) * np.int64(n_rec) + rec_idx.astype(np.int64)
    unique, inverse = np.unique(key, return_inverse=True)
    log(f"unique text pairs {len(unique)} from {len(s1_idx)}")
    got = _text_block(unique // np.int64(n_rec), unique % np.int64(n_rec),
                      names_s1, addr_s1, country_s1, names_rec, addr_rec, vocab)
    return {name: values[inverse] for name, values in got.items()}


def text_matrix(rows, left, right, blend, best, second, top1, n_rec, names_s1, addr_s1,
                country_s1, names_rec, addr_rec, vocab):
    """Graded text columns aligned with ``rows``. Relation codes use -1 when missing."""
    self_text = _scatter_unique(left[rows], right[rows], n_rec, names_s1, addr_s1, country_s1,
                                names_rec, addr_rec, vocab)
    rival = rival_of(rows, left, right, blend, best, second, top1)
    valid = rival >= 0
    rival_text = {name: np.full(len(rows), np.nan, dtype=np.float32) for name in TEXT}
    if valid.any():
        log(f"rival rows {int(valid.sum())}")
        got = _scatter_unique(rival[valid], right[rows][valid], n_rec, names_s1, addr_s1,
                              country_s1, names_rec, addr_rec, vocab)
        for name in TEXT:
            rival_text[name][valid] = got[name]
    street_gap = self_text["street_token_jaccard"] - rival_text["street_token_jaccard"]
    block = np.column_stack([
        self_text["digit_relation"], self_text["n_same_length_substitutions"],
        self_text["n_insertions"], self_text["n_deletions"], self_text["max_group_edit_distance"],
        self_text["name_token_jaccard"], self_text["street_token_jaccard"],
        rival_text["digit_relation"], rival_text["n_same_length_substitutions"],
        rival_text["name_token_jaccard"], rival_text["street_token_jaccard"], street_gap,
    ]).astype(np.float32)
    for column in (0, 7):
        missing = ~np.isfinite(block[:, column])
        block[missing, column] = -1.0
    return block


def as_frame(matrix):
    frame = pd.DataFrame(matrix, columns=FEATURES)
    for name in CAT:
        frame[name] = frame[name].astype(np.int32)
    return frame


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
    names_s1 = ref.business_name.to_numpy(dtype=object)
    addr_s1 = ref.business_address.to_numpy(dtype=object)
    country_s1 = ref.country.to_numpy(dtype=object)
    names_rec = pool.business_name.to_numpy(dtype=object)
    addr_rec = pool.business_address.to_numpy(dtype=object)
    del ref, pool

    best, second, top1 = rival_owners(left, right, blend, n_rec)
    cand_rows = pairs.cand_rows
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

    from catboost import CatBoostClassifier

    prior = CatBoostClassifier()
    prior.load_model(str(PRIOR / "model.cbm"))
    prior_prob = predict(prior, pairs, len(left))
    prior_accept = decide(prior_prob, left, right, PRIOR_THRESHOLD, n_s1)
    prior_tune = evaluate(prior_accept, left, label, true_count, population=s1_tune)["macro_f05"]
    log(f"reloaded pair tree A-tune {prior_tune:.6f}")
    if abs(prior_tune - EXPECTED_PRIOR_ATUNE) > 1e-5:
        raise SystemExit("pair-tree A-tune did not reproduce; refusing to score a drifted split")

    text_path = OUT / "text_features.npy"
    text_shape = (len(cand_rows), len(FEATURES) - len(SCORE_FEATURES))
    text = None
    if text_path.exists():
        try:
            loaded = np.load(text_path)
        except (OSError, ValueError):
            loaded = None
        if loaded is not None and loaded.shape == text_shape:
            text = loaded
            log(f"reused text features {text.shape}")
    if text is None:
        text = text_matrix(cand_rows, left, right, blend, best, second, top1, n_rec,
                           names_s1, addr_s1, country_s1, names_rec, addr_rec, default_vocabulary())
        np.save(text_path, text)
        log(f"saved text features {text.shape}")
    del names_s1, addr_s1, country_s1, names_rec, addr_rec

    def rows_matrix(rows):
        at = np.searchsorted(cand_rows, rows)
        if not np.array_equal(cand_rows[at], rows):
            raise RuntimeError("feature rows are not candidate rows")
        return np.hstack([pairs.features(rows), text[at]])

    fit_rows = cand_rows[s1_fit[left[cand_rows]]]
    if len(fit_rows) > FIT_ROWS:
        fit_rows = np.sort(rng.choice(fit_rows, size=FIT_ROWS, replace=False))
    log(f"fit rows {len(fit_rows)}")
    model = CatBoostClassifier(
        iterations=800, depth=8, learning_rate=0.08, loss_function="Logloss",
        thread_count=THREADS, random_seed=SEED, verbose=100, allow_writing_files=False,
        cat_features=CAT)
    model.fit(as_frame(rows_matrix(fit_rows)), label[fit_rows].astype(np.int8))
    model.save_model(str(OUT / "model.cbm"))
    importance = dict(zip(FEATURES, map(float, model.get_feature_importance())))
    log(f"importance {importance}")

    prob = np.zeros(len(left), dtype=np.float32)
    for start in range(0, len(cand_rows), BATCH):
        chunk = cand_rows[start:start + BATCH]
        prob[chunk] = model.predict_proba(as_frame(rows_matrix(chunk)))[:, 1].astype(np.float32)
        log(f"predict {min(start + BATCH, len(cand_rows))}/{len(cand_rows)}")

    sweep = []
    chosen_spec = None
    for threshold in THRESHOLDS:
        accept = decide(prob, left, right, float(threshold), n_s1)
        score = evaluate(accept, left, label, true_count, population=s1_tune)["macro_f05"]
        sweep.append({"threshold": float(threshold), "a_tune_f05": score})
        if chosen_spec is None or score > chosen_spec["a_tune_f05"]:
            chosen_spec = {"threshold": float(threshold), "a_tune_f05": score}
    log(f"A-tune pair tree {prior_tune:.6f}; chosen {chosen_spec}")

    chosen = decide(prob, left, right, chosen_spec["threshold"], n_s1)
    before = evaluate(prior_accept, left, label, true_count, population=s1_brest, return_per_s1=True)
    after = evaluate(chosen, left, label, true_count, population=s1_brest, return_per_s1=True)
    paired = component_bootstrap(before["per_s1"], after["per_s1"], component, n=300, seed=0,
                                 population=s1_brest)
    fm_before = false_merge(prior_accept, left, s1_brest, true_count)
    fm_after = false_merge(chosen, left, s1_brest, true_count)
    gate = (paired["delta"] >= 0.0005 and paired["ci_low"] > 0
            and fm_after["rate"] - fm_before["rate"] <= 0.002)
    lengths = np.bincount(left[chosen], minlength=n_s1)
    report = {
        "question": "graded digit and street features versus the rival, on top of the pair tree",
        "changed_variable": ("digit relation, digit edits, name and street Jaccard, "
                             "and the rival claimant's digit relation, substitutions, name and street Jaccard"),
        "held_fixed": "pair-tree candidates, 3M fit rows, seed 0, 80/20 A components, one owner, list cap 11",
        "fuzzy_admin": False,
        "features": FEATURES,
        "categorical": CAT,
        "feature_importance": importance,
        "fit": {"rows": int(len(fit_rows)), "positives": int(label[fit_rows].sum()),
                "iterations": 800, "depth": 8, "learning_rate": 0.08, "seed": SEED},
        "prior_pair_tree_a_tune_f05": prior_tune,
        "prior_threshold": PRIOR_THRESHOLD,
        "a_tune_sweep": sweep,
        "chosen_on_a_tune": chosen_spec,
        "b_rest_pair_tree": slim(before),
        "b_rest_chosen": slim(after),
        "paired_bootstrap_chosen_minus_pair_tree": paired,
        "no_match_false_merges": {"pair_tree": fm_before, "chosen": fm_after},
        "longest_list": int(lengths.max()),
        "contested_records": int((np.bincount(right[chosen], minlength=n_rec) > 1).sum()),
        "passes_promotion_gate": bool(gate),
        "holdout2_rows_read": 0,
        "slice_read": False,
        "elapsed_seconds": round(time.time() - STARTED, 1),
    }
    np.save(OUT / "probabilities.npy", prob)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    log(json.dumps({key: report[key] for key in (
        "chosen_on_a_tune", "b_rest_pair_tree", "b_rest_chosen",
        "paired_bootstrap_chosen_minus_pair_tree", "no_match_false_merges",
        "passes_promotion_gate")}, indent=2))


if __name__ == "__main__":
    main()
