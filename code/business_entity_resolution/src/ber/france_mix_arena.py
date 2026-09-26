"""Mixed France pseudo-label + US/India replay arena for a short C2 fine-tune.

``build`` writes an arena that ``ber.fast_trainer --command train --mode c2``
loads unchanged (manifest, reference/pool/pairs, v1_blend/v1_tree, labels dir,
integer partition 0=A, 1=B_REST, 2=SLICE).

French part (test France, no ground truth):
  * the strict pseudo rule audited on arena US/India by ``france_transfer_audit``
    and ``france_transfer_rivals``: P+ = unique S1 exact key equal to the record
    key, street-token Jaccard >= 0.5, v1 blend >= 0.9, no other claim >= 0.5,
    non-empty record address; P- = every other French S1 candidate pair to a P+
    record (rival of the P+ owner). The unaudited template negatives of
    ``france_pseudo_labels`` are not used.
  * only pseudo-labelled pairs are kept, with their v1 test blend/tree scores.
  * ``true_count`` = number of P+ pairs of the S1 (an approximation: French
    truth may hold links the rule does not see; S1s with only P- pairs get 0).
  * ``rec_owner`` = the P+ owner of each record.
  * 10% pseudo-holdout (``france_pseudo_labels.holdout_flags``) -> SLICE. A rival
    pair whose S1 and P+ owner fall on different sides of that split is dropped.
Replay part: a seeded random sample of arena partition-A S1s with all their
pairs and real TRAIN labels; the sample size is set from the aggregate arena
positives per S1 so French positives are about ``--french-share`` of the
training positives (per-pair labels are not read for sampling).
Monitor part: a seeded random sample of arena B_REST S1s (code 1), all pairs.
Owner stubs: arena S1s that own a record in the mixed pool but were not sampled
are added as partition-A reference rows with no pairs, so ``rec_owner`` (and the
C2 contest-loser typing) is exact. They contribute no training pair.

No HOLDOUT-2 label is read. Test labels do not exist. Nothing here is French
accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber.forensic_features import (EMPTY, default_vocabulary, exact_keys, forensic_features,
                                   normalize_names, s1_multiplicities)
from ber.france_pseudo_labels import holdout_flags
from ber.v2_ce_train import file_sha256, sealed_ids

COUNTRY = "France"
P_BLEND = 0.9
RIVAL_BLEND = 0.5
STREET_JACCARD = 0.5
CODE_A, CODE_B_REST, CODE_SLICE = np.uint8(0), np.uint8(1), np.uint8(2)
SRC_TEST, SRC_ARENA = np.uint8(0), np.uint8(1)
TEXT_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
CHUNK = 4_000_000


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def _json(path: Path, value) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _npy(path: Path, value) -> None:
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, value)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# French strict pseudo-labels (label-free)
# ---------------------------------------------------------------------------

def strict_france(test: Path) -> dict:
    """Recompute the audited strict rule on test France from the v1 test scores."""
    ref = pq.read_table(test / "dense_test/reference.parquet", columns=TEXT_COLUMNS).to_pandas()
    french = (ref.country == COUNTRY).to_numpy()
    fr_ref = np.flatnonzero(french)
    s1 = ref.iloc[fr_ref].reset_index(drop=True)
    mult = s1_multiplicities(s1.business_name, s1.business_address, COUNTRY)
    local_s1 = np.full(len(ref), -1, np.int64)
    local_s1[fr_ref] = np.arange(len(fr_ref))
    log(f"test S1 {len(ref)}; France {len(fr_ref)}")

    pool = pq.read_table(test / "dense_test/pool.parquet", columns=TEXT_COLUMNS).to_pandas()
    fr_pool = np.flatnonzero((pool.country == COUNTRY).to_numpy())
    rec = pool.iloc[fr_pool].reset_index(drop=True)
    local_rec = np.full(len(pool), -1, np.int64)
    local_rec[fr_pool] = np.arange(len(fr_pool))
    rec_key = exact_keys(rec.business_name, rec.business_address,
                         normalized_names=normalize_names(rec.business_name))
    log(f"test records {len(pool)}; France {len(fr_pool)}")

    left_all = np.load(test / "index/left.npy", mmap_mode="r")
    right_all = np.load(test / "index/right.npy", mmap_mode="r")
    blend_all = np.load(test / "probabilities.npy", mmap_mode="r")
    tree_all = np.load(test / "base_probabilities.npy", mmap_mode="r")
    n_pairs = len(left_all)
    if not (len(right_all) == len(blend_all) == len(tree_all) == n_pairs):
        raise ValueError("test pair indices and v1 scores are not aligned")
    claims = np.zeros(len(pool), np.int32)
    rows_parts = []
    for begin in range(0, n_pairs, CHUNK):
        left = np.asarray(left_all[begin:begin + CHUNK])
        right = np.asarray(right_all[begin:begin + CHUNK])
        blend = np.asarray(blend_all[begin:begin + CHUNK])
        claims += np.bincount(right[blend >= RIVAL_BLEND], minlength=len(pool)).astype(np.int32)
        rows_parts.append(np.flatnonzero(french[left]) + begin)
    rows = np.concatenate(rows_parts)
    left = np.asarray(left_all[rows], dtype=np.int64)
    right = np.asarray(right_all[rows], dtype=np.int64)
    blend = np.asarray(blend_all[rows], dtype=np.float32)
    tree = np.asarray(tree_all[rows], dtype=np.float32)
    ps, pr = local_s1[left], local_rec[right]
    if (ps < 0).any() or (pr < 0).any():
        raise ValueError("a French S1 candidate points outside the French pool")
    log(f"French candidate pairs {len(rows)}")

    s1_key = mult.exact_key.to_numpy()
    codes, _ = pd.factorize(pd.Series(np.concatenate([s1_key, rec_key])))
    s1_code, rec_code = codes[:len(s1)], codes[len(s1):]
    unique_key = (mult.key_count.to_numpy() == 1) & (s1_key != "")
    funnel = {"french_candidate_pairs": int(len(rows))}
    cand = np.flatnonzero(unique_key[ps] & (s1_code[ps] == rec_code[pr]))
    funnel["unique_key_match"] = int(len(cand))
    cand = cand[blend[cand] >= P_BLEND]
    funnel["and_blend_ge_0.9"] = int(len(cand))
    feats = forensic_features(s1.business_name.to_numpy()[ps[cand]], s1.business_address.to_numpy()[ps[cand]],
                              rec.business_name.to_numpy()[pr[cand]], rec.business_address.to_numpy()[pr[cand]],
                              COUNTRY, default_vocabulary())
    street = feats.street_token_jaccard.to_numpy() >= STREET_JACCARD
    nonempty = feats.address_kind.to_numpy() != EMPTY
    own = (blend[cand] >= RIVAL_BLEND).astype(np.int32)
    no_rival = (claims[right[cand]] - own) == 0
    funnel["and_street_jaccard_ge_0.5"] = int((street).sum())
    funnel["and_no_rival_ge_0.5"] = int((street & no_rival).sum())
    positive = cand[street & no_rival & nonempty]
    funnel["and_record_address_not_empty"] = int(len(positive))
    if len(np.unique(right[positive])) != len(positive):
        raise ValueError("a record has two P+ owners")
    owner = np.full(len(pool), -1, np.int64)
    owner[right[positive]] = left[positive]
    rival = np.flatnonzero((owner[right] >= 0) & (owner[right] != left))
    label = np.zeros(len(rows), bool)
    label[positive] = True
    keep = np.zeros(len(rows), bool)
    keep[positive] = True
    keep[rival] = True
    log(f"strict P+ {len(positive)}  rival P- {len(rival)}")
    return {"ref": ref, "french": french, "rows": rows, "left": left, "right": right, "blend": blend,
            "tree": tree, "label": label, "keep": keep, "owner": owner, "pool": pool,
            "n_pairs": n_pairs, "funnel": funnel, "p_plus": int(len(positive)), "p_minus": int(len(rival)),
            "fr_ref": fr_ref, "tree_ge_001_french_pairs": int((tree >= .01).sum())}


def cross_check(fr: dict, parquet: Path) -> dict:
    """Compare the recomputed strict sets with the local france_pseudo_labels output."""
    table = pq.read_table(parquet, columns=["label", "reasons", "pair_row", "left_idx", "right_idx"]).to_pandas()
    rows, left, right = fr["rows"], fr["left"], fr["right"]
    order = np.argsort(rows)
    pos_rows = rows[fr["keep"] & fr["label"]]
    neg_rows = rows[fr["keep"] & ~fr["label"]]
    old_pos = table.pair_row.to_numpy()[table.label.to_numpy() == 1]
    rival_mask = (table.label.to_numpy() == 0) & table.reasons.str.contains("rival_of_pplus_owner").to_numpy()
    old_rival = table.pair_row.to_numpy()[rival_mask]
    template_only = int(((table.label.to_numpy() == 0) & ~rival_mask).sum())
    at = order[np.searchsorted(rows, table.pair_row.to_numpy(), sorter=order).clip(0, len(rows) - 1)]
    aligned = bool(np.array_equal(rows[at], table.pair_row.to_numpy())
                   and np.array_equal(left[at], table.left_idx.to_numpy())
                   and np.array_equal(right[at], table.right_idx.to_numpy()))
    return {"parquet": str(parquet), "parquet_sha256": file_sha256(parquet),
            "parquet_p_plus": int(len(old_pos)), "parquet_p_minus": int((table.label == 0).sum()),
            "parquet_p_minus_rival": int(len(old_rival)), "parquet_p_minus_template_only": template_only,
            "pair_rows_aligned_with_index": aligned,
            "p_plus_identical": bool(np.array_equal(np.sort(old_pos), np.sort(pos_rows))),
            "rival_identical": bool(np.array_equal(np.sort(old_rival), np.sort(neg_rows))),
            "verdict": ("parquet P+ and rival P- equal the strict recomputation; its template-only "
                        "negatives are outside the audited rule and are excluded")}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _take_text(path: Path, positions: np.ndarray) -> pd.DataFrame:
    table = pq.read_table(path, columns=TEXT_COLUMNS)
    return table.take(pa.array(positions, type=pa.int64())).to_pandas()


def build(args) -> dict:
    started = time.monotonic()
    out = Path(args.output)
    arena_out, label_out = out / "arena", out / "labels"
    for path in (arena_out, label_out):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"output must be new or empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
    arena, labels_dir, test = Path(args.arena), Path(args.labels), Path(args.test_inputs)
    summary = json.loads((labels_dir / "labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise ValueError("refusing arena labels that read HOLDOUT-2")
    forbidden, sealed_sha = sealed_ids(Path(args.sealed_reservation))

    # --- France -----------------------------------------------------------------
    fr = strict_france(test)
    check = cross_check(fr, Path(args.pseudo_labels)) if args.pseudo_labels else None
    if check:
        log(json.dumps(check))
    ref_t = fr["ref"]
    fr_ref = fr["fr_ref"]
    hold_s1 = np.zeros(len(ref_t), bool)
    hold_s1[fr_ref] = holdout_flags(ref_t.entity_id.to_numpy()[fr_ref].tolist())
    collide = np.zeros(len(ref_t), bool)
    collide[fr_ref] = ref_t.entity_id.iloc[fr_ref].isin(forbidden).to_numpy()
    keep = fr["keep"].copy()
    left_t, right_t, owner_t = fr["left"], fr["right"], fr["owner"]
    pair_owner = owner_t[right_t]
    kept_before_split = int(keep.sum())
    cross = keep & (pair_owner >= 0) & (hold_s1[left_t] != hold_s1[np.maximum(pair_owner, 0)])
    keep &= ~cross
    sealed_drop = keep & (collide[left_t] | collide[np.maximum(pair_owner, 0)])
    keep &= ~sealed_drop
    rows_t = fr["rows"][keep]
    left_t, right_t = left_t[keep], right_t[keep]
    label_t, blend_t, tree_t = fr["label"][keep], fr["blend"][keep], fr["tree"][keep]
    fr_s1 = np.unique(left_t)
    fr_rec = np.unique(right_t)
    fr_true = np.bincount(left_t[label_t], minlength=len(ref_t))[fr_s1].astype(np.int16)
    fr_train_pos = int((label_t & ~hold_s1[left_t]).sum())
    log(f"France kept pairs {len(rows_t)} (cross-split rivals dropped {int(cross.sum())}, "
        f"sealed-ID collisions dropped {int(sealed_drop.sum())}); S1 {len(fr_s1)}; train P+ {fr_train_pos}")

    # --- Replay / monitor sample --------------------------------------------------
    part = np.load(args.arena_partition)
    n_ref_a = len(part)
    rng = np.random.default_rng(args.seed)
    per_s1 = summary["positive_pairs"] / n_ref_a
    target_pos = fr_train_pos * (1 - args.french_share) / args.french_share
    n_replay = int(round(target_pos / per_s1))
    a_ids = np.flatnonzero(part == CODE_A)
    b_ids = np.flatnonzero(part == CODE_B_REST)
    replay = np.sort(rng.choice(a_ids, n_replay, replace=False))
    monitor = np.sort(rng.choice(b_ids, args.monitor_s1, replace=False))
    chosen = np.zeros(n_ref_a, bool)
    chosen[replay] = chosen[monitor] = True
    log(f"replay A S1 {len(replay)} (aggregate positives/S1 {per_s1:.4f}); B_REST monitor {len(monitor)}")

    pairs_file = pq.ParquetFile(arena / "pairs.parquet")
    sel_rows, sel_left, sel_right, offset = [], [], [], 0
    for batch in pairs_file.iter_batches(batch_size=CHUNK, columns=["left_idx", "right_idx"]):
        left = batch.column(0).to_numpy()
        right = batch.column(1).to_numpy()
        hit = np.flatnonzero(chosen[left])
        sel_rows.append(hit + offset)
        sel_left.append(left[hit])
        sel_right.append(right[hit])
        offset += len(left)
    rows_a = np.concatenate(sel_rows)
    left_a = np.concatenate(sel_left).astype(np.int64)
    right_a = np.concatenate(sel_right).astype(np.int64)
    blend_src = np.load(arena / "probabilities.npy", mmap_mode="r")
    tree_src = np.load(arena / "base_probabilities.npy", mmap_mode="r")
    if offset != len(blend_src) or offset != len(tree_src):
        raise ValueError("arena pairs and v1 scores are not aligned")
    blend_a = np.asarray(blend_src[rows_a], dtype=np.float32)
    tree_a = np.asarray(tree_src[rows_a], dtype=np.float32)
    label_a = np.asarray(np.load(labels_dir / "label.npy", mmap_mode="r")[rows_a], dtype=bool)
    owner_src = np.load(labels_dir / "rec_owner.npy", mmap_mode="r")
    true_src = np.load(labels_dir / "true_count.npy", mmap_mode="r")
    ar_rec = np.unique(right_a)
    ar_owner = np.asarray(owner_src[ar_rec], dtype=np.int64)
    stubs = np.setdiff1d(np.unique(ar_owner[ar_owner >= 0]), np.flatnonzero(chosen))
    ar_s1 = np.concatenate([replay, monitor, stubs])
    log(f"arena pairs {len(rows_a)}; records {len(ar_rec)}; owner stubs {len(stubs)}")

    # --- Assemble -----------------------------------------------------------------
    n_fr, n_ar = len(fr_s1), len(ar_s1)
    new_t = np.full(len(ref_t), -1, np.int64)
    new_t[fr_s1] = np.arange(n_fr)
    new_a = np.full(n_ref_a, -1, np.int64)
    new_a[ar_s1] = n_fr + np.arange(n_ar)
    rec_t = np.full(len(fr["pool"]), -1, np.int64)
    rec_t[fr_rec] = np.arange(len(fr_rec))
    rec_a = np.full(len(owner_src), -1, np.int64)
    rec_a[ar_rec] = len(fr_rec) + np.arange(len(ar_rec))

    left = np.concatenate([new_t[left_t], new_a[left_a]])
    right = np.concatenate([rec_t[right_t], rec_a[right_a]])
    source = np.concatenate([np.full(len(rows_t), SRC_TEST), np.full(len(rows_a), SRC_ARENA)])
    source_row = np.concatenate([rows_t, rows_a]).astype(np.int64)
    label = np.concatenate([label_t, label_a])
    blend = np.concatenate([blend_t, blend_a]).astype(np.float32)
    tree = np.concatenate([tree_t, tree_a]).astype(np.float32)
    if (left < 0).any() or (right < 0).any():
        raise ValueError("unmapped pair index")
    order = np.lexsort((right, left))
    left, right, source, source_row = left[order], right[order], source[order], source_row[order]
    label, blend, tree = label[order], blend[order], tree[order]
    if len(left) > 1 and np.any((left[1:] == left[:-1]) & (right[1:] <= right[:-1])):
        raise ValueError("duplicate (left, right) pair")

    owner_fr = owner_t[fr_rec]
    owner_new = np.concatenate([np.where(owner_fr >= 0, new_t[np.maximum(owner_fr, 0)], -1),
                                np.where(ar_owner >= 0, new_a[np.maximum(ar_owner, 0)], -1)]).astype(np.int32)
    if np.any(owner_new < -1) or not np.array_equal(label, owner_new[right] == left):
        raise ValueError("owners do not reproduce the pair labels")
    true_count = np.concatenate([fr_true, np.asarray(true_src[ar_s1], dtype=np.int16)]).astype(np.int16)
    if np.any(np.bincount(left[label], minlength=len(true_count)) > true_count):
        raise ValueError("retrieved positives exceed true_count")
    partition = np.concatenate([
        np.where(hold_s1[fr_s1], CODE_SLICE, CODE_A),
        np.full(len(replay), CODE_A), np.full(len(monitor), CODE_B_REST), np.full(len(stubs), CODE_A),
    ]).astype(np.uint8)
    role = np.concatenate([np.full(n_fr, "france"), np.full(len(replay), "replay_a"),
                           np.full(len(monitor), "monitor_b_rest"), np.full(len(stubs), "owner_stub")])

    reference = pd.concat([ref_t.iloc[fr_s1].reset_index(drop=True),
                           _take_text(arena / "reference.parquet", ar_s1)], ignore_index=True)
    reference["source"] = np.r_[np.full(n_fr, SRC_TEST), np.full(n_ar, SRC_ARENA)].astype(np.uint8)
    reference["source_idx"] = np.r_[fr_s1, ar_s1].astype(np.int64)
    reference["role"] = role
    pool = pd.concat([fr["pool"].iloc[fr_rec].reset_index(drop=True),
                      _take_text(arena / "pool.parquet", ar_rec)], ignore_index=True)
    pool["source"] = np.r_[np.full(len(fr_rec), SRC_TEST), np.full(len(ar_rec), SRC_ARENA)].astype(np.uint8)
    pool["source_idx"] = np.r_[fr_rec, ar_rec].astype(np.int64)
    sealed_hits = int(reference.entity_id.isin(forbidden).sum())
    if sealed_hits:
        raise ValueError(f"{sealed_hits} reference IDs are sealed")
    dup_ref = int(reference.entity_id.duplicated().sum())
    dup_pool = int(pool.entity_id.duplicated().sum())

    reference.to_parquet(arena_out / "reference.parquet", index=False, row_group_size=1_000_000)
    pool.to_parquet(arena_out / "pool.parquet", index=False, row_group_size=1_000_000)
    pq.write_table(pa.table({"left_idx": left.astype(np.int32), "right_idx": right.astype(np.int32),
                             "source": source, "source_pair_row": source_row}),
                   arena_out / "pairs.parquet", row_group_size=1_000_000)
    _npy(arena_out / "v1_blend.npy", blend)
    _npy(arena_out / "v1_tree.npy", tree)
    names = ("reference.parquet", "pool.parquet", "pairs.parquet", "v1_blend.npy", "v1_tree.npy")
    manifest = {
        "schema": 1, "source": "ber.france_mix_arena: strict France pseudo-labels on test + arena US/India replay",
        "references": int(len(reference)), "pool_records": int(len(pool)), "pairs": int(len(left)),
        "labels_read": False, "holdout2_excluded": True, "pair_order": "left_then_right",
        "pair_selection": ("France: only strict P+/rival P- pairs (not retrieval-complete); arena: all "
                           "candidate pairs of seeded random A/B_REST S1 samples; no truth injected"),
        "sha256": {name: file_sha256(arena_out / name) for name in names},
    }
    _json(arena_out / "manifest.json", manifest)

    _npy(label_out / "label.npy", label)
    _npy(label_out / "true_count.npy", true_count)
    _npy(label_out / "rec_owner.npy", owner_new)
    french_rows = source == SRC_TEST
    labels_summary = {
        "pairs": int(len(label)), "positive_pairs": int(label.sum()), "true_links": int(true_count.sum()),
        "singletons": int((true_count == 0).sum()), "holdout2_rows_read": 0,
        "french_pairs": int(french_rows.sum()), "french_positive_pairs": int((label & french_rows).sum()),
        "arena_pairs": int((~french_rows).sum()), "arena_positive_pairs": int((label & ~french_rows).sum()),
        "french_true_count_rule": ("number of strict P+ pairs of the S1; approximate (unseen French truth "
                                   "is not counted, S1s with only rival pairs count 0)"),
        "arena_labels": "arena TRAIN labels (label.npy / true_count.npy / rec_owner.npy), remapped",
    }
    _json(label_out / "labels_summary.json", labels_summary)
    _npy(out / "s1_partition.npy", partition)
    population = np.zeros(len(ref_t), bool)
    population[fr_ref] = True
    _npy(out / "test_france_s1.npy", population)

    train_pairs = partition[left] == CODE_A
    report = {
        "state": "complete", "seed": args.seed, "elapsed_s": round(time.monotonic() - started, 1),
        "inputs": {"test_inputs": str(test), "arena": str(arena), "labels": str(labels_dir),
                   "arena_partition": str(args.arena_partition),
                   "arena_partition_sha256": file_sha256(Path(args.arena_partition)),
                   "sealed_reservation_sha256": sealed_sha,
                   "test_hashes": {name: file_sha256(test / name) for name in
                                   ("probabilities.npy", "base_probabilities.npy", "index/left.npy",
                                    "index/right.npy", "dense_test/reference.parquet", "dense_test/pool.parquet")}},
        "france_rule": {"funnel": fr["funnel"], "p_plus": fr["p_plus"], "p_minus_rival": fr["p_minus"],
                        "kept_before_split": kept_before_split,
                        "cross_split_rivals_dropped": int(cross.sum()),
                        "sealed_id_collision_pairs_dropped": int(sealed_drop.sum()),
                        "french_s1_sealed_id_collisions": int(collide.sum()),
                        "tree_ge_0.01_french_candidate_pairs": fr["tree_ge_001_french_pairs"],
                        "cross_check": check},
        "counts": {
            "references": int(len(reference)), "pool_records": int(len(pool)), "pairs": int(len(left)),
            "french_s1": int(n_fr), "french_s1_slice": int(hold_s1[fr_s1].sum()),
            "replay_a_s1": int(len(replay)), "monitor_b_rest_s1": int(len(monitor)), "owner_stub_s1": int(len(stubs)),
            "partition": {name: int((partition == code).sum()) for code, name in
                          ((CODE_A, "A"), (CODE_B_REST, "B_REST"), (CODE_SLICE, "SLICE"))},
            "train_A": {"pairs": int(train_pairs.sum()), "positives": int((label & train_pairs).sum()),
                        "french_positives": int((label & train_pairs & french_rows).sum()),
                        "replay_positives": int((label & train_pairs & ~french_rows).sum()),
                        "french_negatives": int((~label & train_pairs & french_rows).sum()),
                        "replay_negatives": int((~label & train_pairs & ~french_rows).sum())},
            "slice_france": {"pairs": int((partition[left] == CODE_SLICE).sum()),
                             "positives": int((label & (partition[left] == CODE_SLICE)).sum())},
            "b_rest_monitor": {"pairs": int((partition[left] == CODE_B_REST).sum()),
                               "positives": int((label & (partition[left] == CODE_B_REST)).sum()),
                               "true_links": int(true_count[partition == CODE_B_REST].sum())},
            "duplicate_reference_ids": dup_ref, "duplicate_pool_ids": dup_pool,
        },
        "outputs": {"arena": str(arena_out), "labels": str(label_out), "partition": str(out / "s1_partition.npy"),
                    "partition_sha256": file_sha256(out / "s1_partition.npy"),
                    "manifest_sha256": file_sha256(arena_out / "manifest.json"),
                    "labels_summary_sha256": file_sha256(label_out / "labels_summary.json"),
                    "test_france_population": str(out / "test_france_s1.npy")},
    }
    tp = report["counts"]["train_A"]
    tp["french_positive_share"] = tp["french_positives"] / max(1, tp["positives"])
    _json(out / "build_report.json", report)
    log(json.dumps(report["counts"]))
    return report


# ---------------------------------------------------------------------------
# Loader dry-run (CPU, no model weights)
# ---------------------------------------------------------------------------

def dry_run(args) -> dict:
    from ber.fast_trainer import (ArenaBatchSampler, DEFAULT_SEED, NegativeType, classify_negative_types,
                                  load_arena, mode_config, stake_weights, tokenize_batch)
    from ber.text_views import record_texts

    out = Path(args.output)
    started = time.monotonic()
    data = load_arena(out / "arena", out / "labels", partition_file=out / "s1_partition.npy",
                      sealed_reservation=Path(args.sealed_reservation))
    log(f"load_arena ok ({time.monotonic() - started:.0f}s)")
    if not np.any(data.partition == "A") or np.all(data.partition == "A"):
        raise ValueError("partition does not separate A from evaluation S1s")
    profile = mode_config("c2", batch_size=args.batch_size, steps=args.steps)
    src = pd.read_parquet(out / "arena/pairs.parquet", columns=["source"]).source.to_numpy()
    french = src == SRC_TEST
    in_train = data.partition[data.left] == "A"
    positives = np.flatnonzero(in_train & data.label)
    negative = in_train & ~data.label
    t0 = time.monotonic()
    flags = classify_negative_types(data.reference, data.pool, data.left, data.right, data.label,
                                    data.tree, data.blend, data.record_owner)
    flags[~in_train] = 0
    gated = np.flatnonzero(negative & (data.tree >= .01) & (flags == 0))
    typing_s = time.monotonic() - t0
    sampler = ArenaBatchSampler(positives, flags, gated, batch_size=profile.batch_size,
                                positive_fraction=profile.positive_fraction, seed=DEFAULT_SEED)
    draws = {"positive": [0, 0], "negative": [0, 0]}
    first = None
    for step in range(args.batches):
        ids, target = sampler.take()
        if first is None:
            first = (ids, target)
        for name, mask in (("positive", target == 1), ("negative", target == 0)):
            draws[name][0] += int(french[ids[mask]].sum())
            draws[name][1] += int((~french[ids[mask]]).sum())
    typed = {kind.name: {"france": int(((flags & int(kind)) != 0)[french].sum()),
                         "replay": int(((flags & int(kind)) != 0)[~french].sum())} for kind in NegativeType}
    left_text = record_texts(data.reference, canonical=False, field_tagged=False)
    right_text = record_texts(data.pool, canonical=False, field_tagged=False)
    ids, target = first
    weights = stake_weights(target.astype(bool), data.true_count[data.left[ids]])
    report = {
        "loader": "fast_trainer.load_arena with partition + sealed reservation: OK",
        "identity": data.identity, "profile": profile.__dict__,
        "sampler": {"positives_train": int(len(positives)),
                    "positives_train_france": int(french[positives].sum()),
                    "gated_random_negatives": int(len(gated)),
                    "gated_random_negatives_france": int(french[gated].sum()),
                    "typed_any": int((flags != 0).sum()), "typed_any_france": int(((flags != 0) & french).sum()),
                    "typed_by_kind": typed, "typing_seconds": round(typing_s, 1),
                    "per_batch": {"positives": sampler.positive_count, "typed": sampler.typed_count,
                                  "random": sampler.random_count}},
        "draws_over_batches": {"batches": args.batches,
                               "positive_france_replay": draws["positive"],
                               "negative_france_replay": draws["negative"]},
        "first_batch_stake_weights": {"min": float(weights.min()), "max": float(weights.max()),
                                      "mean_pos": float(weights[target == 1].mean()),
                                      "mean_neg": float(weights[target == 0].mean())},
        "text_examples": [f"{left_text[data.left[i]]}  <>  {right_text[data.right[i]]}  label={int(data.label[i])}"
                          for i in ids[:4]],
    }
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        batch = tokenize_batch(tokenizer, left_text, right_text, ids, target, weights, data.left, data.right,
                               max_length=256, microbatch=args.batch_size, length_buckets=(64, 128, 256))
        report["tokenized_first_batch"] = [list(mb["input_ids"].shape) for mb in batch.microbatches]
    if args.model_dir:
        from ber.fast_trainer import model_identity
        identity = model_identity(Path(args.model_dir), Path(args.allowlist))
        report["model_identity_check"] = {"path": identity["path"], "licence": identity["licence"],
                                          "files": sorted(identity["files_sha256"])}
    report["elapsed_s"] = round(time.monotonic() - started, 1)
    _json(out / "dry_run.json", report)
    return report


def write_population(args) -> dict:
    """Per-S1 mask of the mixed arena's evaluation S1s (B_REST monitor + French SLICE)."""
    out = Path(args.output)
    partition = np.load(out / "s1_partition.npy")
    mask = partition != CODE_A
    _npy(out / "eval_s1.npy", mask)
    return {"eval_s1": str(out / "eval_s1.npy"), "s1": int(mask.sum()),
            "sha256": file_sha256(out / "eval_s1.npy")}


def evaluate_mix(args) -> dict:
    """Fixed-rule monitor: v1 vs checkpoint blends on B_REST (real labels) and French SLICE (pseudo)."""
    from ber.arena_decision import evaluate
    from ber.pair_tree_arena import decide, false_merge

    out = Path(args.output)
    summary = json.loads((out / "labels/labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise ValueError("labels include HOLDOUT-2 rows")
    pairs = pq.read_table(out / "arena/pairs.parquet", columns=["left_idx", "right_idx"])
    left = pairs.column(0).to_numpy().astype(np.int64)
    right = pairs.column(1).to_numpy().astype(np.int64)
    label = np.load(out / "labels/label.npy")
    true_count = np.load(out / "labels/true_count.npy")
    partition = np.load(out / "s1_partition.npy")
    country = pq.read_table(out / "arena/reference.parquet", columns=["country"]).column(0).to_numpy(zero_copy_only=False)
    tree = np.load(out / "arena/v1_tree.npy")
    v1 = np.load(out / "arena/v1_blend.npy")
    n_s1 = len(true_count)
    runs = {"v1_0.775": v1}
    for name, directory in (("candidate", args.scores), ("baseline", args.baseline_scores)):
        if directory is None:
            continue
        scores = np.nan_to_num(np.load(Path(directory) / "scores.npy").astype(np.float32))
        mask = np.load(Path(directory) / "scored_mask.npy")
        if scores.shape != label.shape:
            raise ValueError(f"{name} scores are not aligned with the mixed arena pairs")
        need = (partition[left] != CODE_A) & (tree >= args.gate)
        if np.any(need & ~mask):
            raise ValueError(f"{name} scores miss gated evaluation pairs")
        runs[name] = np.where(tree >= args.gate, args.weight * scores + (1 - args.weight) * tree, 0).astype(np.float32)
    report = {"rule": {"weight": args.weight, "threshold": args.threshold, "gate": args.gate,
                       "decode": "one owner, cap 11 (pair_tree_arena.decide); v1 uses blend >= 0.775"},
              "caveat": "SLICE is French pseudo-truth on pseudo-labelled pairs only; not French accuracy"}
    for code, part in ((CODE_B_REST, "B_REST"), (CODE_SLICE, "SLICE_france_pseudo")):
        population = partition == code
        rows = population[left]
        block, per = {}, {}
        for name, prob in runs.items():
            threshold = .775 if name == "v1_0.775" else args.threshold
            accept = np.zeros(len(left), bool)
            accept[rows] = decide(prob[rows], left[rows], right[rows], threshold, n_s1)
            result = evaluate(accept, left, label, true_count, population=population, country=country,
                              return_per_s1=True)
            per[name] = np.asarray(result.pop("per_s1"))[population]
            result["false_merge"] = false_merge(accept, left, population, true_count)
            block[name] = result
        for name in ("candidate", "baseline"):
            if name in per:
                diff = per[name] - per["v1_0.775"]
                block[f"{name}_minus_v1"] = {"mean": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff)))}
        if "candidate" in per and "baseline" in per:
            diff = per["candidate"] - per["baseline"]
            block["candidate_minus_baseline"] = {"mean": float(diff.mean()),
                                                 "se": float(diff.std(ddof=1) / np.sqrt(len(diff)))}
        report[part] = block
    _json(Path(args.report), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "dry-run", "population", "evaluate"))
    parser.add_argument("--scores", help="evaluate: score_external output on the mixed arena (candidate)")
    parser.add_argument("--baseline-scores", help="evaluate: score_external output of the baseline checkpoint")
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--gate", type=float, default=.01)
    parser.add_argument("--report", help="evaluate: output JSON path")
    parser.add_argument("--output", type=Path, default=Path("/workspace/az-ml-new/france_mix"))
    parser.add_argument("--test-inputs", type=Path, default=Path("/opt/az-ml-new/test_inputs"))
    parser.add_argument("--arena", type=Path, default=Path("/workspace/az-ml-new/arena/v1"))
    parser.add_argument("--labels", type=Path, default=Path("/workspace/az-ml-new/arena/labels"))
    parser.add_argument("--arena-partition", type=Path,
                        default=Path("/workspace/az-ml-new/arena/finalize_v1/s1_partition.npy"))
    parser.add_argument("--sealed-reservation", type=Path, default=Path("/workspace/az-ml-new/v2_holdout2"))
    parser.add_argument("--pseudo-labels", type=Path, help="Optional france_pseudo_labels parquet to cross-check")
    parser.add_argument("--french-share", type=float, default=.30)
    parser.add_argument("--monitor-s1", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--tokenizer", help="Checkpoint dir whose tokenizer the dry-run uses")
    parser.add_argument("--model-dir", help="Checkpoint dir to pass through fast_trainer.model_identity")
    parser.add_argument("--allowlist", default="/opt/az-ml-new/repo/problem-statement/models.yaml")
    args = parser.parse_args()
    if not 0 < args.french_share < 1:
        parser.error("--french-share must be in (0, 1)")
    if args.command == "evaluate" and (not args.scores or not args.report):
        parser.error("evaluate requires --scores and --report")
    result = {"build": build, "dry-run": dry_run, "population": write_population,
              "evaluate": evaluate_mix}[args.command](args)
    print(json.dumps(result, indent=2, default=str)[:20000], flush=True)


if __name__ == "__main__":
    main()
