"""Twenty CPU decision screens on the arena.

Thresholds are chosen on partition A only. Each rule is scored once on B_REST.
SLICE is never read. HOLDOUT-2 labels are refused.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from ber.arena_decision import (
    component_bootstrap,
    evaluate,
    expected_f05_decode,
    iterated_best_response,
    one_owner,
)

ART = Path(os.environ.get("ARENA", "/opt/az-ml-new/artifacts/arena"))
OUT = Path(os.environ.get("WAVE_OUT", "/opt/az-ml-new/runs/cpu_wave"))
A, B_REST = 0, 1
BOOT = 300


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def guard() -> None:
    summary_path = ART / "labels" / "labels_summary.json"
    if "holdout2" in str(summary_path).lower():
        raise SystemExit("refusing HOLDOUT-2 path")
    summary = json.loads(summary_path.read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise SystemExit("label summary does not prove HOLDOUT-2 was unread")


def build_indices(n_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    OUT.mkdir(parents=True, exist_ok=True)
    left_path, right_path = OUT / "left.npy", OUT / "right.npy"
    if left_path.stat().st_size == n_pairs * 4 and right_path.stat().st_size == n_pairs * 4 if left_path.exists() and right_path.exists() else False:
        return np.load(left_path, mmap_mode="r"), np.load(right_path, mmap_mode="r")
    log(f"indexing {n_pairs} pairs")
    left = np.lib.format.open_memmap(OUT / "left.npy.tmp", mode="w+", dtype=np.int32, shape=(n_pairs,))
    right = np.lib.format.open_memmap(OUT / "right.npy.tmp", mode="w+", dtype=np.int32, shape=(n_pairs,))
    pos = 0
    parquet = pq.ParquetFile(ART / "v1" / "pairs.parquet")
    for batch in parquet.iter_batches(batch_size=2_000_000, columns=["left_idx", "right_idx"]):
        n = batch.num_rows
        left[pos:pos + n] = batch.column("left_idx").to_numpy()
        right[pos:pos + n] = batch.column("right_idx").to_numpy()
        pos += n
    if pos != n_pairs:
        raise SystemExit(f"indexed {pos} pairs, file has {n_pairs}")
    left.flush()
    right.flush()
    del left, right
    os.replace(OUT / "left.npy.tmp", left_path)
    os.replace(OUT / "right.npy.tmp", right_path)
    return np.load(left_path, mmap_mode="r"), np.load(right_path, mmap_mode="r")


def pack(name: str, accept, left, label, true_count, partition, component, country, baseline_per, note: str) -> dict:
    pop = partition == B_REST
    scored = evaluate(accept, left, label, true_count, population=pop, return_per_s1=True)
    per = scored.pop("per_s1")
    boot = component_bootstrap(baseline_per, per, component, n=BOOT, seed=0, population=pop)
    us = partition == B_REST
    row = {
        "experiment": name,
        "note": note,
        "b_rest": {k: scored[k] for k in ("s1", "macro_f05", "precision", "recall", "false_links", "singleton_accuracy")},
        "delta_vs_baseline": boot["delta"],
        "ci95": [boot["ci_low"], boot["ci_high"]],
        "p_le_zero": boot["p_le_zero"],
        "slice_scored": False,
        "holdout2_rows_read": 0,
    }
    if country is not None:
        for code in ("US", "India"):
            mask = pop & (country == code)
            part = evaluate(accept, left, label, true_count, population=mask)
            row[code] = part["macro_f05"]
    del us
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(row, indent=2) + "\n")
    log(f"{name} f05={row['b_rest']['macro_f05']:.6f} delta={row['delta_vs_baseline']:+.6f} ci={row['ci95']}")
    return row


def choose_on_a(cands, left, label, true_count, partition):
    """cands: list of (label, accept). Return the accept with the best partition-A macro F0.5."""
    best_s, best_a, best_name = -1.0, None, None
    pop = partition == A
    for name, accept in cands:
        score = evaluate(accept, left, label, true_count, population=pop)["macro_f05"]
        if score > best_s:
            best_s, best_a, best_name = score, accept, name
    return best_name, best_a, best_s


def main() -> None:
    guard()
    OUT.mkdir(parents=True, exist_ok=True)
    blend = np.load(ART / "v1" / "probabilities.npy", mmap_mode="r")
    tree = np.load(ART / "v1" / "base_probabilities.npy", mmap_mode="r")
    label = np.load(ART / "labels" / "label.npy", mmap_mode="r")
    true_count = np.load(ART / "labels" / "true_count.npy")
    partition = np.load(ART / "finalize_v1" / "s1_partition.npy")
    component = np.load(ART / "finalize_v1" / "component_ids.npy")
    n_pairs = int(pq.ParquetFile(ART / "v1" / "pairs.parquet").metadata.num_rows)
    if len(blend) != n_pairs or len(label) != n_pairs or len(partition) != len(true_count):
        raise SystemExit("arena arrays are not aligned")
    left, right = build_indices(n_pairs)
    n_s1 = len(partition)
    country = pq.read_table(ART / "v1" / "reference.parquet", columns=["country"]).column("country").to_numpy()
    if len(country) != n_s1:
        raise SystemExit("reference country is not aligned to S1s")

    log("baseline one-owner 0.775")
    baseline = one_owner(blend >= 0.775, left, right, blend)
    base_scored = evaluate(baseline, left, label, true_count, population=partition == B_REST, return_per_s1=True)
    baseline_per = base_scored.pop("per_s1")
    pack("01_baseline_one_owner_775", baseline, left, label, true_count, partition, component, country,
         baseline_per, "Frozen decision: blend >= 0.775 then one owner. Reference for every delta.")

    # Precompute tops for list rules.
    log("per-S1 and per-record tops")
    top1 = np.full(n_s1, -np.inf, np.float32)
    np.maximum.at(top1, left, blend)
    below = blend < top1[left]
    top2 = np.full(n_s1, -np.inf, np.float32)
    np.maximum.at(top2, left[below], blend[below])
    n_rec = int(right.max()) + 1
    rec_max = np.full(n_rec, -np.inf, np.float32)
    np.maximum.at(rec_max, right, blend)

    def at_threshold(t):
        return one_owner(blend >= t, left, right, blend)

    grid = []
    for t in (0.70, 0.74, 0.78, 0.82, 0.86, 0.90):
        grid.append((f"{t:.2f}", at_threshold(t)))
    name, chosen, a_score = choose_on_a(grid, left, label, true_count, partition)
    pack("02_threshold_chosen_on_A", chosen, left, label, true_count, partition, component, country,
         baseline_per, f"One-owner threshold {name} chosen on A (A F0.5 {a_score:.6f}), scored once on B_REST.")

    # Gate band the 0.01 tree gate never sent to a high blend.
    band = (tree >= 0.003) & (tree < 0.01)
    band_grid = []
    for t in (0.30, 0.45, 0.60):
        raw = ((blend >= 0.775) & (tree >= 0.01)) | (band & (blend >= t))
        band_grid.append((f"band>={t:.2f}", one_owner(raw, left, right, blend)))
    name, chosen, a_score = choose_on_a(band_grid, left, label, true_count, partition)
    pack("03_gate_band_003_010", chosen, left, label, true_count, partition, component, country,
         baseline_per, f"Rescue tree scores in [0.003, 0.01). Selected {name} on A ({a_score:.6f}).")

    if os.environ.get("WAVE_SKIP_HEAVY") != "1":
        log("expected F0.5 decode")
        decode_grid = []
        for lam in (0.02, 0.055, 0.10):
            accept, _info = expected_f05_decode(blend, left, lam=lam, n_s1=n_s1)
            decode_grid.append((f"lam={lam}", accept))
        name, chosen, a_score = choose_on_a(decode_grid, left, label, true_count, partition)
        pack("04_expected_f05", chosen, left, label, true_count, partition, component, country,
             baseline_per, f"Bayes prefix decode. Selected {name} on A ({a_score:.6f}).")
        lam = float(name.split("=")[1])
        owned, info = iterated_best_response(blend, left, right, lam=lam, n_s1=n_s1)
        pack("05_iterated_best_response", owned, left, label, true_count, partition, component, country,
             baseline_per, f"Expected-F0.5 decode plus one owner per record, {name}. rounds={len(info['rounds'])}.")
    else:
        log("skipping decode; the uncalibrated run already lost")

    # Mutual agreement and reverse ownership.
    s1_best_score = top1
    rec_best_is_this = np.isclose(blend, rec_max[right], rtol=0, atol=1e-5)
    s1_best_is_this = np.isclose(blend, s1_best_score[left], rtol=0, atol=1e-5)
    mutual_grid = []
    for t in (0.50, 0.70, 0.775, 0.85):
        mutual_grid.append((f"mutual>={t:.2f}", (blend >= t) & rec_best_is_this & s1_best_is_this))
    name, chosen, a_score = choose_on_a(mutual_grid, left, label, true_count, partition)
    pack("06_mutual_top1", chosen, left, label, true_count, partition, component, country,
         baseline_per, f"Keep a pair only when it is both the S1's best and the record's best. {name} on A ({a_score:.6f}).")

    reverse_grid = []
    for t in (0.50, 0.70, 0.775, 0.85):
        reverse_grid.append((f"reverse>={t:.2f}", rec_best_is_this & (blend >= t)))
    name, chosen, a_score = choose_on_a(reverse_grid, left, label, true_count, partition)
    pack("07_reverse_record_owner", chosen, left, label, true_count, partition, component, country,
         baseline_per, f"Each record is given to its highest-scoring S1. {name} on A ({a_score:.6f}).")

    unowned_grid = []
    for t in (0.40, 0.55, 0.70):
        owned_rec = rec_max >= t
        raw = (blend >= 0.775) & owned_rec[right]
        unowned_grid.append((f"unowned_below_{t:.2f}", one_owner(raw, left, right, blend)))
    name, chosen, a_score = choose_on_a(unowned_grid, left, label, true_count, partition)
    pack("08_unowned_record_veto", chosen, left, label, true_count, partition, component, country,
         baseline_per, f"Drop every claim on a record whose best score is below t. {name} on A ({a_score:.6f}).")

    margin = rec_max[right] - blend
    # second-best record score is not stored; margin here is how far this pair sits below the record max.
    # For the winner, margin is ~0. Use S1 top1-top2 as the list margin.
    gap = top1 - np.where(np.isfinite(top2), top2, 0)
    margin_grid = []
    for g in (0.02, 0.05, 0.10, 0.20):
        weak = gap < g
        accept = baseline.copy()
        accept[weak[left]] = False
        margin_grid.append((f"drop_gap<{g:.2f}", accept))
    name, chosen, a_score = choose_on_a(margin_grid, left, label, true_count, partition)
    pack("09_small_gap_drop", chosen, left, label, true_count, partition, component, country,
         baseline_per, f"Empty an S1 list when top1-top2 is below g. {name} on A ({a_score:.6f}).")

    empty_grid = []
    for t in (0.60, 0.70, 0.80, 0.90):
        accept = baseline.copy()
        accept[top1[left] < t] = False
        empty_grid.append((f"empty_if_top<{t:.2f}", accept))
    name, chosen, a_score = choose_on_a(empty_grid, left, label, true_count, partition)
    pack("10_nomatch_if_top_weak", chosen, left, label, true_count, partition, component, country,
         baseline_per, f"Force an empty list when the best blend is below t. {name} on A ({a_score:.6f}).")

    cap_grid = []
    rows0 = np.flatnonzero(baseline)
    rank_key = np.lexsort((rows0, -blend[rows0], left[rows0]))
    rows0 = rows0[rank_key]
    first = np.empty(len(rows0), dtype=bool)
    first[0] = True
    first[1:] = left[rows0][1:] != left[rows0][:-1]
    group_start = np.flatnonzero(first)
    within = np.arange(len(rows0)) - np.repeat(group_start, np.diff(np.append(group_start, len(rows0))))
    for k in (1, 2, 3, 4, 6, 8):
        accept = np.zeros(n_pairs, dtype=bool)
        accept[rows0[within < k]] = True
        cap_grid.append((f"cap_{k}", accept))
    name, chosen, a_score = choose_on_a(cap_grid, left, label, true_count, partition)
    pack("11_list_cap", chosen, left, label, true_count, partition, component, country,
         baseline_per, f"Cap the baseline list at k. {name} on A ({a_score:.6f}).")

    # Corroboration: baseline links whose S1 also has a second pair above 0.5.
    second = np.zeros(n_pairs, dtype=bool)
    ge = blend >= 0.50
    counts = np.bincount(left[ge], minlength=n_s1)
    corr_grid = []
    for t in (0.40, 0.50, 0.60):
        support = np.bincount(left[blend >= t], minlength=n_s1) >= 2
        accept = baseline & support[left]
        corr_grid.append((f"two_above_{t:.2f}", accept))
    name, chosen, a_score = choose_on_a(corr_grid, left, label, true_count, partition)
    pack("12_list_corroboration", chosen, left, label, true_count, partition, component, country,
         baseline_per, f"Keep baseline links only if the S1 has two pairs above t. {name} on A ({a_score:.6f}).")

    # Fixed predeclared rules, no B_REST tuning.
    predeclared = {
        "13_fixed_threshold_070": at_threshold(0.70),
        "14_fixed_threshold_090": at_threshold(0.90),
        "15_tree_or_blend": one_owner((blend >= 0.775) | ((tree >= 0.20) & (blend >= 0.55)), left, right, blend),
    }
    notes = {
        "13_fixed_threshold_070": "Predeclared one-owner at 0.70.",
        "14_fixed_threshold_090": "Predeclared one-owner at 0.90.",
        "15_tree_or_blend": "Predeclared: blend >= 0.775, or tree >= 0.20 and blend >= 0.55, then one owner.",
    }
    for key, accept in predeclared.items():
        pack(key, accept, left, label, true_count, partition, component, country, baseline_per, notes[key])

    # Shadow transfer: fit the one-owner threshold on one country's A rows, score the other country's B_REST.
    us_a = (partition == A) & (country == "US")
    in_a = (partition == A) & (country == "India")

    def best_threshold(pop):
        best_t, best_s = 0.775, -1.0
        for t in (0.70, 0.74, 0.775, 0.82, 0.86, 0.90):
            score = evaluate(at_threshold(t), left, label, true_count, population=pop)["macro_f05"]
            if score > best_s:
                best_t, best_s = t, score
        return best_t, best_s

    t_us, s_us = best_threshold(us_a)
    accept = at_threshold(t_us)
    india_b = (partition == B_REST) & (country == "India")
    india_score = evaluate(accept, left, label, true_count, population=india_b)["macro_f05"]
    base_india = evaluate(baseline, left, label, true_count, population=india_b)["macro_f05"]
    row = {"experiment": "16_shadow_us_threshold_on_india", "threshold_fit_on": "US partition A",
           "threshold": t_us, "fit_f05": s_us, "india_b_rest": india_score,
           "india_baseline": base_india, "delta": india_score - base_india, "slice_scored": False}
    (OUT / "16_shadow_us_threshold_on_india.json").write_text(json.dumps(row, indent=2) + "\n")
    log(f"16 shadow US->India t={t_us} india={india_score:.6f} delta={india_score - base_india:+.6f}")

    t_in, s_in = best_threshold(in_a)
    accept = at_threshold(t_in)
    us_b = (partition == B_REST) & (country == "US")
    us_score = evaluate(accept, left, label, true_count, population=us_b)["macro_f05"]
    base_us = evaluate(baseline, left, label, true_count, population=us_b)["macro_f05"]
    row = {"experiment": "17_shadow_india_threshold_on_us", "threshold_fit_on": "India partition A",
           "threshold": t_in, "fit_f05": s_in, "us_b_rest": us_score, "us_baseline": base_us,
           "delta": us_score - base_us, "slice_scored": False}
    (OUT / "17_shadow_india_threshold_on_us.json").write_text(json.dumps(row, indent=2) + "\n")
    log(f"17 shadow India->US t={t_in} us={us_score:.6f} delta={us_score - base_us:+.6f}")

    # Source complement, only when pool ids are present.
    pool_path = ART / "v1" / "pool.parquet"
    if pool_path.is_file() and pool_path.stat().st_size > 600_000_000:
        log("source complement")
        ids = pq.read_table(pool_path, columns=["entity_id"]).column("entity_id")
        import pyarrow.compute as pc
        is_s3 = pc.starts_with(ids, "S3-").fill_null(False).to_numpy(zero_copy_only=False)
        if len(is_s3) != n_rec:
            log(f"skip source complement: pool {len(is_s3)} != records {n_rec}")
        else:
            s3 = is_s3[right]
            source_grid = []
            for t in (0.55, 0.65, 0.75):
                accept = baseline.copy()
                # S1s whose baseline links are all one source, plus their best other-source pair >= t.
                base_s2 = np.bincount(left[baseline & ~s3], minlength=n_s1)
                base_s3 = np.bincount(left[baseline & s3], minlength=n_s1)
                one_sided = (base_s2 > 0) ^ (base_s3 > 0)
                want_s3 = one_sided & (base_s2 > 0)
                cand = (blend >= t) & ((want_s3[left] & s3) | ((one_sided & (base_s3 > 0))[left] & ~s3))
                # one extra per S1: the best such candidate, if one_owner still allows it
                if cand.any():
                    rows = np.flatnonzero(cand)
                    pick = np.lexsort((rows, -blend[rows], left[rows]))
                    rows = rows[pick]
                    first = np.empty(len(rows), dtype=bool)
                    first[0] = True
                    first[1:] = left[rows][1:] != left[rows][:-1]
                    accept[rows[first]] = True
                    accept = one_owner(accept, left, right, blend)
                source_grid.append((f"add_other_source>={t:.2f}", accept))
            name, chosen, a_score = choose_on_a(source_grid, left, label, true_count, partition)
            pack("18_cross_source_complement", chosen, left, label, true_count, partition, component, country,
                 baseline_per, f"If a list has only S2 or only S3, add the best other-source pair. {name} on A ({a_score:.6f}).")
    else:
        log("pool parquet not ready; source complement skipped")

    # Claim propagation: a record owned at >= 0.95 suppresses rival baseline claims below 0.90.
    rival = rec_best_is_this & (blend >= 0.95)
    strong_owner = np.zeros(n_rec, dtype=bool)
    strong_owner[right[rival]] = True
    suppressed = baseline & strong_owner[right] & ~rec_best_is_this & (blend < 0.90)
    accept = baseline.copy()
    accept[suppressed] = False
    pack("19_claim_propagation", accept, left, label, true_count, partition, component, country,
         baseline_per, "A record with a >= 0.95 owner loses rival baseline claims below 0.90.")

    # Transductive-style EM on arena probabilities: one update of responsibilities.
    # r_i = p_i / sum_{claims of record} p, then decode those shares. Fit nothing on B_REST.
    share = np.zeros(n_pairs, dtype=np.float32)
    mass = np.bincount(right, weights=blend.astype(np.float64), minlength=n_rec)
    share = blend / np.maximum(mass[right], 1e-8)
    accept, _info = expected_f05_decode(share, left, lam=0.055, n_s1=n_s1)
    accept = one_owner(accept, left, right, share)
    pack("20_responsibility_decode", accept, left, label, true_count, partition, component, country,
         baseline_per, "One EM step: pair probability divided by the record's total claim mass, then expected-F0.5 and one owner.")

    log("WAVE_DONE")


if __name__ == "__main__":
    main()
