"""P0.2/P0.3: fixed TUNE AMP parity and length-sorted BF16 throughput.

Read the completed, full-target-pool medium dense candidates and v1 BGE1800
checkpoint without changing either. Screen 2,000 eligible TUNE pairs, then
rerank *all* eligible TUNE pairs in both precisions at the frozen threshold.
The 4,000-pair throughput probe is the exact saved v1 probe, not a new sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .cross_encoder_experiment import blend_probabilities, detailed_score, load_reranker
from .experiment import load_data
from .full_embeddings import atomic_json, checkpoint_identity, file_digest
from .hf_experiment import frame_digest
from .metrics import pair_labels
from .rerank_inference import score_reranker_batched, score_reranker_selective_fp32


TUNE_ID_SHA256 = "5bb5759fa52ac19df8839520c7ccf147bae01a5499a2900af5d1b4ebeb1dbaa2"
FULL_TARGET_COUNT = 10_320_219
ALPHA, GATE, THRESHOLD = 0.75, 0.01, 0.775
MAX_LENGTH, PAD_MULTIPLE = 256, 8


def accept_comparison(first: np.ndarray, second: np.ndarray) -> dict:
    """Report both exact pair decisions and Jaccard of accepted pair identities."""
    a, b = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("Decision vectors must contain the same aligned pairs")
    common, union = int(np.sum(a & b)), int(np.sum(a | b))
    differences = int(np.sum(a != b))
    return {"pairs": len(a), "accepted_first": int(a.sum()), "accepted_second": int(b.sum()),
            "accepted_intersection": common, "accepted_union": union,
            "accept_overlap_jaccard": common / union if union else 1.0,
            "identical_decision_fraction": 1 - differences / len(a) if len(a) else 1.0,
            "decision_disagreements": differences,
            "first_only": int(np.sum(a & ~b)), "second_only": int(np.sum(b & ~a))}


def amp_parity_pass(f05_delta: float, *comparisons: dict) -> bool:
    """The 2k stress probe AND full TUNE must meet the accept-set gate."""
    return f05_delta <= 0.0005 and all(row["accept_overlap_jaccard"] >= 0.999 for row in comparisons)


def false_links_by_country(reference: pd.DataFrame, pairs: pd.DataFrame,
                           true_pairs: np.ndarray, accepted: np.ndarray) -> dict[str, int]:
    if len(pairs) != len(true_pairs) or len(pairs) != len(accepted):
        raise ValueError("Truth and accepted decisions must align with the saved candidates")
    countries = reference.country.to_numpy()[pairs.left_idx.to_numpy(dtype=np.int64)]
    return {country: int(np.sum((countries == country) & np.asarray(accepted) & ~np.asarray(true_pairs, dtype=bool)))
            for country in sorted(np.unique(countries))}


def select_probe(eligible: np.ndarray, saved_blend: np.ndarray, *, seed: int, count: int = 2000) -> np.ndarray:
    """Half closest to the fixed threshold, half random; never use labels."""
    indices = np.flatnonzero(np.asarray(eligible, dtype=bool))
    if len(indices) < count or count < 2 or count % 2 or len(saved_blend) != len(eligible):
        raise ValueError("Probe requires enough aligned eligible pairs and an even count")
    distances = np.abs(np.asarray(saved_blend, dtype=np.float64)[indices] - THRESHOLD)
    near = indices[np.argsort(distances, kind="stable")[:count // 2]]
    remaining = indices[~np.isin(indices, near)]
    random = np.random.default_rng(seed).choice(remaining, count // 2, replace=False)
    return np.sort(np.r_[near, random])


def compact_pairs(reference: pd.DataFrame, pool: pd.DataFrame, pairs: pd.DataFrame):
    """Compact records, retaining every supplied pair in exactly its input order."""
    left, left_idx = np.unique(pairs.left_idx.to_numpy(dtype=np.int64), return_inverse=True)
    right, right_idx = np.unique(pairs.right_idx.to_numpy(dtype=np.int64), return_inverse=True)
    return (reference.iloc[left].reset_index(drop=True), pool.iloc[right].reset_index(drop=True),
            pd.DataFrame({"left_idx": left_idx, "right_idx": right_idx}))


def timed_scores(torch, model, tokenizer, frames, *, amp: str, batch_size: int, device: str,
                 defer_cpu_transfer: bool = False) -> tuple[np.ndarray, dict]:
    reference, pool, pairs = frames
    options = SimpleNamespace(device=device, amp=amp, max_length=MAX_LENGTH,
                              inference_batch_size=batch_size, length_bucketing=True,
                              pad_to_multiple_of=PAD_MULTIPLE, defer_cpu_transfer=defer_cpu_transfer)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    values = score_reranker_batched(model, tokenizer, reference, pool, pairs,
                                   np.ones(len(pairs), dtype=bool), options)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return values, {"seconds": seconds, "pairs_per_second": len(pairs) / seconds,
                     "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()}


def timed_selective_scores(torch, model, tokenizer, frames, base, *, band: float, device: str):
    reference, pool, pairs = frames
    options = SimpleNamespace(device=device, amp="bf16", max_length=MAX_LENGTH,
                              inference_batch_size=128, length_bucketing=True,
                              pad_to_multiple_of=PAD_MULTIPLE, defer_cpu_transfer=False)
    torch.cuda.reset_peak_memory_stats()
    timings = {}
    values, near, bf16 = score_reranker_selective_fp32(
        model, tokenizer, reference, pool, pairs, np.ones(len(pairs), dtype=bool), options,
        base, alpha=ALPHA, threshold=THRESHOLD, band=band, timings=timings)
    timings["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    timings["all_pairs_per_second"] = len(pairs) / timings["total_seconds"]
    return values, near, bf16, timings


def accepted_at_threshold(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) >= THRESHOLD


def decisions(base: np.ndarray, cross: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = blend_probabilities(base, cross, selected, ALPHA)
    return values, accepted_at_threshold(values)


def verify_inputs(data: Path, saved: Path, probe: Path, checkpoint: Path, expected_checkpoint_sha: str):
    """Verify reused split, pair order, full-pool mining and frozen model identity."""
    metadata = json.loads((data / "metadata.json").read_text())
    complete = json.loads((data / "complete.json").read_text())
    config = json.loads((saved / "experiment_config.json").read_text())
    prior = json.loads((probe / "report.json").read_text())
    if (complete.get("state") != "complete" or complete.get("metadata") != metadata
            or complete["config"].get("labels_used_for_mining") is not False
            or metadata.get("positive_candidate_injection") is not False
            or metadata.get("truth_targets_retained_after_mining") is not True
            or metadata.get("mining_full_pool_records") != FULL_TARGET_COUNT
            or config.get("full_pool_records") != FULL_TARGET_COUNT
            or config.get("evaluation_used") is not False):
        raise ValueError("Expected the committed TRAIN/TUNE candidates mined without injected edges against the full pool")
    if (config.get("seed") != prior.get("seed") or config.get("max_length") != MAX_LENGTH
            or config.get("rerank_cutoff") != GATE or prior.get("alpha") != ALPHA
            or prior.get("gate") != GATE or prior.get("threshold") != THRESHOLD
            or prior.get("max_length") != MAX_LENGTH or prior.get("labels_accessed") is not False):
        raise ValueError("Saved probe, gate or checkpoint experiment differs from the frozen decision rule")
    identity = checkpoint_identity(checkpoint)
    if identity["sha256"] != expected_checkpoint_sha or Path(prior["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("Checkpoint identity differs from the fixed BGE1800 probe")
    reference, pool, truth = load_data(data)
    pairs = pd.read_parquet(saved / "pairs.parquet")
    if (len(reference) != metadata["references"] or len(pool) != metadata["pool_records"]
            or len(pairs) != metadata["candidate_pairs"] or not reference.entity_id.is_unique
            or not pool.entity_id.is_unique or pairs.duplicated(["left_idx", "right_idx"]).any()
            or pairs.left_idx.min() < 0 or pairs.left_idx.max() >= len(reference)
            or pairs.right_idx.min() < 0 or pairs.right_idx.max() >= len(pool)):
        raise ValueError("Completed pair/record counts, IDs or indices disagree")
    if (frame_digest(reference, ["entity_id", "business_name", "business_address", "country", "split"])
            != config["reference_digest"] or
            frame_digest(pool, ["entity_id", "business_name", "business_address", "country"])
            != config["pool_digest"] or
            frame_digest(pairs, ["left_idx", "right_idx"]) != config["pair_digest"] or
            file_digest(data / "pairs.parquet") != file_digest(saved / "pairs.parquet")):
        raise ValueError("Full-pool reference, records or ordered candidates differ from saved CE inputs")
    tune = reference.split.eq("tune").to_numpy()
    tune_ids = reference.entity_id.to_numpy()[tune]
    split_sha = hashlib.sha256("\n".join(sorted(tune_ids)).encode()).hexdigest()
    if (len(tune_ids) != 12_000 or split_sha != TUNE_ID_SHA256
            or not set(tune_ids) <= truth.keys() or set(reference.split) != {"train", "tune"}):
        raise ValueError("Saved TUNE split or singleton truth coverage changed")
    positions = np.flatnonzero(tune[pairs.left_idx.to_numpy(dtype=np.int64)])
    if len(positions) != 600_000:
        raise ValueError("Expected exactly 50 saved full-pool candidates per TUNE reference")
    base = np.load(saved / "base_probabilities.npy", mmap_mode="r")
    cross = np.load(saved / "cross_probabilities.npy", mmap_mode="r")
    blended = np.load(saved / "probabilities.npy", mmap_mode="r")
    selected = np.load(saved / "reranked_mask.npy", mmap_mode="r")
    if (any(x.shape != (len(pairs),) for x in (base, cross, blended, selected))
            or any(x.dtype != np.float32 for x in (base, cross, blended))
            or selected.dtype != bool or
            hashlib.sha256(np.asarray(base).tobytes()).hexdigest() != config["base_probability_digest"]):
        raise ValueError("Saved cross/base probability arrays or row alignment changed")
    base_tune, cross_tune = np.asarray(base[positions]), np.asarray(cross[positions])
    mask_tune = np.asarray(selected[positions])
    if (not np.array_equal(mask_tune, base_tune >= GATE)
            or not np.array_equal(np.asarray(blended[positions]),
                                  blend_probabilities(base_tune, cross_tune, mask_tune, ALPHA))):
        raise ValueError("Saved probabilities do not reproduce the fixed tree gate and blend")
    sample_positions = np.load(probe / "original_pair_positions.npy")
    sample_pairs = pd.read_parquet(probe / "pairs.parquet")
    sample_ref = pd.read_parquet(probe / "reference.parquet")
    sample_pool = pd.read_parquet(probe / "pool.parquet")
    sample_base = np.load(probe / "base_probabilities.npy")
    if (sample_positions.shape != (4000,) or len(np.unique(sample_positions)) != 4000
            or sample_positions.min() < 0 or sample_positions.max() >= len(pairs)
            or len(sample_pairs) != 4000 or frame_digest(sample_pairs, ["left_idx", "right_idx"]) != prior["pair_digest"]
            or sample_pairs.left_idx.min() < 0 or sample_pairs.left_idx.max() >= len(sample_ref)
            or sample_pairs.right_idx.min() < 0 or sample_pairs.right_idx.max() >= len(sample_pool)
            or np.any(~tune[pairs.left_idx.to_numpy(dtype=np.int64)[sample_positions]])
            or not np.array_equal(sample_base, np.asarray(base[sample_positions]))
            or not np.array_equal(sample_ref.entity_id.to_numpy()[sample_pairs.left_idx.to_numpy(dtype=np.int64)],
                                  reference.entity_id.to_numpy()[pairs.left_idx.to_numpy(dtype=np.int64)[sample_positions]])
            or not np.array_equal(sample_pool.entity_id.to_numpy()[sample_pairs.right_idx.to_numpy(dtype=np.int64)],
                                  pool.entity_id.to_numpy()[pairs.right_idx.to_numpy(dtype=np.int64)[sample_positions]])):
        raise ValueError("Fixed 4,000-pair probe differs from saved full-pool TUNE candidates")
    return (metadata, config, identity, reference, pool, truth, pairs, positions,
            base_tune, cross_tune, mask_tune, sample_positions, sample_ref, sample_pool, sample_pairs, sample_base)


def run(args) -> dict:
    import torch

    data, saved, probe, checkpoint = map(Path, (args.data, args.saved, args.probe, args.checkpoint))
    output, durable = Path(args.output), Path(args.durable)
    if output.resolve() == durable.resolve() or any(path.exists() and any(path.iterdir()) for path in (output, durable)):
        raise ValueError("Use separate, empty scratch and durable benchmark directories")
    if not args.device.startswith("cuda") or not torch.cuda.is_bf16_supported():
        raise ValueError("The BF16/FP32 comparison requires BF16-capable CUDA")
    if args.repeats < 2 or args.threads < 1:
        raise ValueError("Use at least two timing repeats and one CPU thread")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    (metadata, source_config, identity, reference, pool, truth, pairs, positions, base, saved_cross,
     selected, sample_positions, sample_ref, sample_pool, sample_pairs, sample_base) = verify_inputs(
         data, saved, probe, checkpoint, args.checkpoint_sha256)
    tune_pairs = pairs.iloc[positions][["left_idx", "right_idx"]].reset_index(drop=True)
    tune_mask = reference.split.eq("tune")
    upper = blend_probabilities(base, np.ones(len(base), dtype=np.float32), selected, ALPHA)
    eligible = selected & (np.asarray(upper, dtype=np.float64) >= THRESHOLD)
    if not np.any(eligible):
        raise ValueError("No candidates pass the label-free upper-bound CE gate")
    saved_values, saved_decisions = decisions(base, saved_cross, selected)
    _, short_decisions = decisions(base, saved_cross, eligible)
    if not np.array_equal(saved_decisions, short_decisions):
        raise ValueError("Upper-bound shortcut changed saved TUNE decisions")
    output.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    np.save(output / "tune_pair_positions.npy", positions)
    tune_pairs.to_parquet(output / "tune_pairs.parquet", index=False)
    np.save(output / "tune_base_probabilities.npy", base)
    np.save(output / "eligible_mask.npy", eligible)
    source_files = {"data/metadata.json": data / "metadata.json", "data/complete.json": data / "complete.json",
                    "data/reference.parquet": data / "reference.parquet", "data/pool.parquet": data / "pool.parquet",
                    "data/truth.json": data / "truth.json", "data/pairs.parquet": data / "pairs.parquet",
                    "saved/experiment_config.json": saved / "experiment_config.json",
                    "saved/pairs.parquet": saved / "pairs.parquet",
                    "saved/base_probabilities.npy": saved / "base_probabilities.npy",
                    "saved/cross_probabilities.npy": saved / "cross_probabilities.npy",
                    "saved/probabilities.npy": saved / "probabilities.npy",
                    "saved/reranked_mask.npy": saved / "reranked_mask.npy",
                    "probe/pairs.parquet": probe / "pairs.parquet",
                    "probe/reference.parquet": probe / "reference.parquet",
                    "probe/pool.parquet": probe / "pool.parquet",
                    "probe/base_probabilities.npy": probe / "base_probabilities.npy",
                    "probe/shared_length_batch128_cross.npy": probe / "shared_length_batch128_cross.npy",
                    "probe/original_pair_positions.npy": probe / "original_pair_positions.npy",
                    "probe/report.json": probe / "report.json"}
    config = {"schema": 1, "seed": args.seed, "data": str(data.resolve()), "saved": str(saved.resolve()),
              "fixed_probe": str(probe.resolve()), "checkpoint": identity,
              "source_config_sha256": file_digest(saved / "experiment_config.json"),
              "source_sha256": {name: file_digest(path) for name, path in source_files.items()},
              "source_reference_digest": source_config["reference_digest"],
              "source_pool_digest": source_config["pool_digest"],
              "source_pair_digest": source_config["pair_digest"],
              "tune_pair_digest": frame_digest(tune_pairs, ["left_idx", "right_idx"]),
              "tune_id_sha256": TUNE_ID_SHA256,
              "tune_truth_sha256": hashlib.sha256(json.dumps({entity: sorted(truth[entity])
                             for entity in sorted(reference.entity_id[tune_mask])}, sort_keys=True).encode()).hexdigest(),
              "original_target_pool_records": metadata["mining_full_pool_records"],
              "compacted_record_count": len(pool), "positive_candidate_injection": False,
              "truth_targets_retained_after_mining": True,
              "reference_count": int(tune_mask.sum()), "candidate_pairs": len(tune_pairs),
              "eligible_ce_pairs": int(eligible.sum()), "probe_pairs": 2000,
              "fixed_throughput_pairs": len(sample_pairs), "alpha": ALPHA, "gate": GATE,
              "threshold": THRESHOLD, "max_length": MAX_LENGTH, "pad_to_multiple_of": PAD_MULTIPLE,
              "length_bucketing": True, "device": args.device, "threads": args.threads,
              "timing_repeats": args.repeats,
              "amp_gate": {"max_macro_f05_delta": 0.0005, "min_accept_overlap_jaccard": 0.999},
              "batch256_gate": {"min_pairs_per_second": 1700, "min_identical_decision_fraction": 0.999}}
    atomic_json(output / "config.json", config)

    load_start = time.perf_counter()
    model, tokenizer = load_reranker(str(checkpoint), args.device)
    load_seconds = time.perf_counter() - load_start
    baseline = detailed_score(reference, pool, tune_pairs, saved_values, truth, THRESHOLD, tune_mask)
    true_pairs = pair_labels(reference, pool, tune_pairs, truth)
    saved_report = json.loads((saved / "results.json").read_text())["selected_by_tune"]["tune"]
    if abs(baseline["macro_f05"] - saved_report["macro_f05"]) > 1e-12:
        raise ValueError("The saved full-TUNE score cannot be reproduced on the same candidates")

    # Screen includes hard boundary cases and a label-free random sample. Outside
    # the probe, both hybrids retain the *same* original saved BF16 probabilities.
    chosen = select_probe(eligible, saved_values, seed=args.seed)
    np.save(output / "probe_original_pair_positions.npy", positions[chosen])
    probe_frame = tune_pairs.iloc[chosen].copy().reset_index(drop=True)
    probe_frame.insert(0, "original_pair_position", positions[chosen])
    probe_frame["source1_entity_id"] = reference.entity_id.to_numpy()[probe_frame.left_idx]
    probe_frame["target_entity_id"] = pool.entity_id.to_numpy()[probe_frame.right_idx]
    probe_frame.to_parquet(output / "probe_pairs.parquet", index=False)
    screen_frames = compact_pairs(reference, pool, tune_pairs.iloc[chosen])
    screen_cross, screen_times = {}, {}
    for amp in ("fp32", "bf16"):
        screen_cross[amp], screen_times[amp] = timed_scores(
            torch, model, tokenizer, screen_frames, amp=amp, batch_size=128, device=args.device)
        np.save(output / f"probe_cross_{amp}.npy", screen_cross[amp])
    screen_values = {}
    screen_scores = {}
    for amp in ("fp32", "bf16"):
        hybrid_cross = saved_cross.copy()
        hybrid_cross[chosen] = screen_cross[amp]
        screen_values[amp], _ = decisions(base, hybrid_cross, selected)
        screen_scores[amp] = detailed_score(reference, pool, tune_pairs, screen_values[amp], truth,
                                            THRESHOLD, tune_mask)
    sample_first = decisions(base[chosen], screen_cross["fp32"], np.ones(len(chosen), bool))[1]
    sample_second = decisions(base[chosen], screen_cross["bf16"], np.ones(len(chosen), bool))[1]
    sample_compare = accept_comparison(sample_first, sample_second)
    screen_compare = accept_comparison(accepted_at_threshold(screen_values["fp32"]),
                                       accepted_at_threshold(screen_values["bf16"]))
    screen_delta = abs(screen_scores["fp32"]["macro_f05"] - screen_scores["bf16"]["macro_f05"])
    screen = {"scope": "2,000 saved eligible TUNE pairs rescored; remaining TUNE pairs use identical saved BF16 scores",
              "probe_pair_digest": frame_digest(probe_frame, ["original_pair_position", "left_idx", "right_idx"]),
              "pairs": len(chosen), "near_threshold_pairs": len(chosen) // 2,
              "inference": screen_times, "sample_accept_comparison": sample_compare,
              "full_tune_hybrid_accept_comparison": screen_compare,
              "full_tune_hybrid": screen_scores, "macro_f05_abs_delta": screen_delta,
              "passed": amp_parity_pass(screen_delta, sample_compare, screen_compare)}
    atomic_json(output / "probe_screen.json", screen)

    # This is the complete 12,000-reference TUNE comparison: all 600,000
    # retrieved candidates stay present, every potentially accepted CE pair is
    # newly scored in *each* precision, and missed truth links stay in the metric.
    eligible_rows = np.flatnonzero(eligible)
    np.save(output / "eligible_tune_pair_positions.npy", eligible_rows)
    full_frames = compact_pairs(reference, pool, tune_pairs.iloc[eligible_rows])
    full_values, full_scores, full_times = {}, {}, {}
    for amp in ("fp32", "bf16"):
        cross, full_times[amp] = timed_scores(torch, model, tokenizer, full_frames,
                                              amp=amp, batch_size=128, device=args.device)
        np.save(output / f"full_tune_eligible_cross_{amp}.npy", cross)
        full_cross = np.zeros(len(base), dtype=np.float32)
        full_cross[eligible_rows] = cross
        full_values[amp], _ = decisions(base, full_cross, eligible)
        np.save(output / f"full_tune_probabilities_{amp}.npy", full_values[amp])
        full_scores[amp] = detailed_score(reference, pool, tune_pairs, full_values[amp], truth,
                                          THRESHOLD, tune_mask)
    full_compare = accept_comparison(accepted_at_threshold(full_values["fp32"]),
                                    accepted_at_threshold(full_values["bf16"]))
    full_delta = abs(full_scores["fp32"]["macro_f05"] - full_scores["bf16"]["macro_f05"])
    full = {"scope": "all 12,000 TUNE S1s; 600,000 saved candidates retrieved from 10,320,219 targets; all upper-bound-eligible CE pairs recomputed",
            "candidate_pairs": len(tune_pairs), "eligible_ce_pairs": len(eligible_rows),
            "saved_bf16_batch64": baseline, "fp32": full_scores["fp32"], "bf16": full_scores["bf16"],
            "country_false_positive_links": {
                "saved_bf16_batch64": false_links_by_country(reference, tune_pairs, true_pairs, saved_decisions),
                **{amp: false_links_by_country(reference, tune_pairs, true_pairs, accepted_at_threshold(full_values[amp]))
                   for amp in ("fp32", "bf16")}},
            "inference": full_times, "accept_comparison": full_compare, "macro_f05_abs_delta": full_delta,
            "passed": amp_parity_pass(full_delta, full_compare)}
    atomic_json(output / "full_tune.json", full)

    # Fixed v1 4k sample: identical tokenization/order, BF16, pad8 and model.
    # Interleave runs to reduce order-dependent timing bias. Include tokenization
    # and synchronous host/device copies, but not checkpoint loading/warmup.
    sample_frames = (sample_ref, sample_pool, sample_pairs)
    warmup = (sample_ref, sample_pool, sample_pairs.iloc[:256].reset_index(drop=True))
    timed_scores(torch, model, tokenizer, warmup, amp="bf16", batch_size=128, device=args.device)
    runs, batch_cross = {128: [], 256: []}, {}
    for repeat in range(args.repeats):
        for batch in ((128, 256) if repeat % 2 == 0 else (256, 128)):
            scores, timing = timed_scores(torch, model, tokenizer, sample_frames,
                                           amp="bf16", batch_size=batch, device=args.device)
            runs[batch].append(timing)
            if batch not in batch_cross:
                batch_cross[batch] = scores
                np.save(output / f"batch{batch}_bf16_cross.npy", scores)
            elif not np.array_equal(decisions(sample_base, scores, np.ones(len(scores), bool))[1],
                                    decisions(sample_base, batch_cross[batch], np.ones(len(scores), bool))[1]):
                raise ValueError("Repeated inference changed a fixed-threshold decision")
    saved_128 = np.load(probe / "shared_length_batch128_cross.npy")
    sample_selected = np.ones(len(sample_base), dtype=bool)
    batch_compare = accept_comparison(decisions(sample_base, batch_cross[128], sample_selected)[1],
                                      decisions(sample_base, batch_cross[256], sample_selected)[1])
    comparison_saved = accept_comparison(decisions(sample_base, saved_128, sample_selected)[1],
                                         decisions(sample_base, batch_cross[128], sample_selected)[1])
    throughput = {str(batch): {"runs": runs[batch], "median_seconds": float(np.median(
                       [row["seconds"] for row in runs[batch]])),
                       "median_pairs_per_second": 4000 / float(np.median([row["seconds"] for row in runs[batch]]))}
                  for batch in (128, 256)}
    batch256_speed = throughput["256"]["median_pairs_per_second"]
    sample_countries = reference.country.to_numpy()[pairs.left_idx.to_numpy(dtype=np.int64)[sample_positions]]
    country_names, country_counts = np.unique(sample_countries, return_counts=True)
    batch = {"scope": "same 4,000 saved, gated TUNE pairs as frozen v1 throughput probe; BF16 length-sorted pad8",
             "original_pair_positions_sha256": hashlib.sha256(sample_positions.tobytes()).hexdigest(),
             "pair_digest": frame_digest(sample_pairs, ["left_idx", "right_idx"]),
             "country_pair_counts": {name: int(count) for name, count in zip(country_names, country_counts)},
             "inference": throughput, "accept_comparison_128_vs_256": batch_compare,
             "accept_comparison_saved_128_vs_new_128": comparison_saved,
             "passes_minimum": batch256_speed >= 1700 and batch_compare["identical_decision_fraction"] >= 0.999,
             "batch256_faster_than_128": batch256_speed >= throughput["128"]["median_pairs_per_second"]}
    atomic_json(output / "batch_benchmark.json", batch)
    result = {"schema": 1, "config": str((output / "config.json").resolve()),
              "source_checkpoint_sha256": identity["sha256"], "seed": args.seed,
              "scope": "TRAIN/TUNE-only full-pool dense retrieval; no eval or official test labels",
              "checkpoint_load_seconds": load_seconds,
              "probe_screen": screen, "full_tune": full, "batch_benchmark": batch,
              "adopt_bf16_for_v2": screen["passed"] and full["passed"],
              "adopt_batch256_for_v2": (screen["passed"] and full["passed"] and batch["passes_minimum"]
                                        and batch["batch256_faster_than_128"]),
              "retain_batch_size": 256 if (screen["passed"] and full["passed"] and batch["passes_minimum"]
                                           and batch["batch256_faster_than_128"]) else 128}
    atomic_json(output / "report.json", result)
    artifact_hashes = {path.name: file_digest(path) for path in output.iterdir() if path.is_file()}
    manifest = {"schema": 1, "scratch": str(output.resolve()), "durable": str(durable.resolve()),
                "artifacts_sha256": artifact_hashes, "source_checkpoint_sha256": identity["sha256"]}
    atomic_json(output / "manifest.json", manifest)
    for path in sorted(output.iterdir()):
        if path.is_file():
            target = durable / path.name
            temporary = target.with_name(target.name + ".tmp")
            shutil.copyfile(path, temporary)
            if file_digest(temporary) != file_digest(path):
                raise ValueError(f"Durable output copy differs: {path.name}")
            temporary.replace(target)
    return result


def saved_band_sweep(base: np.ndarray, bf16: np.ndarray, fp32: np.ndarray,
                     widths: tuple[float, ...]) -> list[dict]:
    """Label-free screen; neither pair truth nor a new candidate edge is used."""
    from .rerank_inference import selective_fp32_mask

    selected = np.ones(len(base), dtype=bool)
    target = decisions(base, fp32, selected)[1]
    rows = []
    for band in widths:
        near = selective_fp32_mask(base, bf16, selected, alpha=ALPHA, threshold=THRESHOLD, band=band)
        cross = bf16.copy()
        cross[near] = fp32[near]
        rows.append({"band": band, "rescored_pairs": int(near.sum()),
                     "rescore_fraction": float(near.mean()),
                     "vs_full_fp32": accept_comparison(decisions(base, cross, selected)[1], target)})
    return rows


def run_selective(args) -> dict:
    """Use exact prior pair IDs and FP32 controls; time only new selective work."""
    import torch

    prior = Path(args.selective_from)
    data, saved, probe, checkpoint = map(Path, (args.data, args.saved, args.probe, args.checkpoint))
    output, durable = Path(args.output), Path(args.durable)
    if (output.resolve() in {prior.resolve(), durable.resolve()} or durable.resolve() == prior.resolve()
            or any(path.exists() and any(path.iterdir()) for path in (output, durable))):
        raise ValueError("Use separate, empty follow-up scratch and durable directories")
    if not 0 <= args.rescore_band <= 1 or not args.device.startswith("cuda") or not torch.cuda.is_bf16_supported():
        raise ValueError("The rescore band must lie in [0,1] on a BF16-capable CUDA device")
    if args.repeats < 2 or args.threads < 1:
        raise ValueError("At least two throughput repeats and a positive thread count are required")
    prior_manifest = json.loads((prior / "manifest.json").read_text())
    if prior_manifest["scratch"] != str(prior.resolve()):
        raise ValueError("Prior scratch location does not match its preservation manifest")
    for name, digest in prior_manifest["artifacts_sha256"].items():
        if file_digest(prior / name) != digest:
            raise ValueError(f"Prior AMP artifact changed: {name}")
    previous = json.loads((prior / "report.json").read_text())
    previous_config = json.loads((prior / "config.json").read_text())
    (metadata, source_config, identity, reference, pool, truth, pairs, positions, base, saved_cross,
     selected, sample_positions, sample_ref, sample_pool, sample_pairs, sample_base) = verify_inputs(
         data, saved, probe, checkpoint, args.checkpoint_sha256)
    tune_pairs = pairs.iloc[positions][["left_idx", "right_idx"]].reset_index(drop=True)
    probe_original = np.load(prior / "probe_original_pair_positions.npy")
    probe_rows = np.searchsorted(positions, probe_original)
    eligible = np.load(prior / "eligible_mask.npy")
    upper = blend_probabilities(base, np.ones(len(base), np.float32), selected, ALPHA)
    if (args.seed != previous_config["seed"] or previous_config["checkpoint"]["sha256"] != identity["sha256"]
            or previous_config["source_pair_digest"] != source_config["pair_digest"]
            or previous_config["tune_id_sha256"] != TUNE_ID_SHA256
            or not np.array_equal(np.load(prior / "tune_pair_positions.npy"), positions)
            or not pd.read_parquet(prior / "tune_pairs.parquet").equals(tune_pairs)
            or not np.array_equal(np.load(prior / "tune_base_probabilities.npy"), base)
            or not np.array_equal(eligible, selected & (np.asarray(upper, dtype=np.float64) >= THRESHOLD))
            or probe_original.shape != (2000,) or probe_original.min() < 0
            or probe_original.max() >= len(pairs) or not np.array_equal(positions[probe_rows], probe_original)
            or not np.asarray(eligible[probe_rows]).all()):
        raise ValueError("Follow-up must reuse identical saved TUNE pairs, split, checkpoint and label-free gate")
    eligible_rows = np.flatnonzero(eligible)
    if not np.array_equal(np.load(prior / "eligible_tune_pair_positions.npy"), eligible_rows):
        raise ValueError("Full-TUNE upper-bound CE subset differs from the prior benchmark")
    base_eligible, base_probe = base[eligible_rows], base[probe_rows]
    fp32_probe = np.load(prior / "probe_cross_fp32.npy")
    fp32_full = np.load(prior / "full_tune_eligible_cross_fp32.npy")
    prior_full_values = np.load(prior / "full_tune_probabilities_fp32.npy")
    if (fp32_probe.shape != (2000,) or fp32_full.shape != base_eligible.shape
            or prior_full_values.shape != (len(tune_pairs),)
            or np.any(prior_full_values[~eligible] != 0)):
        raise ValueError("Saved full-FP32 reference lost candidate order or gate")

    output.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    widths = tuple(sorted(set((.002, .005, .02, args.rescore_band))))
    screen_from_saved = {
        "scope": "same fixed 2,000 eligible TUNE pairs and all 43,511 eligible TUNE pairs; no labels",
        "probe": saved_band_sweep(base_probe, np.load(prior / "probe_cross_bf16.npy"), fp32_probe, widths),
        "full_tune_eligible": saved_band_sweep(base_eligible,
                                               np.load(prior / "full_tune_eligible_cross_bf16.npy"), fp32_full, widths)}
    atomic_json(output / "saved_band_screen.json", screen_from_saved)
    config = {"schema": 1, "seed": args.seed, "hypothesis": "FP32 only for BF16 blend near the frozen threshold",
              "band": args.rescore_band, "band_selection": "label-free saved FP32/BF16 decision differences; fixed before fresh inference",
              "alpha": ALPHA, "threshold": THRESHOLD, "gate": GATE,
              "max_length": MAX_LENGTH, "pad_to_multiple_of": PAD_MULTIPLE, "length_bucketing": True,
              "original_target_pool_records": metadata["mining_full_pool_records"],
              "positive_candidate_injection": False, "compacted_target_records": len(pool),
              "tune_reference_count": 12000, "tune_candidate_pairs": len(tune_pairs),
              "eligible_ce_pairs": len(eligible_rows), "probe_pairs": len(probe_rows),
              "fixed_throughput_pairs": len(sample_pairs), "checkpoint": identity,
              "tune_id_sha256": TUNE_ID_SHA256, "tune_pair_digest": previous_config["tune_pair_digest"],
              "tune_truth_sha256": previous_config["tune_truth_sha256"],
              "probe_original_pair_positions_sha256": file_digest(prior / "probe_original_pair_positions.npy"),
              "fixed_4000_pair_digest": frame_digest(sample_pairs, ["left_idx", "right_idx"]),
              "prior_config_sha256": file_digest(prior / "config.json"),
              "prior_report_sha256": file_digest(prior / "report.json"),
              "prior_manifest_sha256": file_digest(prior / "manifest.json"),
              "prior_artifacts_sha256": prior_manifest["artifacts_sha256"],
              "source_code_sha256": {"benchmark": file_digest(Path(__file__)),
                                     "scorer": file_digest(Path(score_reranker_selective_fp32.__code__.co_filename)),
                                     "trainer": file_digest(Path(load_reranker.__code__.co_filename))},
              "device": args.device, "threads": args.threads, "timing_repeats": args.repeats,
              "amp_gate": {"max_full_tune_macro_f05_delta": .0005, "min_sample_and_full_accept_overlap": .999,
                           "min_speedup_over_full_fp32": 2.0, "max_overhead_vs_same_pass_bf16": .25},
              "batch256_gate": {"min_pairs_per_second": 1700, "min_identical_decision_fraction": .999}}
    atomic_json(output / "config.json", config)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    load_start = time.perf_counter()
    model, tokenizer = load_reranker(str(checkpoint), args.device)
    load_seconds = time.perf_counter() - load_start
    # The fixed, boundary-enriched probe is evaluated before the complete TUNE.
    screen_frames = compact_pairs(reference, pool, tune_pairs.iloc[probe_rows])
    screen_cross, screen_mask, screen_bf16, screen_timing = timed_selective_scores(
        torch, model, tokenizer, screen_frames, base_probe, band=args.rescore_band, device=args.device)
    np.save(output / "probe_original_pair_positions.npy", probe_original)
    np.save(output / "probe_cross_selective.npy", screen_cross)
    np.save(output / "probe_bf16_firstpass.npy", screen_bf16)
    np.save(output / "probe_fp32_rescore_mask.npy", screen_mask)
    np.save(output / "probe_fp32_rescored_original_positions.npy", probe_original[screen_mask])
    probe_comparison = accept_comparison(decisions(base_probe, screen_cross, np.ones(2000, bool))[1],
                                         decisions(base_probe, fp32_probe, np.ones(2000, bool))[1])
    screen = {"scope": "same fixed, label-free 2,000-pair boundary-enriched TUNE probe",
              "band": args.rescore_band, "bf16_pairs": 2000, "fp32_rescored_pairs": int(screen_mask.sum()),
              "timing": screen_timing, "vs_prior_full_fp32": probe_comparison,
              "bf16_firstpass_vs_prior": accept_comparison(
                  decisions(base_probe, screen_bf16, np.ones(2000, bool))[1],
                  decisions(base_probe, np.load(prior / "probe_cross_bf16.npy"), np.ones(2000, bool))[1]),
              "passed": probe_comparison["accept_overlap_jaccard"] >= .999}
    atomic_json(output / "probe_screen.json", screen)

    full_frames = compact_pairs(reference, pool, tune_pairs.iloc[eligible_rows])
    full_cross, full_mask, full_bf16, full_timing = timed_selective_scores(
        torch, model, tokenizer, full_frames, base_eligible, band=args.rescore_band, device=args.device)
    np.save(output / "eligible_tune_pair_positions.npy", eligible_rows)
    np.save(output / "full_tune_eligible_cross_selective.npy", full_cross)
    np.save(output / "full_tune_bf16_firstpass.npy", full_bf16)
    np.save(output / "full_tune_fp32_rescore_mask.npy", full_mask)
    np.save(output / "full_tune_fp32_rescored_original_positions.npy", positions[eligible_rows[full_mask]])
    all_cross = np.zeros(len(base), dtype=np.float32)
    all_cross[eligible_rows] = full_cross
    full_values, full_decisions = decisions(base, all_cross, eligible)
    np.save(output / "full_tune_probabilities_selective.npy", full_values)
    tune_mask = reference.split.eq("tune")
    tune_score = detailed_score(reference, pool, tune_pairs, full_values, truth, THRESHOLD, tune_mask)
    true_pairs = pair_labels(reference, pool, tune_pairs, truth)
    country_false_links = false_links_by_country(reference, tune_pairs, true_pairs, full_decisions)
    prior_fp32 = previous["full_tune"]["fp32"]
    full_comparison = accept_comparison(full_decisions, accepted_at_threshold(prior_full_values))
    full_delta = abs(tune_score["macro_f05"] - prior_fp32["macro_f05"])
    speedup = previous["full_tune"]["inference"]["fp32"]["seconds"] / full_timing["total_seconds"]
    bf16_overhead = full_timing["total_seconds"] / full_timing["bf16_seconds"] - 1
    full = {"scope": "all 12,000 TUNE S1s, including retrieval misses/singletons; same 600,000 candidates from 10,320,219 targets",
            "candidate_pairs": len(tune_pairs), "bf16_ce_pairs": len(eligible_rows), "fp32_rescored_pairs": int(full_mask.sum()),
            "fixed_threshold": THRESHOLD, "score": tune_score, "full_fp32_reference": prior_fp32,
            "macro_f05_abs_delta": full_delta, "vs_prior_full_fp32": full_comparison,
            "bf16_firstpass_vs_prior": accept_comparison(
                decisions(base_eligible, full_bf16, np.ones(len(eligible_rows), bool))[1],
                decisions(base_eligible, np.load(prior / "full_tune_eligible_cross_bf16.npy"),
                          np.ones(len(eligible_rows), bool))[1]),
            "country_false_positive_links": country_false_links, "timing": full_timing,
            "speedup_vs_prior_full_fp32": speedup, "overhead_vs_same_pass_bf16": bf16_overhead,
            "passed_parity": amp_parity_pass(full_delta, full_comparison),
            "passed_speed": speedup >= 2 and bf16_overhead <= .25}
    atomic_json(output / "full_tune.json", full)

    # Same v1 4k sample, interleaved timing of one bounded synchronization change.
    sample_frames = (sample_ref, sample_pool, sample_pairs)
    sample_tune_rows = np.searchsorted(positions, sample_positions)
    if not np.array_equal(positions[sample_tune_rows], sample_positions):
        raise ValueError("Fixed 4,000-pair probe cannot be aligned with TUNE")
    sample_cross, sample_mask, _, sample_timing = timed_selective_scores(
        torch, model, tokenizer, sample_frames, sample_base, band=args.rescore_band, device=args.device)
    np.save(output / "fixed_4000_cross_selective.npy", sample_cross)
    np.save(output / "fixed_4000_fp32_rescored_original_positions.npy", sample_positions[sample_mask])
    sample_decisions = decisions(sample_base, sample_cross, np.ones(len(sample_base), bool))[1]
    sample_fp32_reference = accepted_at_threshold(prior_full_values[sample_tune_rows])
    four_thousand_comparison = accept_comparison(sample_decisions, sample_fp32_reference)
    warmup = (sample_ref, sample_pool, sample_pairs.iloc[:256].reset_index(drop=True))
    timed_scores(torch, model, tokenizer, warmup, amp="bf16", batch_size=128, device=args.device)
    variants = {"128": (128, False), "256": (256, False), "256_deferred": (256, True)}
    runs, batch_cross = {key: [] for key in variants}, {}
    order = (("128", "256", "256_deferred"), ("256_deferred", "256", "128"),
             ("256", "128", "256_deferred"))
    for repeat in range(args.repeats):
        for name in order[repeat % len(order)]:
            batch, defer = variants[name]
            scores, timing = timed_scores(torch, model, tokenizer, sample_frames,
                                           amp="bf16", batch_size=batch, device=args.device,
                                           defer_cpu_transfer=defer)
            runs[name].append(timing)
            if name not in batch_cross:
                batch_cross[name] = scores
                np.save(output / f"fixed_4000_cross_{name}.npy", scores)
            elif not np.array_equal(decisions(sample_base, scores, np.ones(len(scores), bool))[1],
                                    decisions(sample_base, batch_cross[name], np.ones(len(scores), bool))[1]):
                raise ValueError("Repeated 4,000-pair batch inference changed a decision")
    speeds = {name: {"runs": runs[name], "median_seconds": float(np.median(
                    [row["seconds"] for row in runs[name]])),
                    "median_pairs_per_second": 4000 / float(np.median([row["seconds"] for row in runs[name]]))}
              for name in variants}
    baseline_decisions = decisions(sample_base, batch_cross["128"], np.ones(len(sample_base), bool))[1]
    comparisons = {name: accept_comparison(baseline_decisions,
                   decisions(sample_base, batch_cross[name], np.ones(len(sample_base), bool))[1])
                   for name in ("256", "256_deferred")}
    qualifying = [name for name in ("256", "256_deferred")
                  if speeds[name]["median_pairs_per_second"] >= 1700
                  and speeds[name]["median_pairs_per_second"] >= speeds["128"]["median_pairs_per_second"]
                  and comparisons[name]["identical_decision_fraction"] >= .999]
    winner = max(qualifying, key=lambda name: speeds[name]["median_pairs_per_second"]) if qualifying else None
    batch = {"scope": "exact prior 4,000 gated TUNE pairs; BF16 length-sorted pad8, maxlen256",
             "pair_digest": frame_digest(sample_pairs, ["left_idx", "right_idx"]),
             "country_pair_counts": previous["batch_benchmark"]["country_pair_counts"],
             "fixed_4000_selective": {"rescored_pairs": int(sample_mask.sum()), "timing": sample_timing,
                                      "vs_prior_full_fp32": four_thousand_comparison},
             "inference": speeds, "comparisons_vs_batch128": comparisons,
             "new_128_vs_prior_128": accept_comparison(baseline_decisions,
                 decisions(sample_base, np.load(prior / "batch128_bf16_cross.npy"),
                           np.ones(len(sample_base), bool))[1]),
             "batch256_winner": winner, "adopt_batch256": winner is not None}
    atomic_json(output / "batch_benchmark.json", batch)

    adopt = screen["passed"] and full["passed_parity"] and full["passed_speed"]
    result = {"schema": 1, "config": str((output / "config.json").resolve()),
              "scope": "saved TRAIN/TUNE full-pool candidates and frozen BGE checkpoint; no eval/test labels",
              "checkpoint_load_seconds": load_seconds, "seed": args.seed,
              "band": args.rescore_band, "saved_band_screen": screen_from_saved,
              "probe_screen": screen, "full_tune": full, "batch_benchmark": batch,
              "adopt_selective_bf16_for_v2": adopt,
              "adopt_batch256_for_v2": bool(adopt and batch["adopt_batch256"]),
              "retain_batch_size": 256 if adopt and batch["adopt_batch256"] else 128}
    atomic_json(output / "report.json", result)
    manifest = {"schema": 1, "scratch": str(output.resolve()), "durable": str(durable.resolve()),
                "prior_manifest_sha256": config["prior_manifest_sha256"],
                "checkpoint_sha256": identity["sha256"],
                "artifacts_sha256": {path.name: file_digest(path) for path in output.iterdir() if path.is_file()}}
    atomic_json(output / "manifest.json", manifest)
    for path in sorted(output.iterdir()):
        if path.is_file():
            target = durable / path.name
            temporary = target.with_name(target.name + ".tmp")
            shutil.copyfile(path, temporary)
            if file_digest(temporary) != file_digest(path):
                raise ValueError(f"Durable follow-up copy differs: {path.name}")
            temporary.replace(target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "saved", "probe", "checkpoint", "checkpoint-sha256", "output", "durable"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--selective-from", help="Prior completed AMP benchmark; reuse its exact pairs and FP32 controls")
    parser.add_argument("--rescore-band", type=float, default=.04,
                        help="FP32 rescore if the BF16/tree blend lies within this fixed distance of 0.775")
    args = parser.parse_args()
    if args.selective_from:
        report = run_selective(args)
        print(json.dumps({"probe_jaccard": report["probe_screen"]["vs_prior_full_fp32"]["accept_overlap_jaccard"],
                          "full_tune_f05": report["full_tune"]["score"]["macro_f05"],
                          "full_tune_f05_delta": report["full_tune"]["macro_f05_abs_delta"],
                          "full_tune_fp32_rescored_pairs": report["full_tune"]["fp32_rescored_pairs"],
                          "full_tune_total_seconds": report["full_tune"]["timing"]["total_seconds"],
                          "batch256_best_pps": max(report["batch_benchmark"]["inference"][name]["median_pairs_per_second"]
                                                   for name in ("256", "256_deferred")),
                          "adopt_selective": report["adopt_selective_bf16_for_v2"],
                          "batch_size": report["retain_batch_size"]}))
    else:
        report = run(args)
        print(json.dumps({"probe_passed": report["probe_screen"]["passed"],
                          "full_tune_passed": report["full_tune"]["passed"],
                          "f05_fp32": report["full_tune"]["fp32"]["macro_f05"],
                          "f05_bf16": report["full_tune"]["bf16"]["macro_f05"],
                          "batch128_pps": report["batch_benchmark"]["inference"]["128"]["median_pairs_per_second"],
                          "batch256_pps": report["batch_benchmark"]["inference"]["256"]["median_pairs_per_second"],
                          "batch_size": report["retain_batch_size"]}))


if __name__ == "__main__":
    main()
