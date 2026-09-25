"""Apply a frozen tree/CE blend without fitting, threshold search, or early labels."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .cross_encoder_experiment import blend_probabilities, load_reranker, score_reranker
from .full_embeddings import atomic_json, checkpoint_identity, file_digest
from .predict_streaming import iter_reference_batches, load_records
from .prepare import log
from .uncertainty import paired_bootstrap


def rerank_batch(reference, pool, pairs, base, config, scorer):
    """Score compact gated pairs; optionally omit provably rejected CE calls.

    The shortcut preserves accepted links and candidate rows. Scores for pairs
    that cannot reach the frozen threshold become zero, rather than a full CE
    blend. Compute the upper bound with the same float32 blend as inference.
    """
    base = np.asarray(base, dtype=np.float32)
    if base.shape != (len(pairs),) or not np.isfinite(base).all() or np.any((base != -1) & ((base < 0) | (base > 1))):
        raise ValueError("Tree probabilities must align with pairs and lie in [0,1] or equal anchor sentinel -1")
    selected, cross = base >= config["gate"], np.zeros(len(base), dtype=np.float32)
    if config.get("skip_provably_rejected_pairs", False):
        upper = blend_probabilities(base, np.ones(len(base), dtype=np.float32), selected, config["alpha"])
        selected &= np.asarray(upper, dtype=np.float64) >= config["threshold"]
    if selected.any():
        compact = pairs.loc[selected, ["left_idx", "right_idx"]].copy()
        left, compact["left_idx"] = np.unique(compact.left_idx, return_inverse=True)
        right, compact["right_idx"] = np.unique(compact.right_idx, return_inverse=True)
        values = np.asarray(scorer(reference.iloc[left].reset_index(drop=True),
                                   pool.iloc[right].reset_index(drop=True), compact), dtype=np.float32)
        if values.shape != (int(selected.sum()),) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("CE probabilities must align with gated pairs and lie in [0,1]")
        cross[selected] = values
    return blend_probabilities(base, cross, selected, config["alpha"]), cross, selected


def score_fixed_outputs(reference, matching, truth_path, output):
    """Called only after prediction files and their hashes have been committed."""
    truth = json.loads(Path(truth_path).read_text())
    if set(reference.entity_id) - truth.keys():
        raise ValueError("Truth must explicitly cover every input S1, including singletons")
    rows = []
    with matching.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for entity, country in reference[["entity_id", "country"]].itertuples(index=False, name=None):
            row = next(reader)
            if row[0] != entity:
                raise ValueError("Fixed predictions lost reference order")
            predicted, actual = set(filter(None, row[1].split(","))), set(truth[entity])
            tp, p, t = len(predicted & actual), len(predicted), len(actual)
            rows.append((entity, country, t, p, tp, p-tp, t-tp,
                         5*tp/(4*p+t) if p+t else 1., 2*tp/(p+t) if p+t else 1.))
        if next(reader, None) is not None:
            raise ValueError("Fixed predictions contain unexpected S1 rows")
    frame = pd.DataFrame(rows, columns=["source1_entity_id", "country", "true_links", "predicted_links",
        "true_positive_links", "false_positive_links", "false_negative_links", "macro_f05", "macro_f1"])
    frame.to_csv(output / "entities.tsv", sep="\t", index=False)
    tp, p, t = frame[["true_positive_links", "predicted_links", "true_links"]].sum()
    singletons = frame.loc[frame.true_links == 0]
    report = {"reference_count": len(frame), "macro_f05": float(frame.macro_f05.mean()), "macro_f1": float(frame.macro_f1.mean()),
        "micro_precision": float(tp/p) if p else 0., "micro_recall": float(tp/t) if t else 0.,
        "false_positive_links": int(p-tp), "false_negative_links": int(t-tp),
        "singleton_accuracy": float(singletons.predicted_links.eq(0).mean()) if len(singletons) else None,
        "country": frame.groupby("country")[["macro_f05", "macro_f1"]].mean().to_dict(orient="index")}
    if len(frame) > 1:
        report["uncertainty"] = {name: paired_bootstrap(frame[name].to_numpy()) for name in ("macro_f05", "macro_f1")}
    return report


def run(args, *, scorer=None):
    data, tree, output, pipeline = map(Path, (args.data, args.tree_predictions, args.output, args.pipeline))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory for frozen predictions")
    config, pipeline_hash = json.loads(pipeline.read_text()), file_digest(pipeline)
    for key in ("alpha", "gate", "threshold"):
        valid = config[key] > 0 if key == "threshold" else 0 <= config[key] <= 1
        if not np.isfinite(config[key]) or not valid:
            raise ValueError("Frozen alpha/gate must lie in [0,1], and threshold must be positive")
    if any(type(config[key]) is not int or config[key] <= 0 for key in ("max_length", "inference_batch_size")):
        raise ValueError("Frozen sequence and batch sizes must be positive integers")
    if type(config.get("skip_provably_rejected_pairs", False)) is not bool:
        raise ValueError("Frozen skip_provably_rejected_pairs must be a boolean")
    if "length_bucketing" in config:
        if type(config["length_bucketing"]) is not bool:
            raise ValueError("Frozen length_bucketing must be a boolean")
        multiple = config.get("pad_to_multiple_of", 8)
        if multiple is not None and (type(multiple) is not int or multiple <= 0):
            raise ValueError("Frozen pad_to_multiple_of must be a positive integer or null")
    completed = json.loads((tree / "complete.json").read_text())
    tree_report = json.loads((tree / "report.json").read_text())
    if completed.get("state") != "complete" or completed.get("report") != tree_report or tree_report.get("benchmark_only"):
        raise ValueError("Tree predictions must be a committed complete population")
    if tree_report.get("scorer", "frozen_tree") != "frozen_tree":
        raise ValueError("Input must contain raw tree scores; refusing to rerank an existing CE blend")
    if tree_report["checkpoint_sha256"] != config["tree_checkpoint_sha256"] or file_digest(Path(tree_report["checkpoint"])) != config["tree_checkpoint_sha256"]:
        raise ValueError("Frozen tree checkpoint digest disagrees with tree predictions")
    if Path(tree_report["data"]).resolve() != data.resolve() or tree_report.get("pair_order") != ["left_idx", "right_idx"]:
        raise ValueError("Tree prediction record source/order disagrees with reranking input")
    ce = checkpoint_identity(Path(config["cross_encoder_checkpoint"]))
    if ce["sha256"] != config["cross_encoder_sha256"]:
        raise ValueError("Frozen CE checkpoint digest mismatch")
    reference, pool = load_records(data)
    reference_batch_size = config.get("reference_batch_size", args.reference_batch_size)
    runtime = {"device": args.device, "threads": getattr(args, "threads", 2), "interop_threads": 1,
        "reference_batch_size": reference_batch_size, "max_pairs_per_batch": args.max_pairs_per_batch,
        "inference_batch_size": config["inference_batch_size"], "max_length": config["max_length"],
        "inference_backend": "batched" if "length_bucketing" in config else "legacy",
        "skip_provably_rejected_pairs": config.get("skip_provably_rejected_pairs", False),
        "length_bucketing": config.get("length_bucketing"),
        "pad_to_multiple_of": config.get("pad_to_multiple_of", 8) if "length_bucketing" in config else None}
    pair_path, candidate_path = tree / "scored_pairs.parquet", tree / "candidate_pairs.tsv"
    base = np.load(tree / "probabilities.npy", mmap_mode="r")
    count = pq.ParquetFile(pair_path).metadata.num_rows
    if base.dtype != np.float32 or base.shape != (count,) or count != tree_report["candidate_pairs"] or len(reference) != tree_report["reference_count"]:
        raise ValueError("Tree candidate/probability/reference counts disagree")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pipeline, output / "frozen_pipeline.json")
    manifest = {"pipeline_sha256": pipeline_hash, "pipeline": config, "cross_encoder_identity": ce,
        "tree_prediction_directory": str(tree.resolve()), "labels_accessed": False, "runtime": runtime,
        "input_sha256": {name: file_digest(tree / name) for name in ("probabilities.npy", "scored_pairs.parquet", "candidate_pairs.tsv")}}
    atomic_json(output / "frozen_manifest.json", manifest)
    if scorer is None:
        import torch

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        model, tokenizer = load_reranker(config["cross_encoder_checkpoint"], args.device)
        scoring = SimpleNamespace(device=args.device, max_length=config["max_length"],
                                  inference_batch_size=config["inference_batch_size"],
                                  length_bucketing=config.get("length_bucketing", False),
                                  pad_to_multiple_of=config.get("pad_to_multiple_of", 8))
        score_function = score_reranker
        if "length_bucketing" in config:
            from .rerank_inference import score_reranker_batched

            score_function = score_reranker_batched
        scorer = lambda left, right, pairs: score_function(model, tokenizer, left, right, pairs, np.ones(len(pairs), bool), scoring)
    values_path, cross_path, matching = (output / name for name in ("probabilities.npy.tmp", "cross_probabilities.npy.tmp", "matching_results.tsv.tmp"))
    values = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.float32, shape=(count,))
    cross_values = np.lib.format.open_memmap(cross_path, mode="w+", dtype=np.float32, shape=(count,))
    pool_ids, reference_ids = pool.entity_id.to_numpy(), reference.entity_id.to_numpy()
    offset = gated = 0
    with matching.open("w", newline="", encoding="utf-8") as stream, candidate_path.open(newline="", encoding="utf-8") as cstream:
        writer, candidates = csv.writer(stream, delimiter="\t", lineterminator="\n"), csv.reader(cstream, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        if next(candidates, None) != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError("Unexpected candidate TSV header")
        for batch in iter_reference_batches(pair_path, len(reference), len(pool), reference_batch_size,
                max_pairs_per_batch=args.max_pairs_per_batch, require_sorted_right=True):
            end = offset + len(batch.pairs)
            blended, cross, selected = rerank_batch(reference, pool, batch.pairs, base[offset:end], config, scorer)
            values[offset:end], cross_values[offset:end] = blended, cross
            left, right = batch.pairs.left_idx.to_numpy(), batch.pairs.right_idx.to_numpy()
            boundaries = np.searchsorted(left, np.arange(batch.start, batch.end+1))
            for index, position in enumerate(range(batch.start, batch.end)):
                first, last = boundaries[index:index+2]
                ids, entity = pool_ids[right[first:last]], reference_ids[position]
                if next(candidates, None) != [entity, ",".join(sorted(ids))]:
                    raise ValueError("Candidate TSV lost reference or pair alignment")
                accepted = np.asarray(blended[first:last], dtype=np.float64) >= config["threshold"]
                writer.writerow([entity, ",".join(sorted(ids[accepted]))])
            offset, gated = end, gated + int(selected.sum())
            log(f"Frozen reranking: {batch.end:,}/{len(reference):,} S1; {offset:,} candidates; {gated:,} CE calls")
        if next(candidates, None) is not None or offset != count:
            raise ValueError("Tree prediction rows were not consumed exactly once")
    values.flush()
    cross_values.flush()
    if file_digest(pipeline) != pipeline_hash or file_digest(output / "frozen_pipeline.json") != pipeline_hash:
        raise ValueError("Frozen pipeline changed during inference")
    if checkpoint_identity(Path(config["cross_encoder_checkpoint"]))["sha256"] != ce["sha256"]:
        raise ValueError("CE checkpoint changed during inference")
    for name, digest in manifest["input_sha256"].items():
        if file_digest(tree / name) != digest:
            raise ValueError(f"Tree input changed during inference: {name}")
    for source in (values_path, cross_path, matching):
        source.replace(source.with_suffix(""))
    for source in (candidate_path, pair_path):
        try:
            os.link(source, output / source.name)
        except OSError:
            shutil.copyfile(source, output / source.name)
    report = {"pipeline_sha256": pipeline_hash, "reference_count": len(reference), "candidate_pairs": count,
        "reranked_pairs": gated, "threshold": config["threshold"], "labels_accessed": False,
        "runtime": runtime,
        "probability_order": "scored_pairs.parquet row order", "ready_for_submission": False,
        "output_sha256": {name: file_digest(output / name) for name in ("matching_results.tsv", "candidate_pairs.tsv", "probabilities.npy")}}
    atomic_json(output / "predictions_fixed.json", report)
    if args.truth:
        report["score"] = score_fixed_outputs(reference, output / "matching_results.tsv", args.truth, output)
        report["labels_accessed"] = True
    atomic_json(output / "report.json", report)
    atomic_json(output / "complete.json", {"state": "complete", "report": report})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "tree-predictions", "pipeline", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--truth", help="Optional held-out truth, read only after predictions are committed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--reference-batch-size", type=int, default=1000)
    parser.add_argument("--max-pairs-per-batch", type=int, default=200000)
    print(json.dumps(run(parser.parse_args())))


if __name__ == "__main__":
    main()
