"""Stress a saved matcher against complete training source pools, tuning IDs only.

TSV datasets use train/train_source{2,3}.tsv. Existing controller Parquets are
also supported (train/s{1,2,3}.parquet and train/gt.parquet); their Source 1 IDs
and ground truth must agree with the original quick data before evaluation.
Only one country's pool/index is retained at a time. All candidates belonging
to a reference are passed to feature extraction together, including hard rivals.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .features import build_features
from .metrics import candidate_report, entity_f05, score_pairs
from .models import predict_proba
from .normalization import normalize_text
from .prepare import log


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_chunks(path, columns=None, chunksize=100_000):
    """Bound source parsing memory for either supplied TSVs or cached Parquets."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunksize, columns=columns):
            yield batch.to_pandas().fillna("")
    else:
        yield from pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                               usecols=columns, chunksize=chunksize)


def source_paths(dataset, pool_format="auto"):
    base = Path(dataset)
    if (base / "train").is_dir():
        base /= "train"
    tsv = [base / f"train_source{s}.tsv" for s in (2, 3)]
    parquet = [base / f"s{s}.parquet" for s in (2, 3)]
    if pool_format in ("auto", "tsv") and all(p.is_file() for p in tsv):
        return "tsv", base, tsv
    if pool_format in ("auto", "parquet") and all(p.is_file() for p in parquet):
        return "parquet", base, parquet
    raise FileNotFoundError(f"No complete training Source 2/3 pool found under {base}")


def tuning_references(data, base, pool_format, country, maximum, seed):
    if not 1 <= maximum <= 1000:
        raise ValueError("--max-reference must be between 1 and 1000 for this stress run")
    reference = pd.read_parquet(Path(data) / "reference.parquet").fillna("")
    if "split" not in reference:
        raise ValueError("Quick references must have explicit train/tune/eval splits")
    reference = reference.loc[reference["split"].eq("tune")].copy()
    if country is not None:
        reference = reference.loc[reference.country.map(normalize_text).eq(normalize_text(country))]
    before_filter = len(reference)
    # A controller training cache may exclude and re-ID held-out entities. Restrict
    # by its Source 1 membership before sampling; never infer eligibility from score.
    if pool_format == "parquet":
        source1 = base / "s1.parquet"
        if not source1.is_file():
            raise FileNotFoundError("Parquet stress requires s1.parquet to verify original IDs")
        wanted, found = set(reference.entity_id), set()
        for chunk in read_chunks(source1, ["entity_id"]):
            found.update(chunk.loc[chunk.entity_id.isin(wanted), "entity_id"])
        reference = reference.loc[reference.entity_id.isin(found)]
    eligible = len(reference)
    if not eligible:
        raise ValueError("No eligible tuning references; eval/test references are never substituted")
    if reference.entity_id.duplicated().any():
        raise ValueError("Duplicate tuning Source 1 IDs")
    # Deterministic sampling independent of labels and model predictions.
    reference = reference.sort_values("entity_id")
    if len(reference) > maximum:
        reference = reference.sample(n=maximum, random_state=seed)
    reference = reference.sort_values(["country", "entity_id"]).reset_index(drop=True)
    all_truth = json.loads((Path(data) / "truth.json").read_text(encoding="utf-8"))
    missing = set(reference.entity_id) - set(all_truth)
    if missing:
        raise ValueError(f"Missing original truth for {len(missing)} selected tuning references")
    truth = {x: set(all_truth[x]) for x in reference.entity_id}
    del all_truth
    if pool_format == "parquet":
        gt = base / "gt.parquet"
        if not gt.is_file():
            raise FileNotFoundError("Parquet stress requires gt.parquet to verify original labels")
        seen, inconsistent = set(), set()
        for chunk in read_chunks(gt, ["source1_entity_id", "matched_entity_ids"]):
            for entity_id, ids in chunk.loc[chunk.source1_entity_id.isin(truth)].itertuples(index=False, name=None):
                seen.add(entity_id)
                if set(x for x in ids.split(",") if x) != truth[entity_id]:
                    inconsistent.add(entity_id)
        if seen != set(truth) or inconsistent:
            raise ValueError(f"Original-label mismatch in controller cache: missing={len(set(truth)-seen)}, inconsistent={len(inconsistent)}")
    selection = {"quick_tuning_references": before_filter, "eligible_references": eligible,
                 "selected_references": len(reference), "seed": seed,
                 "source1_id_sha256": hashlib.sha256("\n".join(sorted(reference.entity_id)).encode()).hexdigest()}
    return reference, truth, selection


def load_country_pool(paths, country_key, wanted_truth):
    pieces, true_records, statistics = [], {}, []
    for path in paths:
        total = retained = 0
        for chunk in read_chunks(path, SOURCE_COLUMNS):
            total += len(chunk)
            # Capture selected true targets even when a country label blocks them.
            for row in chunk.loc[chunk.entity_id.isin(wanted_truth)].to_dict("records"):
                true_records[row["entity_id"]] = row
            keep = chunk.country.map(normalize_text).eq(country_key)
            if keep.any():
                part = chunk.loc[keep, SOURCE_COLUMNS].copy()
                retained += len(part)
                pieces.append(part)
        statistics.append({"path": str(path), "full_source_rows": total, "country_rows": retained})
        log(f"{path.name}: {total:,} records scanned; {retained:,} in {country_key}")
    pool = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=SOURCE_COLUMNS)
    del pieces
    absent = wanted_truth - set(true_records)
    if absent:
        raise ValueError(f"Training pool lacks {len(absent)} original true IDs; refusing an incomplete-pool stress score")
    return pool, true_records, statistics


def diagnostics(reference, pool, pairs, probabilities, truth, threshold, true_records, maximum_examples):
    candidate_maps = [dict() for _ in range(len(reference))]
    pool_ids = pool.entity_id.to_numpy()
    for left, right, probability in zip(pairs.left_idx, pairs.right_idx, probabilities):
        candidate_maps[int(left)][pool_ids[int(right)]] = (float(probability), int(right))
    entities, false_positives, missed = [], [], []
    for left, row in enumerate(reference.to_dict("records")):
        entity_id, candidates = row["entity_id"], candidate_maps[left]
        actual = truth[entity_id]
        predicted = {x for x, (p, _) in candidates.items() if p >= threshold}
        found = actual & candidates.keys()
        tp = len(actual & predicted)
        entities.append({"source1_entity_id": entity_id, "country": row["country"],
                         "true_links": len(actual), "candidate_links": len(candidates),
                         "retrieved_true_links": len(found), "predicted_links": len(predicted),
                         "true_positives": tp, "false_positives": len(predicted-actual),
                         "missed_links": len(actual-predicted),
                         "macro_f05": entity_f05(actual, predicted),
                         "oracle_f05": entity_f05(actual, found),
                         "singleton": not bool(actual),
                         "singleton_correct": not actual and not predicted,
                         "complete_match_set_retrieved": found == actual})
        common = {"source1_entity_id": entity_id, "country": row["country"],
                  "source1_name": row["business_name"], "source1_address": row["business_address"]}
        for entity in predicted - actual:
            probability, position = candidates[entity]
            target = pool.iloc[position]
            false_positives.append({**common, "target_entity_id": entity, "probability": probability,
                                    "target_name": target.business_name, "target_address": target.business_address})
        for entity in actual - predicted:
            target = true_records[entity]
            if entity in candidates:
                reason, probability = "below_threshold", candidates[entity][0]
            elif normalize_text(target["country"]) != normalize_text(row["country"]):
                reason, probability = "country_blocked", None
            else:
                reason, probability = "not_retrieved", None
            missed.append({**common, "target_entity_id": entity, "probability": probability,
                           "reason": reason, "target_country": target["country"],
                           "target_name": target["business_name"], "target_address": target["business_address"]})
    false_positives.sort(key=lambda r: (-r["probability"], r["source1_entity_id"], r["target_entity_id"]))
    return entities, false_positives[:maximum_examples], missed


def aggregate(entities):
    frame = pd.DataFrame(entities)
    true = int(frame.true_links.sum())
    predicted = int(frame.predicted_links.sum())
    tp = int(frame.true_positives.sum())
    singletons = frame.loc[frame.singleton]
    matched = frame.loc[frame.true_links.gt(0)]
    return {"reference_count": len(frame), "macro_f05": float(frame.macro_f05.mean()),
            "micro_precision": tp / predicted if predicted else 0.0,
            "micro_recall": tp / true if true else 0.0,
            "true_links": true, "predicted_links": predicted,
            "false_positives": int(frame.false_positives.sum()), "missed_links": int(frame.missed_links.sum()),
            "singleton_accuracy": float(singletons.singleton_correct.mean()) if len(singletons) else None,
            "retrieval_link_recall": float(frame.retrieved_true_links.sum() / true) if true else None,
            "complete_match_set_recall": float(matched.complete_match_set_retrieved.mean()) if len(matched) else None,
            "oracle_macro_f05": float(frame.oracle_f05.mean()),
            "candidate_pairs": int(frame.candidate_links.sum()),
            "pairs_per_reference": float(frame.candidate_links.mean())}


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_summary(path, summary):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run(args):
    started = time.time()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pool_format, base, paths = source_paths(args.dataset, args.pool_format)
    reference, truth, selection = tuning_references(args.data, base, pool_format, args.country,
                                                    args.max_reference, args.seed)
    artifact = joblib.load(args.model)
    required = {"model", "threshold", "feature_columns", "candidate_config"}
    if not isinstance(artifact, dict) or not required <= artifact.keys():
        raise ValueError("--model must be an experiment.py saved baseline artifact")
    threshold = float(artifact["threshold"])
    config = dict(artifact["candidate_config"])
    config.update(batch_size=args.batch_size, n_jobs=args.threads, cache_countries=1)
    summary = {"status": "running", "split": "tune", "model": str(Path(args.model).resolve()),
               "threshold": threshold, "threshold_retuned": False, "pool_format": pool_format,
               "selection": selection, "candidate_config": config, "countries": [],
               "limitations": ["Tuning references only: this is a scale stress check, not a new held-out score.",
                               "Each query sees every training Source 2/3 record in its country.",
                               "Parquet controller pools may differ from the full original TSV release."]}
    all_entities, all_false_positives, all_missed = [], [], []
    keys = reference.country.map(normalize_text)
    for country_key in sorted(keys.unique()):
        country_started = time.time()
        current = reference.loc[keys.eq(country_key)].reset_index(drop=True)
        current_truth = {x: truth[x] for x in current.entity_id}
        wanted = set().union(*current_truth.values())
        log(f"Stress {country_key}: {len(current):,} tuning queries against complete country pool")
        pool, true_records, scale = load_country_pool(paths, country_key, wanted)
        log(f"Building retrieval index over {len(pool):,} records")
        pairs = generate_candidates(current, pool, **config)
        features = build_features(current, pool, pairs)
        missing_columns = set(artifact["feature_columns"]) - set(features.columns)
        if missing_columns:
            raise ValueError(f"Saved model feature columns unavailable: {sorted(missing_columns)}")
        probabilities = predict_proba(artifact["model"], features[artifact["feature_columns"]])
        score = score_pairs(current, pool, pairs, probabilities, current_truth, threshold)
        retrieval = candidate_report(current, pool, pairs, current_truth) if wanted else None
        entities, false_positives, missed = diagnostics(current, pool, pairs, probabilities,
                                                        current_truth, threshold, true_records, args.max_examples)
        all_entities.extend(entities)
        all_false_positives.extend(false_positives)
        all_missed.extend(missed)
        result = {"country_key": country_key, "pool_records": len(pool), "source_scale": scale,
                  "score": score, "retrieval": retrieval, "elapsed_seconds": time.time()-country_started}
        summary["countries"].append(result)
        summary["overall"] = aggregate(all_entities)
        summary["elapsed_seconds"] = time.time()-started
        write_summary(output / "summary.json", summary)
        log(json.dumps(json_safe(result)))
        del pool, true_records, pairs, features, probabilities
        gc.collect()
    all_false_positives.sort(key=lambda r: (-r["probability"], r["source1_entity_id"], r["target_entity_id"]))
    all_missed.sort(key=lambda r: (r["source1_entity_id"], r["target_entity_id"]))
    pd.DataFrame(all_entities).to_csv(output / "entities.tsv", sep="\t", index=False)
    diagnostic_columns = ["source1_entity_id", "country", "source1_name", "source1_address",
                          "target_entity_id", "probability", "target_name", "target_address"]
    pd.DataFrame(all_false_positives[:args.max_examples], columns=diagnostic_columns).to_csv(
        output / "false_positive_examples.tsv", sep="\t", index=False)
    pd.DataFrame(all_missed, columns=diagnostic_columns + ["reason", "target_country"]).to_csv(
        output / "missed_links.tsv", sep="\t", index=False)
    reference[["entity_id", "country", "split"]].to_csv(output / "selected_references.tsv", sep="\t", index=False)
    summary.update(status="complete", elapsed_seconds=time.time()-started,
                   missed_link_reasons=pd.Series([r["reason"] for r in all_missed], dtype=str).value_counts().to_dict())
    write_summary(output / "summary.json", summary)
    log(f"Stress result: {json.dumps(summary['overall'])}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Training dataset root or its train directory")
    parser.add_argument("--data", required=True, help="Quick data containing fixed tune split and original truth")
    parser.add_argument("--model", required=True, help="Saved experiment.py joblib artifact")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-reference", type=int, default=1000)
    parser.add_argument("--country")
    parser.add_argument("--pool-format", choices=["auto", "tsv", "parquet"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1 or args.max_examples < 0:
        parser.error("threads/batch-size must be positive; max-examples must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
