"""One-shot 12k confirmation: label-free preparation/decision, then sealed scoring.

Run prepare -> existing complete-pool dense miner -> frozen tree predictor ->
decide -> confirm. The last command is the only one allowed to open ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unicodedata

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .cross_encoder_experiment import blend_probabilities, load_reranker
from .full_embeddings import checkpoint_identity, file_digest
from .predict_streaming import iter_reference_batches, load_records
from .rerank_inference import score_reranker_selective_fp32
from .uncertainty import paired_bootstrap
from .v2_decision import CountPosterior, assignment, predict_calibration


EXPECTED_IDS = "ec057370b8dc8b3dbef24e277a6920f905a08481ca2b35a5245dcdf9d22f5187"
EXPECTED_FREEZE = "44e284bd3c743025cd640f2afb6ab2ca9cc5db76b13f85bccb5c15ee5d9ba81f"
EXPECTED_P1 = "52cc4982ad90282473fadcc898a61d234037ab6f802c966674cf84159dcb04ec"
EXPECTED_V1 = "9ac4050a72c64b05c032a17ec26f2a13b339c1b8c135791bf07b737534371f5e"


def _ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _write(path: Path, document: dict):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _check_freeze(root: Path, p1: Path, v1: Path | None = None):
    frozen = root / "frozen_decision.json"
    if file_digest(frozen) != EXPECTED_FREEZE or file_digest(p1) != EXPECTED_P1:
        raise ValueError("Frozen selection/P1 calibrator bytes differ; no holdout access permitted")
    if v1 is not None and file_digest(v1) != EXPECTED_V1:
        raise ValueError("Frozen v1 pipeline differs")
    selection = json.loads(frozen.read_text())
    if (selection["reservation_id_sha256"] != EXPECTED_IDS or selection["gate"] != .01
            or selection["alpha"] != .75 or selection["v1_threshold"] != .775
            or selection["p1"]["source_balance_weight"] != 0
            or selection["p1"]["assignment_delta"] != 0
            or selection["ce_runtime"]["selective_fp32_band"] != .04):
        raise ValueError("Decision configuration is not the precommitted one-shot selection")
    return selection


def _reservation(path: Path):
    manifest = json.loads((path / "reservation.json").read_text())
    id_file = path / "reserved_ids.txt"
    ids = id_file.read_text().splitlines()
    if (len(ids) != 12000 or len(set(ids)) != 12000 or _ids(ids) != EXPECTED_IDS
            or file_digest(id_file) != manifest["id_file_sha256"]
            or manifest["reserved_id_sha256"] != EXPECTED_IDS or manifest["labels_read"] is not False):
        raise ValueError("Older independently sealed ID reservation differs")
    return set(ids)


def prepare(root: Path, reservation: Path, raw: Path, p1: Path):
    _check_freeze(root, p1)
    selected = _reservation(reservation)
    query = root / "query_source1.tsv"
    if query.exists() or (root / "prepared.json").exists():
        raise FileExistsError("Refusing to replace confirmation query records")
    seen = set()
    with query.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["entity_id", "business_name", "business_address", "country"])
        for frame in pd.read_csv(raw, sep="\t", dtype=str, keep_default_na=False,
                                 usecols=["entity_id", "business_name", "business_address", "country"],
                                 chunksize=100000):
            rows = frame.loc[frame.entity_id.isin(selected)]
            if seen.intersection(rows.entity_id) or rows.entity_id.duplicated().any():
                raise ValueError("Repeated holdout S1")
            seen.update(rows.entity_id)
            writer.writerows(rows[["entity_id", "business_name", "business_address", "country"]].itertuples(index=False, name=None))
    if seen != selected or file_digest(raw) != "591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973":
        raise ValueError("Holdout references/raw input differ from independent reservation")
    _write(root / "prepared.json", {"reference_count": len(seen), "id_sha256": _ids(seen),
                                   "query_sha256": file_digest(query), "raw_sha256": file_digest(raw),
                                   "labels_read": False})
    print(json.dumps({"prepared": len(seen), "query_sha256": file_digest(query)}), flush=True)


def _checked_inputs(root: Path, reservation: Path, data: Path, tree: Path, p1: Path, v1: Path):
    frozen = _check_freeze(root, p1, v1)
    selected = _reservation(reservation)
    prepared = json.loads((root / "prepared.json").read_text())
    if file_digest(root / "query_source1.tsv") != prepared["query_sha256"]:
        raise ValueError("Prepared query changed")
    reference, pool = load_records(data)
    if len(reference) != 12000 or set(reference.entity_id) != selected or _ids(reference.entity_id) != EXPECTED_IDS:
        raise ValueError("Retrieved reference coverage is not the reserved 12k")
    mined = json.loads((data / "complete.json").read_text())
    if (mined["state"] != "complete" or mined["metadata"]["references"] != 12000
            or mined["metadata"]["mining_full_pool_records"] != 10320219
            or mined["metadata"]["positive_candidate_injection"] is not False
            or mined["metadata"]["labels_accessed"] is not False
            or mined["metadata"]["candidate_pairs"] != 600000):
        raise ValueError("Expected label-free top50 from full 10,320,219 target pool")
    tree_report = json.loads((tree / "complete.json").read_text())["report"]
    if (tree_report["checkpoint_sha256"] != frozen["tree_checkpoint_sha256"]
            or tree_report["candidate_pairs"] != 600000
            or tree_report["reference_count"] != 12000
            or tree_report["scorer"] != "frozen_tree"
            or tree_report["benchmark_only"] or tree_report["labels_accessed"]):
        raise ValueError("Frozen tree output is incomplete or not label-free")
    if not np.array_equal(reference.entity_id.to_numpy(), pd.read_parquet(data / "reference.parquet").entity_id.to_numpy()):
        raise ValueError("Reference row order changed")
    return frozen, reference, pool, tree_report


def _decisions(left, right, base, blend, gate, country, pool_ids, models, prior, n):
    offsets = np.r_[0, np.cumsum(np.bincount(left, minlength=n))]
    if not np.array_equal(left, np.repeat(np.arange(n), np.diff(offsets))):
        raise ValueError("Pair groups not complete and ordered")
    calibrated = predict_calibration(models, blend, gate, left, country)
    model = CountPosterior(np.asarray(prior["p_true_count"]), prior["gate_true_link_discovery"], prior["pair_prevalence"])
    keep = np.zeros(len(left), dtype=bool)
    for entity in range(n):
        start, end = offsets[entity:entity + 2]
        eligible = np.arange(start, end)[gate[start:end]]
        if not len(eligible):
            continue
        order = eligible[np.lexsort((right[eligible], -calibrated[eligible]))]
        unscored = calibrated[start:end][~gate[start:end]]
        count, _ = model.best_prefix(calibrated[order], float(unscored[0]) if len(unscored) else 1e-6, end - start)
        keep[order[:count]] = True
    keep, conflicts = assignment(keep, right, calibrated, left, 0.)
    return keep, calibrated, conflicts


def decide(root: Path, reservation: Path, data: Path, tree: Path, p1: Path, v1: Path, durable: Path):
    frozen, reference, pool, report = _checked_inputs(root, reservation, data, tree, p1, v1)
    if (root / "predictions_fixed.json").exists():
        raise FileExistsError("Predictions already committed")
    if checkpoint_identity(Path(json.loads(v1.read_text())["cross_encoder_checkpoint"]))["sha256"] != frozen["reranker_checkpoint_sha256"]:
        raise ValueError("Reranker checkpoint differs from frozen selection")
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model, tokenizer = load_reranker(json.loads(v1.read_text())["cross_encoder_checkpoint"], "cuda")
    options = SimpleNamespace(device="cuda", max_length=256, inference_batch_size=128,
                              length_bucketing=True, pad_to_multiple_of=8, amp="bf16")
    pair_path = tree / "scored_pairs.parquet"
    if file_digest(pair_path) != file_digest(data / "pairs.parquet"):
        raise ValueError("Tree scored pairs differ from mined candidate pairs")
    base = np.load(tree / "probabilities.npy", mmap_mode="r")
    if base.shape != (600000,) or base.dtype != np.float32:
        raise ValueError("Tree probability shape/order incorrect")
    cross = np.lib.format.open_memmap(root / "cross_probabilities.npy", mode="w+", dtype=np.float32, shape=(len(base),))
    blend = np.lib.format.open_memmap(root / "blend_probabilities.npy", mode="w+", dtype=np.float32, shape=(len(base),))
    near_total = gated_total = offset = 0
    for batch in iter_reference_batches(pair_path, len(reference), len(pool), 1000, require_sorted_right=True):
        size = len(batch.pairs)
        raw = np.asarray(base[offset:offset + size])
        gate = raw >= .01
        if gate.any():
            pairs = batch.pairs
            left_positions, left_inv = np.unique(pairs.left_idx, return_inverse=True)
            right_positions, right_inv = np.unique(pairs.right_idx, return_inverse=True)
            compact = pd.DataFrame({"left_idx": left_inv, "right_idx": right_inv})
            correct, near, _ = score_reranker_selective_fp32(
                model, tokenizer, reference.iloc[left_positions].reset_index(drop=True),
                pool.iloc[right_positions].reset_index(drop=True), compact, gate, options,
                raw, alpha=.75, threshold=.775, band=.04)
            near_total += int(near.sum())
        else:
            correct = np.zeros(size, dtype=np.float32)
        cross[offset:offset + size] = correct
        blend[offset:offset + size] = blend_probabilities(raw, correct, gate, .75)
        offset += size
        gated_total += int(gate.sum())
        print(f"CE confirmation {batch.end}/12000, gated={gated_total}, fp32={near_total}", flush=True)
    if offset != 600000:
        raise ValueError("Not every candidate received an aligned tree/blend score")
    cross.flush(); blend.flush()
    pairs = pd.read_parquet(pair_path, columns=["left_idx", "right_idx"])
    left, right = pairs.left_idx.to_numpy(np.int32), pairs.right_idx.to_numpy(np.int32)
    gate = np.asarray(base >= .01)
    v1_keep = np.asarray(blend, dtype=np.float64) >= .775
    p1_document = json.loads(p1.read_text())
    keep, calibrated, conflicts = _decisions(left, right, np.asarray(base), np.asarray(blend), gate,
        reference.country.to_numpy(), pool.entity_id.to_numpy(),
        p1_document["outer_cross_fit"]["final_calibrator_fitted_on_all_tune"],
        p1_document["count_model"]["training_only_prior"], len(reference))
    for name, array in (("v1_accept.npy", v1_keep), ("p1_accept.npy", keep),
                        ("calibrated_probabilities.npy", calibrated.astype(np.float32))):
        np.save(root / name, array)
    ids = pool.entity_id.to_numpy()
    starts = np.r_[0, np.cumsum(np.bincount(left, minlength=len(reference)))]
    for name, mask in (("v1_matching_results.tsv", v1_keep), ("p1_matching_results.tsv", keep)):
        with (root / name).open("x", newline="", encoding="utf-8") as out:
            writer = csv.writer(out, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "matched_entity_ids"])
            for i, entity in enumerate(reference.entity_id):
                start, end = starts[i:i+2]
                writer.writerow([entity, ",".join(sorted(ids[right[start:end][mask[start:end]]]))])
    outputs = ("v1_matching_results.tsv", "p1_matching_results.tsv", "v1_accept.npy", "p1_accept.npy",
               "calibrated_probabilities.npy", "cross_probabilities.npy", "blend_probabilities.npy")
    fixed = {"labels_accessed": False, "reference_count": len(reference), "candidate_pairs": len(left),
             "full_pool_rows": 10320219, "frozen_sha256": EXPECTED_FREEZE,
             "p1_config_sha256": EXPECTED_P1, "v1_pipeline_sha256": EXPECTED_V1,
             "query_sha256": file_digest(root / "query_source1.tsv"),
             "mined_pair_sha256": file_digest(pair_path),
             "tree_probability_sha256": file_digest(tree / "probabilities.npy"),
             "reranked_pairs": gated_total, "selective_fp32_pairs": near_total,
             "p1_one_to_one": conflicts,
             "files": {name: {"sha256": file_digest(root / name), "bytes": (root / name).stat().st_size} for name in outputs}}
    _write(root / "predictions_fixed.json", fixed)
    durable.mkdir(parents=True, exist_ok=True)
    for name in (*outputs, "predictions_fixed.json"):
        dest = durable / name
        if dest.exists():
            raise FileExistsError(f"Durable confirmation output already exists: {dest}")
        import shutil
        shutil.copyfile(root / name, dest)
        if file_digest(dest) != file_digest(root / name):
            raise ValueError("Durable prediction copy failed SHA verification")
    print(json.dumps({"committed": True, "v1_links": int(v1_keep.sum()),
                      "p1_links": int(keep.sum()), "fp32": near_total}), flush=True)


def _script(text: str):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "no_letters"
    nonlatin = {unicodedata.name(ch, "UNKNOWN").split()[0] for ch in letters
                if "LATIN" not in unicodedata.name(ch, "")}
    return "+".join(sorted(nonlatin)) if nonlatin else "Latin"


def confirm(root: Path, reservation: Path, data: Path, tree: Path, p1: Path, v1: Path,
            durable: Path, truth_path: Path):
    _, reference, pool, _ = _checked_inputs(root, reservation, data, tree, p1, v1)
    fixed_file = root / "predictions_fixed.json"
    if not fixed_file.exists() or file_digest(fixed_file) != file_digest(durable / fixed_file.name):
        raise ValueError("No verified durable pre-truth prediction commitment")
    fixed = json.loads(fixed_file.read_text())
    if fixed["labels_accessed"] or fixed["frozen_sha256"] != EXPECTED_FREEZE:
        raise ValueError("Prediction commitment invalid")
    for name, details in fixed["files"].items():
        for path in (root / name, durable / name):
            if file_digest(path) != details["sha256"] or path.stat().st_size != details["bytes"]:
                raise ValueError(f"Committed prediction altered: {path}")
    if (root / "confirmation_score.json").exists() or (durable / "confirmation_score.json").exists():
        raise FileExistsError("One-shot confirmation already scored")
    # FIRST HOLDOUT LABEL ACCESS: all prediction bytes are now committed twice.
    selected = set(reference.entity_id)
    truth = {}
    columns = None
    for chunk in pd.read_csv(truth_path, sep="\t", dtype=str, keep_default_na=False, chunksize=250000):
        if columns is None:
            columns = chunk.columns.tolist()
        if columns != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("Unexpected official ground truth schema")
        for source, targets in chunk.iloc[:, :2].itertuples(index=False, name=None):
            if source in selected:
                if source in truth:
                    raise ValueError("Duplicate ground truth S1")
                values = [value for value in targets.split(",") if value]
                if len(values) != len(set(values)):
                    raise ValueError("Duplicate true target for one S1")
                truth[source] = set(values)
    if set(truth) != selected:
        raise ValueError("Invalid scored S1 coverage")
    pairs = pd.read_parquet(tree / "scored_pairs.parquet", columns=["left_idx", "right_idx"])
    left, right = pairs.left_idx.to_numpy(np.int32), pairs.right_idx.to_numpy(np.int32)
    pool_ids = pool.entity_id.to_numpy()
    ref_ids = reference.entity_id.to_numpy()
    actual = np.asarray([len(truth[key]) for key in ref_ids], dtype=np.int32)
    found = np.asarray([pool_ids[r] in truth[ref_ids[l]] for l, r in zip(left, right)], dtype=bool)
    if len(set(zip(left.tolist(), right.tolist()))) != len(left):
        raise ValueError("Duplicate candidate pairs")
    if not set(pool_ids[right[found]]).issubset(set(pool_ids)):
        raise ValueError("Invalid target identity")
    country = reference.country.to_numpy()
    scripts = np.asarray([_script(n) for n in reference.business_name])
    def score(mask):
        predicted = np.bincount(left[mask], minlength=len(ref_ids))
        tp = np.bincount(left[mask & found], minlength=len(ref_ids))
        f05 = np.divide(5. * tp, 4 * predicted + actual, out=np.zeros(len(ref_ids)), where=(4*predicted+actual)>0)
        f05[(actual == 0) & (predicted == 0)] = 1.
        f1 = np.divide(2. * tp, predicted + actual, out=np.zeros(len(ref_ids)), where=(predicted+actual)>0)
        f1[(actual == 0) & (predicted == 0)] = 1.
        def summary(which):
            p, t, hit = int(predicted[which].sum()), int(actual[which].sum()), int(tp[which].sum())
            single = which & (actual == 0)
            return {"reference_count": int(which.sum()), "macro_f05": float(f05[which].mean()),
                    "macro_f1": float(f1[which].mean()), "precision": hit / p if p else 0.,
                    "recall": hit / t if t else 0., "predicted_links": p, "true_links": t,
                    "true_positive_links": hit, "false_positive_links": p-hit,
                    "missed_true_links": t-hit, "singletons": int(single.sum()),
                    "singleton_accuracy": float(np.mean(predicted[single] == 0)) if single.any() else None}
        every = np.ones(len(ref_ids), dtype=bool)
        result = summary(every)
        result["by_country"] = {c: summary(country == c) for c in sorted(set(country))}
        result["by_query_name_script"] = {s: summary(scripts == s) for s in sorted(set(scripts))}
        return result, f05, predicted, tp
    v1 = np.load(root / "v1_accept.npy")
    p1_mask = np.load(root / "p1_accept.npy")
    for name, mask in (("v1", v1), ("p1", p1_mask)):
        if mask.shape != found.shape or mask.dtype != np.bool_:
            raise ValueError(f"{name} decision array not aligned")
    v1_score, v1_entity, v1_pred, v1_tp = score(v1)
    p1_score, p1_entity, p1_pred, p1_tp = score(p1_mask)
    # Re-evaluate TSV bytes independently of candidate masks and include empty S1s.
    for name, mask in (("v1", v1), ("p1", p1_mask)):
        with (root / f"{name}_matching_results.tsv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream, delimiter="\t"))
        if rows[0] != ["source1_entity_id", "matched_entity_ids"] or len(rows) != len(ref_ids) + 1:
            raise ValueError("Prediction TSV header/coverage incorrect")
        expected = {}
        for l, r in zip(left[mask], right[mask]):
            expected.setdefault(int(l), []).append(pool_ids[r])
        for i, row in enumerate(rows[1:]):
            if row != [ref_ids[i], ",".join(sorted(expected.get(i, [])))]:
                raise ValueError("Committed TSV differs from scored candidate mask")
    retrieval_hits = int(found.sum())
    result = {"scope": "older independently sealed 12000 S1; full 10320219 training targets; one shot",
              "prediction_commitment_sha256": file_digest(fixed_file), "freeze_sha256": EXPECTED_FREEZE,
              "truth_source_sha256": file_digest(truth_path), "truth_columns": columns,
              "retrieval": {"true_links": int(actual.sum()), "retrieved_true_links": retrieval_hits,
                            "missed_true_links": int(actual.sum())-retrieval_hits,
                            "link_recall": retrieval_hits / int(actual.sum())},
              "v1": v1_score, "p1": p1_score,
              "p1_minus_v1": {"macro_f05": float((p1_entity-v1_entity).mean()),
                              "precision": p1_score["precision"]-v1_score["precision"],
                              "false_positive_links": p1_score["false_positive_links"]-v1_score["false_positive_links"],
                              "improved_entities": int((p1_entity > v1_entity).sum()),
                              "worsened_entities": int((p1_entity < v1_entity).sum()),
                              "unchanged_entities": int((p1_entity == v1_entity).sum())},
              "paired_bootstrap_f05": paired_bootstrap(p1_entity, v1_entity, n_bootstrap=10000, seed=20260925),
              "decision": "P1" if p1_score["macro_f05"] > v1_score["macro_f05"] and p1_score["precision"] >= v1_score["precision"] else "existing_validated_v1"}
    _write(root / "confirmation_score.json", result)
    import shutil
    shutil.copyfile(root / "confirmation_score.json", durable / "confirmation_score.json")
    if file_digest(root / "confirmation_score.json") != file_digest(durable / "confirmation_score.json"):
        raise ValueError("Durable score copy failed")
    print(json.dumps({"v1": v1_score, "p1": p1_score,
                      "ci": result["paired_bootstrap_f05"]["confidence_intervals"]["left_minus_right"],
                      "decision": result["decision"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "decide", "confirm"])
    for name in ("root", "reservation", "p1", "raw", "data", "tree", "v1", "durable", "truth"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    required = {"prepare": ("root", "reservation", "raw", "p1"),
                "decide": ("root", "reservation", "data", "tree", "p1", "v1", "durable"),
                "confirm": ("root", "reservation", "data", "tree", "p1", "v1", "durable", "truth")}
    if any(getattr(args, name) is None for name in required[args.stage]):
        parser.error(f"{args.stage} requires {required[args.stage]}")
    if args.stage == "prepare":
        prepare(args.root, args.reservation, args.raw, args.p1)
    elif args.stage == "decide":
        decide(args.root, args.reservation, args.data, args.tree, args.p1, args.v1, args.durable)
    else:
        confirm(args.root, args.reservation, args.data, args.tree, args.p1, args.v1, args.durable, args.truth)


if __name__ == "__main__":
    main()
