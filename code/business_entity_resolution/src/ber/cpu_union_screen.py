"""CPU-only retrieval ceiling for saved medium dense versus lexical/phonetic edges.

This is an ID-aligned, TRAIN/TUNE-only compatibility screen, not a new matcher:
union-only edges have neither recomputed candidate-dependent features nor CE scores.
Truth edges are joined for accounting after candidate sets are frozen, never injected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .cpu_decision import TUNE_ID_SHA256, file_sha256
from .experiment import json_write, load_data
from .hf_experiment import frame_digest
from .retrieval_ablation import _script
from .uncertainty import paired_bootstrap


RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
REFERENCE_COLUMNS = RECORD_COLUMNS + ["split"]
PAIR_COLUMNS = ["left_idx", "right_idx"]
LEXICAL_CAPS = (20, 30)


def check_pairs(pairs, reference, pool, count):
    if (not set(PAIR_COLUMNS) <= set(pairs) or len(pairs) != count
            or pairs.duplicated(PAIR_COLUMNS).any()
            or pairs.left_idx.min() < 0 or pairs.left_idx.max() >= len(reference)
            or pairs.right_idx.min() < 0 or pairs.right_idx.max() >= len(pool)):
        raise ValueError("Incomplete, duplicated or out-of-range input candidate pairs")


def oracle(reference, tune, truth_pairs, source):
    actual = np.bincount(truth_pairs.left_idx.to_numpy(dtype=np.int64), minlength=len(reference))
    present = truth_pairs[source].to_numpy(dtype=bool)
    found = np.bincount(truth_pairs.left_idx.to_numpy(dtype=np.int64)[present], minlength=len(reference))
    count = int(tune.sum())
    scored = np.divide(5.0 * found, 4 * found + actual, out=np.zeros(len(reference)),
                       where=(4 * found + actual) > 0)
    f1 = np.divide(2.0 * found, found + actual, out=np.zeros(len(reference)),
                   where=(found + actual) > 0)
    singleton = actual == 0
    scored[singleton] = f1[singleton] = 1.0
    nonempty = tune & ~singleton
    result = {"references": count, "true_links": int(actual[tune].sum()),
              "retrieved_true_links": int(found[tune].sum()),
              "link_recall": float(found[tune].sum() / actual[tune].sum()),
              "complete_match_set_recall": float((found[nonempty] == actual[nonempty]).mean()),
              "oracle_macro_f05": float(scored[tune].mean()), "oracle_macro_f1": float(f1[tune].mean()),
              "oracle_micro_precision": 1.0, "oracle_false_links": 0,
              "oracle_singleton_accuracy": 1.0, "true_singletons": int((tune & singleton).sum()),
              "retrieval_missed_true_links": int((actual[tune] - found[tune]).sum())}
    result["by_country"] = {}
    for country in sorted(reference.country[tune].unique()):
        selected = tune & reference.country.eq(country).to_numpy()
        result["by_country"][country] = {
            "references": int(selected.sum()), "true_links": int(actual[selected].sum()),
            "retrieved_true_links": int(found[selected].sum()),
            "link_recall": float(found[selected].sum() / actual[selected].sum()),
            "oracle_macro_f05": float(scored[selected].mean())}
    return result, scored


def run(args):
    started = time.monotonic()
    dense_dir, lex_dir, output = map(Path, (args.dense, args.lexical, args.output))
    dense_pair_path = Path(args.dense_pairs) if args.dense_pairs else dense_dir / "pairs.parquet"
    dense_meta = json.loads((dense_dir / "metadata.json").read_text())
    lex_meta = json.loads((lex_dir / "metadata.json").read_text())
    complete = json.loads((dense_dir / "complete.json").read_text())
    identity = complete["config"]["pool_embedding_identity"]
    if (complete.get("state") != "complete" or complete["metadata"] != dense_meta
            or complete["config"]["labels_used_for_mining"] is not False
            or dense_meta["positive_candidate_injection"] is not False
            or lex_meta["dataset_kind"] != "full_pool_hard_candidates"
            or "no positive candidate injection" not in lex_meta["label_usage"]
            or identity["scope"] != "full_pool" or not identity["target_rows"] == 10320219
            == dense_meta["mining_full_pool_records"] == sum(lex_meta["full_pool_records_by_country"].values())
            or dense_meta["reference_id_sha256"] != lex_meta["reference_id_sha256"]
            or dense_meta["split_counts"] != lex_meta["split_counts"]):
        raise ValueError("Both inputs must be completed label-free mining on the same full target pool")
    dense_ref, dense_pool, dense_truth = load_data(dense_dir)
    lex_ref, lex_pool, lex_truth = load_data(lex_dir)
    if (dense_ref.entity_id.duplicated().any() or lex_ref.entity_id.duplicated().any()
            or dense_pool.entity_id.duplicated().any() or lex_pool.entity_id.duplicated().any()
            or set(lex_ref.split.unique()) != {"train", "tune"} or dense_truth != lex_truth):
        raise ValueError("Reference/pool IDs, splits or complete TRAIN/TUNE truth disagree")
    if (frame_digest(dense_ref, REFERENCE_COLUMNS) != complete["config"]["reference_digest"]
            or hashlib.sha256("\n".join(sorted(lex_ref.entity_id)).encode()).hexdigest()
            != lex_meta["reference_id_sha256"]):
        raise ValueError("Saved reference identity does not match these rows")
    lex_ref_index = pd.Index(lex_ref.entity_id)
    dense_to_lex = lex_ref_index.get_indexer(dense_ref.entity_id)
    if (np.any(dense_to_lex < 0) or not np.array_equal(
            lex_ref.iloc[dense_to_lex][REFERENCE_COLUMNS].to_numpy(),
            dense_ref[REFERENCE_COLUMNS].to_numpy())):
        raise ValueError("Reference texts/countries/splits disagree across retrieval branches")
    lex_pool_index = pd.Index(lex_pool.entity_id)
    shared_idx = lex_pool_index.get_indexer(dense_pool.entity_id)
    shared = shared_idx >= 0
    if not np.array_equal(lex_pool.iloc[shared_idx[shared]][RECORD_COLUMNS].to_numpy(),
                          dense_pool.loc[shared, RECORD_COLUMNS].to_numpy()):
        raise ValueError("Shared target IDs have different text or country")
    del shared_idx, shared
    lex_tune = lex_ref.split.eq("tune").to_numpy()
    if (len(lex_ref) != 47999 or lex_tune.sum() != 12000
            or hashlib.sha256("\n".join(sorted(lex_ref.entity_id[lex_tune])).encode()).hexdigest()
            != TUNE_ID_SHA256):
        raise ValueError("This is not the fixed medium TRAIN/TUNE reference population")
    dense_pairs = pd.read_parquet(dense_pair_path, columns=PAIR_COLUMNS)
    lex_pairs = pd.read_parquet(lex_dir / "candidate_pairs.parquet",
                                columns=PAIR_COLUMNS + ["retrieval_rank", "phonetic_hit", "phonetic_only"])
    check_pairs(dense_pairs, dense_ref, dense_pool, dense_meta["candidate_pairs"])
    check_pairs(lex_pairs, lex_ref, lex_pool, lex_meta["candidate_pairs"])
    if args.dense_score_config:
        score_config = json.loads(Path(args.dense_score_config).read_text())
        if (score_config["reference_digest"] != complete["config"]["reference_digest"]
                or frame_digest(dense_pairs, PAIR_COLUMNS) != score_config["pair_digest"]):
            raise ValueError("Dense candidates do not align with the saved reranker input")
    dense_rows = dense_pairs.left_idx.to_numpy(dtype=np.int64)
    dense_tune = lex_tune[dense_to_lex[dense_rows]]
    lex_rows = lex_pairs.left_idx.to_numpy(dtype=np.int64)
    lex_pair_tune = lex_tune[lex_rows]
    if int(dense_tune.sum()) != 600000 or int(lex_pair_tune.sum()) != 747894:
        raise ValueError("Input TUNE candidate coverage does not match the completed reports")
    dense_edges = pd.DataFrame({
        "left_idx": dense_to_lex[dense_rows[dense_tune]],
        "target_id": dense_pool.entity_id.to_numpy()[dense_pairs.right_idx.to_numpy(dtype=np.int64)[dense_tune]]})
    lex_edges = pd.DataFrame({
        "left_idx": lex_rows[lex_pair_tune],
        "target_id": lex_pool.entity_id.to_numpy()[lex_pairs.right_idx.to_numpy(dtype=np.int64)[lex_pair_tune]],
        "retrieval_rank": lex_pairs.retrieval_rank.to_numpy()[lex_pair_tune],
        "phonetic_hit": lex_pairs.phonetic_hit.to_numpy()[lex_pair_tune],
        "phonetic_only": lex_pairs.phonetic_only.to_numpy()[lex_pair_tune]})
    union = dense_edges.merge(lex_edges, on=["left_idx", "target_id"], how="outer",
                              validate="one_to_one", indicator=True)
    if len(union) != len(dense_edges) + len(lex_edges) - int(union._merge.eq("both").sum()):
        raise ValueError("Duplicate or dropped candidate edges in the union")
    # Keep missed truth *records* for scoring only; no candidate edge is introduced.
    truth_pairs = pd.DataFrame([(idx, target) for idx, ref_id in enumerate(lex_ref.entity_id)
                                if lex_tune[idx] for target in lex_truth[ref_id]],
                               columns=["left_idx", "target_id"])
    truth_pairs = truth_pairs.merge(union, on=["left_idx", "target_id"], how="left",
                                    validate="one_to_one")
    origins = truth_pairs._merge.astype(object).fillna("neither")
    truth_pairs["dense_found"] = origins.isin(["left_only", "both"])
    truth_pairs["lexical_found"] = origins.isin(["right_only", "both"])
    truth_pairs["union_found"] = origins.ne("neither")
    summary, scores = {}, {}
    for label, column in (("dense", "dense_found"), ("lexical_phonetic", "lexical_found"),
                          ("union", "union_found")):
        summary[label], scores[label] = oracle(lex_ref, lex_tune, truth_pairs, column)
    rank_screens = {}
    for cap in LEXICAL_CAPS:
        # Dense remains complete. Only the incremental lexical branch is capped;
        # keep its native-script phonetic hits independently of lexical rank.
        lex_kept = (lex_pairs.retrieval_rank.to_numpy()[lex_pair_tune] <= cap) | lex_pairs.phonetic_hit.to_numpy()[lex_pair_tune]
        selected = union._merge.ne("right_only") | (union.retrieval_rank <= cap) | union.phonetic_hit.fillna(False)
        found = truth_pairs.dense_found | (truth_pairs.lexical_found & (
            (truth_pairs.retrieval_rank <= cap) | truth_pairs.phonetic_hit.fillna(False)))
        truth_pairs[f"capped_{cap}_found"] = found
        score, entity_scores = oracle(lex_ref, lex_tune, truth_pairs, f"capped_{cap}_found")
        rank_screens[f"top{cap}_or_phonetic"] = {
            "lexical_pairs_retained": int(lex_kept.sum()), "union_pairs": int(selected.sum()),
            "new_pairs_vs_dense": int(union._merge.eq("right_only")[selected].sum()),
            "new_true_links_vs_dense": int((found & ~truth_pairs.dense_found).sum()),
            "retrieval": score,
            "paired_oracle_minus_dense": paired_bootstrap(
                entity_scores[lex_tune], scores["dense"][lex_tune],
                n_bootstrap=args.replicates, seed=20260925)}
    for key, expected in (("link_recall", dense_meta["retrieval"]["tune"]["link_recall"]),
                          ("oracle_macro_f05", dense_meta["retrieval"]["tune"]["oracle_macro_f05"])):
        if not np.isclose(summary["dense"][key], expected, rtol=0, atol=1e-12):
            raise ValueError(f"Dense {key} fails to reproduce the saved result")
    target_positions = lex_pool_index.get_indexer(truth_pairs.target_id)
    if np.any(target_positions < 0):
        raise ValueError("A truth record is absent from the retained lexical pool")
    for field in ("business_name", "business_address"):
        scripts = [_script(text) for text in lex_pool.iloc[target_positions][field]]
        truth_pairs[field + "_script"] = scripts
    script_breakdown = {}
    for field in ("business_name", "business_address"):
        for script, group in truth_pairs.groupby(field + "_script"):
            script_breakdown[f"{field}/{script}"] = {
                "true_links": len(group),
                **{f"{name}_found": int(group[column].sum()) for name, column in (
                    ("dense", "dense_found"), ("lexical_phonetic", "lexical_found"),
                    ("union", "union_found"))}}
    paired = paired_bootstrap(scores["union"][lex_tune], scores["dense"][lex_tune],
                              n_bootstrap=args.replicates, seed=20260925)
    lexical_only_truth = truth_pairs[origins.eq("right_only")]
    lexical_only_ranks = {f"rank_le_{cap}": int((lexical_only_truth.retrieval_rank <= cap).sum())
                          for cap in (10, 20, 30, 50)}
    lexical_only_ranks.update(phonetic_hit=int(lexical_only_truth.phonetic_hit.sum()),
                              phonetic_only=int(lexical_only_truth.phonetic_only.sum()))
    diff = scores["union"][lex_tune] - scores["dense"][lex_tune]
    files = {"dense_reference": dense_dir / "reference.parquet", "dense_pool": dense_dir / "pool.parquet",
             "dense_pairs": dense_pair_path, "dense_truth": dense_dir / "truth.json",
             "dense_metadata": dense_dir / "metadata.json", "dense_complete": dense_dir / "complete.json",
             "lex_reference": lex_dir / "reference.parquet", "lex_pool": lex_dir / "pool.parquet",
             "lex_pairs": lex_dir / "candidate_pairs.parquet", "lex_truth": lex_dir / "truth.json",
             "lex_metadata": lex_dir / "metadata.json"}
    if args.dense_score_config:
        files["dense_pair_config"] = Path(args.dense_score_config)
    report = {"schema": 1, "retrieval_only": True, "full_competing_target_pool": identity["target_rows"],
              "reference_count": int(lex_tune.sum()), "tune_id_sha256": TUNE_ID_SHA256,
              "input_sha256": {key: file_sha256(path) for key, path in files.items()},
              "candidate_pairs": {"dense": len(dense_edges), "lexical_phonetic": len(lex_edges),
                                  "union": len(union), "overlap": int(union._merge.eq("both").sum()),
                                  "added_vs_dense": int(union._merge.eq("right_only").sum())},
              "true_link_origins": {name: int(origins.eq(name).sum()) for name in (
                  "both", "left_only", "right_only", "neither")},
              "lexical_only_truth_ranks": lexical_only_ranks,
              "retrieval": summary, "rank_cap_screens": rank_screens,
              "script_of_truth_target": script_breakdown,
              "paired_union_minus_dense_oracle_f05": paired,
              "entities_oracle_improved": int((diff > 0).sum()),
              "entities_oracle_worsened": int((diff < 0).sum()),
              "entities_oracle_unchanged": int((diff == 0).sum()),
              "runtime_seconds": time.monotonic() - started,
              "limitations": "An oracle knows which retrieved pairs are true: its precision=1/false links=0/"
                             "singleton accuracy=1 are definitions, not matcher results. Union-only pairs have no "
                             "aligned tree/CE score or recomputed candidate-dependent features. TUNE is reused; "
                             "no holdout/French/official-test labels were inspected."}
    config = {"schema": 1, "seed": 20260925, "device": "cpu", "threads": 2,
              "hypothesis": "Lexical/phonetic retrieval recovers true medium TUNE links missed by dense top50 "
                            "when both search the same 10,320,219 targets.",
              "changed_variable": "candidate set: dense top50 plus all or top-ranked lexical/phonetic edges",
              "bounded_lexical_rank_caps": list(LEXICAL_CAPS), "phonetic_hits_always_retained": True,
              "dense_source": "artifacts/full_dense_medium (completed TRAIN/TUNE full-pool mining)",
              "lexical_source": "data/medium_hard (completed TRAIN/TUNE full-pool mining)",
              "tune_id_sha256": TUNE_ID_SHA256, "reference_id_sha256": lex_meta["reference_id_sha256"],
              "embedding_checkpoint_sha256": identity["checkpoint"]["sha256"],
              "bootstrap_replicates": args.replicates, "evaluation_used": False,
              "positive_candidate_injection": False, "rebuild_indexes": False,
              "matcher_scores_claimed": False}
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / "experiment_config.json", config)
    json_write(output / "results.json", report)
    print(json.dumps({"candidate_pairs": report["candidate_pairs"],
                      "true_link_origins": report["true_link_origins"],
                      "lexical_only_truth_ranks": lexical_only_ranks,
                      "retrieval": summary, "rank_cap_screens": rank_screens,
                      "paired_union_minus_dense_oracle_f05": paired,
                      "runtime_seconds": report["runtime_seconds"]}, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", required=True)
    parser.add_argument("--dense-pairs", help="Exact completed dense pairs if stored with the reranker")
    parser.add_argument("--dense-score-config", help="Optional saved reranker pair/reference digests")
    parser.add_argument("--lexical", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
