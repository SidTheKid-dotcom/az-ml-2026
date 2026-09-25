"""Synthetic-only P6 tests: complete groups, global assignment and hard gates."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from ber.full_embeddings import checkpoint_identity, file_digest
from ber.v2_decision import TUNE_ID_SHA256, assignment
from ber.v2_release import (_pure_assignment, _saved_claims, _saved_pair_batches,
                            prepare, run, verify_ready,
                            CLAIM_DTYPE)
from ber.v2_tree_train import feature_columns, union_feature_columns
from ber.validate_release import validate_release


OFFICIAL = Path(__file__).resolve().parents[1] / "resources/validate_submission.py"
RECORDS = ("entity_id", "business_name", "business_address", "country")


class FixtureTree:
    def __init__(self, columns):
        self.feature_names_ = columns

    def predict_proba(self, features):
        p = features["dense_score"].to_numpy(dtype=np.float32)
        return np.column_stack((1 - p, p))


def _scores(reference, pool, pairs, selected, amp):
    answer = np.zeros(len(pairs), dtype=np.float32)
    base = pool.entity_id.to_numpy()[pairs.right_idx.to_numpy(dtype=np.int64)]
    answer[selected] = np.where(base[selected] == "S2-x", .91 if amp == "fp32" else .9, .88)
    return answer


class V2ReleaseFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ber_v2_release_fixture_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.test_dir = self.root / "dataset/test"
        self.test_dir.mkdir(parents=True)
        self.candidates = self.root / "test_candidates"
        self.candidates.mkdir()
        self.output = self.root / "submission_v2_fixture"
        self.references = pd.DataFrame([
            ("S1-a", "Acme", "10 Main", "US"),
            ("S1-b", "Acme", "10 Main", "US"),
            ("S1-c", "Libre", "Rue 11", "France"),
            ("S1-d", "Other", "22 Main", "US"),
        ], columns=RECORDS)
        self.raw_targets = pd.DataFrame([
            ("S2-x", "Acme", "10 Main", "US"),
            ("S2-z", "Other", "22 Main", "US"),
            ("S3-y", "Acme", "10 Main", "US"),
        ], columns=RECORDS)
        self.pool = self.raw_targets.copy()
        self.pool["raw_row_idx"] = np.arange(3, dtype=np.int32)
        self.pairs = pd.DataFrame({
            "left_idx": np.array([0, 0, 1, 3], dtype=np.int32),
            "right_idx": np.array([0, 2, 0, 1], dtype=np.int32),
            "right_raw_idx": np.array([0, 2, 0, 1], dtype=np.int32),
            "dense_score": np.array([.94, .92, .83, .005], dtype=np.float32),
            "dense_rank": np.array([1, 2, 1, 1], dtype=np.int32),
            "name_score": np.zeros(4, np.float32),
            "address_score": np.zeros(4, np.float32),
            "retrieval_score": np.array([.94, .92, .83, .005], dtype=np.float32),
            "retrieval_rank": np.array([1, 2, 1, 1], dtype=np.int32),
            "dense_only": np.ones(4, np.float32),
        })
        columns = feature_columns(self.pairs.columns)
        self.tree = self.root / "tree.joblib"
        joblib.dump({"model": FixtureTree(columns), "feature_columns": columns, "threshold": .775}, self.tree)
        self.ce = self.root / "ce"
        self.encoder = self.root / "encoder"
        for directory in (self.ce, self.encoder):
            directory.mkdir()
            (directory / "model.safetensors").write_bytes(b"synthetic checkpoint; no pretrained model")
        self.calibration = self.root / "tune_calibration.json"
        self.config = self.root / "pipeline.json"
        self.pipeline = {
            "schema": 2, "selection": {"scope": "fixture"},
            "fixture_counts": {"references": 4, "targets": 3},
            "retrieval": {"kind": "dense_k50_v1", "top_k": 2,
                          "encoder_checkpoint": str(self.encoder),
                          "encoder_sha256": checkpoint_identity(self.encoder)["sha256"]},
            "tree": {"checkpoint": str(self.tree), "sha256": file_digest(self.tree),
                     "feature_policy": "as_saved"},
            "cross_encoder": {"checkpoint": str(self.ce),
                              "sha256": checkpoint_identity(self.ce)["sha256"]},
            "blend": {"alpha": .75, "tree_gate": .01},
            "runtime": {"amp": "selective_fp32", "device": "cpu", "threads": 1,
                        "reference_batch_size": 1, "inference_batch_size": 2,
                        "max_length": 32, "pad_to_multiple_of": 8, "length_bucketing": True,
                        "fp32_band": .2, "fp32_anchor": .775},
            "decision": {"kind": "p1_count_assignment", "assignment_delta": 0,
                         "source_weight": 0., "calibration": {"kind": "tune_refit",
                             "path": str(self.calibration), "sha256": ""}},
        }
        self.write_calibration()

    def write_inputs(self):
        self.references.to_csv(self.test_dir / "test_source1.tsv", sep="\t", index=False, lineterminator="\n")
        self.raw_targets.iloc[:2].to_csv(self.test_dir / "test_source2.tsv", sep="\t", index=False, lineterminator="\n")
        self.raw_targets.iloc[2:].to_csv(self.test_dir / "test_source3.tsv", sep="\t", index=False, lineterminator="\n")
        self.references.to_parquet(self.candidates / "reference.parquet", index=False)
        self.pool.to_parquet(self.candidates / "pool.parquet", index=False)
        self.pairs.to_parquet(self.candidates / "pairs.parquet", index=False, row_group_size=1)
        metadata = {"mode": "unlabeled_inference", "references": len(self.references),
                    "pool_records": len(self.pool), "mining_full_pool_records": len(self.raw_targets),
                    "candidate_pairs": len(self.pairs), "pair_order": ["left_idx", "right_idx"],
                    "labels_accessed": False, "positive_candidate_injection": False,
                    "all_input_references_retained": True, "top_k": 2}
        (self.candidates / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        config = {"labels_accessed": False, "positive_candidate_injection": False,
                  "query_input": {"expected_rows": len(self.references),
                                  "size_bytes": (self.test_dir / "test_source1.tsv").stat().st_size},
                  "pool_embedding_identity": {"target_rows": len(self.raw_targets),
                                              "checkpoint": {"sha256": checkpoint_identity(self.encoder)["sha256"]}}}
        (self.candidates / "complete.json").write_text(json.dumps({"state": "complete", "config": config,
                                                                      "metadata": metadata}), encoding="utf-8")

    def write_calibration(self):
        model = {"x": [0., 1.], "y": [.001, .999], "p_ungated": .001}
        cal = {"schema": 2, "fit_scope": "TUNE", "tune_id_sha256": TUNE_ID_SHA256,
               "holdout_labels_used": False, "full_competing_target_pool": 10_320_219,
               "prior_source": {"split": "TRAIN", "full_competing_target_pool": 10_320_219,
                                "train_reference_ids_sha256": "a" * 64,
                                "train_truth_sha256": "b" * 64},
               "score_identity": {"tree_sha256": self.pipeline["tree"]["sha256"],
                                  "cross_encoder_sha256": self.pipeline["cross_encoder"]["sha256"],
                                  "retrieval_kind": "dense_k50_v1", "alpha": .75, "gate": .01},
               "calibrators": {"__pooled__": model, "US": model, "India": model},
               "training_only_prior": {"train_references": 100, "train_true_links": 120,
                                       "p_true_count": [.01, .49, .5],
                                       "gate_true_link_discovery": .99, "pair_prevalence": .2},
               "selection": {"source_weight": 0., "assignment_delta": 0}}
        self.calibration.write_text(json.dumps(cal), encoding="utf-8")
        self.pipeline["decision"]["calibration"]["sha256"] = file_digest(self.calibration)
        self.config.write_text(json.dumps(self.pipeline), encoding="utf-8")
        self.write_inputs()  # Complete marker must bind the fixture encoder, never a label file.

    def prepared(self):
        return prepare(self.config, self.candidates, self.test_dir, OFFICIAL, fixture=True)

    def predict(self):
        return run(self.config, self.candidates, self.test_dir, self.output,
                   validator=OFFICIAL, fixture=True, score_fn=_scores)

    def test_dry_run_requires_only_fixture_and_writes_nothing(self):
        result = run(self.config, self.candidates, self.test_dir, self.output,
                     validator=OFFICIAL, fixture=True, dry_run=True)
        self.assertEqual(result["candidate_pairs"], 4)
        self.assertFalse(result["ready_for_submission"])
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "restricted to a tiny synthetic fixture"):
            run(self.config, self.candidates, self.test_dir, self.output, validator=OFFICIAL, dry_run=True)

    def test_unselected_model_cannot_start_an_official_release(self):
        with self.assertRaisesRegex(ValueError, "finalist"):
            prepare(self.config, self.candidates, self.test_dir, OFFICIAL, fixture=False)
        with self.assertRaisesRegex(ValueError, "Injected test scorers"):
            run(self.config, self.candidates, self.test_dir, self.output, validator=OFFICIAL,
                holdout_attestation=self.root / "not_opened.json", score_fn=_scores)
        self.assertFalse(self.output.exists())

    def test_complete_fixture_selective_fp32_and_cross_batch_assignment(self):
        result = self.predict()
        self.assertTrue(result["fixture_validation_passed"])
        self.assertFalse(result["ready_for_submission"])
        self.assertFalse((self.output / "ready.json").exists())
        self.assertEqual(result["source1_rows"], 4)
        self.assertGreater(result["selective_fp32_rescored_pairs"], 0)
        self.assertEqual(result["assignment"]["conflicting_targets"], 1)
        matches = (self.output / "matching_results.tsv").read_text().splitlines()
        candidates = (self.output / "candidate_pairs.tsv").read_text().splitlines()
        self.assertEqual([row.split("\t")[0] for row in matches[1:]],
                         self.references.entity_id.tolist())
        self.assertEqual([row.split("\t")[0] for row in candidates[1:]],
                         self.references.entity_id.tolist())
        self.assertEqual(matches[2], "S1-b\t")  # Loses shared S2 claim across batch boundary.
        self.assertEqual(matches[3], "S1-c\t")  # S1 with no candidates still has one row.
        self.assertEqual(candidates[3], "S1-c\t")
        self.assertEqual(matches[4], "S1-d\t")  # Tree-gated rejection, not dropped reference.
        self.assertEqual(candidates[1], "S1-a\tS2-x,S3-y")
        self.assertEqual(np.load(self.output / "probabilities.npy").shape, (4,))
        self.assertEqual(np.load(self.output / "cross_probabilities.npy")[-1], 0.)
        self.assertNotIn(b"\r\n", (self.output / "matching_results.tsv").read_bytes())
        self.assertFalse(json.loads((self.output / "complete.json").read_text())["labels_accessed"])
        with self.assertRaises((FileNotFoundError, ValueError)):
            verify_ready(self.output)  # Never package a fixture as a full-test release.

    def test_global_pure_assignment_is_exactly_the_p1_tie_rule(self):
        claims = np.array([(0, 1, 0, .90), (2, 2, 5, .91), (1, 1, 3, .90),
                           (2, 1, 6, .80)], dtype=CLAIM_DTYPE)
        actual, details = _pure_assignment(claims)
        expected, _ = assignment(np.ones(4, bool), claims["right"], claims["score"],
                                 claims["left"], 0)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(details["removed_claims"], 2)

    def test_saved_score_stream_prefix_preserves_pair_order_and_non_submittable_scope(self):
        decision = self.prepared().decision
        score = np.array([.91, .88, .85, 0], dtype=np.float32)
        base = np.array([.94, .92, .83, .005], dtype=np.float32)
        path = self.root / "prefix_claims.bin"
        report = _saved_claims(self.candidates / "pairs.parquet", score, base,
                               self.references.country.to_numpy(), len(self.pool), decision,
                               path, expected_pairs=4, stop_reference=2, progress_every=10)
        self.assertEqual(report["candidate_pairs"], 3)
        self.assertEqual(report["references_with_candidates"], 2)
        claims = np.fromfile(path, dtype=CLAIM_DTYPE)
        self.assertTrue((claims["left"] < 2).all())
        self.assertTrue((claims["pair_row"] < 3).all())
        full = self.root / "full_claims.bin"
        complete = _saved_claims(self.candidates / "pairs.parquet", score, base,
                                 self.references.country.to_numpy(), len(self.pool), decision,
                                 full, expected_pairs=4, progress_every=10)
        self.assertEqual(complete["candidate_pairs"], 4)
        self.assertEqual(complete["references_with_candidates"], 3)
        self.assertEqual([len(left) for _, left, _ in _saved_pair_batches(
            self.candidates / "pairs.parquet", 4, 3, read_rows=1)], [2, 1, 1])

    def test_saved_score_stream_rejects_duplicate_or_reordered_pairs_at_row_boundary(self):
        self.pairs.loc[1, "right_idx"] = np.int32(0)
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "duplicate or reordered pair"):
            list(_saved_pair_batches(self.candidates / "pairs.parquet", 4, 3, read_rows=1))

    def test_missing_and_reordered_s1_are_rejected_before_scoring(self):
        for frame in (self.references.iloc[:-1], self.references.iloc[[1, 0, 2, 3]],
                      self.references.iloc[[0, 0, 2, 3]]):
            with self.subTest(ids=frame.entity_id.tolist()):
                frame.to_parquet(self.candidates / "reference.parquet", index=False)
                with self.assertRaisesRegex(ValueError, "Source 1"):
                    self.prepared()

    def test_mismatched_duplicate_and_reordered_target_ids_fail(self):
        for frame in (self.pool.assign(entity_id=["S2-wrong", "S2-z", "S3-y"]),
                      self.pool.assign(entity_id=["S2-x", "S2-x", "S3-y"]),
                      self.pool.iloc[[1, 0, 2]]):
            with self.subTest(ids=frame.entity_id.tolist()):
                frame.to_parquet(self.candidates / "pool.parquet", index=False)
                with self.assertRaisesRegex(ValueError, "target|pool"):
                    self.prepared()

    def test_duplicate_pair_even_across_parquet_row_groups_never_ready(self):
        self.pairs.loc[1, ["left_idx", "right_idx", "right_raw_idx"]] = [0, 0, 0]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Duplicate candidate"):
            self.predict()
        self.assertFalse((self.output / "ready.json").exists())

    def test_reordered_pair_or_mismatched_feature_order_cannot_reach_ready(self):
        for column in ("right_idx", "right_raw_idx"):
            self.pairs.loc[[0, 1], column] = np.array([2, 0], dtype=np.int32)
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "aligned scores cannot be reordered"):
            self.predict()
        self.assertFalse((self.output / "ready.json").exists())
        for column in ("right_idx", "right_raw_idx"):
            self.pairs.loc[[0, 1], column] = np.array([0, 2], dtype=np.int32)
        self.write_inputs()
        columns = list(reversed(feature_columns(self.pairs.columns)))
        joblib.dump({"model": FixtureTree(columns), "feature_columns": columns}, self.tree)
        self.pipeline["tree"]["sha256"] = file_digest(self.tree)
        self.pipeline["decision"] = {"kind": "threshold_assignment", "threshold": .7,
                                     "assignment_delta": 0}
        self.config.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError, "feature order"):
            self.prepared()

    def test_partial_or_unaligned_ce_score_vector_fails_before_validation(self):
        def truncated(left, right, pairs, selected, amp):
            return np.zeros(max(0, len(pairs)-1), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "CE scorer lost selected pair order"):
            run(self.config, self.candidates, self.test_dir, self.output,
                validator=OFFICIAL, fixture=True, score_fn=truncated)
        self.assertFalse((self.output / "ready.json").exists())

    def test_score_binding_gate_and_union_requires_new_test_candidates(self):
        cal = json.loads(self.calibration.read_text())
        cal["score_identity"]["tree_sha256"] = "0" * 64
        self.calibration.write_text(json.dumps(cal))
        self.pipeline["decision"]["calibration"]["sha256"] = file_digest(self.calibration)
        self.config.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError, "score scale"):
            self.prepared()
        cal["score_identity"]["tree_sha256"] = self.pipeline["tree"]["sha256"]
        self.calibration.write_text(json.dumps(cal))
        self.pipeline["decision"]["calibration"]["sha256"] = file_digest(self.calibration)
        self.pipeline["retrieval"]["kind"] = "test_union"
        self.pipeline["tree"]["feature_policy"] = "dense_key_shared"
        self.config.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError, "score scale|NEW"):
            self.prepared()

    def test_saved_p1_calibrator_uses_review_choice_and_original_matcher(self):
        saved = self.root / "p1_experiment_config.json"
        review_path = self.root / "p1_decision_review.json"
        model = {"x": [0., 1.], "y": [.001, .999], "p_ungated": .001}
        artifact = {"tune_id_sha256": TUNE_ID_SHA256, "full_competing_target_pool": 10_320_219,
                    "evaluation_or_old_holdout_used": False, "france_labels_used": False,
                    "tree_gate": .01, "blend_ce_weight": .75,
                    "provenance": {"tree_checkpoint_sha256": self.pipeline["tree"]["sha256"],
                                   "reranker_checkpoint_sha256": self.pipeline["cross_encoder"]["sha256"]},
                    "outer_cross_fit": {"final_calibrator_fitted_on_all_tune": {"__pooled__": model}},
                    "count_model": {"training_only_prior": {
                        "train_references": 100, "train_true_links": 120, "p_true_count": [.01, .49, .5],
                        "gate_true_link_discovery": .99, "pair_prevalence": .2}}}
        saved.write_text(json.dumps(artifact))
        review = {"artifact_sha256": {"experiment_config.json": file_digest(saved)},
                  "recommendation_for_new_holdout": {
                      "final_calibrator": "experiment_config.json/outer_cross_fit/final_calibrator_fitted_on_all_tune",
                      "source_balance_weight": 0., "assignment_margin": 0}}
        review_path.write_text(json.dumps(review))
        self.pipeline["decision"]["calibration"] = {"kind": "p1_saved", "path": str(saved),
                "sha256": file_digest(saved), "review": str(review_path),
                "review_sha256": file_digest(review_path)}
        self.config.write_text(json.dumps(self.pipeline))
        self.assertEqual(self.prepared().decision["kind"], "p1_count_assignment")
        artifact["provenance"]["tree_checkpoint_sha256"] = "0" * 64
        saved.write_text(json.dumps(artifact))
        review["artifact_sha256"]["experiment_config.json"] = file_digest(saved)
        review_path.write_text(json.dumps(review))
        self.pipeline["decision"]["calibration"].update(
            sha256=file_digest(saved), review_sha256=file_digest(review_path))
        self.config.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError, "P1 TUNE calibrator"):
            self.prepared()

    def test_new_union_fixture_recomputes_shared_features_and_validates(self):
        self.pairs["dense_hit"] = np.array([True, True, True, False])
        self.pairs["phonetic_key_hit"] = np.zeros(len(self.pairs), dtype=bool)
        # Legacy lexical scores are not P3 union features; they are zeroed by the adapter.
        self.pairs["name_score"] = np.float32(.77)
        self.pairs["lexical_retrieval_score"] = np.float32(.88)
        self.write_inputs()
        columns = union_feature_columns(self.pairs.columns)
        joblib.dump({"model": FixtureTree(columns), "feature_columns": columns}, self.tree)
        self.pipeline["tree"].update(sha256=file_digest(self.tree), feature_policy="dense_key_shared")
        self.pipeline["retrieval"]["kind"] = "test_union"
        self.pipeline["decision"] = {"kind": "threshold_assignment", "threshold": .75,
                                     "assignment_delta": 0}
        meta_path = self.candidates / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta.update(mode="v2_test_union", retrieval_kind="test_union", feature_policy="dense_key_shared",
                    references_scanned=4, no_test_prefix=True, candidate_groups_complete=True,
                    reference_input_sha256=file_digest(self.test_dir / "test_source1.tsv"))
        meta_path.write_text(json.dumps(meta))
        complete_path = self.candidates / "complete.json"
        complete = json.loads(complete_path.read_text())
        complete["metadata"] = meta
        complete["config"]["encoder_sha256"] = self.pipeline["retrieval"]["encoder_sha256"]
        complete_path.write_text(json.dumps(complete))
        self.config.write_text(json.dumps(self.pipeline))
        result = self.predict()
        self.assertTrue(result["fixture_validation_passed"])
        self.assertEqual(result["retrieval_kind"], "test_union")
        self.assertFalse((self.output / "ready.json").exists())

    def test_validator_fails_when_prediction_is_not_a_candidate(self):
        self.predict()
        path = self.output / "matching_results.tsv"
        rows = path.read_text().splitlines(keepends=True)
        rows[1] = "S1-a\tS2-z\n"
        path.write_text("".join(rows))
        result = validate_release(path, self.output / "candidate_pairs.tsv", self.test_dir,
                                  self.root / "changed_validation.json", validator=OFFICIAL,
                                  expected_rows=4)
        self.assertFalse(result["passed"])
        self.assertTrue(any("subset" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
