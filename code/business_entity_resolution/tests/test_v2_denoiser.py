"""Split sealing, fixed full-pool accounting and conservative denoising gates."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ber import v2_denoiser as den


def fixture(root: Path):
    medium, sealed = root / "medium", root / "sealed"
    medium.mkdir()
    sealed.mkdir()
    reference = pd.DataFrame([
        ("S1-A", "Rose Cafe", "18 Main Road", "US", "train"),
        ("S1-B", "No Match", "2 Elm Road", "US", "train"),
        ("S1-C", "लक्ष्मी स्टोर", "5 Main Road", "India", "train"),
        ("S1-V", "Rose Market", "20 Oak Road", "US", "tune"),
    ], columns=[*den.RECORD_FIELDS, "split"])
    pool = pd.DataFrame([
        ("S2-A", "Rose Cafe", "18 Main Rd", "US"),
        ("S3-B", "Rosé Café", "18 Main Road", "US"),
        ("S2-C", "लक्ष्मी स्टोर", "5 Main Rd", "India"),
        ("S2-X", "Rose Cafe", "99 Main Road", "US"),
    ], columns=list(den.RECORD_FIELDS))
    reference.to_parquet(medium / "reference.parquet", index=False)
    pool.to_parquet(medium / "pool.parquet", index=False)
    # This file is hash-checked against the reservation; no eval labels are read.
    (root / "medium_splits.parquet").write_bytes(b"sealed reference identity")
    meta = {"mining_full_pool_records": 4, "positive_candidate_injection": False,
            "reference_id_sha256": den.ids_sha(reference.entity_id)}
    (medium / "metadata.json").write_text(json.dumps(meta))
    (medium / "complete.json").write_text(json.dumps({
        "state": "complete", "config": {"labels_used_for_mining": False}, "metadata": meta}))
    excluded = sorted([*reference.entity_id, "S1-H", "S1-OLDER"])
    (sealed / "train_tune_excluded_ids.txt").write_text("\n".join(excluded) + "\n")
    (sealed / "reserved_ids.txt").write_text("S1-H\n")
    reservation = {
        "verification": {"labels_read": False, "older_and_new_holdouts_excluded_from_future_train_tune": True},
        "prior_exclusions": {"pairwise_disjoint": True},
        "inputs": {"medium_all_splits": {"file_sha256": den.sha256(root / "medium_splits.parquet")}},
        "future_raw_train_tune": {"excluded_id_file": "train_tune_excluded_ids.txt",
                                  "excluded_file_sha256": den.sha256(sealed / "train_tune_excluded_ids.txt"),
                                  "excluded_count": len(excluded),
                                  "excluded_sorted_id_sha256": den.ids_sha(excluded)},
        "holdout2": {"id_file": "reserved_ids.txt", "id_file_sha256": den.sha256(sealed / "reserved_ids.txt"),
                     "count": 1, "sorted_id_sha256": den.ids_sha(["S1-H"])},
    }
    (sealed / "reservation.json").write_text(json.dumps(reservation))
    truth = root / "train_ground_truth.tsv"
    with truth.open("wb") as stream:
        stream.write(b"source1_entity_id\tmatched_entity_ids\n")
        stream.write(b"S1-A\tS2-A,S3-B\nS1-B\t\nS1-C\tS2-C\n")
        # Invalid UTF-8 proves no non-TRAIN label is even decoded.
        stream.write(b"S1-V\t\xff\nS1-H\t\xff\nS1-OLDER\t\xff\n")
    return medium, sealed, truth


def full_fixture(root: Path, sealed: Path):
    bootstrap = root / "p2_bootstrap"
    (bootstrap / "full").mkdir(parents=True)
    reference = pd.DataFrame([
        ("S1-NEW", "Rose Cafe", "18 Main Road", "US"),
        ("S1-EMPTY", "No Match", "5 Main Road", "India"),
    ], columns=list(den.RECORD_FIELDS))
    pool = pd.DataFrame([
        ("S2-NEW", "Rose Cafe", "18 Main Rd", "US"),
        ("S2-X", "No Match", "7 Main Rd", "India"),
        ("S3-NEW", "Rosé Cafe", "18 Main Road", "US"),
        ("S3-X", "No Match", "8 Main Rd", "India"),
    ], columns=list(den.RECORD_FIELDS))
    reference.to_parquet(bootstrap / "full/reference.parquet", index=False)
    pool.to_parquet(bootstrap / "pool.parquet", index=False)
    files = {}
    for name, path in (("reference", bootstrap / "full/reference.parquet"),
                       ("pool", bootstrap / "pool.parquet")):
        files[name] = {"path": str(path.relative_to(bootstrap)), "sha256": den.sha256(path),
                       "bytes": path.stat().st_size, "rows": len(reference if name == "reference" else pool)}
    manifest = {"state": "complete", "truth_read": False, "positive_candidate_injection": False,
                "reservation_sha256": den.sha256(sealed / "reservation.json"),
                "train_reference_id_sha256": den.ids_sha(reference.entity_id), "files": files}
    (bootstrap / "bootstrap.json").write_text(json.dumps(manifest))
    (bootstrap / "pool_complete.json").write_text(json.dumps({
        "state": "complete", "truth_read": False, "source2_rows": 2,
        "files": {"pool": {"sha256": files["pool"]["sha256"]}}}))
    ground = root / "full_truth.tsv"
    with ground.open("wb") as stream:
        stream.write(b"source1_entity_id\tmatched_entity_ids\n")
        stream.write(b"S1-NEW\tS2-NEW,S3-NEW\nS1-EMPTY\t\n")
        stream.write(b"S1-H\t\xff\nS1-OLDER\t\xff\nS1-V\t\xff\n")
    return bootstrap, ground


class DenoiserScreenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.medium, self.sealed, self.truth = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_reads_only_train_labels_and_retains_uncandidate_truth_records(self):
        output, durable = self.root / "screen", self.root / "durable"
        result = den.prepare(self.medium, self.root / "medium_splits.parquet", self.sealed,
                             self.truth, output, durable, strict=False)
        self.assertEqual(result["train_businesses"], 3)
        self.assertEqual(result["training_pairs"], 3)
        self.assertEqual(result["identity"]["raw_truth"]["non_train_labels_decoded"], 0)
        self.assertFalse(result["identity"]["positive_candidate_injection"])
        self.assertEqual(len(pd.read_parquet(output / "train_pairs.parquet")), 3)
        self.assertEqual(den._prepared(output, durable), result)
        with self.assertRaises(FileExistsError):
            den.prepare(self.medium, self.root / "medium_splits.parquet", self.sealed,
                        self.truth, output, durable, strict=False)

    def test_rejects_modified_seal_or_overlapping_holdout_or_reference(self):
        reservation = json.loads((self.sealed / "reservation.json").read_text())
        (self.sealed / "reserved_ids.txt").write_text("S1-A\n")
        with self.assertRaisesRegex(ValueError, "Sealed ID"):
            den.checked_medium(self.medium, self.root / "medium_splits.parquet", self.sealed, strict=False)
        reservation["holdout2"]["id_file_sha256"] = den.sha256(self.sealed / "reserved_ids.txt")
        reservation["holdout2"]["sorted_id_sha256"] = den.ids_sha(["S1-A"])
        (self.sealed / "reservation.json").write_text(json.dumps(reservation))
        with self.assertRaisesRegex(ValueError, "overlaps"):
            den.checked_medium(self.medium, self.root / "medium_splits.parquet", self.sealed, strict=False)

    def test_guard_format_digits_and_truncation(self):
        raw = "Rose Cafe | 18 Main Rd"
        self.assertEqual(den.safe_canonical_text(raw, "Rosé Café | 18 Main Road"),
                         ("Rosé Café | 18 Main Road", "accepted"))
        self.assertEqual(den.safe_canonical_text(raw, "Rose Cafe | 19 Main Road"),
                         (raw, "address_digits_changed"))
        self.assertEqual(den.safe_canonical_text(raw, "Rose Cafe 18 Main Road"),
                         (raw, "invalid_delimiter"))
        self.assertEqual(den.safe_canonical_text(raw, "Rose Cafe | 18 Main Rd", input_truncated=True),
                         (raw, "input_truncated"))

    def test_full_train_join_streams_only_eligible_labels_and_keeps_pair_order(self):
        bootstrap, ground = full_fixture(self.root, self.sealed)
        output, durable = self.root / "full_pairs", self.root / "full_durable"
        result = den.prepare_full_train(bootstrap, self.sealed, ground, output, durable,
                                        strict=False, batch_links=1)
        self.assertEqual(result["train_businesses"], 2)
        self.assertEqual(result["true_links"], 2)
        self.assertEqual(result["true_singletons"], 1)
        self.assertEqual(result["raw_truth"]["nontrain_payloads_decoded"], 0)
        self.assertFalse(result["training_has_started_on_full_pairs"])
        self.assertFalse(result["retrieval_edges_read_or_modified"])
        self.assertEqual(len(pd.read_parquet(output / "train_truth_pairs.parquet")), 2)
        self.assertEqual(den.sha256(output / "train_truth_pairs.parquet"),
                         den.sha256(durable / "train_truth_pairs.parquet"))
        with self.assertRaises(FileExistsError):
            den.prepare_full_train(bootstrap, self.sealed, ground, output, durable, strict=False)

    def test_full_train_join_rejects_reserved_s1_before_parsing_any_label(self):
        bootstrap, ground = full_fixture(self.root, self.sealed)
        reference_file = bootstrap / "full/reference.parquet"
        reference = pd.read_parquet(reference_file)
        reference.loc[1, "entity_id"] = "S1-H"
        reference.to_parquet(reference_file, index=False)
        manifest = json.loads((bootstrap / "bootstrap.json").read_text())
        manifest["files"]["reference"].update(sha256=den.sha256(reference_file),
                                                bytes=reference_file.stat().st_size)
        manifest["train_reference_id_sha256"] = den.ids_sha(reference.entity_id)
        (bootstrap / "bootstrap.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "reserved"):
            den.prepare_full_train(bootstrap, self.sealed, ground, self.root / "bad_full",
                                    self.root / "bad_durable", strict=False)

    def test_retrieval_oracle_includes_singleton_and_miss_and_probe_never_injects(self):
        reference = pd.DataFrame([
            ("S1-A", "Rose Cafe", "18 Main Road", "US"),
            ("S1-B", "Other", "2 Elm Road", "US"),
        ], columns=list(den.RECORD_FIELDS))
        pool = pd.DataFrame([
            ("S3-MISS", "Rose Cafe", "18 Main Rd", "US"),
            ("S2-WRONG", "Rose Cafe", "19 Main Road", "US"),
        ], columns=list(den.RECORD_FIELDS))
        pairs = pd.DataFrame({"left_idx": [0, 1], "right_idx": [1, 1],
                              "target_entity_id": ["S2-WRONG", "S2-WRONG"],
                              "dense_hit": [True, True], "original_dense_score": [.9, .6]})
        truth = {"S1-A": {"S3-MISS"}, "S1-B": set()}
        expected = {"pairs": 2, "true_links": 1, "retrieved_true_links": 0,
                    "missed_true_links": 1, "true_singletons": 1, "oracle_macro_f05": .5}
        result, labels = den.retrieval_ceiling(reference, pairs, truth, {"retrieval": {
            "dense_k50_saved": expected, "full_key_union": {**expected, "target_name_script": []}}})
        np.testing.assert_array_equal(labels, [False, False])
        self.assertEqual(result["full_key_union"]["oracle_macro_f05"], .5)
        sample = den._probe_sample(reference, pool, pairs, truth, seed=1, n_refs=1)
        self.assertEqual(len(sample), 2)
        self.assertFalse(sample.loc[sample.true_link, "candidate_edge"].any())
        self.assertTrue(sample.loc[~sample.true_link, "candidate_edge"].all())
        sample.to_parquet(self.root / "frozen_sample.parquet", index=False)
        self.assertTrue(pd.read_parquet(self.root / "frozen_sample.parquet").equals(sample))

    def test_frozen_v1_probabilities_align_only_real_dense_union_edges(self):
        v1_ref = pd.DataFrame([
            ("S1-TRAIN", "Training Shop", "1 Main St", "US", "train"),
            ("S1-V0", "Rose Cafe", "18 Main Road", "US", "tune"),
            ("S1-V1", "Hill Cafe", "20 Oak Road", "US", "tune"),
        ], columns=[*den.RECORD_FIELDS, "split"])
        v1_pool = pd.DataFrame([
            ("S2-A", "Training Shop", "1 Main Street", "US"),
            ("S2-B", "Rose Cafe", "18 Main Rd", "US"),
            ("S2-C", "Rose Café", "18 Main Road", "US"),
            ("S2-D", "Hill Cafe", "20 Oak Rd", "US"),
            ("S2-E", "Hill Café", "20 Oak Road", "US"),
        ], columns=list(den.RECORD_FIELDS))
        pairs = pd.DataFrame({"left_idx": [0, 1, 1, 2, 2], "right_idx": [0, 1, 2, 3, 4]})
        probs = np.array([0, .8, .2, .77, .95], dtype=np.float32)
        union_ref = v1_ref.iloc[[2, 1]].reset_index(drop=True)
        union_pool = v1_pool.iloc[[4, 2, 3, 1, 0]].reset_index(drop=True)
        union = pd.DataFrame({"left_idx": [0, 0, 0, 1, 1], "right_idx": [0, 2, 4, 1, 3],
                              "dense_hit": [True, True, False, True, True]})
        aligned, original_rows, mapped = den.align_frozen_dense(
            v1_ref, v1_pool, pairs, probs, union_ref, union_pool, union, dense_k=2)
        np.testing.assert_array_equal(original_rows, [1, 2, 3, 4])
        np.testing.assert_array_equal(mapped, [4, 3, 1, 0])
        np.testing.assert_allclose(aligned[[0, 1, 3, 4]], [.95, .77, .2, .8])
        self.assertTrue(np.isnan(aligned[2]))  # Union-only edge has no v1 score.
        changed = union_pool.copy()
        changed.loc[0, "business_name"] = "Different business"
        with self.assertRaisesRegex(ValueError, "original-script business_name"):
            den.align_frozen_dense(v1_ref, v1_pool, pairs, probs, union_ref, changed, union, dense_k=2)
        with self.assertRaisesRegex(ValueError, "exactly the original dense"):
            den.align_frozen_dense(v1_ref, v1_pool, pairs, probs, union_ref, union_pool,
                                   union.assign(dense_hit=[True, False, False, True, True]), dense_k=2)

    def test_feature_rule_scores_all_s1_singletons_misses_and_false_claims(self):
        references = pd.DataFrame({"entity_id": ["S1-A", "S1-B", "S1-C"],
                                   "country": ["US", "US", "India"]})
        left = np.array([0, 0, 1, 2], dtype=np.int32)
        labels = np.array([True, False, False, False])
        actual = np.array([2, 0, 1])  # S1-C's true target was never retrieved.
        baseline, per_s1 = den.score_candidate_decisions(references, left, labels, actual,
                                                          [True, False, False, False])
        np.testing.assert_allclose(per_s1, [5 / 6, 1., 0.])
        self.assertAlmostEqual(baseline["macro_f05"], (5 / 6 + 1) / 3)
        self.assertEqual(baseline["false_links"], 0)
        self.assertEqual(baseline["missed_links"], 2)
        self.assertEqual(baseline["singleton_accuracy"], 1.)
        promoted, per_s1 = den.score_candidate_decisions(references, left, labels, actual,
                                                          [True, True, True, False])
        np.testing.assert_allclose(per_s1, [.5, 0., 0.])
        self.assertEqual(promoted["false_links"], 2)
        self.assertEqual(promoted["singleton_accuracy"], 0.)
        self.assertEqual(promoted["by_country"]["India"]["macro_f05"], 0.)
        self.assertEqual(den.wilson_lower(0, 0), 0.)
        self.assertGreater(den.wilson_lower(100, 100), .95)

    def test_paired_name_accuracy_and_country_script_breakdowns(self):
        ref = pd.DataFrame([
            ("S1-A", "Rose Cafe", "18 Main Road", "US"),
            ("S1-C", "लक्ष्मी स्टोर", "5 Main Road", "India"),
        ], columns=list(den.RECORD_FIELDS))
        pool = pd.DataFrame([
            ("S2-A", "Rse Café", "18 Main Road", "US"),
            ("S3-C", "लक्ष्मी स्टोर", "5 Main Road", "India"),
            ("S2-F", "Rose Cafe", "19 Main Road", "US"),
        ], columns=list(den.RECORD_FIELDS))
        diagnostic = pd.DataFrame({
            "left_idx": [0, 1, 0], "right_idx": [0, 1, 2],
            "source1_entity_id": ["S1-A", "S1-C", "S1-A"],
            "target_entity_id": ["S2-A", "S3-C", "S2-F"],
            "country": ["US", "India", "US"],
            "target_name_script": ["Latin_or_other", "Devanagari", "Latin_or_other"],
            "true_link": [True, True, False], "candidate_edge": [False, True, True],
            "canonical": ["Rose Cafe | 18 Main Road", "लक्ष्मी स्टोर | 5 Main Road",
                          "Rose Cafe | 19 Main Road"], "fallback_reason": ["accepted"] * 3,
        })
        probe = den._component_metrics(diagnostic, ref, pool)
        report = den._canonical_report(probe, ref, seed=5)
        self.assertEqual(report["sampled_retrieval_misses"], 1)
        self.assertEqual(report["sampled_true_links_including_retrieval_misses"]["full"]["raw_hits"], 1)
        self.assertEqual(report["sampled_true_links_including_retrieval_misses"]["full"]["denoised_hits"], 2)
        self.assertEqual(report["sampled_retrieved_hard_nonlinks"]["full"]["denoised_hits"], 0)
        self.assertEqual(report["by_target_name_script"]["Devanagari"]["negative"]["pairs"], 0)
        self.assertEqual(report["paired_positive_full"]["sampling_unit"], "source1_entity")
        previous = probe.copy()
        previous.loc[previous.true_link, "new_full"] = False
        comparison = den.paired_extension(probe, previous, seed=5)
        self.assertEqual(comparison["positive_exact"]["full"]["new_hits"], 2)
        self.assertEqual(comparison["positive_exact"]["full"]["old_hits"], 0)
        self.assertTrue(comparison["new_checkpoint_wins_screen_gate"])
        previous.loc[0, "right_idx"] = 2
        with self.assertRaisesRegex(ValueError, "same labeled pair"):
            den.paired_extension(probe, previous, seed=5)


if __name__ == "__main__":
    unittest.main()
