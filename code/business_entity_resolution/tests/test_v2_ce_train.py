"""P3 data gates, exact score accounting, sampler/resume and bounded augmentation."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ber import v2_ce_train as ce
from ber.metrics import score_pairs


def make_fixture(root: Path, *, long_text: bool = False) -> argparse.Namespace:
    """A synthetic P2 full-pool join with a missed true link and a TUNE singleton."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train, tune, sealed = (root / name for name in ("p2", "tune", "sealed"))
    for path in (train, tune, sealed):
        path.mkdir()
    ref = pd.DataFrame([
        ("S1-A", "Market Rose", "10 Elm Road", "US", "train"),
        ("S1-B", "Market Lily", "20 Elm Road", "US", "train"),
        ("S1-C", "लक्ष्मी स्टोर", "5 Main Road", "India", "train"),
        ("S1-D", "Market Palm", "30 Elm Road", "US", "train"),
    ], columns=["entity_id", "business_name", "business_address", "country", "split"])
    pool = pd.DataFrame([
        ("S2-A", "Market Rose", "10 Elm Rd", "US"),
        ("S3-B", "Market Rosé", "10 Elm Road", "US"),
        ("S2-C", "Market Lily", "20 Elm Rd", "US"),
        ("S2-D", "Market Rose", "30 Elm Rd", "US"),
        ("S3-E", "लक्ष्मी स्टोर", "5 Main Road", "India"),
        ("S2-F", "लक्ष्मी स्टोर", "6 Main Road", "India"),
    ], columns=["entity_id", "business_name", "business_address", "country"])
    if long_text:
        # Exercise the actual 256-token truncation/pad8 CUDA training path.
        suffix = " " + " ".join(["Broadway"] * 100)
        ref["business_address"] += suffix
        pool["business_address"] += suffix
    positives = pd.DataFrame({"left_idx": [0, 0, 1, 2], "right_idx": [0, 1, 2, 4]})
    candidate = pd.DataFrame({
        "left_idx": [0, 0, 0, 1, 1, 2, 3], "right_idx": [0, 1, 3, 2, 3, 5, 3],
        "label": [1, 1, 0, 1, 0, 0, 0],
        "tree_probability": [.9, .8, .96, .8, .02, .6, .7],
        "dense_rank": [1, 2, 1, 1, 0, 3, 0],
        "same_name": [True, True, True, True, False, True, True],
        "same_digits": [True, True, False, True, False, False, True],
    })
    train_files = {"reference": ref, "pool": pool, "candidate_pairs": candidate,
                   "train_truth_pairs": positives}
    for name, table in train_files.items():
        table.to_parquet(train / f"{name}.parquet", index=False)
    ready = {"state": "complete", "split": "train", "dense_mining_complete": True,
             "train_truth_join_complete": True, "positive_candidate_injection": False,
             "full_pool_records": 6, "train_reference_id_sha256": ce.id_sha256(ref.entity_id.tolist()),
             "files": {name: {"path": f"{name}.parquet", "sha256": ce.file_sha256(train / f"{name}.parquet"),
                               "rows": len(table)} for name, table in train_files.items()}}
    (train / "ready.json").write_text(json.dumps(ready))

    tref = pd.DataFrame([
        ("S1-V1", "Market Rose", "10 Elm Road", "US", "tune"),
        ("S1-V2", "Market Lily", "70 Elm Road", "US", "tune"),
        ("S1-V3", "लक्ष्मी स्टोर", "5 Main Road", "India", "tune"),
    ], columns=ref.columns)
    tpool = pd.DataFrame([
        ("S2-P", "Market Rose", "10 Elm Rd", "US"),
        ("S2-Q", "Market Rose", "90 Elm Rd", "US"),
        ("S3-R", "लक्ष्मी स्टोर", "6 Main Rd", "India"),
        ("S3-M", "लक्ष्मी स्टोर", "5 Main Rd", "India"),
    ], columns=pool.columns)
    if long_text:
        tref["business_address"] += suffix
        tpool["business_address"] += suffix
    tpairs = pd.DataFrame({"left_idx": [0, 0, 1, 2], "right_idx": [0, 1, 1, 2]})
    tref.to_parquet(tune / "reference.parquet", index=False)
    tpool.to_parquet(tune / "pool.parquet", index=False)
    tpairs.to_parquet(tune / "pairs.parquet", index=False)
    (tune / "truth.json").write_text(json.dumps({"S1-V1": ["S2-P"], "S1-V2": [], "S1-V3": ["S3-M"]}))
    metadata = {"reference_id_sha256": ce.id_sha256(tref.entity_id.tolist()),
                "split_counts": {"tune/India": 1, "tune/US": 2},
                "full_pool_records": 6, "candidate_pairs": len(tpairs),
                "positive_candidate_injection": False}
    (tune / "metadata.json").write_text(json.dumps(metadata))
    (tune / "complete.json").write_text(json.dumps({"state": "complete", "config": {
        "labels_used_for_mining": False}, "metadata": metadata}))
    np.save(tune / "tree_scores.npy", np.array([.92, .4, .4, .6], dtype=np.float32))
    tpairs.to_parquet(tune / "tree_pairs.parquet", index=False)
    (tune / "tree_checkpoint.joblib").write_bytes(b"synthetic selected tree - smoke only")

    excluded, held = sorted(["S1-H2", *tref.entity_id.tolist()]), ["S1-H2"]
    (sealed / "train_tune_excluded_ids.txt").write_text("\n".join(excluded) + "\n")
    (sealed / "reserved_ids.txt").write_text("S1-H2\n")
    reservation = {"verification": {"labels_read": False},
                   "future_raw_train_tune": {"excluded_id_file": "train_tune_excluded_ids.txt",
                                             "excluded_count": len(excluded),
                                             "excluded_file_sha256": ce.file_sha256(sealed / "train_tune_excluded_ids.txt"),
                                             "excluded_sorted_id_sha256": ce.id_sha256(excluded)},
                   "holdout2": {"id_file": "reserved_ids.txt", "count": len(held),
                                "id_file_sha256": ce.file_sha256(sealed / "reserved_ids.txt"),
                                "sorted_id_sha256": ce.id_sha256(held)}}
    (sealed / "reservation.json").write_text(json.dumps(reservation))
    return argparse.Namespace(train_ready=train / "ready.json", sealed_reservation=sealed,
                              tune_data=tune, tune_pairs=None, tune_base_dir=None, tune_scores=tune / "tree_scores.npy",
                              tune_score_pairs=tune / "tree_pairs.parquet",
                              tune_tree_checkpoint=tune / "tree_checkpoint.joblib",
                              smoke=True, smoke_full_pool_rows=6, output=root / "screen",
                              tree_gate=.01, dense_top_k=50)


def refresh_ready_file(args: argparse.Namespace, name: str) -> None:
    ready = json.loads(args.train_ready.read_text())
    file = args.train_ready.parent / ready["files"][name]["path"]
    ready["files"][name]["sha256"] = ce.file_sha256(file)
    ready["files"][name]["rows"] = len(pd.read_parquet(file))
    args.train_ready.write_text(json.dumps(ready))


class V2CETrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = make_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_completed_join_keeps_unretrieved_positive_only_in_training_pool(self):
        result = ce.prepare(self.args)
        self.assertEqual(result["positive_pool"], 4)
        self.assertEqual(result["candidate_rows"], 7)
        self.assertEqual(result["missed_train_truth_links"], 1)
        self.assertEqual(result["hard_negative_pool"], 4)
        self.assertEqual(result, ce.prepare(self.args))  # verified, reusable preparation
        positives = np.load(self.args.output / "positive_pairs.npy")
        negatives = np.load(self.args.output / "negative_pairs.npy")
        self.assertIn([2, 4], positives.tolist())
        self.assertNotIn((2, 4), [(int(row["left"]), int(row["right"])) for row in negatives])
        channels = ce.hard_negative_indices(negatives)
        self.assertTrue(all(len(item) for item in channels.values()))
        sampled = ce.sample_hard_negatives(channels, np.random.default_rng(7), 128)
        self.assertEqual(len(sampled), 128)
        self.assertEqual(result["identity"]["scope"], "compatibility_smoke")

    def test_no_start_without_completed_dense_and_train_truth_join(self):
        for key in ("state", "dense_mining_complete", "train_truth_join_complete",
                    "positive_candidate_injection", "full_pool_records"):
            with self.subTest(key=key):
                document = json.loads(self.args.train_ready.read_text())
                expected = document[key]
                document[key] = "pending" if key == "state" else not expected
                self.args.train_ready.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    ce.prepare(self.args)
                document[key] = expected
                self.args.train_ready.write_text(json.dumps(document))

    def test_join_disagrees_with_complete_truth_and_rejects_duplicate_edges(self):
        path = self.args.train_ready.parent / "candidate_pairs.parquet"
        table = pd.read_parquet(path)
        table.loc[0, "label"] = 0
        table.to_parquet(path, index=False)
        refresh_ready_file(self.args, "candidate_pairs")
        with self.assertRaisesRegex(ValueError, "disagree"):
            ce.prepare(self.args)
        table.loc[0, "label"] = 1
        pd.concat([table, table.iloc[:1]], ignore_index=True).to_parquet(path, index=False)
        refresh_ready_file(self.args, "candidate_pairs")
        # A previous failed preparation intentionally leaves an uncommitted output.
        self.args.output = self.root / "duplicate-screen"
        with self.assertRaisesRegex(ValueError, "Duplicate retrieved candidate"):
            ce.prepare(self.args)

    def test_sealed_holdout_and_tune_disjoint_gates(self):
        path = self.args.train_ready.parent / "reference.parquet"
        table = pd.read_parquet(path)
        table.loc[0, "entity_id"] = "S1-H2"
        table.to_parquet(path, index=False)
        ready = json.loads(self.args.train_ready.read_text())
        ready["train_reference_id_sha256"] = ce.id_sha256(table.entity_id.tolist())
        self.args.train_ready.write_text(json.dumps(ready))
        refresh_ready_file(self.args, "reference")
        with self.assertRaisesRegex(ValueError, "sealed"):
            ce.prepare(self.args)

    def test_positive_cycle_and_noise_preserve_measured_coverage_and_digits(self):
        cycle = ce.PositiveCycle(5, 20260925)
        first = cycle.take(3)
        restarted = ce.PositiveCycle(**cycle.state())
        np.testing.assert_array_equal(cycle.take(9), restarted.take(9))
        self.assertEqual(len(np.unique(first)), 3)
        self.assertLess(40_000 * 128, 7_638_365)  # plan's all-positive exposure conflict
        ref = pd.read_parquet(self.args.train_ready.parent / "reference.parquet")
        pool = pd.read_parquet(self.args.train_ready.parent / "pool.parquet")
        profile = ce.estimate_train_noise(ref, pool, np.array([[0, 0], [0, 1], [1, 2], [2, 4]]),
                                          seed=44, sample_size=4)
        self.assertEqual(profile["sample_count"], 4)
        self.assertEqual(sum(profile["name_op_counts"].values()), 4)
        profile["name_op_probabilities"] = {"script_or_accent": 1.}
        profile["accented_latin_subrate"] = 1.
        name, addr, op = ce.synthetic_positive("Cafe East", "12 West Road", profile, np.random.default_rng(4))
        self.assertEqual(op, "accented_latin")
        self.assertNotEqual(name, "Cafe East")
        self.assertEqual(ce._DIGITS.findall(addr), ["12"])
        other = ce.synthetic_different_digit("Cafe East", "12 West Road", np.random.default_rng(5))
        self.assertEqual(other[0], "Cafe East")
        self.assertNotEqual(ce._DIGITS.findall(other[1]), ["12"])
        self.assertIsNone(ce.synthetic_different_digit("Cafe", "West Road", np.random.default_rng(5)))

    def test_full_tune_scoring_counts_singletons_and_retrieval_misses(self):
        prepared = ce.prepare(self.args)
        forbidden, _ = ce.sealed_ids(self.args.sealed_reservation, smoke=True)
        data = ce.load_tune(self.args, forbidden, prepared)
        report, entities, probabilities = ce.tune_predictions(data, np.array([.9, .1, .1, .2], dtype=np.float32))
        self.assertEqual(report["candidate_pairs"], 4)
        self.assertEqual(report["retrieved_true_links"], 1)
        self.assertEqual(report["missed_retrieval_links"], 1)
        self.assertAlmostEqual(report["retrieval_oracle_macro_f05"], 2 / 3)
        self.assertEqual(report["retrieval_complete_set_recall"], .5)
        self.assertEqual(len(entities), 3)
        self.assertEqual(entities.set_index("source1_entity_id").loc["S1-V3", "macro_f05"], 0.)
        self.assertEqual(report["selected_by_tune"]["tune"]["singleton_accuracy"], 1.)
        official = score_pairs(data.reference, data.pool, data.pairs, probabilities,
                               data.truth, report["selected_by_tune"]["tune"]["threshold"])
        self.assertAlmostEqual(official["macro_f05"], entities.macro_f05.mean(), places=12)
        self.assertEqual(official["true_links"], 2)
        chosen = report["selected_by_tune"]["tune"]
        self.assertEqual(sum(group["references"] for group in chosen["by_country"].values()), 3)
        self.assertEqual(sum(group["references"] for group in chosen["by_script"].values()), 3)
        native = [row for row in report["by_target_name_script"] if row["target_name_script"] == "Devanagari"]
        self.assertEqual(len(native), 1)
        self.assertEqual((native[0]["true_links"], native[0]["retrieved_true_links"]), (1, 0))

    def test_weighted_augmentation_is_capped_and_counts_only_real_exposure(self):
        prepared = ce.prepare(self.args)
        positives = np.load(self.args.output / "positive_pairs.npy")
        negative = np.load(self.args.output / "negative_pairs.npy")
        reference = pd.read_parquet(self.args.train_ready.parent / "reference.parquet")
        pool = pd.read_parquet(self.args.train_ready.parent / "pool.parquet")
        profile = ce.estimate_train_noise(reference, pool, positives, seed=71, sample_size=4)
        args = argparse.Namespace(effective_batch=256, noise_positive_fraction=.1,
                                  noise_negative_fraction=.05, noise_weight=.2)
        cycle = ce.PositiveCycle(len(positives), 6)
        seen, counters = np.zeros(len(positives), dtype=bool), Counter()
        sample_rng, augmentation_rng = np.random.default_rng(71), np.random.default_rng(72)
        left, right, labels, weights = ce._batch_pairs(
            positives, negative, cycle, ce.hard_negative_indices(negative), sample_rng, augmentation_rng,
            seen, reference, ce.record_texts(reference), ce.record_texts(pool), profile, args, counters)
        self.assertEqual((len(left), len(right), len(labels), len(weights)), (256,) * 4)
        self.assertEqual(int(labels.sum()), 128)
        self.assertEqual(counters["real_positive"] + counters["synthetic_positive"], 128)
        self.assertEqual(int(seen.sum()), counters["distinct_real_positive_seen"])
        self.assertLessEqual(int(seen.sum()), prepared["positive_pool"])
        self.assertTrue(np.logical_or(np.isclose(weights, .2), np.isclose(weights, 1.)).all())
        self.assertGreater(counters["synthetic_positive"], 0)
        control = argparse.Namespace(effective_batch=256, noise_positive_fraction=0.,
                                     noise_negative_fraction=0., noise_weight=.2)
        baseline_cycle = ce.PositiveCycle(len(positives), 6)
        baseline_sample, baseline_noise = np.random.default_rng(71), np.random.default_rng(72)
        baseline_left, _, _, _ = ce._batch_pairs(
            positives, negative, baseline_cycle, ce.hard_negative_indices(negative),
            baseline_sample, baseline_noise, np.zeros(len(positives), dtype=bool),
            reference, ce.record_texts(reference), ce.record_texts(pool), None, control, Counter())
        self.assertEqual(left, baseline_left)  # augmentation never changes sampled S1/order
        self.assertEqual(sample_rng.bit_generator.state, baseline_sample.bit_generator.state)
        self.assertEqual(cycle.state(), baseline_cycle.state())

    def test_reject_misaligned_tree_probability_order(self):
        prepared = ce.prepare(self.args)
        path = self.args.tune_score_pairs
        reordered = pd.read_parquet(path).iloc[::-1]
        reordered.to_parquet(path, index=False)
        forbidden, _ = ce.sealed_ids(self.args.sealed_reservation, smoke=True)
        with self.assertRaisesRegex(ValueError, "exactly align"):
            ce.load_tune(self.args, forbidden, prepared)

    def test_fixed_medium_tune_id_digest_matches_saved_split(self):
        self.assertEqual(len(ce.TUNE_IDS_SHA256), 64)
        self.assertEqual(ce.TUNE_IDS_SHA256,
                         "5bb5759fa52ac19df8839520c7ccf147bae01a5499a2900af5d1b4ebeb1dbaa2")

    def _p2_tune_union(self) -> Path:
        """Reproduce P2's context-scored TUNE layout without opening another split."""
        legacy = self.args.tune_data
        union = self.root / "p2-tune-union"
        union.mkdir()
        for name in ("reference.parquet", "pool.parquet", "truth.json"):
            (union / name).write_bytes((legacy / name).read_bytes())
        source = legacy / "pairs.parquet"
        raw = union / "candidate_pairs_raw.parquet"
        mapped = union / "candidate_pairs.parquet"
        context = union / "candidate_pairs.context.parquet"
        for path in (raw, mapped, context):
            path.write_bytes(source.read_bytes())
        complete = {"state": "complete", "references": 3, "full_target_pool_rows": 6,
                    "reference_ids_sha256": ce.id_sha256(pd.read_parquet(union / "reference.parquet").entity_id),
                    "positive_candidate_injection": False, "truth_used_in_retrieval": False,
                    "raw_pairs": {"sha256": ce.file_sha256(raw), "rows": 4}}
        (union / "complete.json").write_text(json.dumps(complete))
        manifest = {"state": "complete", "positive_candidate_injection": False,
                    "inputs": {"pairs": {"sha256": ce.file_sha256(mapped)}},
                    "pair_file": {"sha256": ce.file_sha256(context)}}
        context.with_suffix(".context.json").write_text(json.dumps(manifest))
        self.args.tune_data, self.args.tune_pairs = union, context
        return context

    def test_p2_context_tune_and_separate_train_tree_scores_align_exactly(self):
        self._p2_tune_union()
        candidate_path = self.args.train_ready.parent / "candidate_pairs.parquet"
        candidate = pd.read_parquet(candidate_path)
        scores = candidate.tree_probability.to_numpy(dtype=np.float32)
        candidate["training_label"] = candidate.pop("label")
        candidate["hard_negative_priority"] = [ -1, -1, 4, -1, 4, 4, 0 ]
        candidate = candidate.drop(columns=["tree_probability", "same_name", "same_digits"])
        candidate.to_parquet(candidate_path, index=False)
        refresh_ready_file(self.args, "candidate_pairs")
        np.save(self.root / "train_scores.npy", scores)
        candidate[["left_idx", "right_idx"]].to_parquet(self.root / "train_score_pairs.parquet", index=False)
        (self.root / "selected_tree.joblib").write_bytes(b"synthetic checked tree")
        self.args.train_scores = self.root / "train_scores.npy"
        self.args.train_score_pairs = self.root / "train_score_pairs.parquet"
        self.args.train_tree_checkpoint = self.root / "selected_tree.joblib"
        prepared = ce.prepare(self.args)
        self.assertEqual(prepared["positive_pool"], 4)
        self.assertEqual(prepared["hard_negative_pool"], 4)
        self.assertEqual(prepared["identity"]["tune"]["dataset_kind"],
                         "completed_p2_tune_union_with_scored_context")
        forbidden, _ = ce.sealed_ids(self.args.sealed_reservation, smoke=True)
        tune = ce.load_tune(self.args, forbidden, prepared)
        self.assertEqual(len(tune.reference), 3)
        self.assertEqual(int(tune.labels.sum()), 1)

        # A SHA-valid but re-ordered score vector must not silently train on
        # another S1-target edge or reuse the earlier prepared pool.
        pairs = pd.read_parquet(self.args.train_score_pairs).iloc[::-1]
        pairs.to_parquet(self.args.train_score_pairs, index=False)
        self.args.output = self.root / "misaligned-train-screen"
        with self.assertRaisesRegex(ValueError, "probability order differs"):
            ce.prepare(self.args)

    def test_p2_context_manifest_missing_is_rejected_before_tune_truth(self):
        pair_path = self._p2_tune_union()
        pair_path.with_suffix(".context.json").unlink()
        with self.assertRaises(FileNotFoundError):
            ce.prepare(self.args)

    def test_p3_tree_full_checkpoint_binds_train_and_tune_pair_sources(self):
        directory = self.root / "p3-tree"
        directory.mkdir()
        checkpoint = directory / "tree_bundle.joblib"
        checkpoint.write_bytes(b"synthetic selected checkpoint")
        (directory / "tune_pairs.parquet").write_bytes((self.args.tune_data / "tree_pairs.parquet").read_bytes())
        (directory / "tune_probabilities.npy").write_bytes(self.args.tune_scores.read_bytes())
        (directory / "results.json").write_text(json.dumps({
            "tune_preparation": {"pairs": {"sha256": ce.file_sha256(directory / "tune_pairs.parquet")}},
            "evaluation": {"tune_pair_probabilities": {
                "sha256": ce.file_sha256(directory / "tune_probabilities.npy")}}}))
        config = {"mode": "full", "input_train": {"ready": {"sha256": ce.file_sha256(self.args.train_ready)}},
                  "input_tune": {"pairs": {"sha256": ce.file_sha256(self.args.tune_data / "pairs.parquet")}}}
        (directory / "experiment_config.json").write_text(json.dumps(config))
        (directory / "complete.json").write_text(json.dumps({"state": "complete",
            "tree_checkpoint_sha256": ce.file_sha256(checkpoint),
            "results_sha256": ce.file_sha256(directory / "results.json")}))
        identity = ce._full_tree_identity(checkpoint, train_ready=self.args.train_ready,
                                          tune_pair_sha=ce.file_sha256(self.args.tune_data / "pairs.parquet"),
                                          score_pairs=directory / "tune_pairs.parquet",
                                          scores=directory / "tune_probabilities.npy")
        self.assertEqual(identity["sha256"], ce.file_sha256(checkpoint))
        config["input_train"]["ready"]["sha256"] = "different-p2"
        (directory / "experiment_config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "another P2"):
            ce._full_tree_identity(checkpoint, train_ready=self.args.train_ready)

    def test_p3_completed_tree_scores_permuted_tune_edges_by_exact_keys(self):
        context = self._p2_tune_union()
        scores = np.load(self.args.tune_scores)
        pairs = pd.read_parquet(context)
        tree = self.root / "p3-permuted"
        tree.mkdir()
        perm = [1, 0, 2, 3]
        pair_path = tree / "tune_pairs.parquet"
        prob_path = tree / "tune_probabilities.npy"
        pairs.iloc[perm].to_parquet(pair_path, index=False)
        np.save(prob_path, scores[perm])
        checkpoint = tree / "tree_bundle.joblib"
        checkpoint.write_bytes(b"synthetic full P3.1 tree")
        (tree / "experiment_config.json").write_text(json.dumps({"mode": "full",
            "input_tune": {"pairs": {"sha256": ce.file_sha256(context)}}}))
        (tree / "results.json").write_text(json.dumps({
            "tune_preparation": {"pairs": {"sha256": ce.file_sha256(pair_path)}},
            "evaluation": {"tune_pair_probabilities": {"sha256": ce.file_sha256(prob_path)}}}))
        (tree / "complete.json").write_text(json.dumps({"state": "complete",
            "tree_checkpoint_sha256": ce.file_sha256(checkpoint),
            "results_sha256": ce.file_sha256(tree / "results.json")}))
        self.args.tune_scores, self.args.tune_score_pairs = prob_path, pair_path
        self.args.tune_tree_checkpoint = checkpoint
        prepared = ce.prepare(self.args)
        forbidden, _ = ce.sealed_ids(self.args.sealed_reservation, smoke=True)
        data = ce.load_tune(self.args, forbidden, prepared)
        np.testing.assert_array_equal(data.base, scores)
        self.assertEqual(data.identity["tree_provenance"]["pair_alignment"],
                         "verified_exact_key_permutation_from_completed_p3_tree")
        pairs.to_parquet(pair_path, index=False)
        with self.assertRaisesRegex(ValueError, "changed after scoring"):
            ce.load_tune(self.args, forbidden, prepared)

    def test_cpu_tree_scorer_restores_unsorted_p2_pair_order_across_row_groups(self):
        from ber.predict import feature_batch

        candidate = pd.read_parquet(self.args.train_ready.parent / "candidate_pairs.parquet")
        # P2 may sort by target ID rather than compact right_idx. The scorer
        # must use P3.1's sorted context but output probabilities in P2 order.
        candidate = pd.concat([candidate.iloc[[2, 0, 1]], candidate.iloc[3:]], ignore_index=True)
        path = self.root / "unsorted_within_s1.parquet"
        candidate.to_parquet(path, index=False, row_group_size=2)
        batches = list(ce._original_pair_batches(path, 4, 6, reference_batch=2,
                                                  read_batch=2, max_pairs=20))
        self.assertEqual([tuple(x) for _, _, frame in batches for x in frame[["left_idx", "right_idx"]].to_numpy()],
                         [tuple(x) for x in candidate[["left_idx", "right_idx"]].to_numpy()])

        class ToyTree:
            def predict_proba(self, features, *, thread_count):
                self.columns = list(features.columns)
                values = features.address_token_sort.to_numpy(dtype=np.float32)
                return np.column_stack((1 - values, values))

        ref = pd.read_parquet(self.args.train_ready.parent / "reference.parquet")
        pool = pd.read_parquet(self.args.train_ready.parent / "pool.parquet")
        model = ToyTree()
        values = np.concatenate([ce._score_tree_batch(model, ["address_token_sort"], ref, pool,
                                                     frame, start, stop, 2)
                                 for start, stop, frame in batches])
        expected = feature_batch(ref, pool, candidate, ["address_token_sort"]).address_token_sort.to_numpy()
        np.testing.assert_allclose(values, expected)
        self.assertEqual(model.columns, ["address_token_sort"])
        self.assertNotEqual(float(values[0]), float(values[1]))

    def test_cpu_tree_scorer_refuses_incomplete_p2_or_unselected_tree(self):
        args = argparse.Namespace(smoke=False, train_tree_checkpoint=self.root / "not-selected.joblib")
        with self.assertRaisesRegex(ValueError, "completed P3.1"):
            ce.materialize_train_tree_scores(args)

    def test_full_train_score_manifest_binds_all_p2_pairs_and_selected_tree(self):
        ready = json.loads(self.args.train_ready.read_text())
        candidate = self.args.train_ready.parent / "candidate_pairs.parquet"
        tree = self.root / "selected-full-tree"
        tree.mkdir()
        checkpoint = tree / "tree_bundle.joblib"
        checkpoint.write_bytes(b"toy selected tree")
        (tree / "results.json").write_text("{}")
        (tree / "experiment_config.json").write_text(json.dumps({"mode": "full",
            "input_train": {"ready": {"sha256": ce.file_sha256(self.args.train_ready)}}}))
        (tree / "complete.json").write_text(json.dumps({"state": "complete",
            "tree_checkpoint_sha256": ce.file_sha256(checkpoint),
            "results_sha256": ce.file_sha256(tree / "results.json")}))
        scores_root = self.root / "full-train-scores"
        scores_root.mkdir()
        scores = scores_root / "train_probabilities.npy"
        np.save(scores, pd.read_parquet(candidate).tree_probability.to_numpy(dtype=np.float32))
        order = ce._pair_order_sha256(candidate)
        marker = {"state": "complete", "pairs": ready["files"]["candidate_pairs"]["rows"],
                  "scores_sha256": ce.file_sha256(scores), "pair_order_sha256": order,
                  "score_pairs_path": str(candidate.resolve()), "identity": {
                      "p2_ready_sha256": ce.file_sha256(self.args.train_ready),
                      "tree_checkpoint": {"sha256": ce.file_sha256(checkpoint)}}}
        (scores_root / "complete.json").write_text(json.dumps(marker))
        args = argparse.Namespace(train_ready=self.args.train_ready, train_scores=scores,
                                  train_score_pairs=candidate, train_tree_checkpoint=checkpoint, smoke=False)
        named = {name: self.args.train_ready.parent / value["path"] for name, value in ready["files"].items()}
        values, source = ce._train_score_source(args, ready, named)
        self.assertEqual(values.shape, (7,))
        self.assertEqual(source["pair_order_sha256"], order)
        self.assertIsNotNone(source["materialization_complete_sha256"])
        marker["scores_sha256"] = "stale-tree-probabilities"
        (scores_root / "complete.json").write_text(json.dumps(marker))
        with self.assertRaisesRegex(ValueError, "completed, aligned"):
            ce._train_score_source(args, ready, named)

    def test_full_training_requires_explicit_gate(self):
        parser = self.args
        parser.schedule_steps, parser.eval_steps, parser.max_steps = 40_000, [10_000, 20_000, 40_000], 10_000
        parser.effective_batch, parser.microbatch = 256, 32
        parser.max_length, parser.learning_rate, parser.weight_decay = 256, 2e-5, .01
        parser.threads, parser.inference_batch = 2, 128
        parser.save_every, parser.warmup_fraction = 10_000, .06
        parser.plateau_min_delta, parser.near_tie = .0005, .00025
        parser.noise_positive_fraction, parser.noise_negative_fraction, parser.noise_weight = 0., 0., .2
        parser.variant, parser.control_output, parser.resume, parser.init_from = "baseline", None, False, None
        parser.prepare_only, parser.launch_full = False, False
        parser.device, parser.amp = "cuda", "bf16"
        parser.smoke, parser.smoke_full_pool_rows = False, 0
        with self.assertRaisesRegex(ValueError, "launch-full"):
            ce._validate_training_args(parser)
        parser.launch_full = True
        initial = self.root / "baseline-checkpoint"
        initial.mkdir()
        (initial / "complete.json").write_text(json.dumps({"state": "complete", "step": 10_000}))
        parser.init_from, parser.max_steps, parser.eval_steps = initial, 13_000, [13_000]
        with self.assertRaisesRegex(ValueError, "<=2k"):
            ce._validate_training_args(parser)
        parser.max_steps, parser.eval_steps = 12_000, [12_000]
        ce._validate_training_args(parser)  # two equal-budget branches may be scored once

    def test_paired_noise_ablation_keeps_every_tune_entity(self):
        for name, variant, scores in (("noise", "noise", [.8, 1., 0.]),
                                      ("baseline", "baseline", [.6, 1., 0.])):
            output = self.root / name
            step = output / "tune/step_000002"
            step.mkdir(parents=True)
            config = {key: "same" for key in ("prepared_identity", "tune_identity", "model_identity",
                      "seed", "schedule_steps", "effective_batch", "positive_fraction", "microbatch",
                      "init_checkpoint_marker_sha256", "max_length", "amp", "learning_rate", "weight_decay",
                      "warmup_fraction", "tree_gate", "dense_top_k", "near_tie",
                      "gradient_checkpointing", "eval_steps", "source_sha256", "sampling_rng_streams")}
            config["variant"] = variant
            (output / "training_config.json").write_text(json.dumps(config))
            (step / "report.json").write_text(json.dumps({"step": 2, "smoke": True,
                "selected_by_tune": {"tune": {"macro_f05": sum(scores) / 3}}}))
            pd.DataFrame({"source1_entity_id": ["a", "b", "c"], "country": ["US", "US", "India"],
                          "true_links": [1, 0, 1], "predicted_links": [1, 0, 0],
                          "false_positive_links": [0, 0, 0], "macro_f05": scores}).to_csv(
                              step / "entities.tsv", sep="\t", index=False)
        report = ce.compare_noise_control(self.root / "noise", self.root / "baseline", 2, replicates=100)
        self.assertAlmostEqual(report["paired_f05"]["estimates"]["left_minus_right"], .2 / 3)
        self.assertEqual(report["paired_f05"]["reference_count"], 3)
        self.assertTrue((self.root / "noise/noise_ablation_step_000002.json").exists())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch checkpoint smoke runs on the GPU pod")
    def test_optimizer_rng_sampler_checkpoint_round_trip(self):
        import torch

        class TinyModel:
            def __init__(self):
                self.linear = torch.nn.Linear(2, 1)

            def save_pretrained(self, directory, *, safe_serialization=True):
                torch.save(self.linear.state_dict(), directory / "tiny.pt")

        class TinyTokenizer:
            def save_pretrained(self, directory):
                (directory / "tokenizer.json").write_text("{}")

        torch.manual_seed(17)
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.linear.parameters(), lr=2e-5)
        optimizer.zero_grad()
        model.linear(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        cycle = ce.PositiveCycle(9, 17)
        cycle.take(4)
        rng = np.random.default_rng(18)
        rng.integers(100)
        augmentation_rng = np.random.default_rng(19)
        augmentation_rng.integers(100)
        seen = np.array([True] * 4 + [False] * 5)
        output = self.root / "tiny-output"
        output.mkdir()
        checkpoint = ce._save_checkpoint(model, TinyTokenizer(), optimizer, cycle, rng, augmentation_rng, seen,
                                         Counter(real_positive=4), {"schema": 1}, "same-config", 3,
                                         output / "checkpoints", output, 2.5, .3, torch)
        self.assertEqual(ce._checkpoint_marker(checkpoint)["step"], 3)
        expected_order = cycle.take(5)
        expected_rng = rng.integers(1_000)
        expected_augmentation = augmentation_rng.integers(1_000)
        expected_torch_rng = torch.rand(1)
        restored = torch.optim.AdamW(model.linear.parameters(), lr=2e-5)
        state, again, random, restored_aug, bits, counters = ce._restore_checkpoint(
            checkpoint, model, restored, "same-config", 9, 17, fork=False, torch=torch)
        self.assertEqual(state["completed_steps"], 3)
        np.testing.assert_array_equal(again.take(5), expected_order)
        self.assertEqual(random.integers(1_000), expected_rng)
        self.assertEqual(restored_aug.integers(1_000), expected_augmentation)
        torch.testing.assert_close(torch.rand(1), expected_torch_rng)
        np.testing.assert_array_equal(bits, seen)
        self.assertEqual(counters["real_positive"], 4)
        self.assertEqual(len(restored.state), len(optimizer.state))
        with self.assertRaisesRegex(ValueError, "does not belong"):
            ce._restore_checkpoint(checkpoint, model, restored, "different", 9, 17, fork=False, torch=torch)


if __name__ == "__main__":
    unittest.main()
