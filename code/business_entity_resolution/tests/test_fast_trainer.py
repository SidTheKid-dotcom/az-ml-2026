import argparse
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from ber.fast_trainer import (ArenaBatchSampler, LegacyBatchSampler, NegativeType, TokenizationPrefetcher,
                              _config_digest, adapt_g0b, classify_negative_types,
                              evaluate_scores, file_sha256, g0_matrix, learning_rate,
                              load_arena, load_score_data, mode_config, model_identity, stake_weights,
                              _validate_pair_order,
                              verified_checkpoint)
from ber.models import HeuristicModel
from ber.cross_encoder_experiment import _ShuffledCycle


class FastTrainerTests(unittest.TestCase):
    def test_stake_formula_class_normalization_and_validation(self):
        labels = np.array([1, 1, 0, 0, 0], dtype=bool)
        counts = np.array([1, 3, 1, 3, 0])
        raw = stake_weights(labels, counts, normalize=False)
        np.testing.assert_allclose(raw, [1, 1 / 11, 4 / 9, 4 / 19, 1])
        normalized = stake_weights(labels, counts)
        self.assertAlmostEqual(float(normalized[labels].mean()), 1.0, places=6)
        self.assertAlmostEqual(float(normalized[~labels].mean()), 1.0, places=6)
        with self.assertRaises(ValueError):
            stake_weights([True], [0])

    def test_typed_negative_flags_cover_arena_mistakes(self):
        reference = pd.DataFrame({"business_name": ["Cafe", "Cafe", "Other"],
                                  "business_address": ["12 Road", "13 Road", ""]})
        pool = pd.DataFrame({"business_name": ["Cafe", "Cafe", "Other"],
                             "business_address": ["13 Road", "", ""]})
        flags = classify_negative_types(
            reference, pool, [0, 1, 2], [0, 1, 2], [False] * 3,
            [.9, .9, .9], [.9, .4, .4], [1, -1, -1])
        self.assertTrue(flags[0] & NegativeType.FALSE_ACCEPT)
        self.assertTrue(flags[0] & NegativeType.CONTEST_LOSER)
        self.assertTrue(flags[0] & NegativeType.DIGIT_SUBSTITUTION)
        self.assertTrue(flags[1] & NegativeType.UNOWNED_LOOKALIKE)
        self.assertTrue(flags[1] & NegativeType.EMPTY_ADDRESS_SIBLING)

    def test_sampler_is_exact_1_to_3_duplicate_free_and_resumable(self):
        positives = np.arange(8)
        flags = np.zeros(80, dtype=np.uint8)
        negative_rows = np.arange(44, 80)
        kinds = list(NegativeType)
        for offset, row in enumerate(range(8, 44)):
            flags[row] = int(kinds[offset % len(kinds)])
        first = ArenaBatchSampler(positives, flags, negative_rows, batch_size=16, seed=9)
        ids, labels = first.take()
        self.assertEqual((int(labels.sum()), int((labels == 0).sum())), (4, 12))
        self.assertEqual(len(np.unique(ids)), 16)
        self.assertEqual(int(np.sum((ids >= 8) & (ids < 44))), 6)
        self.assertEqual(int(np.sum(ids >= 44)), 6)
        state = first.state()
        expected = first.take()
        resumed = ArenaBatchSampler(positives, flags, negative_rows, batch_size=16, seed=9, state=state)
        actual = resumed.take()
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_prefetch_retains_state_at_each_consumed_batch_boundary(self):
        positives = np.arange(8)
        flags = np.zeros(80, dtype=np.uint8)
        flags[8:44] = int(NegativeType.FALSE_ACCEPT)
        sampler = ArenaBatchSampler(positives, flags, np.arange(44, 80), batch_size=16, seed=5)

        def batches():
            for _ in range(4):
                pair_ids, labels = sampler.take()
                yield pair_ids, labels, sampler.state()

        prefetched = TokenizationPrefetcher(batches(), lambda ids, labels, state: state, depth=4)
        first_state = next(iter(prefetched))
        self.assertEqual(first_state["step"], 1)
        resumed = ArenaBatchSampler(positives, flags, np.arange(44, 80), batch_size=16,
                                    seed=5, state=first_state)
        expected = ArenaBatchSampler(positives, flags, np.arange(44, 80), batch_size=16, seed=5)
        expected.take()
        np.testing.assert_array_equal(resumed.take()[0], expected.take()[0])

    def test_modes_lock_g0_and_legacy_parity(self):
        self.assertEqual(len(g0_matrix()), 6)
        self.assertEqual({row["batch_size"] for row in g0_matrix()}, {32, 64, 128})
        parity = mode_config("g0b")
        self.assertEqual((parity.steps, parity.batch_size, parity.positive_fraction), (1800, 32, .5))
        self.assertFalse(parity.canonical_text)
        self.assertFalse(parity.field_tagged)
        with self.assertRaises(ValueError):
            mode_config("g0b", steps=1799)
        self.assertAlmostEqual(learning_rate(5, base=2e-5, warmup_steps=10,
                                             total_steps=100, schedule="constant"), 1e-5)
        self.assertAlmostEqual(learning_rate(11, base=2e-5, warmup_steps=10,
                                              total_steps=100, schedule="constant"), 2e-5)
        positives, negatives = np.arange(8), np.arange(8, 24)
        legacy = LegacyBatchSampler(positives, negatives, batch_size=8, seed=7)
        ids, labels = legacy.take()
        self.assertEqual((int(labels.sum()), len(labels)), (4, 8))
        rng = np.random.default_rng(7)
        old_positive, old_negative = _ShuffledCycle(positives, rng), _ShuffledCycle(negatives, rng)
        expected_ids = rng.permutation(np.r_[old_positive.take(4), old_negative.take(4)])
        np.testing.assert_array_equal(ids, expected_ids)
        resumed = LegacyBatchSampler(positives, negatives, batch_size=8, seed=7, state=legacy.state())
        np.testing.assert_array_equal(resumed.take()[0], legacy.take()[0])

    def test_arena_loader_rejects_holdout_or_misalignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arena, labels = root / "arena", root / "labels"
            arena.mkdir()
            labels.mkdir()
            reference = pd.DataFrame({"entity_id": ["S1-A"], "business_name": ["A"],
                                      "business_address": ["1 Rd"], "country": ["US"]})
            pool = pd.DataFrame({"entity_id": ["S2-A"], "business_name": ["A"],
                                 "business_address": ["1 Rd"], "country": ["US"]})
            pairs = pd.DataFrame({"left_idx": [0], "right_idx": [0]})
            reference.to_parquet(arena / "reference.parquet", index=False)
            pool.to_parquet(arena / "pool.parquet", index=False)
            pairs.to_parquet(arena / "pairs.parquet", index=False)
            np.save(arena / "v1_tree.npy", np.array([.9], np.float32))
            np.save(arena / "v1_blend.npy", np.array([.9], np.float32))
            np.save(labels / "label.npy", np.array([True]))
            np.save(labels / "true_count.npy", np.array([1], np.int16))
            np.save(labels / "rec_owner.npy", np.array([0], np.int32))
            (labels / "labels_summary.json").write_text(
                '{"holdout2_rows_read": 0, "positive_pairs": 1, "true_links": 1, "singletons": 0}')
            manifest = {"holdout2_excluded": True, "labels_read": False,
                        "references": 1, "pool_records": 1, "pairs": 1,
                        "sha256": {name: file_sha256(arena / name) for name in
                                   ("reference.parquet", "pool.parquet", "pairs.parquet")}}
            (arena / "manifest.json").write_text(json.dumps(manifest))
            loaded = load_arena(arena, labels)
            self.assertEqual(set(loaded.identity), {"manifest_sha256", "labels_summary_sha256",
                                                    "label_sha256", "true_count_sha256",
                                                    "record_owner_sha256", "tree_scores_sha256",
                                                    "blend_scores_sha256", "partition_sha256",
                                                    "reference_sha256", "pool_sha256", "pairs_sha256",
                                                    "pair_order_sha256", "sealed_reservation_sha256",
                                                    "references", "pool_records", "pairs"})
            manifest["holdout2_excluded"] = False
            (arena / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "HOLDOUT-2"):
                load_arena(arena, labels)

    def test_model_and_completed_checkpoint_bind_bytes_license_and_text_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{"num_labels": 1}')
            (model / "model.safetensors").write_bytes(b"weights")
            allowlist = root / "models.yaml"
            allowlist.write_text("models:\n  - {id: BAAI/bge-reranker-v2-m3, role: reranker, licence: apache-2.0, params_b: 0.568}\n")
            identity = model_identity(model, allowlist)
            self.assertEqual(identity["licence"], "apache-2.0")
            self.assertIn("model.safetensors", identity["files_sha256"])

            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            config = {"schema": 2, "profile": {"canonical_text": True, "field_tagged": True},
                      "max_length": 256, "initial_model": identity}
            (checkpoint / "training_config.json").write_text(json.dumps(config, indent=2))
            (checkpoint / "model.safetensors").write_bytes(b"trained")
            files = {path.name: file_sha256(path) for path in checkpoint.iterdir()}
            (checkpoint / "complete.json").write_text(json.dumps({"state": "complete", "step": 20,
                "config_sha256": _config_digest(config), "files_sha256": files}))
            restored, checkpoint_identity = verified_checkpoint(checkpoint)
            self.assertTrue(restored["profile"]["canonical_text"])
            self.assertEqual(checkpoint_identity["step"], 20)
            (checkpoint / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "changed"):
                verified_checkpoint(checkpoint)

    def test_fixed_evaluation_uses_complete_true_count_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arena, labels, scored, output = (root / name for name in ("arena", "labels", "scores", "eval"))
            arena.mkdir()
            labels.mkdir()
            scored.mkdir()
            reference = pd.DataFrame({"entity_id": ["S1-A", "S1-B", "S1-C"],
                                      "business_name": ["A", "B", "C"],
                                      "business_address": ["1", "2", "3"], "country": ["US"] * 3})
            pool = pd.DataFrame({"entity_id": ["S2-A", "S2-X"], "business_name": ["A", "X"],
                                 "business_address": ["1", "9"], "country": ["US", "US"]})
            pairs = pd.DataFrame({"left_idx": [0, 1], "right_idx": [0, 1]})
            for name, frame in (("reference.parquet", reference), ("pool.parquet", pool),
                                ("pairs.parquet", pairs)):
                frame.to_parquet(arena / name, index=False)
            manifest = {"holdout2_excluded": True, "labels_read": False, "references": 3,
                        "pool_records": 2, "pairs": 2,
                        "sha256": {name: file_sha256(arena / name) for name in
                                   ("reference.parquet", "pool.parquet", "pairs.parquet")}}
            (arena / "manifest.json").write_text(json.dumps(manifest))
            np.save(arena / "v1_tree.npy", np.array([.8, .8], np.float32))
            np.save(arena / "v1_blend.npy", np.array([.8, .8], np.float32))
            np.save(labels / "label.npy", np.array([True, False]))
            np.save(labels / "true_count.npy", np.array([2, 0, 0], np.int16))
            np.save(labels / "rec_owner.npy", np.array([0, -1], np.int32))
            (labels / "labels_summary.json").write_text(
                '{"holdout2_rows_read": 0, "positive_pairs": 1, "true_links": 2, "singletons": 2}')
            partition = root / "partition.npy"
            np.save(partition, np.array(["B", "B", "B"]))
            loaded = load_arena(arena, labels, partition_file=partition)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            checkpoint_config = {"schema": 2,
                                 "profile": {"canonical_text": True, "field_tagged": True},
                                 "max_length": 256, "initial_model": {
                                     "model_id": "BAAI/bge-reranker-v2-m3", "licence": "apache-2.0",
                                     "files_sha256": {"model.safetensors": "initial"}}}
            (checkpoint / "training_config.json").write_text(json.dumps(checkpoint_config))
            (checkpoint / "model.safetensors").write_bytes(b"trained")
            checkpoint_files = {path.name: file_sha256(path) for path in checkpoint.iterdir()}
            (checkpoint / "complete.json").write_text(json.dumps({"state": "complete", "step": 20,
                "config_sha256": _config_digest(checkpoint_config), "files_sha256": checkpoint_files}))
            _, checkpoint_identity = verified_checkpoint(checkpoint)
            np.save(scored / "scores.npy", np.array([.9, .1], np.float32))
            np.save(scored / "scored_mask.npy", np.ones(2, dtype=bool))
            score_manifest = {"state": "complete", "inputs": {
                key: loaded.identity[key] for key in ("manifest_sha256", "reference_sha256", "pool_sha256",
                                                       "pairs_sha256", "pair_order_sha256", "partition_sha256",
                                                       "sealed_reservation_sha256",
                                                       "references", "pool_records", "pairs")},
                "checkpoint": checkpoint_identity,
                "files": {name: file_sha256(scored / name) for name in ("scores.npy", "scored_mask.npy")}}
            (scored / "complete.json").write_text(json.dumps(score_manifest))
            args = argparse.Namespace(arena=arena, labels=labels, partition=partition, scores=scored,
                                      output=output, eval_partition="B", choice=["fixed:1:0.5"],
                                      tree_gate=.01, one_owner=False, sealed_reservation=None)
            report = evaluate_scores(args)
            # S1-A misses one of its two true links (5/6), S1-B is a correct singleton,
            # and candidate-less singleton S1-C is retained: mean = 17/18.
            self.assertAlmostEqual(report["results"][0]["metrics"]["macro_f05"], 17 / 18)
            incomplete = json.loads((scored / "complete.json").read_text())
            incomplete["inputs"].pop("pair_order_sha256")
            (scored / "complete.json").write_text(json.dumps(incomplete))
            args.output = root / "incomplete-manifest-eval"
            with self.assertRaisesRegex(ValueError, "exact arena inputs"):
                evaluate_scores(args)
            (scored / "complete.json").write_text(json.dumps(score_manifest))
            values = np.load(scored / "scores.npy")
            values[0] = .1
            np.save(scored / "scores.npy", values)
            args.output = root / "tampered-eval"
            with self.assertRaisesRegex(ValueError, "changed"):
                evaluate_scores(args)

    def test_g0b_adapter_excludes_eval_and_saves_exact_sampling_pools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, base, output = root / "medium", root / "base", root / "adapter"
            data.mkdir()
            base.mkdir()
            reference = pd.DataFrame({"entity_id": ["S1-A", "S1-B", "S1-V", "S1-E"],
                                      "business_name": ["A", "B", "V", "E"],
                                      "business_address": ["1", "2", "3", "4"],
                                      "country": ["US"] * 4,
                                      "split": ["train", "train", "tune", "eval"]})
            pool = pd.DataFrame({"entity_id": ["S2-A", "S2-B", "S2-X", "S2-Y"],
                                 "business_name": ["A", "B", "X", "Y"],
                                 "business_address": ["1", "2", "8", "9"], "country": ["US"] * 4})
            reference.to_parquet(data / "reference.parquet", index=False)
            pool.to_parquet(data / "pool.parquet", index=False)
            (data / "truth.json").write_text(json.dumps({"S1-A": ["S2-A"], "S1-B": ["S2-B"],
                                                          "S1-V": [], "S1-E": []}))
            pairs = pd.DataFrame({"left_idx": [0, 0, 0, 0, 1, 1, 1, 1, 2, 3],
                                  "right_idx": [0, 1, 2, 3, 1, 0, 2, 3, 2, 3]})
            pairs.to_parquet(base / "pairs.parquet", index=False)
            columns = ["core_token_sort", "name_token_sort", "address_token_sort",
                       "number_disjoint", "long_number_disjoint", "country_equal"]
            features = pd.DataFrame(np.zeros((len(pairs), len(columns))), columns=columns)
            features["country_equal"] = 1
            features["core_token_sort"] = np.linspace(.99, .1, len(pairs))
            features["name_token_sort"] = features.core_token_sort
            features["address_token_sort"] = features.core_token_sort
            features.to_parquet(base / "base_features.parquet", index=False)
            pd.DataFrame({"num_unit_conflict": np.zeros(len(pairs))}).to_parquet(
                base / "numeric_features.parquet", index=False)
            joblib.dump({"model": HeuristicModel(columns), "threshold": .5,
                         "feature_columns": columns}, base / "best.joblib")
            report = adapt_g0b(argparse.Namespace(medium_data=data, base_dir=base, output=output))
            adapted = pd.read_parquet(output / "arena/reference.parquet")
            self.assertEqual(set(adapted.split), {"train", "tune"})
            self.assertEqual(report["positive_pool"], 2)
            self.assertEqual(report["negative_pool"], 6)
            self.assertTrue((output / "arena/g0b_positive.npy").is_file())
            self.assertTrue((output / "arena/g0b_negative.npy").is_file())
            loaded = load_score_data(output / "arena", partition_file=output / "partition.npy")
            self.assertEqual(len(loaded.pairs), 9)
            manifest_path = output / "arena/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["pair_order"] = "left_then_right"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "strictly sorted"):
                load_score_data(output / "arena", partition_file=output / "partition.npy")

    def test_grouped_historical_order_detects_nonadjacent_duplicate_with_bounded_groups(self):
        historical = {"pair_order": "left_grouped_original_right_order",
                      "max_pairs_per_reference": 3}
        _validate_pair_order(np.array([0, 0, 1, 1]), np.array([2, 0, 1, 0]), historical)
        with self.assertRaisesRegex(ValueError, "Duplicate candidate pair"):
            _validate_pair_order(np.array([0, 0, 0]), np.array([1, 0, 1]), historical)
        with self.assertRaisesRegex(ValueError, "strictly sorted"):
            _validate_pair_order(np.array([0, 0]), np.array([1, 0]), {"pair_order": "left_then_right"})


if __name__ == "__main__":
    unittest.main()
