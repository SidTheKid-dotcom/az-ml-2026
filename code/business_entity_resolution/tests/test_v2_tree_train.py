"""P3.1's complete-group feature, split, quantization and score contracts."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa

from ber.predict import feature_batch
from ber.v2_tree_train import (
    Dataset, FULL_POOL_ROWS, _tune_union_files, check_reference_feature_order, effective_available_bytes, entity_scores,
    feature_columns, ids_digest, load_dataset, load_p2_ready, mandatory_negative_rows, no_swap_verified_pairs,
    pair_budget, pair_labels,
    prepare_exact_features, sampled_negatives, score_tune, sha256_file, supervised_full_fit, train_tree,
    union_feature_columns, run,
)


class ChunkedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame({
            "entity_id": ["S1-a", "S1-b", "S1-c"],
            "business_name": ["Coral Bank", "Acme Foods", "Empty"],
            "business_address": ["22 River Road", "11 Park Lane", ""],
            "country": ["US", "US", "India"],
            "split": ["train", "train", "tune"],
        })
        self.pool = pd.DataFrame({
            "entity_id": ["S2-a", "S3-b", "S2-c", "S3-miss"],
            "business_name": ["Coral Bank", "Coral Bank Co", "Acme Foods", "Empty"],
            "business_address": ["22 River Rd", "99 River Rd", "11 Park Lane", ""],
            "country": ["US", "US", "US", "India"],
        })
        self.pairs = pd.DataFrame({"left_idx": [0, 0, 1], "right_idx": [0, 1, 2],
                                   "name_score": [.9, .5, .8],
                                   "address_score": [.8, .1, .8],
                                   "retrieval_score": [.95, .4, .9],
                                   "retrieval_rank": [1, 2, 1],
                                   "dense_score": [.95, .4, .9],
                                   "dense_only": [1., 1., 1.]})
        self.truth = {"S1-a": ["S2-a", "S3-miss"], "S1-b": ["S2-c"], "S1-c": []}
        self.data = Dataset(Path("unused"), Path("unused"), {}, {},
                            pa.Table.from_pandas(self.reference, preserve_index=False),
                            pa.Table.from_pandas(self.pool, preserve_index=False),
                            self.reference.entity_id.tolist(), self.reference.country.to_numpy(),
                            self.reference.split.to_numpy(), self.truth, feature_columns(self.pairs.columns))

    def test_complete_groups_and_reindexed_record_table_are_exact(self):
        expected = feature_batch(self.reference, self.pool, self.pairs, self.data.features)
        first = prepare_exact_features(self.data, 0, 1, self.pairs.iloc[:2])
        second = prepare_exact_features(self.data, 1, 3, self.pairs.iloc[2:])
        observed = pd.concat([first, second], ignore_index=True)
        np.testing.assert_array_equal(observed.to_numpy(), expected.to_numpy())
        self.assertEqual(list(observed.columns), self.data.features)
        self.assertEqual(len(observed.columns), 105)
        self.assertAlmostEqual(first.candidate_log_count.iat[0], np.log1p(2))
        # An early negative filter would corrupt candidate-dependent features.
        partial = prepare_exact_features(self.data, 0, 1, self.pairs.iloc[:1])
        self.assertNotEqual(partial.candidate_log_count.iat[0], first.candidate_log_count.iat[0])

    def test_label_join_never_injects_missing_positive_pair(self):
        labels, targets = pair_labels(self.data, self.pairs.iloc[:2])
        np.testing.assert_array_equal(labels, [True, False])
        self.assertEqual(targets.tolist(), ["S2-a", "S3-b"])
        self.assertEqual(len(self.truth["S1-a"]), 2)

    def test_v1_column_order_check_catches_drift(self):
        with TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "original.joblib"
            joblib.dump({"feature_columns": self.data.features,
                         "record": {"depth": 10, "negative_weight": 2}}, checkpoint)
            record = check_reference_feature_order(self.data.features, checkpoint)
            self.assertEqual(record["depth"], 10)
            self.assertEqual(record["extra_union_columns"], [])
            with self.assertRaisesRegex(ValueError, "feature order"):
                check_reference_feature_order(self.data.features[::-1], checkpoint)

    def test_shared_p2_union_feature_semantics_remove_tune_only_legacy_scores(self):
        pairs = self.pairs.assign(dense_hit=[True, False, True],
                                  phonetic_key_hit=[False, True, False],
                                  phonetic_score=[.8, .4, .9],
                                  lexical_retrieval_score=[.7, .4, .8],
                                  lexical_scores_missing=[False, False, False])
        self.data.features = union_feature_columns(pairs.columns)
        self.data.pair_policy = "dense_key_shared"
        selected = prepare_exact_features(self.data, 0, 1, pairs.iloc[:2])
        self.assertEqual(len(self.data.features), 109)
        np.testing.assert_array_equal(selected.retrieval_name, [0., 0.])
        np.testing.assert_array_equal(selected.retrieval_address, [0., 0.])
        np.testing.assert_array_equal(selected.phonetic_score, [0., 0.])
        np.testing.assert_array_equal(selected.lexical_retrieval_score, [0., 0.])
        np.testing.assert_array_equal(selected.dense_only, [1., 0.])
        np.testing.assert_array_equal(selected.phonetic_only, [0., 1.])
        self.assertTrue(selected.lexical_scores_missing.eq(1).all())
        self.assertAlmostEqual(selected.candidate_log_count.iat[0], np.log1p(2))
        with self.assertRaisesRegex(ValueError, "dense cosine"):
            prepare_exact_features(self.data, 0, 1, pairs.iloc[:2].assign(retrieval_score=[.3, .4]))


class MemoryAndScoringTests(unittest.TestCase):
    def test_pod_memory_availability_uses_cgroup_not_host_free(self):
        self.assertEqual(effective_available_bytes(200 * 1024 ** 3, 50 * 1024 ** 3,
                         40 * 1024 ** 3, 25 * 1024 ** 3), 35 * 1024 ** 3)
        self.assertEqual(effective_available_bytes(8 * 1024 ** 3, 50 * 1024 ** 3,
                         40 * 1024 ** 3, 25 * 1024 ** 3), 8 * 1024 ** 3)
    def test_negative_sampling_keeps_positives_and_is_chunk_invariant(self):
        pairs = pd.DataFrame({"left_idx": np.repeat(np.arange(6), 9),
                              "right_idx": np.tile(np.arange(9), 6)})
        labels = ((pairs.left_idx == 2) & (pairs.right_idx == 4)).to_numpy()
        together = sampled_negatives(pairs, labels, .35, seed=20260925)
        split = np.r_[sampled_negatives(pairs.iloc[:18], labels[:18], .35, 20260925),
                      sampled_negatives(pairs.iloc[18:], labels[18:], .35, 20260925)]
        np.testing.assert_array_equal(together, split)
        self.assertTrue(together[labels].all())
        self.assertTrue(np.all(np.bincount(pairs.left_idx.to_numpy()[together & ~labels], minlength=6) >= 1))

    def test_mandatory_negative_is_hardest_per_s1_and_keeps_singletons(self):
        pairs = pd.DataFrame({"left_idx": [0, 0, 1, 1, 2, 2], "right_idx": [8, 9, 5, 6, 4, 3],
                              "dense_score": [.99, .9, .8, .7, .6, .5],
                              "hard_negative_priority": [3, 0, -1, 0, 2, 2]})
        labels = np.array([False, False, True, False, False, False])
        mandatory = mandatory_negative_rows(pairs, labels)
        np.testing.assert_array_equal(mandatory, [False, True, False, True, True, False])
        np.testing.assert_array_equal(sampled_negatives(pairs, labels, 0., 20260925, mandatory),
                                      labels | mandatory)

    def test_ram_cap_never_silently_drops_retrieved_positives(self):
        args = SimpleNamespace(ram_budget_gb=.01, min_available_gb=.001, max_train_pairs=60_000_000)
        with patch("ber.v2_tree_train.resource_snapshot", return_value={"effective_available_bytes": 2 * 1024 ** 3}):
            budget = pair_budget(1, 10_000, 105, args)
            self.assertGreater(budget["actual_pair_cap"], 1)
            with self.assertRaisesRegex(MemoryError, "do not silently drop positives"):
                pair_budget(10_000, 10_000, 105, args)

    def test_local_full_is_capped_by_measured_no_swap_rows(self):
        report = {"scope": "TRAIN-only medium dense full-target-pool memory probe; no TUNE fitting",
                  "no_tune_or_holdout_model_score": True, "measurements": [
                      {"train_pairs": 750_000, "status": "pass", "exitcode": 0,
                       "peak_child_swap_bytes": 0, "system_swap_out_delta_bytes": 0,
                       "stages": [{"stage": "quantized"}, {"stage": "fitted"}]},
                      {"train_pairs": 1_500_000, "status": "unsafe_or_failed", "exitcode": -15,
                       "peak_child_swap_bytes": 0, "system_swap_out_delta_bytes": 190_000_000}]}
        self.assertEqual(no_swap_verified_pairs(report), 750_000)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "resource_results.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            args = SimpleNamespace(mode="full", execution_site="local", resource_report=str(path),
                                   ram_budget_gb=6., min_available_gb=3., max_train_pairs=60_000_000)
            with patch("ber.v2_tree_train.resource_snapshot", return_value={"effective_available_bytes": 8 * 1024 ** 3}):
                with self.assertRaisesRegex(MemoryError, "do not silently drop positives"):
                    pair_budget(7_000_000, 80_000_000, 109, args)

    def test_pod_cogpu_profile_caps_pairs_and_reserves_cgroup_memory(self):
        args = SimpleNamespace(mode="full", execution_site="pod-cpu", threads=3,
                               ram_budget_gb=22., min_available_gb=12., max_process_rss_gb=22.,
                               max_train_pairs=60_000_000)
        measured = {"effective_available_bytes": 35 * 1024 ** 3,
                    "cgroup_limit_bytes": 50 * 1024 ** 3, "cgroup_cpu_quota": 7.65}
        with patch("ber.v2_tree_train.resource_snapshot", return_value=measured):
            budget = pair_budget(7_000_000, 90_000_000, 109, args, mandatory_negatives=2_072_821)
            self.assertEqual(budget["actual_pair_cap"], 10_000_000)
            self.assertLess(budget["negative_keep_probability"], .02)
            self.assertEqual(budget["mandatory_negative_s1s"], 2_072_821)
            args.threads = 4
            with self.assertRaisesRegex(ValueError, "co-residency"):
                pair_budget(7_000_000, 90_000_000, 109, args, mandatory_negatives=2_072_821)

    def test_singletons_and_retrieval_misses_remain_in_macro_score(self):
        left = np.array([0, 0, 2])
        labels = np.array([True, False, False])
        actual = np.array([1, 1, 0, 0])
        f05, _, predicted, tp = entity_scores(left, labels, np.array([True, False, False]), actual)
        np.testing.assert_array_equal(f05, [1., 0., 1., 1.])
        np.testing.assert_array_equal(predicted, [1, 0, 0, 0])
        np.testing.assert_array_equal(tp, [1, 0, 0, 0])

    def test_tune_report_includes_empty_refs_and_oracle_misses(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)
            refs = pd.DataFrame({"entity_id": ["a", "b", "c", "d"],
                                 "country": ["US", "India", "India", "US"]})
            pairs = pd.DataFrame({"left_idx": [0, 0, 2], "right_idx": [0, 1, 2],
                                  "reference_local_idx": [0, 0, 2],
                                  "source1_entity_id": ["a", "a", "c"],
                                  "candidate_entity_id": ["p", "wrong", "wrong"],
                                  "is_match": [True, False, False]})
            pairs.to_parquet(output / "tune_pairs.parquet", index=False)
            np.save(output / "tune_probabilities.npy", np.array([.8, .05, .2], dtype=np.float32))
            pool = pa.table({"entity_id": ["p", "wrong", "missed"],
                             "business_name": ["A", "X", "बैंक"],
                             "business_address": ["1 Rd", "2 Rd", "3 Rd"],
                             "country": ["US", "India", "India"]})
            data = SimpleNamespace(ref_ids=refs.entity_id.tolist(), country=refs.country.to_numpy(),
                                   truth={"a": ["p"], "b": ["missed"], "c": [], "d": []}, pool=pool)
            result, f05 = score_tune(data, np.ones(4, dtype=bool), output,
                                     np.load(output / "tune_probabilities.npy"))
            self.assertEqual(result["tune"]["reference_count"], 4)
            self.assertEqual(result["tune"]["macro_f05"], .75)
            self.assertEqual(result["retrieval_ceiling"]["link_recall"], .5)
            self.assertEqual(result["retrieval_ceiling"]["oracle_macro_f05"], .75)
            self.assertEqual(result["tune"]["singleton_accuracy"], 1.0)
            native = [row for row in result["tune"]["by_target_name_script"]
                      if row["target_name_script"] == "Devanagari"]
            self.assertEqual(native[0]["retrieved_true_links"], 0)
            self.assertEqual(native[0]["true_links"], 1)
            np.testing.assert_array_equal(f05, [1., 0., 1., 1.])
            self.assertEqual(len(pd.read_parquet(output / "tune_entities.parquet")), 4)


class ArtifactGateTests(unittest.TestCase):
    def _tiny_input(self, directory: Path, split: str, kind: str = "full_pool_dense_candidates"):
        reference = pd.DataFrame({"entity_id": ["S1-a"], "business_name": ["A"],
                                  "business_address": ["1 Road"], "country": ["US"], "split": [split]})
        pool = pd.DataFrame({"entity_id": ["S2-a"], "business_name": ["A"],
                             "business_address": ["1 Rd"], "country": ["US"]})
        reference.to_parquet(directory / "reference.parquet", index=False)
        pool.to_parquet(directory / "pool.parquet", index=False)
        pd.DataFrame({"left_idx": [0], "right_idx": [0], "dense_score": [.9],
                      "dense_only": [1.]}).to_parquet(directory / "pairs.parquet", index=False)
        (directory / "truth.json").write_text("invalid -- should never be read", encoding="utf-8")
        metadata = {"dataset_kind": kind, "references": 1, "pool_records": 1,
                    "candidate_pairs": 1, "positive_candidate_injection": False,
                    "mining_full_pool_records": FULL_POOL_ROWS}
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (directory / "complete.json").write_text(json.dumps({"state": "complete", "metadata": metadata}), encoding="utf-8")

    def test_holdout_split_fails_before_truth_is_opened(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self._tiny_input(root, "eval")
            with self.assertRaisesRegex(ValueError, "Eval/HOLDOUT-2"):
                load_dataset(root, None, root / "records", require_union=False, min_available_gb=.01)

    def test_full_mode_rejects_dense_only_before_truth_is_opened(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self._tiny_input(root, "train")
            with self.assertRaisesRegex(ValueError, "P2 candidate union"):
                load_dataset(root, None, root / "records", require_union=True, min_available_gb=.01)

    def test_full_train_ready_uses_all_truth_links_without_injecting_missed_pairs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            refs = pd.DataFrame({"entity_id": ["S1-a", "S1-b"], "business_name": ["A", "B"],
                                 "business_address": ["1 Rd", "2 Rd"], "country": ["US", "US"],
                                 "split": ["train", "train"]})
            pool = pd.DataFrame({"entity_id": ["S2-a", "S2-x", "S3-missed"],
                                 "business_name": ["A", "X", "A"],
                                 "business_address": ["1 Rd", "3 Rd", "1 Rd"],
                                 "country": ["US"] * 3})
            pairs = pd.DataFrame({"left_idx": [0, 1], "right_idx": [0, 1], "label": [1, 0],
                                  "dense_score": [.9, .8], "retrieval_score": [.9, .8],
                                  "retrieval_rank": [1, 1], "dense_only": [1., 1.],
                                  "dense_hit": [True, True], "phonetic_key_hit": [False, False]})
            truth = pd.DataFrame({"left_idx": [0, 0], "right_idx": [0, 2]})
            frames = {"reference": refs, "pool": pool, "candidate_pairs": pairs, "train_truth_pairs": truth}
            files = {}
            for name, frame in frames.items():
                path = root / f"{name}.parquet"
                frame.to_parquet(path, index=False)
                files[name] = {"path": path.name, "rows": len(frame), "sha256": sha256_file(path)}
            ready = {"state": "complete", "split": "train", "dense_mining_complete": True,
                     "train_truth_join_complete": True, "positive_candidate_injection": False,
                     "full_pool_records": FULL_POOL_ROWS,
                     "train_reference_id_sha256": ids_digest(refs.entity_id), "files": files}
            marker = root / "ready.json"
            marker.write_text(json.dumps(ready), encoding="utf-8")
            data = load_p2_ready(marker, root / "records", .01)
            np.testing.assert_array_equal(data.truth_counts, [2, 0])
            labels, targets = pair_labels(data, pairs)
            np.testing.assert_array_equal(labels, [True, False])
            self.assertEqual(targets.tolist(), ["S2-a", "S2-x"])
            fast_labels, omitted_ids = pair_labels(data, pairs, include_target_ids=False)
            np.testing.assert_array_equal(fast_labels, labels)
            self.assertIsNone(omitted_ids)
            with self.assertRaisesRegex(ValueError, "labels disagree"):
                pair_labels(data, pairs.assign(label=[0, 0]))

    def test_incomplete_p2_ready_fails_before_any_file_or_label_read(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "ready.json"
            marker.write_text(json.dumps({"state": "mining", "split": "train"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "atomic"):
                load_p2_ready(marker, root / "records", .01)

    def test_full_fit_rejects_completed_p2_screen_before_loading_large_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "ready.json"
            marker.write_text(json.dumps({"mode": "screen_10k", "screen_only": True,
                                          "files": {"reference": {"rows": 10_000}}}), encoding="utf-8")
            args = SimpleNamespace(mode="full", output=str(root / "tree"), threads=3,
                                   iterations=650, max_train_pairs=10_000_000,
                                   reference_batch_size=1000, pair_read_batch_size=100000,
                                   max_pairs_per_batch=250000, ram_budget_gb=22.,
                                   min_available_gb=12., negative_weight=1.,
                                   screen_train_references=1200, execution_site="pod-cpu",
                                   train_ready=str(marker), train_data=None, train_pairs=None)
            with self.assertRaisesRegex(ValueError, "FULL ready marker"):
                run(args)
            self.assertFalse((root / "tree").exists())

    def test_tune_union_requires_same_complete_scored_context(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            pairs = pd.DataFrame({"left_idx": [0], "right_idx": [0], "dense_score": [.9]})
            raw = root / "candidate_pairs_raw.parquet"
            original = root / "candidate_pairs.parquet"
            scored = root / "scored_union.parquet"
            for path in (raw, original, scored):
                pairs.to_parquet(path, index=False)
            pd.DataFrame({"entity_id": ["S2-a"]}).to_parquet(root / "pool.parquet", index=False)
            complete = {"state": "complete", "references": 12_000,
                        "reference_ids_sha256": "5bb5759fa52ac19df8839520c7ccf147bae01a5499a2900af5d1b4ebeb1dbaa2",
                        "full_target_pool_rows": FULL_POOL_ROWS,
                        "positive_candidate_injection": False, "truth_used_in_retrieval": False,
                        "raw_pairs": {"rows": 1, "sha256": sha256_file(raw)}}
            context = {"state": "complete", "positive_candidate_injection": False,
                       "inputs": {"pairs": {"sha256": sha256_file(original)}},
                       "pair_file": {"sha256": sha256_file(scored)}}
            manifest = scored.with_suffix(".context.json")
            manifest.write_text(json.dumps(context), encoding="utf-8")
            metadata = _tune_union_files(root, scored, complete)
            self.assertEqual(metadata["candidate_pairs"], 1)
            context["inputs"]["pairs"]["sha256"] = "changed"
            manifest.write_text(json.dumps(context), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "another union"):
                _tune_union_files(root, scored, complete)

    def test_training_from_quantized_numeric_tsv_uses_exact_names(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)
            columns = ["name_ratio", "num_short_exact_any"]
            (output / "train_features.cd").write_text(
                "0\tLabel\n1\tNum\tname_ratio\n2\tNum\tnum_short_exact_any\n", encoding="utf-8")
            sample = np.linspace(0, 1, 40)
            np.savetxt(output / "train_features.tsv",
                       np.column_stack(((sample > .5).astype(int), sample, (sample > .5).astype(int))),
                       fmt="%.9g", delimiter="\t")
            args = SimpleNamespace(threads=2, seed=20260925, ram_budget_gb=1.0,
                                   min_available_gb=.01, iterations=8, depth=3, negative_weight=1.0)
            model, report = train_tree(SimpleNamespace(features=columns), output, args)
            self.assertEqual(list(model.feature_names_), columns)
            self.assertEqual(report["tree_count"], 8)
            self.assertTrue((output / "train_pool.quantized").is_file())
            self.assertEqual(len(report["native_tree"]["sha256"]), 64)

    def test_full_fit_runs_isolated_and_returns_exact_ordered_checkpoint(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)
            columns = ["name_ratio", "dense_score"]
            (output / "train_features.cd").write_text(
                "0\tLabel\n1\tNum\tname_ratio\n2\tNum\tdense_score\n", encoding="utf-8")
            values = np.linspace(0, 1, 50)
            np.savetxt(output / "train_features.tsv", np.column_stack((values > .5, values, values)),
                       fmt="%.9g", delimiter="\t")
            args = SimpleNamespace(threads=2, iterations=8, depth=3, seed=20260925,
                                   negative_weight=1., ram_budget_gb=1., min_available_gb=.01,
                                   max_process_rss_gb=2., execution_site="local", max_swap_out_mb=1024.)
            model, report = supervised_full_fit(SimpleNamespace(features=columns), output, args)
            self.assertEqual(list(model.feature_names_), columns)
            self.assertEqual(report["tree_count"], 8)
            self.assertEqual(report["bounded_cpu_resource"]["peak_child_swap_bytes"], 0)
            self.assertTrue((output / "fit_resource.json").is_file())


if __name__ == "__main__":
    unittest.main()
