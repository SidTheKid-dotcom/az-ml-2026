import tempfile
import unittest
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ber.cross_encoder_experiment import conflict_priorities, detailed_score, load_base_bundle, training_pool, verify_full_pool_data
from ber.models import HeuristicModel


class CrossEncoderAdapterTests(unittest.TestCase):
    def test_union_provenance_requires_committed_matching_parents_and_no_eval(self):
        reference = pd.DataFrame({"entity_id": ["a", "b"], "split": ["train", "tune"], "country": ["India", "US"]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("lexical", "dense", "encoder", "union"):
                (root / name).mkdir()
            shared = {"reference_id_sha256": hashlib.sha256(b"a\nb").hexdigest(),
                      "split_counts": {"train/India": 1, "tune/US": 1}}
            lexical = {**shared, "full_pool_records_by_country": {"India": 400, "US": 600}}
            dense = {**shared, "mining_full_pool_records": 1000}
            identity = {"scope": "full_pool", "target_rows": 1000,
                        "checkpoint": {"sha256": "verified", "path": str(root / "encoder/checkpoint")}}
            metadata = {**shared, "dataset_kind": "full_pool_lexical_phonetic_dense_union",
                        "full_pool_records": 1000, "checkpoint_sha256": "verified",
                        "parent_lexical": str(root / "lexical"), "parent_dense": str(root / "dense"),
                        "positive_candidate_injection": False, "original_splits_preserved": True}
            (root / "lexical/metadata.json").write_text(json.dumps(lexical))
            (root / "dense/complete.json").write_text(json.dumps({"state": "complete", "metadata": dense,
                "config": {"pool_embedding_identity": identity}}))
            (root / "encoder/results.json").write_text(json.dumps({"model_id": "intfloat/multilingual-e5-small"}))
            (root / "union/complete.json").write_text(json.dumps({"state": "complete", "metadata": metadata}))
            self.assertEqual(verify_full_pool_data(root / "union", metadata, reference), "intfloat/multilingual-e5-small")
            with self.assertRaises(ValueError):
                verify_full_pool_data(root / "union", metadata, reference.assign(split=["train", "eval"]))
            dense["reference_id_sha256"] = "different_population"
            (root / "dense/complete.json").write_text(json.dumps({"state": "complete", "metadata": dense,
                "config": {"pool_embedding_identity": identity}}))
            with self.assertRaises(ValueError):
                verify_full_pool_data(root / "union", metadata, reference)

    def test_conflict_sampling_uses_only_training_labels(self):
        reference = pd.DataFrame({"split": ["train", "tune", "eval"]})
        pairs = pd.DataFrame({"left_idx": [0, 0, 0, 0, 0, 1, 1, 2, 2]})
        labels = np.array([1, 0, 0, 0, 0, 1, 0, 1, 0])
        scores = np.array([.4, .99, .98, .97, .96, .9, 1., .9, 1.])
        conflicts = np.array([0, 0, 0, 1, 1, 0, 100, 0, 100])
        positive, negative = training_pool(pairs, reference, labels, scores, 2,
                                            conflicts=conflicts, conflict_fraction=.5)
        self.assertEqual(positive.tolist(), [0])
        self.assertEqual(set(negative), {1, 3})

    def test_conflict_priority_covers_units_and_substantive_name_changes(self):
        features = pd.DataFrame({"num_unit_conflict": [1, 0, 0],
                                 "address_token_sort": [.9, .9, 1.],
                                 "core_token_sort": [.8, .6, 1.], "core_exact": [1, 0, 1],
                                 "name_token_left_coverage": [1, .5, 1],
                                 "name_token_right_coverage": [1, .5, 1]})
        np.testing.assert_array_equal(conflict_priorities(features), [1, 1, 0])

    def test_completed_tabular_bundle_recomputes_selected_model_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            columns = ["core_token_sort", "name_token_sort", "address_token_sort",
                       "number_disjoint", "long_number_disjoint", "country_equal"]
            base = pd.DataFrame([[1, 1, 1, 0, 0, 1], [.4, .4, .4, 0, 0, 1]], columns=columns)
            numeric = pd.DataFrame({"num_unit_conflict": [0, 1]})
            joblib.dump({"model": HeuristicModel(columns), "threshold": .8,
                         "feature_columns": columns}, path / "best.joblib")
            base.to_parquet(path / "base_features.parquet", index=False)
            numeric.to_parquet(path / "numeric_features.parquet", index=False)
            np.save(path / "best_probabilities.npy", np.zeros(2))  # deliberately stale
            with self.assertRaises(FileNotFoundError):
                load_base_bundle(path, "tabular")
            pd.DataFrame({"left_idx": [0, 1], "right_idx": [0, 1]}).to_parquet(path / "pairs.parquet", index=False)
            package, pairs, scores, features, provenance = load_base_bundle(path)
            np.testing.assert_allclose(scores, [1, .4])
            self.assertEqual(provenance["format"], "tabular")
            self.assertIn("num_unit_conflict", features)

    def test_reject_all_threshold_keeps_false_positive_counts_consistent(self):
        reference = pd.DataFrame({"entity_id": ["a"], "country": ["US"]})
        pool = pd.DataFrame({"entity_id": ["wrong"]})
        pairs = pd.DataFrame({"left_idx": [0], "right_idx": [0]})
        scores = np.array([.999], dtype=np.float32)
        threshold = np.nextafter(float(scores[0]), np.inf)
        report = detailed_score(reference, pool, pairs, scores, {"a": set()}, threshold, [True])
        self.assertEqual(report["macro_f05"], 1.0)
        self.assertEqual(report["false_positive_links"], 0)


if __name__ == "__main__":
    unittest.main()
