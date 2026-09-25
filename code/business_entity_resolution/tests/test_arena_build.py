"""C1 arena partitioning, provenance, and label-free fidelity tests."""
from __future__ import annotations

import json
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber.arena_build import (PARTITION_A, PARTITION_B_REST, PARTITION_SLICE, capped_claim_components,
                             fidelity_profile, finalize, labels, make_partitions, sha256,
                             FROZEN_V1_PIPELINE_SHA256)


def _score_artifact(directory, data, pairs, n_references, n_pool):
    directory.mkdir()
    pairs_path = directory / "scored_pairs.parquet"
    blend, tree = directory / "probabilities.npy", directory / "base_probabilities.npy"
    pq.write_table(pairs, pairs_path)
    np.save(blend, np.full(len(pairs), 0.9, dtype=np.float32))
    np.save(tree, np.full(len(pairs), 0.8, dtype=np.float32))
    report = {
        "reference_count": n_references, "full_reference_count": n_references,
        "pool_count": n_pool, "candidate_pairs": len(pairs), "scorer": "frozen_tree_ce",
        "benchmark_only": False, "all_selected_references_written": True, "labels_accessed": False,
        "probability_order": "scored_pairs.parquet row order",
        "base_probability_order": "scored_pairs.parquet row order",
        "pipeline_sha256": FROZEN_V1_PIPELINE_SHA256,
        "pipeline": {"procedure": "medium_dense_bge_ce1800", "alpha": 0.75,
                     "gate": 0.01, "threshold": 0.775},
        "input_identities": {
            "reference": {"size_bytes": (data / "reference.parquet").stat().st_size},
            "pool": {"size_bytes": (data / "pool.parquet").stat().st_size},
            "pairs": {"size_bytes": pairs_path.stat().st_size},
        },
    }
    report_path, complete_path = directory / "report.json", directory / "complete.json"
    report_path.write_text(json.dumps(report))
    complete_path.write_text(json.dumps({"state": "complete", "report": report}))
    hashes = directory / "source.sha256"
    hashes.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in (blend, tree, pairs_path)))
    return SimpleNamespace(pairs=pairs_path, blend=blend, tree=tree, complete=complete_path, hashes=hashes)


class ComponentPartitionTests(unittest.TestCase):
    def test_threshold_stays_at_point_zero_five_when_cap_passes(self):
        component, threshold, attempts = capped_claim_components(
            np.array([0, 1, 2]), np.array([10, 10, 11]), np.array([0.8, 0.01, 0.9]), 10,
            max_component_fraction=0.2)
        self.assertEqual(threshold, 0.05)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(np.unique(component)), 10)

    def test_threshold_is_raised_only_after_largest_component_fails(self):
        s1 = np.arange(10)
        component, threshold, attempts = capped_claim_components(
            s1, np.zeros(10, dtype=int), np.full(10, 0.09), 10,
            max_component_fraction=0.2, threshold_step=0.05)
        self.assertEqual(threshold, 0.1)
        self.assertEqual([row["threshold"] for row in attempts], [0.05, 0.1])
        self.assertEqual(attempts[0]["largest_component_s1"], 10)
        self.assertEqual(len(np.unique(component)), 10)

    def test_threshold_rises_until_forced_components_are_disjoint(self):
        force_a = np.array([True, False, False, False])
        force_b = np.array([False, True, False, False])
        component, threshold, attempts = capped_claim_components(
            np.array([0, 1]), np.array([7, 7]), np.array([0.09, 0.09]), 4,
            force_a=force_a, force_b=force_b, max_component_fraction=0.75,
            threshold_step=0.05, max_threshold=0.15)
        self.assertEqual(threshold, 0.1)
        self.assertEqual([row["forced_conflict_components"] for row in attempts], [1, 0])
        self.assertNotEqual(component[0], component[1])

    def test_threshold_fails_only_after_documented_safe_range(self):
        force_a = np.array([True, False])
        force_b = np.array([False, True])
        with self.assertRaisesRegex(ValueError, r"safe threshold range \[0.05, 0.1\]"):
            capped_claim_components(np.array([0, 1]), np.array([7, 7]), np.array([0.9, 0.9]), 2,
                                    force_a=force_a, force_b=force_b, max_component_fraction=1.0,
                                    threshold_step=0.05, max_threshold=0.1)

    def test_forced_components_and_exact_component_disjoint_slice(self):
        component = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        force_a = np.array([True, False, False, False, False, False, False, False, False, False])
        force_b = np.array([False, False, True, False, False, False, False, False, False, False])
        first, counts = make_partitions(component, force_a, force_b, a_fraction=0.6, slice_size=2, seed=7)
        second, _ = make_partitions(component, force_a, force_b, a_fraction=0.6, slice_size=2, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[component == 0] == PARTITION_A))
        self.assertNotEqual(first[2], PARTITION_A)
        self.assertEqual(counts["SLICE"], 2)
        for comp in np.unique(component):
            self.assertEqual(len(np.unique(first[component == comp])), 1)

    def test_forced_component_conflict_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "share 1 claim components"):
            make_partitions(np.array([0, 0, 1]), np.array([True, False, False]),
                            np.array([False, True, False]), slice_size=0)


class FidelityTests(unittest.TestCase):
    @staticmethod
    def frames():
        reference = pd.DataFrame({
            "entity_id": ["S1-a", "S1-b", "S1-c"],
            "business_name": ["Alpha LLC", "Beta", "Gamma"],
            "business_address": ["1 A", "2 B", "3 C"],
            "country": ["US", "US", "India"],
        })
        pool = pd.DataFrame({
            "entity_id": ["S2-a", "S2-g", "S3-b"],
            "business_name": ["Alpha", "Gamma", "Beta"],
            "business_address": ["<NULL>", "3 C", ""],
            "country": ["US", "India", "US"],
        })
        return reference, pool

    def test_arena_truth_is_separate_from_label_free_test_profile(self):
        reference, pool = self.frames()
        left = np.array([0, 1, 1, 2])
        right = np.array([0, 0, 2, 1])
        blend = np.array([0.8, 0.9, 0.1, 0.8])
        labels = np.array([True, False, True, True])
        arena = fidelity_profile(left, right, blend, reference, pool, label=labels,
                                 true_count=np.array([2, 1, 1]))
        test = fidelity_profile(left, right, blend, reference, pool)
        self.assertTrue(arena["labels_used"])
        self.assertFalse(test["labels_used"])
        self.assertEqual(test["v1_pre_ownership_contests"]["removed_claim_wrong_rate"], None)
        self.assertNotIn("arena_truth_metrics", test)
        self.assertEqual(arena["arena_truth_metrics"]["one_owner"]["true_links"], 4)
        self.assertEqual(arena["arena_truth_metrics"]["one_owner"]["retrieved_true_links"], 3)
        self.assertEqual(arena["name_only_changes"]["added"], 2)
        self.assertIn("empty", test["one_owner"]["US"]["accepted_by_address_kind"])


class LabelTests(unittest.TestCase):
    def test_labels_preserve_unretrieved_truth_and_skip_excluded_payload(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            data, raw, out = root / "data", root / "raw", root / "labels"
            data.mkdir(), raw.mkdir()
            pd.DataFrame({"entity_id": ["S1-a", "S1-b"]}).to_parquet(data / "reference.parquet", index=False)
            pd.DataFrame({"entity_id": ["S2-a", "S2-x"]}).to_parquet(data / "pool.parquet", index=False)
            pq.write_table(pa.table({"left_idx": np.array([0], dtype=np.int32),
                                     "right_idx": np.array([0], dtype=np.int32)}), data / "pairs.parquet")
            (raw / "train_ground_truth.tsv").write_text(
                "source1_entity_id\tmatched_entity_ids\n"
                "S1-a\tS2-a,S3-missed\nS1-sealed\tS2-x\nS1-b\t\n", encoding="utf-8")
            result = labels(SimpleNamespace(data=data, raw=raw, out=out))
            np.testing.assert_array_equal(np.load(out / "true_count.npy"), [2, 0])
            np.testing.assert_array_equal(np.load(out / "label.npy"), [True])
            np.testing.assert_array_equal(np.load(out / "rec_owner.npy"), [0, -1])
            self.assertEqual(result["true_links"], 2)
            self.assertEqual(result["holdout2_rows_read"], 0)


class FinalizeTests(unittest.TestCase):
    def test_finalize_writes_atomic_arrays_and_hash_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            data, labels, test_data, out = (root / name for name in ("arena", "labels", "test", "out"))
            for directory in (data, labels, test_data):
                directory.mkdir()
            reference = pd.DataFrame({
                "entity_id": [f"S1-{i}" for i in range(8)],
                "business_name": [f"Name {i}" for i in range(8)],
                "business_address": [f"{i} Road" for i in range(8)],
                "country": ["US"] * 4 + ["India"] * 4,
            })
            pool = pd.DataFrame({
                "entity_id": [f"S2-{i}" for i in range(8)],
                "business_name": [f"Name {i}" for i in range(8)],
                "business_address": ["" if i % 2 == 0 else f"{i} Road" for i in range(8)],
                "country": ["US"] * 4 + ["India"] * 4,
            })
            pairs = pa.table({"left_idx": np.arange(8, dtype=np.int32),
                              "right_idx": np.arange(8, dtype=np.int32)})
            reference.to_parquet(data / "reference.parquet", index=False)
            pool.to_parquet(data / "pool.parquet", index=False)
            pq.write_table(pairs, data / "pairs.parquet")
            manifest = {"labels_read": False, "holdout2_excluded": True,
                        "sha256": {name: sha256(data / name) for name in
                                   ("reference.parquet", "pool.parquet", "pairs.parquet")}}
            (data / "manifest.json").write_text(json.dumps(manifest))
            np.save(labels / "label.npy", np.ones(8, dtype=bool))
            np.save(labels / "true_count.npy", np.ones(8, dtype=np.int16))
            (labels / "labels_summary.json").write_text(json.dumps({"holdout2_rows_read": 0}))
            arena_scores = _score_artifact(root / "arena_scores", data, pairs, 8, 8)

            reference.to_parquet(test_data / "reference.parquet", index=False)
            pool.to_parquet(test_data / "pool.parquet", index=False)
            test_scores = _score_artifact(root / "test_scores", test_data, pairs, 8, 8)
            quick = pd.DataFrame({"entity_id": ["S1-0"], "split": ["train"]})
            medium = pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"],
                                   "split": ["train", "tune", "eval"]})
            quick_path, medium_path = root / "quick.parquet", root / "medium.parquet"
            quick.to_parquet(quick_path, index=False)
            medium.to_parquet(medium_path, index=False)

            result = finalize(SimpleNamespace(
                data=data, labels=labels, blend=arena_scores.blend, tree=arena_scores.tree,
                score_complete=arena_scores.complete, score_hashes=arena_scores.hashes,
                quick_reference=quick_path, medium_reference=medium_path, test_data=test_data,
                test_pairs=test_scores.pairs, test_blend=test_scores.blend, test_tree=test_scores.tree,
                test_score_complete=test_scores.complete, test_score_hashes=test_scores.hashes,
                out=out, component_threshold=0.05, max_component_fraction=0.2, threshold_step=0.05,
                max_component_threshold=1.0, accept_threshold=0.775, a_fraction=0.5,
                slice_size=2, seed=11))
            component = np.load(out / "component_ids.npy")
            partition = np.load(out / "s1_partition.npy")
            self.assertEqual(len(component), 8)
            self.assertEqual(int((partition == PARTITION_SLICE).sum()), 2)
            self.assertEqual(partition[0], PARTITION_A)
            self.assertEqual(partition[1], PARTITION_A)
            self.assertIn(partition[2], (PARTITION_B_REST, PARTITION_SLICE))
            self.assertIn(partition[3], (PARTITION_B_REST, PARTITION_SLICE))
            self.assertEqual(result["outputs"]["component_ids.npy"]["sha256"], sha256(out / "component_ids.npy"))
            fidelity = json.loads((out / "fidelity.json").read_text())
            self.assertFalse(fidelity["test"]["labels_used"])
            self.assertTrue(fidelity["arena"]["labels_used"])
            self.assertFalse(any(path.name.startswith(".") for path in out.iterdir()))

    def test_finalize_refuses_nonempty_output_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            out = Path(tmp_name) / "out"
            out.mkdir()
            sentinel = out / "keep.txt"
            sentinel.write_text("prior evidence")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                finalize(SimpleNamespace(data="missing", labels="missing", out=out))
            self.assertEqual(sentinel.read_text(), "prior evidence")

    def test_finalize_rejects_incomplete_score_provenance_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            data, labels, test_data = (root / name for name in ("arena", "labels", "test"))
            for directory in (data, labels, test_data):
                directory.mkdir()
            frame = pd.DataFrame({"entity_id": ["S1-0"], "business_name": ["A"],
                                  "business_address": ["1 A"], "country": ["US"]})
            pool = frame.assign(entity_id=["S2-0"])
            pairs = pa.table({"left_idx": np.array([0], dtype=np.int32),
                              "right_idx": np.array([0], dtype=np.int32)})
            for directory in (data, test_data):
                frame.to_parquet(directory / "reference.parquet", index=False)
                pool.to_parquet(directory / "pool.parquet", index=False)
            pq.write_table(pairs, data / "pairs.parquet")
            score = _score_artifact(root / "scores", data, pairs, 1, 1)
            score.complete.write_text(json.dumps({"state": "running", "report": {}}))
            out = root / "fresh-output"
            with self.assertRaisesRegex(ValueError, "final complete.json"):
                finalize(SimpleNamespace(
                    data=data, labels=labels, blend=score.blend, tree=score.tree,
                    score_complete=score.complete, score_hashes=score.hashes, out=out))
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
