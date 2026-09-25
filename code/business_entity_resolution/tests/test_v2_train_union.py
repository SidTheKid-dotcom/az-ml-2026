"""P2.5 raw-row union, holdout-safe truth, and exact P3 handoff contracts."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from ber import v2_train_union as p2
from ber.v2_ce_train import validate_ready
from ber.v2_tree_train import (feature_columns, load_p2_ready, pair_labels,
                               prepare_exact_features, union_feature_columns)


class FullPoolJoinTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.durable = self.root / "durable"
        self.durable.mkdir()
        self.stage = self.root / "staged" / "candidate_pairs.parquet"
        self.stage.mkdir(parents=True)
        self.raw_dir = self.root / "raw_input"
        self.raw_dir.mkdir()
        self.patches = [patch.object(p2, name, value) for name, value in {
            "POOL_ROWS": 52, "SOURCE_ROWS": (51, 1), "TRAIN_REFERENCES": 2,
            "SHARD_REFERENCES": 2, "TRUTH_ROWS": 3,
            "TRAIN_IDS_SHA256": p2.ids_sha256(["S1-a", "S1-b"])}.items()]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.references = pd.DataFrame({
            "entity_id": ["S1-a", "S1-b"], "business_name": ["Oasis", "River"],
            "business_address": ["11 Main Street", "22 Main Street"],
            "country": ["US", "US"], "split": ["train", "train"]})
        self.pool = pd.DataFrame({
            "entity_id": [*(f"S2-{i}" for i in range(51)), "S3-51"],
            "business_name": ["Oasis"] * 25 + ["River"] * 25 + ["Oasis", "Missed"],
            "business_address": ["11 Main Street"] * 25 + ["22 Main Street"] * 25 +
                                ["11 Main Street", "99 Main Street"],
            "country": ["US"] * 52})
        self.pool.to_parquet(self.durable / "pool.parquet", index=False)
        (self.durable / "full").mkdir()
        self.references.to_parquet(self.durable / "full" / "reference.parquet", index=False)
        key_schema = pa.schema([("core", pa.string()), ("digit_set", pa.string())])
        with pa.OSFile(str(self.durable / "pool_keys.arrow"), "wb") as sink, ipc.new_file(sink, key_schema) as writer:
            pairs = [p2._text_rank_keys(name, address) for name, address in
                     self.pool[["business_name", "business_address"]].itertuples(index=False, name=None)]
            writer.write_batch(pa.RecordBatch.from_arrays([pa.array([row[i] for row in pairs])
                                                           for i in (0, 1)], schema=key_schema))
        vectors = np.zeros((52, 2), dtype=np.float16)
        vectors[:25, 0] = 1
        vectors[25:50, 1] = 1
        vectors[50] = [.8, .6]
        vectors[51] = [.6, .8]
        queries = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        dense_rows = np.vstack((np.arange(50), np.r_[np.arange(25, 50), np.arange(25)]))
        dense_scores = np.einsum("ijk,ik->ij", vectors[dense_rows].astype(np.float32), queries)
        self.inputs = SimpleNamespace(
            durable=self.durable, staged=self.stage.parent, reference=self.references,
            mapping=np.arange(2, dtype=np.int32), raw=dense_rows.astype(np.int32),
            scores=dense_scores.astype(np.float32), queries=queries, vectors=vectors,
            countries=np.zeros(52, dtype=np.int32), query_country=np.zeros(2, dtype=np.int8),
            bootstrap={"dense_complete_sha256": "d" * 64, "dense_candidate_pairs_sha256": "e" * 64,
                       "lexical_reference_sha256": "l" * 64, "index_manifest_sha256": "i" * 64,
                       "reservation_sha256": "r" * 64,
                       "files": {"fit_to_dense": {"sha256": "m" * 64}}})
        key = pd.DataFrame({
            "left_idx": [0, 0, 1], "right_raw_idx": [0, 50, 50],
            "canonical_raw_idx": [0, 50, 50], "canonical_group_size": [1, 1, 1],
            "lexical_key_score": [.95, .9, .8], "name_exact_hit": [True, True, False],
            "phonetic_hit": [False, True, True], "digit_sequence_hit": [True] * 3,
            "digit_set_hit": [True] * 3, "composite_key_hit": [True, True, False]})
        key_path = self.stage / "part_000000.parquet"
        key.to_parquet(key_path, index=False)
        self.part = {"name": key_path.name, "start": 0, "end": 2,
                     "rows": len(key), "bytes": key_path.stat().st_size,
                     "sha256": p2.file_sha256(key_path)}
        # The invalid byte is the held-out label value. Parsing it would fail;
        # only its S1 ID may be read, and it cannot enter any selected truth set.
        (self.raw_dir / "train_ground_truth.tsv").write_bytes(
            b"source1_entity_id\tmatched_entity_ids\n"
            b"S1-holdout2\t\xff\nS1-b\t\nS1-a\tS2-50,S3-51\n")

    def _fixed_raw(self):
        raw = p2.union_shard(self.inputs, 0, 2, [self.part])
        output = self.durable / "screen"
        info = p2._write_parquet(raw, output / "raw" / "part_000000.parquet")
        p2.atomic_json(output / "raw_progress.json", {
            "state": "complete", "identity": p2._raw_identity(self.inputs),
            "references_done": 2, "truth_read": False, "positive_candidate_injection": False,
            "parts": [{"start": 0, "end": 2, "file": {**info, "path": "part_000000.parquet"}}]})
        return raw, output

    def test_raw_union_maps_original_target_rows_and_scores_new_only(self):
        raw = p2.union_shard(self.inputs, 0, 2, [self.part])
        self.assertEqual(len(raw), 102)  # 50 dense for each S1 + one new key hit each.
        self.assertFalse(raw.duplicated(["left_idx", "right_idx"]).any())
        self.assertTrue(raw.lexical_scores_missing.all())
        self.assertEqual(raw.dense_hit.sum(), 100)
        self.assertEqual(raw.loc[(raw.left_idx == 0) & (raw.right_idx == 0), "dense_hit"].tolist(), [True])
        new = raw.loc[(raw.left_idx == 0) & (raw.right_idx == 50)].iloc[0]
        self.assertFalse(new.dense_hit)
        self.assertEqual(int(new.dense_rank), 0)
        self.assertAlmostEqual(float(new.dense_score), .8, places=3)
        self.assertEqual(int(new.right_raw_idx), 50)
        self.assertEqual(feature_columns(raw.columns), feature_columns(
            list(raw.columns) + ["label", "hard_negative_priority"]))

    def test_rejects_duplicate_or_cross_country_before_join(self):
        key_path = self.stage / self.part["name"]
        repeated = pd.read_parquet(key_path)
        pd.concat([repeated, repeated.iloc[:1]], ignore_index=True).to_parquet(key_path, index=False)
        corrupt = {**self.part, "bytes": key_path.stat().st_size,
                   "sha256": p2.file_sha256(key_path), "rows": 4}
        with self.assertRaisesRegex(ValueError, "duplicat"):
            p2.union_shard(self.inputs, 0, 2, [corrupt])
        repeated.to_parquet(key_path, index=False)
        self.inputs.countries[50] = 1
        with self.assertRaisesRegex(ValueError, "country"):
            p2.union_shard(self.inputs, 0, 2, [self.part])

    def test_near_twin_ranks_ahead_of_higher_cosine_and_positive_is_never_capped(self):
        raw = p2.union_shard(self.inputs, 0, 2, [self.part])
        candidates = raw.loc[raw.left_idx.eq(0) & raw.right_idx.isin([1, 25, 50])].copy()
        candidates.loc[candidates.right_idx.eq(1), "dense_score"] = .10
        candidates.loc[candidates.right_idx.eq(25), "dense_score"] = .99
        target_keys = ipc.open_file(pa.memory_map(str(self.durable / "pool_keys.arrow"), "r")).read_all()
        truth = p2._positional_keys(np.array([0], dtype=np.int32), np.array([50], dtype=np.int32))
        compact, stats = p2.compact_shard(candidates, self.references, target_keys, truth,
                                          start=0, end=1, max_negatives=1)
        self.assertEqual(compact.right_idx.tolist(), [1, 50])
        self.assertEqual(compact.label.tolist(), [False, True])
        self.assertEqual(compact.hard_negative_priority.tolist(), [0, -1])
        self.assertEqual(stats["same_name_same_digit_negatives"], 1)
        self.assertEqual(stats["retrieved_true_links"], 1)

    def test_truth_is_gated_until_raw_pairs_fixed(self):
        _, output = self._fixed_raw()
        state = json.loads((output / "raw_progress.json").read_text())
        state["state"] = "merging"
        p2.atomic_json(output / "raw_progress.json", state)
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            p2.selected_truth(self.inputs, self.raw_dir, output, references=2)
        self.assertFalse((output / "train_truth_pairs.parquet").exists())
        # A crash after the last immutable shard but before the completion
        # marker is recoverable without rebuilding or changing candidate edges.
        repaired = p2.produce_raw(self.inputs, output, screen=True)
        self.assertEqual(repaired["state"], "complete")
        self.assertEqual(repaired["raw_candidate_pairs"], 102)

    def test_immutable_full_raw_seal_survives_stale_geesefs_progress(self):
        _, output = self._fixed_raw()
        stage = self.stage.parent
        self.references.assign(population="train_tune_eligible").to_parquet(
            stage / "reference.parquet", index=False)
        (stage / "source_index_complete.json").write_text('{"rows": 52}', encoding="utf-8")
        reference_sha = p2.file_sha256(stage / "reference.parquet")
        index_sha = p2.file_sha256(stage / "source_index_complete.json")
        p2.atomic_json(stage / "complete.json", {
            "state": "complete", "references": 2, "pairs": 3, "truth_read": False,
            "pair_file": {"parts": [self.part]}})
        p2.atomic_json(stage / "staging_manifest.json", {
            "state": "staging", "references_total": 2, "references_staged": 2,
            "train_reference_id_sha256": p2.ids_sha256(self.references.entity_id),
            "reference_sha256": reference_sha, "index_sha256": index_sha,
            "lexical_identity": {"reference_ids_sha256": p2.ids_sha256(self.references.entity_id),
                                 "splits": ["train_tune_eligible"],
                                 "partition_column": "population", "partitioned": True,
                                 "batch_size": 1000},
            "parts": [self.part],
            "source_complete": {"sha256": p2.file_sha256(stage / "complete.json"), "pairs": 3},
            "truth_read": False, "positive_candidate_injection": False})
        self.inputs.bootstrap["lexical_reference_sha256"] = reference_sha
        self.inputs.bootstrap["index_manifest_sha256"] = index_sha
        raw_path = output / "raw_progress.json"
        raw = json.loads(raw_path.read_text())
        raw["identity"] = p2._raw_identity(self.inputs)
        raw["state"] = "merging"  # This is the old mutable object geesefs may return.
        p2.atomic_json(raw_path, raw)
        with self.assertRaisesRegex(ValueError, "raw candidate set"):
            p2.selected_truth(self.inputs, self.raw_dir, output, references=2)
        _, seal = p2._sealed_raw(self.inputs, output, 2, screen=False, create=True)
        self.assertEqual(seal["state"], "complete")
        self.assertEqual(seal["raw_candidate_pairs"], 102)
        self.assertEqual(json.loads(raw_path.read_text())["state"], "merging")
        keys, counts, truth = p2.selected_truth(self.inputs, self.raw_dir, output, references=2)
        np.testing.assert_array_equal(counts, [2, 0])
        self.assertEqual(len(keys), 2)
        self.assertEqual(truth["raw_complete_sha256"], p2.file_sha256(output / "raw_complete.json"))
        # Completion is tied to the immutable part manifest, not the mutable
        # progress marker's potentially stale state/mtime.
        again, _, _ = p2.selected_truth(self.inputs, self.raw_dir, output, references=2)
        np.testing.assert_array_equal(again, keys)
        raw["parts"][0]["file"]["sha256"] = "0" * 64
        p2.atomic_json(raw_path, raw)
        with self.assertRaisesRegex(ValueError, "raw completion"):
            p2.selected_truth(self.inputs, self.raw_dir, output, references=2)

    def test_real_join_keeps_retrieved_positive_and_missed_record_for_p3(self):
        raw, output = self._fixed_raw()
        ready = p2.join_and_publish(self.inputs, self.raw_dir, output,
                                    screen=True, max_negatives=1)
        self.assertTrue((output / "ready.json").is_file())
        self.assertEqual(ready["files"]["reference"]["rows"], 2)
        self.assertEqual(ready["files"]["pool"]["rows"], 52)
        self.assertEqual(ready["files"]["train_truth_pairs"]["rows"], 2)
        self.assertEqual(ready["files"]["candidate_pairs"]["rows"], 3)
        self.assertEqual(ready["truth_links_not_in_candidate_pairs"], 1)
        self.assertEqual(ready["retrieval"]["retrieval"]["union"]["all"]["true_singletons"], 1)
        self.assertAlmostEqual(ready["retrieval"]["retrieval"]["union"]["all"]["oracle_macro_f05"],
                               (5 / 6 + 1) / 2)
        self.assertFalse(ready["positive_candidate_injection"])
        self.assertFalse(ready["truth_used_in_retrieval"])
        pairs = pd.read_parquet(output / "candidate_pairs.parquet")
        self.assertEqual(pairs.label.sum(), 1)
        self.assertTrue(pairs.loc[pairs.label, "right_idx"].eq(50).all())
        self.assertFalse(pairs.right_idx.eq(51).any())
        self.assertTrue(pairs.lexical_scores_missing.all())
        with patch("ber.v2_tree_train.FULL_POOL_ROWS", 52):
            loaded = load_p2_ready(output / "ready.json", self.root / "p3_cache", .01)
        np.testing.assert_array_equal(loaded.truth_counts, [2, 0])
        labels, target_ids = pair_labels(loaded, pairs)
        np.testing.assert_array_equal(labels, pairs.label.to_numpy())
        self.assertIn("S2-50", target_ids)
        self.assertNotIn("S3-51", target_ids)  # In full pool/truth, never injected as edge.
        self.assertEqual(len(loaded.pool), 52)
        self.assertEqual(loaded.features, feature_columns(pairs.columns))
        self.assertEqual(loaded.features, union_feature_columns(pairs.columns))
        self.assertEqual(loaded.pair_policy, "dense_key_shared")
        numeric = prepare_exact_features(loaded, 0, 2, pairs)
        self.assertEqual(list(numeric.columns), loaded.features)
        self.assertTrue(np.isfinite(numeric.to_numpy()).all())
        np.testing.assert_array_equal(numeric.retrieval_score, numeric.dense_score)
        self.assertTrue(numeric.lexical_scores_missing.eq(1).all())
        ce_ready, ce_files = validate_ready(output / "ready.json", full_pool_rows=52)
        self.assertEqual(ce_ready["train_reference_id_sha256"], p2.ids_sha256(["S1-a", "S1-b"]))
        self.assertEqual(ce_files["candidate_pairs"].resolve(), (output / "candidate_pairs.parquet").resolve())
        self.assertEqual(p2.join_and_publish(self.inputs, self.raw_dir, output,
                                              screen=True, max_negatives=1)["files"], ready["files"])

    def test_fixed_raw_country_error_rejected_before_truth_file_open(self):
        _, output = self._fixed_raw()
        self.inputs.countries[0] = 1
        with self.assertRaisesRegex(ValueError, "country"):
            p2.join_and_publish(self.inputs, self.raw_dir, output, screen=True, max_negatives=1)
        self.assertFalse((output / "truth_complete.json").exists())


class DisjointLexicalHandoffTests(unittest.TestCase):
    def test_sealed_prefix_stages_only_contiguous_global_raw_row_parts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "full_eligible_keys"
            shard = root / "full_eligible_shards" / "shard_1000_2501"
            for folder in (source, shard):
                (folder / "candidate_pairs.parquet").mkdir(parents=True)
            identifiers = [f"S1-{i}" for i in range(2501)]
            digest = p2.ids_sha256(identifiers)
            index_sha = "f" * 64
            references = pd.DataFrame({"entity_id": identifiers,
                                       "population": ["train_tune_eligible"] * 2501})
            references.to_parquet(source / "reference.parquet", index=False)
            references.iloc[1000:].to_parquet(shard / "reference.parquet", index=False)
            reference_sha = p2.file_sha256(source / "reference.parquet")
            prefix_file = source / "candidate_pairs.parquet/part_000000.parquet"
            pd.DataFrame({"left_idx": [0], "right_raw_idx": [0]}).to_parquet(prefix_file, index=False)
            def part(path: Path, start: int, end: int) -> dict:
                return {"name": path.name, "start": start, "end": end,
                        "rows": 1, "bytes": path.stat().st_size, "sha256": p2.file_sha256(path)}
            first = part(prefix_file, 0, 1000)
            prefix = {"state": "mining", "references_total": 2501,
                      "references_done": 1000, "pairs": 1, "parts": [first],
                      "reference_output_sha256": reference_sha,
                      "identity": {"reference_ids_sha256": digest,
                                   "index_manifest_sha256": index_sha,
                                   "splits": ["train_tune_eligible"],
                                   "partition_column": "population", "partitioned": True,
                                   "batch_size": 1000}}
            p2.atomic_json(source / "mining_progress.json", prefix)
            seal = {"state": "sealed_prefix", "references_total": 2501,
                    "references_done": 1000, "partition_count": 1,
                    "candidate_pairs": 1, "progress_sha256": p2.file_sha256(source / "mining_progress.json"),
                    "full_reference_sha256": reference_sha, "parent_reference_ids_sha256": digest,
                    "full_pool_index_manifest_sha256": index_sha, "full_pool_rows": 5,
                    "truth_read": False, "positive_candidate_injection": False}
            p2.atomic_json(source / "prefix_sealed.json", seal)
            shard_file = shard / "candidate_pairs.parquet/part_000000.parquet"
            pd.DataFrame({"left_idx": [1000], "right_raw_idx": [1]}).to_parquet(shard_file, index=False)
            state = {"state": "mining", "references_total": 1501,
                     "references_done": 1000, "pairs": 1, "parts": [part(shard_file, 0, 1000)],
                     "reference_output_sha256": p2.file_sha256(shard / "reference.parquet"),
                     "identity": {"parent_references": 2501,
                                  "parent_reference_ids_sha256": digest,
                                  "reference_file_sha256": reference_sha,
                                  "left_offset": 1000, "left_stop": 2501,
                                  "index_manifest_sha256": index_sha,
                                  "splits": ["train_tune_eligible"],
                                  "partitioned": True, "batch_size": 1000}}
            p2.atomic_json(shard / "mining_progress.json", state)
            with (patch.object(p2, "POOL_ROWS", 5),
                  patch.object(p2, "TRAIN_REFERENCES", 2501),
                  patch.object(p2, "TRAIN_IDS_SHA256", digest)):
                committed, paths = p2._committed_lexical_prefix(source, prefix, index_sha)
                self.assertEqual([(p["name"], p["start"], p["end"]) for p in committed],
                                 [("part_000000.parquet", 0, 1000),
                                  ("part_000001.parquet", 1000, 2000)])
                self.assertEqual(paths[committed[-1]["name"]], shard_file)
                state["identity"]["parent_reference_ids_sha256"] = "0" * 64
                p2.atomic_json(shard / "mining_progress.json", state)
                with self.assertRaisesRegex(ValueError, "sealed TRAIN-only source"):
                    p2._committed_lexical_prefix(source, prefix, index_sha)
                state["identity"]["parent_reference_ids_sha256"] = digest
                last_file = shard / "candidate_pairs.parquet/part_000001.parquet"
                pd.DataFrame({"left_idx": [2000], "right_raw_idx": [2]}).to_parquet(last_file, index=False)
                state["parts"].append(part(last_file, 1000, 1501))
                state["references_done"], state["pairs"] = 1501, 2
                p2.atomic_json(shard / "mining_progress.json", state)
                completed, _ = p2._committed_lexical_prefix(source, prefix, index_sha)
                self.assertEqual(completed[-1]["name"], "part_000002.parquet")
                self.assertEqual(completed[-1]["end"], 2501)
                # When the lexical owner publishes these very bytes globally,
                # the staged descriptors must be identical and never recopied.
                prefix.update(parts=completed, references_done=2501, pairs=3,
                              assembly_manifest_sha256="a" * 64)
                p2.atomic_json(source / "mining_progress.json", prefix)
                published, _ = p2._committed_lexical_prefix(source, prefix, index_sha)
                self.assertEqual(completed, published)


if __name__ == "__main__":
    unittest.main()
