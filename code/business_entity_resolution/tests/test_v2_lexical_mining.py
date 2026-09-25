"""Lossless canonicalization and label-free full-pool retrieval invariants."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ber.v2_lexical_mining import (FullPoolIndex, build_pool_index, ids_digest,
                                   mine_reference, seal_completed_prefix, assemble_shards,
                                   publish_assembled_prefix, wait_and_publish,
                                   text_keys)
from ber.v2_decision import composite_key
from ber.phonetic_ablation import phonetic_text


class LexicalMiningTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.s2 = pd.DataFrame([
            ("S2-01", "Acme Robotics Inc", "Unit 9, 500 Market St", "US"),
            ("S2-02", "Acme Robotics Inc", "Unit 9, 500 Market St", "US"),
            ("S2-03", "Acme Robotics", "Unit 10, 500 Market St", "US"),
            ("S2-04", "Acme Robotics", "500 Market St Unit 9", "US"),
            ("S2-05", "Acme Robotics Inc", "Unit 9, 500 Market St", "India"),
        ], columns=["entity_id", "business_name", "business_address", "country"])
        self.s3 = pd.DataFrame([
            ("S3-01", "Acme Robotics Inc", "Unit 9, 500 Market St", "US"),
            ("S3-02", "Acme Robots", "9, 500 Market St", "US"),
            ("S3-03", "Ācme Robotics Ltd", "500 Market St Unit 9", "US"),
            ("S3-04", "Something", "", "US"),
            ("S3-05", "Acme Robotics Inc", "UNIT 9, 500 MARKET ST", "US"),
        ], columns=self.s2.columns)
        self.s2.to_csv(self.raw / "train_source2.tsv", sep="\t", index=False)
        self.s3.to_csv(self.raw / "train_source3.tsv", sep="\t", index=False)
        self.refs = pd.DataFrame([
            ("S1-a", "Acme Robotics LLC", "Unit 9, 500 Market St", "US", "tune"),
            ("S1-b", "No Business", "", "US", "tune"),
            ("S1-c", "Acme Robotics", "Unit 9, 500 Market St", "India", "tune"),
        ], columns=["entity_id", "business_name", "business_address", "country", "split"])
        self.ref_file = self.root / "reference.parquet"
        self.refs.to_parquet(self.ref_file, index=False)
        self.index_path = self.root / "index"
        self.index_report = build_pool_index(self.raw, self.index_path, chunksize=2, expected_rows=(5, 5))

    def test_keys_agree_with_decision_module(self):
        for name, address, country in zip(self.refs.business_name, self.refs.business_address, self.refs.country):
            keys = text_keys(name, address, country)
            decision = composite_key(name, address, country)
            self.assertEqual((country, keys.core, tuple(keys.digit_seq.split("|"))) if keys.core and keys.digit_seq else None,
                             decision)
            self.assertEqual(keys.phonetic, phonetic_text(name, "harmonized_core"))
        self.assertEqual(text_keys("Acme", "9 - 500 - 9", "US").digit_set, "500|9")

    def test_exact_canonical_groups_expand_all_original_source_ids(self):
        self.assertEqual(self.index_report["rows"], 10)
        self.assertEqual(self.index_report["duplicate_rows"], 2)
        index = FullPoolIndex(self.index_path, verify_hashes=True)
        self.addCleanup(index.close)
        self.assertEqual(index.expand(0).tolist(), [0, 1, 5])
        self.assertEqual(index.expand(1).tolist(), [0, 1, 5])
        self.assertEqual(index.expand(4).tolist(), [4])  # India is a different record
        self.assertEqual(index.expand(9).tolist(), [9])  # Raw case is *not* canonicalized
        ids = index.lookup_rows([0, 1, 5, 9])
        self.assertEqual([ids[x][1] for x in [0, 1, 5]], ["S2-01", "S2-02", "S3-01"])
        self.assertEqual(int(index.lookup_ids(["S3-01"])["S3-01"][0]), 5)
        self.assertTrue(np.array_equal(np.sort(index.members), np.arange(10)))

    def test_composite_and_digit_set_expansion_before_any_labels(self):
        index = FullPoolIndex(self.index_path)
        self.addCleanup(index.close)
        rows, stats = index.search("Acme Robotics LLC", "Unit 9, 500 Market St", "US", max_scan=20)
        ids = {raw: value[1] for raw, value in index.lookup_rows([row["right_raw_idx"] for row in rows]).items()}
        self.assertEqual(len(rows), len(ids))
        self.assertTrue({"S2-01", "S2-02", "S3-01"} <= set(ids.values()))
        for row in rows:
            if ids[row["right_raw_idx"]] in {"S2-01", "S2-02", "S3-01"}:
                self.assertTrue(row["composite_key_hit"])
                self.assertEqual(row["canonical_group_size"], 3)
        self.assertTrue(any(row["digit_set_hit"] and ids[row["right_raw_idx"]] == "S2-04" for row in rows))
        self.assertFalse(any(ids[row["right_raw_idx"]] == "S2-05" for row in rows))
        self.assertGreaterEqual(stats["duplicate_expansion_pairs"], 2)

    def test_mining_saves_all_references_and_never_opens_truth(self):
        output = self.root / "candidates"
        report = mine_reference(self.ref_file, self.index_path, output, batch_size=2, max_scan=4,
                                expected_id_sha256=ids_digest(self.refs.entity_id))
        self.assertEqual(report["references"], 3)
        self.assertFalse(report["truth_read"])
        self.assertFalse(report["positive_candidate_injection"])
        self.assertEqual(len(pd.read_parquet(output / "reference.parquet")), 3)
        pairs = pd.read_parquet(output / "candidate_pairs.parquet")
        self.assertEqual(len(pairs), report["pairs"])
        self.assertFalse(pairs.duplicated(["left_idx", "right_raw_idx"]).any())
        self.assertTrue(set(pairs.loc[pairs.left_idx.eq(2), "right_raw_idx"]) <= {4})
        self.assertEqual(json.loads((output / "mining_progress.json").read_text())["references_done"], 3)

    def test_disallows_eval_reference_and_stale_index(self):
        with self.assertRaisesRegex(ValueError, "TRAIN/TUNE"):
            mine_reference(self.ref_file, self.index_path, self.root / "out", splits=("eval",))
        self.s2.loc[0, "business_name"] = "Modified"
        self.s2.to_csv(self.raw / "train_source2.tsv", sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "different full-pool identity"):
            build_pool_index(self.raw, self.index_path, chunksize=2, expected_rows=(5, 5))

    def test_shared_population_can_mine_fit_eligible_without_holdout(self):
        shared = self.refs.drop(columns="split").assign(
            population=["train_tune_eligible", "holdout2", "train_tune_eligible"])
        path = self.root / "shared.parquet"
        shared.to_parquet(path, index=False)
        output = self.root / "full"
        fit_ids = shared.entity_id[shared.population.eq("train_tune_eligible")]
        report = mine_reference(path, self.index_path, output,
                                splits=("train_tune_eligible",), expected_id_sha256=ids_digest(fit_ids),
                                batch_size=2, max_scan=20)
        self.assertEqual(report["references"], 2)
        self.assertEqual(report["partition_column"], "population")
        self.assertNotIn("S1-b", pd.read_parquet(output / "reference.parquet").entity_id.tolist())
        with self.assertRaisesRegex(ValueError, "TRAIN/TUNE"):
            mine_reference(path, self.index_path, self.root / "forbidden", splits=("holdout2",),
                           expected_id_sha256=ids_digest(["S1-b"]))
        holdout = mine_reference(path, self.index_path, self.root / "holdout_text_only",
                                 splits=("holdout2",), expected_id_sha256=ids_digest(["S1-b"]),
                                 allow_holdout2_unlabeled=True, batch_size=1, max_scan=20)
        self.assertEqual(holdout["references"], 1)
        self.assertTrue(holdout["holdout2_text_only"])
        self.assertFalse(holdout["truth_read"])
        self.assertEqual(holdout["pair_file"]["type"], "parquet_partition_directory")
        self.assertEqual(pd.read_parquet(self.root / "holdout_text_only" / "reference.parquet").entity_id.tolist(), ["S1-b"])

    def test_disjoint_row_shard_emits_parent_global_left_index(self):
        shared = self.refs.drop(columns="split").assign(population="train_tune_eligible")
        path = self.root / "shared_fit.parquet"
        shared.to_parquet(path, index=False)
        digest = ids_digest(shared.entity_id)
        shard = mine_reference(path, self.index_path, self.root / "shard",
                               splits=("train_tune_eligible",), expected_id_sha256=digest,
                               start_row=2, stop_row=3, batch_size=1, max_scan=20)
        self.assertEqual((shard["left_offset"], shard["left_stop"], shard["parent_references"]), (2, 3, 3))
        self.assertEqual(shard["parent_reference_ids_sha256"], digest)
        self.assertEqual(shard["reference_ids_sha256"], ids_digest(["S1-c"]))
        self.assertEqual(set(pd.read_parquet(self.root / "shard" / "candidate_pairs.parquet").left_idx), {2})
        with self.assertRaisesRegex(ValueError, "Row shards"):
            mine_reference(self.ref_file, self.index_path, self.root / "bad_shard", start_row=1)

    def test_full_mining_resumes_only_completed_entity_partitions(self):
        output = self.root / "partitioned"
        original = FullPoolIndex.search
        calls = 0

        def interrupted(instance, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption")
            return original(instance, *args, **kwargs)

        with patch.object(FullPoolIndex, "search", interrupted):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                mine_reference(self.ref_file, self.index_path, output,
                               expected_id_sha256=ids_digest(self.refs.entity_id),
                               partitioned=True, batch_size=1, max_scan=20)
        first = json.loads((output / "mining_progress.json").read_text())
        self.assertEqual(first["references_done"], 1)
        self.assertEqual(len(first["parts"]), 1)
        result = mine_reference(self.ref_file, self.index_path, output,
                                expected_id_sha256=ids_digest(self.refs.entity_id),
                                partitioned=True, batch_size=1, max_scan=20)
        self.assertEqual(result["pair_file"]["type"], "parquet_partition_directory")
        self.assertEqual(len(result["pair_file"]["parts"]), 3)
        pairs = pd.read_parquet(output / "candidate_pairs.parquet")
        self.assertEqual(len(pairs), result["pairs"])
        self.assertFalse(pairs.duplicated(["left_idx", "right_raw_idx"]).any())

    def test_sealed_prefix_and_shard_assemble_identically_to_unsharded_mining(self):
        shared = self.refs.drop(columns="split").assign(population="train_tune_eligible")
        path = self.root / "fit_population.parquet"
        shared.to_parquet(path, index=False)
        digest = ids_digest(shared.entity_id)
        prefix = self.root / "prefix"
        original = FullPoolIndex.search
        calls = 0

        def stop_after_committed_part(instance, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("stop at saved prefix")
            return original(instance, *args, **kwargs)

        with patch.object(FullPoolIndex, "search", stop_after_committed_part):
            with self.assertRaisesRegex(RuntimeError, "saved prefix"):
                mine_reference(path, self.index_path, prefix, splits=("train_tune_eligible",),
                               expected_id_sha256=digest, batch_size=1, max_scan=20, partitioned=True)
        seal = seal_completed_prefix(prefix)
        self.assertEqual((seal["references_done"], seal["full_pool_rows"]), (1, 10))
        with self.assertRaisesRegex(FileExistsError, "sealed"):
            mine_reference(path, self.index_path, prefix, splits=("train_tune_eligible",),
                           expected_id_sha256=digest, batch_size=1, max_scan=20, partitioned=True)
        shard = self.root / "shard_after_prefix"
        mine_reference(prefix / "reference.parquet", self.index_path, shard,
                       splits=("train_tune_eligible",), expected_id_sha256=digest,
                       start_row=1, stop_row=3, batch_size=1, max_scan=20, partitioned=True)
        assembled = self.root / "assembled"
        result = assemble_shards(prefix, [shard], assembled)
        self.assertEqual(result["references"], 3)
        self.assertEqual(result["full_pool_rows"], 10)
        self.assertEqual([part["start"] for part in result["pair_partitions"]], [0, 1, 2])
        self.assertTrue(result["holdout2_excluded_from_training_references"])
        monolithic = self.root / "monolithic"
        mine_reference(path, self.index_path, monolithic, splits=("train_tune_eligible",),
                       expected_id_sha256=digest, batch_size=1, max_scan=20, partitioned=True)
        cols = ["left_idx", "right_raw_idx", "composite_key_hit", "digit_set_hit"]
        actual = pd.read_parquet(assembled / "candidate_pairs.parquet", columns=cols)
        expected = pd.read_parquet(monolithic / "candidate_pairs.parquet", columns=cols)
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
        self.assertEqual(result["candidate_pairs"], len(actual))
        with self.assertRaisesRegex(ValueError, "overlap or leave uncovered"):
            assemble_shards(prefix, [shard, shard], self.root / "overlap")
        published = publish_assembled_prefix(prefix, assembled)
        self.assertEqual(published["pairs"], len(actual))
        self.assertEqual(len(json.loads((prefix / "mining_progress.json").read_text())["parts"]), 3)
        self.assertTrue((prefix / "candidate_pairs.parquet" / "part_000001.parquet").samefile(
            assembled / "candidate_pairs.parquet" / "part_000001.parquet"))
        pd.testing.assert_frame_equal(
            pd.read_parquet(prefix / "candidate_pairs.parquet", columns=cols).reset_index(drop=True),
            expected.reset_index(drop=True))
        self.assertEqual(publish_assembled_prefix(prefix, assembled), published)
        replay = wait_and_publish(prefix, [shard], assembled, worker_pids=[99999999],
                                  poll_seconds=.01, max_wait_seconds=1)
        self.assertEqual(replay["candidate_pairs"], len(actual))
        self.assertEqual(Path(replay["p2_5_source_manifest"]), prefix / "complete.json")


if __name__ == "__main__":
    unittest.main()
