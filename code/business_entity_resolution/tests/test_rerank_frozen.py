import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ber.cross_encoder_experiment import blend_probabilities
from ber.full_embeddings import checkpoint_identity, file_digest
from ber import rerank_frozen as frozen


class FrozenRerankerTests(unittest.TestCase):
    def fixture(self, root, *, empty=False):
        data, tree, ce = [root / name for name in ("data", "tree", "ce")]
        for path in (data, tree, ce):
            path.mkdir()
        reference = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(4)], "business_name": list("ABCD"),
            "business_address": ["1", "2", "3", "4"], "country": ["India"]*4})
        pool = pd.DataFrame({"entity_id": [f"S2-{i}" for i in range(3)], "business_name": list("XYZ"),
            "business_address": ["5", "6", "7"], "country": ["India"]*3})
        pairs = pd.DataFrame({"left_idx": pd.Series([0, 0, 2, 2], dtype="int64"),
                              "right_idx": pd.Series([0, 2, 0, 1], dtype="int64")})
        base = np.array([.9, .005, .6, -1], dtype=np.float32)
        if empty:
            pairs, base = pairs.iloc[:0], base[:0]
        reference.to_parquet(data / "reference.parquet", index=False)
        pool.to_parquet(data / "pool.parquet", index=False)
        pairs.to_parquet(tree / "scored_pairs.parquet", index=False, row_group_size=1)
        np.save(tree / "probabilities.npy", base)
        with (tree / "candidate_pairs.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            for i, entity in enumerate(reference.entity_id):
                writer.writerow([entity, ",".join(sorted(pool.entity_id.iloc[pairs.loc[pairs.left_idx == i, "right_idx"]]))])
        (root / "tree.bin").write_bytes(b"frozen tree")
        (ce / "model.safetensors").write_bytes(b"frozen CE")
        config = {"tree_checkpoint_sha256": file_digest(root / "tree.bin"), "cross_encoder_checkpoint": str(ce),
            "cross_encoder_sha256": checkpoint_identity(ce)["sha256"], "alpha": .5, "gate": .01,
            "threshold": .6, "max_length": 160, "inference_batch_size": 4}
        (root / "pipeline.json").write_text(json.dumps(config))
        report = {"checkpoint": str(root / "tree.bin"), "checkpoint_sha256": config["tree_checkpoint_sha256"],
            "data": str(data), "pair_order": ["left_idx", "right_idx"], "reference_count": 4,
            "candidate_pairs": len(pairs), "benchmark_only": False}
        (tree / "report.json").write_text(json.dumps(report))
        (tree / "complete.json").write_text(json.dumps({"state": "complete", "report": report}))
        args = SimpleNamespace(data=data, tree_predictions=tree, pipeline=root / "pipeline.json", output=root / "output",
            truth=None, device="cpu", reference_batch_size=1, max_pairs_per_batch=20)
        return args, reference, pool, pairs, base, config

    @staticmethod
    def scorer(left, right, pairs):
        lookup = {("S1-0", "S2-0"): .2, ("S1-2", "S2-0"): .8}
        return np.array([lookup[left.entity_id.iat[l], right.entity_id.iat[r]]
                         for l, r in pairs.itertuples(index=False, name=None)], dtype=np.float32)

    def test_exact_blend_gating_compact_alignment_and_all_reference_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args, reference, pool, pairs, base, config = self.fixture(Path(directory))
            expected = blend_probabilities(base, np.array([.2, 0, .8, 0], np.float32), base >= .01, .5)
            for alpha in (0., .25, .5, 1.):
                result, cross, selected = frozen.rerank_batch(reference, pool, pairs, base, {**config, "alpha": alpha}, self.scorer)
                np.testing.assert_array_equal(result, blend_probabilities(base, cross, selected, alpha))
            report = frozen.run(args, scorer=self.scorer)
            np.testing.assert_array_equal(np.load(args.output / "probabilities.npy"), expected)
            self.assertEqual(report["reranked_pairs"], 2)
            self.assertEqual((args.output / "candidate_pairs.tsv").read_bytes(), (args.tree_predictions / "candidate_pairs.tsv").read_bytes())
            self.assertEqual((args.output / "matching_results.tsv").read_text().splitlines()[1:],
                             ["S1-0\t", "S1-1\t", "S1-2\tS2-0", "S1-3\t"])

    def test_no_candidates_scores_every_reference_after_predictions_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            args, *unused = self.fixture(Path(directory), empty=True)
            args.truth = Path(directory) / "truth.json"
            args.truth.write_text(json.dumps({f"S1-{i}": (["S2-0"] if i == 0 else []) for i in range(4)}))
            original = frozen.score_fixed_outputs
            def after_fixed(*values):
                self.assertTrue((args.output / "predictions_fixed.json").exists())
                self.assertFalse(json.loads((args.output / "predictions_fixed.json").read_text())["labels_accessed"])
                return original(*values)
            with patch.object(frozen, "score_fixed_outputs", side_effect=after_fixed), patch.object(self, "scorer") as scorer:
                report = frozen.run(args, scorer=scorer)
                scorer.assert_not_called()
            self.assertEqual(report["score"]["reference_count"], 4)
            self.assertEqual(report["score"]["macro_f05"], .75)
            self.assertEqual(report["score"]["macro_f1"], .75)
            self.assertIn("uncertainty", report["score"])

    def test_unsorted_right_indices_are_rejected_without_reordering_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            args, reference, pool, pairs, base, config = self.fixture(Path(directory))
            pairs.iloc[[1, 0, 2, 3]].to_parquet(args.tree_predictions / "scored_pairs.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "right_idx"):
                frozen.run(args, scorer=self.scorer)
            self.assertFalse((args.output / "complete.json").exists())

    def test_bad_probability_alignment_or_checkpoint_hash_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            args, reference, pool, pairs, base, config = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "align"):
                frozen.rerank_batch(reference, pool, pairs, base[:-1], config, self.scorer)
            (Path(config["cross_encoder_checkpoint"]) / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest"):
                frozen.run(args, scorer=self.scorer)

    def test_candidate_tsv_alignment_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            args, *unused = self.fixture(Path(directory))
            path = args.tree_predictions / "candidate_pairs.tsv"
            path.write_text(path.read_text().replace("S2-0,S2-2", "S2-0"))
            with self.assertRaisesRegex(ValueError, "alignment"):
                frozen.run(args, scorer=self.scorer)


if __name__ == "__main__":
    unittest.main()
