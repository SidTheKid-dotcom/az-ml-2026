import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber import resolver as rs  # noqa: E402


def _pairs(path):
    table = pq.read_table(path)
    return table.column("left_idx").to_numpy(), table.column("right_idx").to_numpy()


class ResolverTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.paths = rs.make_fixture(self.root / "fixture")
        self.n_pairs = pq.ParquetFile(self.paths["pairs"]).metadata.num_rows
        self.partition = np.load(self.paths["partition"])

    def tearDown(self):
        shutil.rmtree(self.root)

    def _run(self, out="run", **kwargs):
        p = self.paths
        options = dict(model="logistic", iterations=200, threads=1, batch_rows=37, bootstrap=100,
                       reference=p["reference"])
        options.update(kwargs)
        return rs.run(p["pairs"], p["scores"], p["labels"], p["partition"], p["components"], self.root / out, **options)

    def _rewrite_pairs(self, left, right):
        pq.write_table(pa.table({"left_idx": left, "right_idx": right}), self.paths["pairs"], row_group_size=16)

    def test_stream_matches_full_read_across_batch_boundaries(self):
        left, right = _pairs(self.paths["pairs"])
        scores = rs.open_scores(self.paths["scores"], self.n_pairs)
        label = np.load(self.paths["labels"] / "label.npy")
        out = self.root / "stream"
        out.mkdir()
        stats = rs.stream_pairs(self.paths["pairs"], scores, label, len(self.partition), out, batch_rows=7)
        np.testing.assert_array_equal(np.load(out / "pairs_left_idx.npy"), left)
        np.testing.assert_array_equal(np.load(out / "pairs_right_idx.npy"), right)
        np.testing.assert_array_equal(stats["count"], np.bincount(left, minlength=len(self.partition)))
        np.testing.assert_array_equal(stats["retrieved"],
                                      np.bincount(left[label], minlength=len(self.partition)))
        blend = np.load(self.paths["scores"]["blend"])
        expected = np.full(len(self.partition), -np.inf, dtype=np.float32)
        np.maximum.at(expected, left, blend)
        np.testing.assert_array_equal(stats["s1_max"]["blend"], expected)
        self.assertEqual([p.name for p in out.iterdir() if p.name.startswith(".")], [])

    def test_logistic_run_preserves_every_s1_and_writes_aligned_outputs(self):
        meta = self._run()
        out = self.root / "run"
        prob = np.load(out / "prob.npy")
        accept = np.load(out / "accept.npy")
        self.assertEqual(prob.shape, (self.n_pairs,))
        self.assertEqual(accept.shape, (self.n_pairs,))
        self.assertTrue(np.all((prob >= 0) & (prob <= 1)))
        self.assertFalse(meta["holdout2_labels_opened"])
        self.assertEqual(meta["counts"]["s1_without_candidates"], 3)
        for rule, by_pop in meta["results"].items():
            if rule != "expected_f05":
                self.assertEqual(by_pop["records_with_multiple_owners"], 0)
            for name, code in (("B_REST", 1), ("SLICE", 2)):
                self.assertEqual(by_pop[name]["s1"], int((self.partition == code).sum()))
        left, right = _pairs(self.paths["pairs"])
        self.assertFalse(accept[self.partition[left] == rs.PARTITIONS["A"]].any())
        self.assertEqual(len(np.unique(right[accept])), int(accept.sum()))
        on_disk = json.loads((out / "metadata.json").read_text())
        self.assertEqual(on_disk["selected_rule"], meta["selected_rule"])
        self.assertEqual(on_disk["outputs"]["prob.npy"]["sha256"], rs.sha256(out / "prob.npy"))
        self.assertEqual([p.name for p in out.iterdir() if p.name.startswith(".")], [])
        self.assertIn(meta["chosen_threshold"], rs.DEFAULT_THRESHOLDS)
        self.assertGreater(meta["training"]["early_stop_components"], 0)

    def test_catboost_smoke(self):
        meta = rs.smoke(self.root / "smoke")
        self.assertTrue((self.root / "smoke" / "run" / "model.cbm").is_file())
        self.assertIn(meta["selected_rule"], ("threshold_one_owner", "expected_f05", "iterated_best_response"))
        self.assertEqual(meta["comparator"], "baseline")
        self.assertIn("B_REST", meta["paired_bootstrap_selected_minus_comparator"])

    def test_training_is_bounded_and_component_disjoint(self):
        count = np.bincount(_pairs(self.paths["pairs"])[0], minlength=len(self.partition))
        component = np.load(self.paths["components"])
        fit, es, info = rs.select_training(self.partition, component, count, train_code=0, max_rows=60,
                                           early_stop_fraction=0.25, seed=3)
        self.assertLessEqual(int(count[fit | es].sum()), 60)
        self.assertTrue(np.all(self.partition[fit | es] == 0))
        self.assertEqual(len(np.intersect1d(component[fit], component[es])), 0)
        self.assertEqual(info["fit_s1"], int(fit.sum()))

    def test_rejects_component_spanning_partitions(self):
        component = np.load(self.paths["components"])
        partition = self.partition.copy()
        first_a = np.flatnonzero(partition == 0)[0]
        partition[np.flatnonzero(component == component[first_a])[-1]] = 1
        with self.assertRaisesRegex(ValueError, "span the training partition"):
            rs.select_training(partition, component, np.ones(len(partition)), train_code=0, max_rows=100,
                               early_stop_fraction=0.1, seed=0)

    def test_rejects_duplicate_pair(self):
        left, right = _pairs(self.paths["pairs"])
        self._rewrite_pairs(np.insert(left, 10, left[10]), np.insert(right, 10, right[10]))
        for name, path in self.paths["scores"].items():
            np.save(path, np.insert(np.load(path), 10, 0.5))
        label = np.load(self.paths["labels"] / "label.npy")
        np.save(self.paths["labels"] / "label.npy", np.insert(label, 10, label[10]))
        summary_path = self.paths["labels"] / "labels_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["pairs"] = len(label) + 1
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "duplicate or mismatched pair order"):
            self._run()
        self.assertEqual(list((self.root / "run").iterdir()), [])

    def test_rejects_out_of_order_pairs_across_batches(self):
        left, right = _pairs(self.paths["pairs"])
        i = np.flatnonzero(left[1:] != left[:-1])[5] + 1
        order = np.arange(len(left))
        order[[i - 1, i]] = order[[i, i - 1]]
        self._rewrite_pairs(left[order], right[order])
        with self.assertRaisesRegex(ValueError, "duplicate or mismatched pair order"):
            self._run(batch_rows=int(i))

    def test_rejects_misaligned_score_array(self):
        np.save(self.paths["scores"]["tree"], np.load(self.paths["scores"]["tree"])[:-1])
        with self.assertRaisesRegex(ValueError, "Score array tree"):
            self._run()

    def test_never_opens_holdout2_labels(self):
        sealed = self.root / "v2" / "holdout2"
        shutil.copytree(self.paths["labels"], sealed)
        with mock.patch.object(rs.np, "load", side_effect=AssertionError("label file opened")):
            with self.assertRaisesRegex(ValueError, "HOLDOUT-2"):
                rs.guard_labels(sealed, self.n_pairs, len(self.partition))
        summary_path = self.paths["labels"] / "labels_summary.json"
        summary = json.loads(summary_path.read_text())
        for bad in ({**summary, "holdout2_rows_read": 5}, {k: v for k, v in summary.items() if k != "holdout2_rows_read"}):
            summary_path.write_text(json.dumps(bad))
            with mock.patch.object(rs.np, "load", side_effect=AssertionError("label file opened")):
                with self.assertRaisesRegex(ValueError, "HOLDOUT-2"):
                    rs.guard_labels(self.paths["labels"], self.n_pairs, len(self.partition))

    def test_refuses_nonempty_output_and_writes_json_atomically(self):
        out = self.root / "run"
        out.mkdir()
        (out / "keep.txt").write_text("x")
        with self.assertRaises(FileExistsError):
            self._run()
        target = self.root / "meta.json"
        rs.atomic_json(target, {"nan": float("nan"), "arr": np.arange(3), "n": np.int64(4)})
        self.assertEqual(json.loads(target.read_text()), {"arr": [0, 1, 2], "n": 4, "nan": None})
        self.assertEqual([p.name for p in self.root.iterdir() if p.name.startswith(".")], [])


if __name__ == "__main__":
    unittest.main()
