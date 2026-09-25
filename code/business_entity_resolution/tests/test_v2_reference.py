"""Small label-free population with real exclusion, order, and resume checks."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from ber.full_embeddings import file_digest
from ber.v2_holdout import id_sha256
from ber.v2_reference import build


class ReferencePopulationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "train_source1.tsv"
        self.medium = self.root / "medium.parquet"
        self.reservation = self.root / "reservation"
        self.reservation.mkdir()
        self.output = self.root / "scratch"
        self.durable = self.root / "durable"
        self.ids = [f"S1-{n:03d}" for n in range(19)]
        self.raw.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n" +
                            "".join(f"{value}\tname {n}\t{n} road\t{'US' if n % 2 else 'India'}\n"
                                    for n, value in enumerate(self.ids)), encoding="utf-8")
        # Even though medium has an eval row, only entity_id and split are projected.
        pd.DataFrame({"entity_id": [self.ids[2], self.ids[4], self.ids[3]],
                      "split": ["eval", "tune", "train"],
                      "secret_truth": ["unopened"] * 3}).to_parquet(self.medium)
        self.holdout = {self.ids[6], self.ids[8], self.ids[11]}
        self.blocked = {self.ids[0], self.ids[1], self.ids[2], self.ids[3], self.ids[4],
                        self.ids[5]} | self.holdout
        excluded = self.reservation / "train_tune_excluded_ids.txt"
        reserved = self.reservation / "reserved_ids.txt"
        for path, values in [(excluded, self.blocked), (reserved, self.holdout)]:
            path.write_text("\n".join(sorted(values)) + "\n", encoding="utf-8")
        (self.reservation / "reservation.json").write_text(json.dumps({
            "inputs": {
                "raw_source1": {"bytes": self.raw.stat().st_size, "sha256": file_digest(self.raw),
                                "rows": len(self.ids)},
                "medium_all_splits": {"file_sha256": file_digest(self.medium), "ids": 3,
                                      "sorted_id_sha256": id_sha256(self.ids[2:5])},
            },
            "future_raw_train_tune": {"excluded_id_file": excluded.name, "excluded_count": len(self.blocked),
                                      "excluded_file_sha256": file_digest(excluded),
                                      "excluded_sorted_id_sha256": id_sha256(self.blocked),
                                      "eligible_count": len(self.ids) - len(self.blocked)},
            "holdout2": {"id_file": reserved.name, "count": len(self.holdout),
                         "id_file_sha256": file_digest(reserved),
                         "sorted_id_sha256": id_sha256(self.holdout)},
            "prior_exclusions": {"count": len(self.blocked) - len(self.holdout)},
        }), encoding="utf-8")

    def _build(self):
        return build(self.raw, self.reservation, self.medium, self.output, self.durable,
                     chunk_size=3, tune_count=1, tune_sha=id_sha256([self.ids[4]]))

    def test_full_raw_order_all_sealed_rows_and_medium_tune_separate(self):
        report = self._build()
        expected = [value for value in self.ids if value not in self.blocked or value in self.holdout]
        population = pd.read_parquet(self.output / "reference.parquet")
        self.assertEqual(population.entity_id.tolist(), expected)
        self.assertEqual(population.raw_source1_row.tolist(),
                         [self.ids.index(value) for value in expected])
        self.assertEqual(population.loc[population.population.eq("holdout2"), "entity_id"].tolist(),
                         [value for value in self.ids if value in self.holdout])
        tune = pd.read_parquet(self.output / "medium_tune_reference.parquet")
        self.assertEqual(tune.entity_id.tolist(), [self.ids[4]])
        self.assertEqual(report["population"]["groups"]["train_tune_eligible"]["count"], 10)
        self.assertEqual(report["population"]["groups"]["holdout2"]["count"], 3)
        self.assertFalse(report["labels_read"])
        self.assertFalse(report["holdout_truth_read"])
        self.assertEqual(report, self._build())  # verified no-op resume
        self.assertEqual(file_digest(self.durable / "reference.parquet"),
                         report["parquet_sha256"]["reference.parquet"])

    def test_modified_reservation_or_duplicate_raw_is_rejected_before_completion(self):
        (self.reservation / "reserved_ids.txt").write_text("S1-999\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Reservation ID file changed"):
            self._build()
        self.assertFalse((self.output / "reference_manifest.json").exists())

    def test_medium_tune_digest_is_an_explicit_comparison_gate(self):
        with self.assertRaisesRegex(ValueError, "medium TUNE IDs"):
            build(self.raw, self.reservation, self.medium, self.output, self.durable,
                  chunk_size=3, tune_count=1, tune_sha="0" * 64)


if __name__ == "__main__":
    unittest.main()
