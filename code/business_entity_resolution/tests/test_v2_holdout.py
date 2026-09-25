"""ID-only reservation and future-split leakage guards, using synthetic S1 rows."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from ber import v2_holdout


class HoldoutReservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "train_source1.tsv"
        self.quick = self.root / "quick.parquet"
        self.medium = self.root / "medium.parquet"
        self.older = self.root / "old" / "reserved_ids.txt"
        self.older.parent.mkdir()
        self.ids = [f"S1-{i:03d}" for i in range(30)]
        self.quick_ids = self.ids[:2]
        self.medium_ids = self.ids[2:5]
        self.older_ids = self.ids[5:7]
        self.raw.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n" +
            "".join(f"{entity_id}\tunused name\tunused address\t{'India' if i % 2 else 'US'}\n"
                    for i, entity_id in enumerate(self.ids)), encoding="utf-8"
        )
        pd.DataFrame({"entity_id": self.quick_ids, "business_name": ["unused"] * 2}).to_parquet(self.quick)
        pd.DataFrame({"entity_id": self.medium_ids, "split": ["train", "tune", "eval"],
                      "business_address": ["unused"] * 3}).to_parquet(self.medium)
        self.older.write_text("\n".join(self.older_ids) + "\n", encoding="utf-8")
        self.spec = v2_holdout.ReservationSpec(
            seed=20260929, raw_rows=len(self.ids), raw_sha256=v2_holdout.file_sha256(self.raw),
            raw_bytes=self.raw.stat().st_size, quick_count=2,
            quick_sha256=v2_holdout.id_sha256(self.quick_ids), medium_count=3,
            medium_sha256=v2_holdout.id_sha256(self.medium_ids),
            medium_splits=(("train", 1), ("tune", 1), ("eval", 1)), older_count=2,
            older_sha256=v2_holdout.id_sha256(self.older_ids), holdout_count=6,
        )
        self._write_older_manifest()
        self.output = self.root / "artifacts" / "v2_holdout2"

    def _write_older_manifest(self):
        (self.older.parent / "reservation.json").write_text(json.dumps({
            "source_sha256": self.spec.raw_sha256,
            "excluded_quick_id_sha256": self.spec.quick_sha256,
            "excluded_medium_id_sha256": self.spec.medium_sha256,
            "exclusion_reference_sha256": {
                "quick": v2_holdout.file_sha256(self.quick),
                "medium": v2_holdout.file_sha256(self.medium),
            },
            "reserved_ids": self.spec.older_count,
            "reserved_id_sha256": self.spec.older_sha256,
            "id_file_sha256": v2_holdout.file_sha256(self.older),
            "labels_read": False,
        }), encoding="utf-8")

    def _reserve(self, output=None):
        return v2_holdout.reserve(self.raw, self.quick, self.medium, self.older,
                                  output or self.output, spec=self.spec)

    def test_reserve_verifies_all_groups_and_guards_future_train_tune(self):
        manifest = self._reserve()
        self.assertEqual(manifest, v2_holdout.verify(self.output, spec=self.spec))
        relocated_raw = self.root / "relocated_raw.tsv"
        relocated_raw.write_bytes(self.raw.read_bytes())
        self.assertEqual(manifest, v2_holdout.verify(self.output, raw=relocated_raw, spec=self.spec))
        selected = (self.output / "reserved_ids.txt").read_text().splitlines()
        blocked = (self.output / "train_tune_excluded_ids.txt").read_text().splitlines()
        self.assertEqual(len(selected), 6)
        self.assertEqual(len(blocked), 13)
        self.assertEqual(selected, sorted(selected))
        self.assertEqual(set(blocked), set(self.quick_ids + self.medium_ids + self.older_ids + selected))
        self.assertEqual(manifest["holdout2"]["sorted_id_sha256"], v2_holdout.id_sha256(selected))
        self.assertEqual(manifest["prior_exclusions"]["count"], 7)
        self.assertEqual(sum(manifest["holdout2"]["country_counts"].values()), 6)
        self.assertEqual(manifest["future_raw_train_tune"]["eligible_count"], 17)
        self.assertFalse(manifest["verification"]["labels_read"])
        self.assertFalse((self.output / "reserved_ids.txt").stat().st_mode & 0o222)

        allowed = sorted(set(self.ids) - set(blocked))
        train, tune = self.root / "train_ids.txt", self.root / "tune_ids.txt"
        train.write_text("\n".join(allowed[:2]) + "\n", encoding="utf-8")
        tune.write_text(allowed[2] + "\n", encoding="utf-8")
        self.assertEqual(v2_holdout.check_train_tune(self.output, train, tune)["disjoint"], True)
        for forbidden_id in (self.older_ids[0], selected[0], self.medium_ids[2], self.quick_ids[0]):
            tune.write_text(forbidden_id + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reserved/development"):
                v2_holdout.check_train_tune(self.output, train, tune)

    def test_row_order_independent_with_deterministic_hash_rank_ties(self):
        # Force equal ranks to exercise the lexical tie break, even though a
        # real BLAKE2b-128 collision is extraordinarily unlikely.
        with patch.object(v2_holdout, "_rank", return_value=1):
            self._reserve()
            expected = self.ids[7:13]
            self.assertEqual((self.output / "reserved_ids.txt").read_text().splitlines(), expected)
            lines = self.raw.read_text(encoding="utf-8").splitlines(keepends=True)
            self.raw.write_text(lines[0] + "".join(reversed(lines[1:])), encoding="utf-8")
            self.spec = replace(self.spec, raw_sha256=v2_holdout.file_sha256(self.raw))
            self._write_older_manifest()
            second = self.root / "other_reservation"
            self._reserve(second)
            self.assertEqual((second / "reserved_ids.txt").read_text().splitlines(), expected)

    def test_reject_modified_artifacts_and_overwrite(self):
        self._reserve()
        with self.assertRaises(FileExistsError):
            self._reserve()
        selected = self.output / "reserved_ids.txt"
        selected.chmod(0o644)
        selected.write_text("S1-tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ID file differs"):
            v2_holdout.verify(self.output, spec=self.spec)

    def test_reject_old_manifest_mismatch_and_missing_medium_eval(self):
        old_manifest_path = self.older.parent / "reservation.json"
        old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        old_manifest["reserved_id_sha256"] = hashlib.sha256(b"wrong").hexdigest()
        old_manifest_path.write_text(json.dumps(old_manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sealed reservation manifest"):
            self._reserve()
        self.assertFalse(self.output.exists())

        self._write_older_manifest()
        pd.DataFrame({"entity_id": self.medium_ids, "split": ["train", "tune", "tune"]}).to_parquet(self.medium)
        self._write_older_manifest()
        with self.assertRaisesRegex(ValueError, "TRAIN/TUNE/EVAL"):
            self._reserve()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
