"""Name-only ownership rule (A0): normalization, pricing and an end-to-end synthetic rewrite."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ber.name_only import (build_test_file, expected_row_delta, is_empty_address, name_multiplicity, norm_name,
                           per_link_value)


class NormalizationTests(unittest.TestCase):
    def test_norm_name(self):
        self.assertEqual(norm_name("Zephay Labs Inc"), "zephay labs")
        self.assertEqual(norm_name("Nantes Café SARL"), "nantes cafe")
        self.assertEqual(norm_name("The A.B.C. Co., Ltd."), "a b c")
        self.assertEqual(norm_name("S/R Wheeler, LLC"), "s r wheeler")
        self.assertEqual(norm_name("Smart Broking Pvt. Limited"), "smart broking")
        self.assertEqual(norm_name("राम ट्रेडर्स"), "")
        self.assertEqual(norm_name(None), "")

    def test_empty_address(self):
        for value in (None, "", "   ", "<NULL>", " ## ", "##", "\t<NULL>\t"):
            self.assertTrue(is_empty_address(value), value)
        for value in ("12 Main St", "<NULL> x", "#", "N/A"):
            self.assertFalse(is_empty_address(value), value)

    def test_multiplicity_is_per_country(self):
        mult = name_multiplicity(["US", "US", "India", "US"], ["a", "a", "a", "b"])
        np.testing.assert_array_equal(mult, [2, 2, 1, 1])


class PricingTests(unittest.TestCase):
    def test_single_changes_match_manual_table(self):
        self.assertAlmostEqual(expected_row_delta(1, [0.0], []), 4 / 9)       # remove wrong, k=2
        self.assertAlmostEqual(expected_row_delta(0, [1.0], []), -1.0)        # remove only true link
        self.assertAlmostEqual(expected_row_delta(0, [0.0], []), 1.0)         # singleton restored
        self.assertAlmostEqual(expected_row_delta(1, [], [1.0]), 1 / 6)       # add true, k=1
        self.assertAlmostEqual(expected_row_delta(0, [], [0.0]), -1.0)        # false merge on a singleton
        self.assertAlmostEqual(expected_row_delta(2, [], [0.0]), 10 / 14 - 1)  # add wrong, k=2
        for k in range(0, 6):
            for p in (0.1, 0.5, 0.978):
                self.assertAlmostEqual(expected_row_delta(k, [], [p]), per_link_value(k, p, "add"))
                self.assertAlmostEqual(expected_row_delta(k, [p], []), per_link_value(k + 1, p, "remove"))

    def test_joint_row_expectation(self):
        # Two certain additions to an abstaining S1 with T=2: 0 -> 1.
        self.assertAlmostEqual(expected_row_delta(0, [], [1.0, 1.0]), 1.0)
        # Break-even of an addition to a one-link list is 0.727.
        self.assertAlmostEqual(expected_row_delta(1, [], [8 / 11]), 0.0)


def _write(path, header, rows, newline="\n"):
    path.write_text(newline.join([header, *rows]) + newline, encoding="utf-8")


class EndToEndTests(unittest.TestCase):
    def test_synthetic_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            test_dir = tmp / "dataset" / "test"
            test_dir.mkdir(parents=True)
            head = "entity_id\tbusiness_name\tbusiness_address\tcountry"
            _write(test_dir / "test_source1.tsv", head, [
                "S1-1\tAlpha Traders\t1 Main St\tUS", "S1-2\tBeta Co\t2 Main St\tUS",
                "S1-3\tBeta\t3 Main St\tUS", "S1-4\tGamma LLC\t4 Road\tIndia",
                "S1-5\tराम ट्रेडर्स\t5 Road\tIndia", "S1-6\tதமிழ் கடை\t6 Road\tIndia"])
            _write(test_dir / "test_source2.tsv", head, [
                "S2-10\tAlpha Traders Inc\t<NULL>\tUS", "S2-11\tbeta\t\tUS", "S2-13\tGamma\t##\tIndia",
                "S2-14\tDelta\t9 Road\tUS", "S2-15\tराम ट्रेडर्स\t\tIndia",
                "S2-16\tதமிழ் கடை\t\tIndia"])
            _write(test_dir / "test_source3.tsv", head, ["S3-12\tAlpha Traders\t1 Main St\tUS"])
            base = tmp / "base.tsv"
            _write(base, "source1_entity_id\tmatched_entity_ids", [
                "S1-1\tS3-12", "S1-2\tS2-11", "S1-3\t", "S1-4\tS2-10",
                "S1-5\tS2-15", "S1-6\tS2-16"])
            cand = tmp / "cand.tsv"
            _write(cand, "source1_entity_id\tcandidate_entity_ids",
                    ["S1-1\tS3-12,S2-10", "S1-2\tS2-11,S2-14", "S1-3\tS2-11", "S1-4\tS2-10,S2-13",
                     "S1-5\tS2-15", "S1-6\tS2-16"], newline="\r\n")
            out = tmp / "out"
            report = build_test_file(base, cand, test_dir, out, run_official=False, expected_rows=6)
            lines = (out / "matching_results.tsv").read_bytes().decode().split("\n")
            self.assertEqual(lines, ["source1_entity_id\tmatched_entity_ids", "S1-1\tS3-12,S2-10", "S1-2\t",
                                     "S1-3\t", "S1-4\tS2-13", "S1-5\tS2-15", "S1-6\tS2-16", ""])
            us, india = report["by_group"]["US"], report["by_group"]["India"]
            self.assertEqual((us["links_added"], us["links_dropped"], us["owner_changes"]), (1, 1, 0))
            self.assertEqual((india["links_added"], india["owner_changes"], india["rows_newly_empty"]), (1, 1, 0))
            self.assertEqual(us["rows_newly_empty"], 1)
            self.assertEqual(report["contested_ids"], 0)
            self.assertTrue(json.loads((out / "report.json").read_text())["strict_validation"]["passed"])


if __name__ == "__main__":
    unittest.main()
