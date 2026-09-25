"""Tiny fixtures exercise the original validator and the stricter release gate."""
import json
import tempfile
import unittest
from pathlib import Path

from ber.validate_release import file_sha256, validate_release

OFFICIAL = Path(__file__).resolve().parents[3] / "data/raw/utils/validate_submission.py"


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ber_release_gate_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for source, ids in ((1, ("S1-a", "S1-b")), (2, ("S2-x",)), (3, ("S3-y",))):
            (self.root / f"test_source{source}.tsv").write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                + "".join(f"{entity}\tName\tAddress\tFrance\n" for entity in ids), encoding="utf-8")

    def run_gate(self, matching="S1-a\tS2-x\nS1-b\t\n", candidate="S1-a\tS2-x,S3-y\nS1-b\t\n"):
        matches, candidates = self.root / "matching.tsv", self.root / "candidate.tsv"
        matches.write_text("source1_entity_id\tmatched_entity_ids\n" + matching, encoding="utf-8")
        candidates.write_text("source1_entity_id\tcandidate_entity_ids\n" + candidate, encoding="utf-8")
        result = validate_release(matches, candidates, self.root, self.root / "report.json", validator=OFFICIAL, expected_rows=2)
        self.assertEqual(json.loads((self.root / "report.json").read_text()), result)
        self.assertEqual(result["original_validator"]["sha256"], file_sha256(OFFICIAL))
        return result

    def test_success_includes_empty_singleton_and_audit_hashes(self):
        result = self.run_gate()
        self.assertTrue(result["passed"])
        self.assertEqual(result["counts"], dict(rows_checked=2, matched_links=1, candidate_links=2, empty_match_rows=1))
        self.assertFalse(result["metric_score_computed"])

    def test_missing_source1_fails(self):
        self.assertFalse(self.run_gate(matching="S1-a\tS2-x\n")["passed"])

    def test_official_subset_warning_is_a_hard_failure(self):
        result = self.run_gate(candidate="S1-a\tS3-y\nS1-b\t\n")
        self.assertEqual(result["original_validator"]["exit_code"], 0)
        self.assertTrue(result["original_validator"]["warnings"])
        self.assertFalse(result["passed"])
        self.assertTrue(any("subset" in error for error in result["errors"]))

    def test_duplicate_rows_and_ids_fail(self):
        for rows in ("S1-a\tS2-x\nS1-a\t\n", "S1-a\tS2-x,S2-x\nS1-b\t\n"):
            with self.subTest(rows=rows):
                self.assertFalse(self.run_gate(matching=rows)["passed"])

    def test_wrong_order_unknown_and_empty_ids_fail(self):
        for rows in ("S1-b\t\nS1-a\tS2-x\n", "S1-a\tS2-unknown\nS1-b\t\n", "S1-a\tS2-x,\nS1-b\t\n"):
            with self.subTest(rows=rows):
                self.assertFalse(self.run_gate(matching=rows)["passed"])


if __name__ == "__main__":
    unittest.main()
