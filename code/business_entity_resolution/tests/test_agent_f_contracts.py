"""Focused contracts for France pseudo-labeling, vocabulary, and diagnostics."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from ber.derive_forensic_vocab import parse_countries
from ber.forensic_features import StreetVocabulary
from ber.france_pseudo_labels import stream_french_pairs, summarize, verify_score_artifacts
from ber.label_free_diagnostics import diagnose, parse_report_names


EMPTY_VOCAB = StreetVocabulary()


class PseudoLabelContractTests(unittest.TestCase):
    def test_score_manifest_is_required_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, payload in (("probabilities.npy", b"p"),
                                  ("base_probabilities.npy", b"b"),
                                  ("scored_pairs.parquet", b"s")):
                (root / name).write_bytes(payload)
            import hashlib
            lines = [f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}"
                     for name in ("probabilities.npy", "base_probabilities.npy", "scored_pairs.parquet")]
            (root / "source.sha256").write_text("\n".join(lines) + "\n")
            self.assertEqual(set(verify_score_artifacts(root)), set(name.split()[-1] for name in lines))
            (root / "probabilities.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_score_artifacts(root)

    def test_stream_validates_scores_and_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs = Path(directory) / "pairs.parquet"
            pd.DataFrame({"left_idx": [0, 1], "right_idx": [0, 1]}).to_parquet(pairs, index=False)
            french = np.array([True, False])
            frame, claims = stream_french_pairs(
                pairs, np.array([0.9, 0.4]), np.array([0.8, 0.3]), french, 2,
                expected_pairs=2)
            self.assertEqual(frame.pair_row.tolist(), [0])
            self.assertEqual(claims.tolist(), [1, 0])

            with self.assertRaisesRegex(ValueError, "finite probabilities"):
                stream_french_pairs(pairs, np.array([np.nan, 0.4]), np.array([0.8, 0.3]),
                                    french, 2, expected_pairs=2)
            pd.DataFrame({"left_idx": [-1], "right_idx": [0]}).to_parquet(pairs, index=False)
            with self.assertRaisesRegex(ValueError, "outside"):
                stream_french_pairs(pairs, np.array([0.9]), np.array([0.8]), french, 2,
                                    expected_pairs=1)

    def test_stream_handles_no_french_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs = Path(directory) / "pairs.parquet"
            pd.DataFrame({"left_idx": [0], "right_idx": [0]}).to_parquet(pairs, index=False)
            frame, claims = stream_french_pairs(
                pairs, np.array([0.1]), np.array([0.2]), np.array([False]), 1,
                expected_pairs=1)
            self.assertTrue(frame.empty)
            self.assertEqual(claims.tolist(), [0])

    def test_empty_summary_is_strict_json(self):
        result = pd.DataFrame({
            "label": pd.Series(dtype=np.int8), "holdout": pd.Series(dtype=bool),
            "reasons": pd.Series(dtype=str), "v1_blend": pd.Series(dtype=np.float32),
            "s1_entity_id": pd.Series(dtype=str), "record_entity_id": pd.Series(dtype=str),
        })
        s1 = pd.DataFrame({"holdout": [False]})
        counts = summarize(result, s1, pd.DataFrame(index=[0]), {}, {}, 0)
        encoded = json.dumps(counts, allow_nan=False)
        self.assertIn('"p_minus_reject_rate": null', encoded)


class DiagnosticContractTests(unittest.TestCase):
    @staticmethod
    def populations():
        s1 = {
            "id": np.array([1_000_000_000_001], dtype=np.int64),
            "country": np.array(["US"], dtype=object),
            "key": np.array([0], dtype=np.uint64),
            "text": pd.DataFrame(columns=["entity_id", "business_name", "business_address"]),
        }
        records = {
            "id": np.array([2_000_000_000_001], dtype=np.int64),
            "country": np.array(["US"], dtype=object),
            "key": np.array([0], dtype=np.uint64),
            "text": pd.DataFrame(columns=["entity_id", "business_name", "business_address"]),
        }
        return s1, records

    def test_empty_countries_emit_null_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            matching = Path(directory) / "matching.tsv"
            matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\t\n")
            s1, records = self.populations()
            report = diagnose(matching, s1, records, EMPTY_VOCAB)
            self.assertIsNone(report["countries"]["France"]["empty_rate"])
            self.assertIsNone(report["countries"]["France"]["longest_list"])
            self.assertIsNone(report["countries"]["France"]["street_mismatch"]
                              ["street_mismatch_rate_among_accepted"])
            json.dumps(report, allow_nan=False)

    def test_duplicate_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            matching = Path(directory) / "matching.tsv"
            matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S2-1\n")
            s1, records = self.populations()
            with self.assertRaisesRegex(ValueError, "duplicate"):
                diagnose(matching, s1, records, EMPTY_VOCAB)

    def test_report_names_are_exact_unique_basenames(self):
        paths = [Path("a/matching.tsv"), Path("b/matching.tsv")]
        self.assertEqual(parse_report_names("first, second", paths), ["first", "second"])
        for invalid in ("one", "one,one", "../one,two", "one,"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_report_names(invalid, paths)


class VocabularyContractTests(unittest.TestCase):
    def test_country_parser(self):
        self.assertEqual(parse_countries("France, US"), ["France", "US"])
        for invalid in ("France,", "France,France", "Canada"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_countries(invalid)


if __name__ == "__main__":
    unittest.main()
