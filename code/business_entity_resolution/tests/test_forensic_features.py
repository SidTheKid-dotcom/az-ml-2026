"""Hand-built cases for every forensic feature class, plus a 200k-pair speed check."""
from __future__ import annotations

import os
import time
import unittest

import numpy as np

from ber import forensic_features as ff
from ber.forensic_features import (
    REL_CAND_ADDED_SOME, REL_CAND_DROPPED_SOME, REL_CAND_NO_DIGITS, REL_NO_DIGITS,
    REL_OTHER_MISMATCH, REL_SAME_COUNT_SUBSTITUTED, REL_SAME_SEQUENCE, REL_SAME_SET_REORDERED,
    StreetVocabulary, align_digit_groups, forensic_features)

# A hand-made vocabulary so the tests do not depend on the derived resource file.
VOCAB = StreetVocabulary.from_dict({"countries": {
    "France": {"admin_phrases": ["nantes", "lille", "pays de la loire", "loire atlantique",
                                 "hauts de france", "nord", "saint nazaire", "st nazaire"],
               "admin_level": {"nantes": "city", "lille": "city", "saint nazaire": "city",
                               "st nazaire": "city", "pays de la loire": "region",
                               "loire atlantique": "region", "hauts de france": "region",
                               "nord": "region"},
               "generic_tokens": ["rue", "avenue", "boulevard", "allee", "impasse", "chemin",
                                  "place", "faubourg", "de", "du", "des", "la", "le", "bis", "n", "no"],
               "abbreviations": {"r": "rue", "av": "avenue", "bd": "boulevard", "pl": "place",
                                 "chem": "chemin", "all": "allee", "imp": "impasse",
                                 "fbg": "faubourg", "st": "saint"}},
    "US": {"admin_phrases": ["albuquerque", "nm", "new mexico"],
           "generic_tokens": ["unit", "street", "road", "court"],
           "abbreviations": {"ct": "court", "st": "street"}},
    "India": {"admin_phrases": ["new delhi", "delhi"], "generic_tokens": ["floor", "sector"],
              "abbreviations": {}},
}})


def features(left_name, left_address, right_name, right_address, country, vocab=VOCAB):
    return forensic_features([left_name], [left_address], [right_name], [right_address],
                             [country], vocab).iloc[0]


class AddressKindTests(unittest.TestCase):
    def test_empty_placeholders_and_digits(self):
        kinds = ff.address_kinds(["", "   ", "<NULL>", "##", None, "## <null> ##",
                                  "Rue Coustou, Nantes", "5 Rue Coustou", "कोलकाता"])
        self.assertEqual(kinds.tolist(), [0, 0, 0, 0, 0, 0, 1, 2, 1])
        self.assertEqual(kinds.dtype, np.int8)
        self.assertEqual(ff.ADDRESS_KINDS[kinds[3]], "empty")

    def test_right_side_kind_in_pair_frame(self):
        row = features("A", "5 Rue X, Nantes", "A", "<NULL>", "France")
        self.assertEqual(row.address_kind, ff.EMPTY)
        self.assertEqual(row.left_address_kind, ff.HAS_DIGITS)
        self.assertEqual(row.digit_relation, REL_CAND_NO_DIGITS)


class DigitRelationTests(unittest.TestCase):
    def test_every_relation_class(self):
        cases = {
            ((), ()): REL_NO_DIGITS,
            (("5",), ()): REL_CAND_NO_DIGITS,
            (("5", "118"), ("5", "118")): REL_SAME_SEQUENCE,
            (("5", "118"), ("118", "5")): REL_SAME_SET_REORDERED,
            (("9509", "11"), ("9509",)): REL_CAND_DROPPED_SOME,
            (("9509",), ("9509", "11")): REL_CAND_ADDED_SOME,
            ((), ("12",)): REL_CAND_ADDED_SOME,
            (("5",), ("7",)): REL_SAME_COUNT_SUBSTITUTED,
            (("5", "12"), ("7",)): REL_OTHER_MISMATCH,
        }
        for (left, right), expected in cases.items():
            with self.subTest(left=left, right=right):
                self.assertEqual(align_digit_groups(left, right)[0], expected)
                self.assertEqual(ff.DIGIT_RELATIONS[expected], ff.DIGIT_RELATIONS[align_digit_groups(left, right)[0]])

    def test_same_length_substitution_is_the_decoy_pattern(self):
        # "5 Rue Coustou" vs "7 Rue Coustou": one same-length substitution.
        rel, same, change, pad, dels, ins, dist, trans = align_digit_groups(("5",), ("7",))
        self.assertEqual((same, change, pad, dels, ins, dist, trans), (1, 0, 0, 0, 0, 1, 0))

    def test_length_change_zero_pad_transposition(self):
        self.assertEqual(align_digit_groups(("7000",), ("000",))[1:], (0, 1, 0, 0, 0, 1, 0))
        self.assertEqual(align_digit_groups(("29",), ("0029",))[1:], (0, 0, 1, 0, 0, 0, 0))
        self.assertEqual(align_digit_groups(("118", "12"), ("12", "181"))[1:], (1, 0, 0, 0, 0, 2, 1))
        self.assertEqual(align_digit_groups(("118",), ("119",))[7], 0)

    def test_deletions_insertions_alignment(self):
        # 3 left groups vs 1 right group: best pair by distance, two deletions.
        out = align_digit_groups(("1064", "11", "52"), ("1065",))
        self.assertEqual(out[0], REL_OTHER_MISMATCH)
        self.assertEqual(out[1:], (1, 0, 0, 2, 0, 1, 0))
        out = align_digit_groups(("5",), ("5", "12", "3"))
        self.assertEqual(out[0], REL_CAND_ADDED_SOME)
        self.assertEqual(out[4:6], (0, 2))

    def test_india_and_us_noise_keeps_same_sequence(self):
        row = features("Good Consulting Pvt Ltd", "F-118, Sector 5, New Delhi, Delhi",
                       "Good Consulting Private Limited", "F-##118, SECTOR 5, New Delhi", "India")
        self.assertEqual(row.digit_relation, REL_SAME_SEQUENCE)
        self.assertEqual(row.n_same_length_substitutions + row.n_length_changing_edits, 0)
        self.assertEqual(row.exact_key_equal, 1)
        row = features("Pinnacle Staffing Corp", "9509 Snyder Court, Unit 7309, Albuquerque, NM",
                       "Pinnacle Staffing Corporation", "# 7309, 9509 SNYDER CT, Albuquerque, New Mexico", "US")
        self.assertEqual(row.digit_relation, REL_SAME_SET_REORDERED)
        self.assertEqual(row.street_tokens_equal, 1)
        self.assertEqual(row.norm_name_equal, 1)
        # Digits concatenated in order differ ("95097309" vs "73099509"), so the key does too.
        self.assertEqual(row.exact_key_equal, 0)


class NameTests(unittest.TestCase):
    def test_normalization(self):
        names = ff.normalize_names(["Nantes Club EURL", "NANTES CLUB E.U.R.L.", "Nantes  Club S.A.S",
                                    "Àmicale Cœur SARL", "The Foyer SNC", "Private Limited", None,
                                    "ഗുഡ് കൺസൾട്ടിംഗ്"])
        self.assertEqual(names.tolist(), ["nantes club", "nantes club", "nantes club", "amicale coeur",
                                          "foyer snc", "", "", ""])

    def test_equality_and_jaccard(self):
        row = features("Nantes Club EURL", "5 Rue Coustou, Nantes", "NANTES CLUB SARL", "5 R COUSTOU", "France")
        self.assertEqual(row.norm_name_equal, 1)
        self.assertAlmostEqual(row.name_token_jaccard, 1.0)
        row = features("Nantes Club EURL", "5 Rue Coustou, Nantes", "Nantes Conseil EURL", "5 Rue Coustou", "France")
        self.assertEqual(row.norm_name_equal, 0)
        self.assertAlmostEqual(row.name_token_jaccard, 1 / 3, places=6)
        # Empty (native-only) names never count as equal.
        row = features("कंपनी", "5 X", "कंपनी", "5 X", "India")
        self.assertEqual(row.norm_name_equal, 0)
        self.assertEqual(row.exact_key_equal, 0)
        self.assertTrue(np.isnan(row.name_token_jaccard))

    def test_template_similarity_is_positional_and_preserves_multiplicity(self):
        self.assertTrue(ff.template_similar("alpha club", "alpha conseil"))
        self.assertFalse(ff.template_similar("alpha alpha club", "alpha conseil conseil"))
        self.assertFalse(ff.template_similar("alpha club", "club alpha"))


class ExactKeyTests(unittest.TestCase):
    def test_key_and_pair_flag(self):
        keys = ff.exact_keys(["Nantes Club EURL", "Nantes Club", ""], ["5 bis Rue Coustou 44000", "R. Coustou 5-44000", "5"])
        self.assertEqual(keys.tolist(), ["nantes club|544000", "nantes club|544000", ""])
        row = features("Nantes Club EURL", "5 Rue Coustou, Nantes", "Nantes Club", "Nantes, 5 R. Bisson", "France")
        self.assertEqual(row.exact_key_equal, 1)  # key ignores the street: needs the street view too
        self.assertEqual(row.street_token_jaccard, 0.0)

    def test_aligned_lengths_are_validated(self):
        with self.assertRaisesRegex(ValueError, "name and address"):
            ff.exact_keys(["A"], ["1 X", "2 X"])
        with self.assertRaisesRegex(ValueError, "normalized_names"):
            ff.exact_keys(["A"], ["1 X"], normalized_names=np.array([]))


class StreetTests(unittest.TestCase):
    def test_french_street_discriminates_lookalikes(self):
        same = features("Nantes Club EURL", "5 Rue Coustou, Nantes, Pays de la Loire",
                        "NANTES CLUB SARL", "5 R. COUSTOU, NANTES, Loire-Atlantique", "France")
        self.assertEqual(same.street_tokens_equal, 1)
        self.assertEqual(same.street_token_jaccard, 1.0)
        other = features("Nantes Club EURL", "5 Rue Coustou, Nantes, Pays de la Loire",
                         "Nantes Club EURL", "Pays de la Loire, 5 Rue Bisson, Nantes", "France")
        self.assertEqual(other.street_tokens_equal, 0)
        self.assertEqual(other.street_token_jaccard, 0.0)
        self.assertLess(other.street_token_ratio, 0.5)
        self.assertEqual(other.digit_relation, REL_SAME_SEQUENCE)

    def test_abbreviations_accents_admin(self):
        streets, admins = ff.parse_addresses(
            ["12 av. du Général Leclerc, Lille, Hauts-de-France", "12 AVENUE DU GENERAL LECLERC, LILLE, Nord",
             "3 Bd Victor Hugo, St-Nazaire", "4 Fbg de Paris, Lile", "7 chem. des Prés, Nantes"],
            "France", VOCAB)
        self.assertEqual(streets[0], frozenset({"general", "leclerc"}))
        self.assertEqual(streets[0], streets[1])
        self.assertEqual(admins[0], frozenset({"lille", "hauts de france"}))
        self.assertEqual(streets[2], frozenset({"victor", "hugo"}))
        self.assertEqual(admins[2], frozenset({"saint nazaire"}))  # expanded spelling
        self.assertEqual(streets[3], frozenset({"paris", "lile"}))  # "lile" < 5 chars: no fuzzy admin
        self.assertEqual(streets[4], frozenset({"pres"}))

    def test_fuzzy_admin_and_empty_streets(self):
        row = features("Pinnacle", "9509 Snyder, Albuquerque, NM", "Pinnacle", "New Mexico, Albuqureque, 9509 Snyder", "US")
        self.assertEqual(row.street_tokens_equal, 1)
        row = features("A", "5 Rue, Nantes", "A", "<NULL>", "France")
        self.assertTrue(np.isnan(row.street_token_jaccard))
        self.assertEqual(row.street_tokens_equal, -1)
        self.assertEqual(row.n_right_street_tokens, 0)

    def test_derived_resource_handles_french_examples(self):
        if not ff.DEFAULT_VOCAB_PATH.exists():
            self.skipTest("derived vocabulary not generated")
        vocab = StreetVocabulary.load()
        self.assertIn("hauts de france", vocab["France"].admin)
        self.assertEqual(vocab["France"].abbreviations.get("r"), "rue")
        rows = forensic_features(["Nantes Club EURL"] * 2, ["5 Rue Coustou, Nantes, Pays de la Loire"] * 2,
                                 ["NANTES CLUB SARL", "Nantes Club EURL"],
                                 ["5 R. COUSTOU, NANTES, Loire-Atlantique", "5 Rue Bisson, Nantes"], "France", vocab)
        self.assertEqual(rows.street_tokens_equal.tolist(), [1, 0])


class NativeScriptTests(unittest.TestCase):
    def test_flags(self):
        self.assertEqual(ff.native_script(["गुड कंसल्टिंग", "ഗുഡ്", "Good", "Àmicale", None]).tolist(), [1, 1, 0, 0, 0])
        row = features("Good Consulting", "Kochi, Kerala", "ഗുഡ് കൺസൾട്ടിംഗ്", "കൊച്ചി, കേരളം", "India")
        self.assertEqual((row.left_name_native, row.right_name_native, row.right_address_native), (0, 1, 1))


class FrameContractTests(unittest.TestCase):
    def test_dtypes_columns_and_lengths(self):
        frame = forensic_features(["A", "B"], ["1 X", ""], ["A", "C"], ["1 X", None], "US", VOCAB)
        self.assertEqual(tuple(frame.columns), ff.FEATURE_COLUMNS)
        for column in frame.columns:
            self.assertIn(frame[column].dtype, (np.int8, np.float32), column)
        with self.assertRaises(ValueError):
            forensic_features(["A"], ["1"], ["A", "B"], ["1"], "US", VOCAB)
        self.assertEqual(len(forensic_features([], [], [], [], [], VOCAB)), 0)


class MultiplicityTests(unittest.TestCase):
    def test_counts_within_country(self):
        frame = ff.s1_multiplicities(
            ["Nantes Club EURL", "Nantes Club SARL", "Nantes Club", "Other", ""],
            ["5 Rue Coustou, Nantes", "Nantes, 5 Rue Coustou", "5 Rue Coustou, Nantes", "5 Rue Coustou, Nantes", "x"],
            ["France", "France", "US", "France", "France"])
        self.assertEqual(frame.name_count.tolist(), [2, 2, 1, 1, 0])
        self.assertEqual(frame.address_count.tolist(), [3, 3, 1, 3, 1])
        self.assertEqual(frame.key_count.tolist(), [2, 2, 1, 1, 0])

    def test_aligned_lengths_are_validated(self):
        with self.assertRaisesRegex(ValueError, "name and address"):
            ff.s1_multiplicities(["A"], ["1", "2"], ["US"])
        with self.assertRaisesRegex(ValueError, "country length"):
            ff.s1_multiplicities(["A"], ["1"], [])


def synthetic_pairs(n_s1: int = 4_000, k: int = 50, seed: int = 7):
    rng = np.random.default_rng(seed)
    cities = ["Nantes", "Lille", "Bordeaux", "Calais", "Albuquerque", "New Delhi"]
    streets = [f"{w} {v}" for w in ("Rue", "Avenue", "Bd", "Allée", "Road", "Street")
               for v in ("Coustou", "Bisson", "Victor Hugo", "Jean Jaurès", "Snyder", "Pasteur", "Gambetta")]
    words = ["Club", "Conseil", "Centre", "Sport", "Amicale", "Staffing", "Labs", "Traders"]
    suffixes = ["EURL", "SARL", "SAS", "Inc", "LLC", "Private Limited", ""]
    s1_name = [f"{rng.choice(cities)} {rng.choice(words)} {rng.choice(suffixes)}" for _ in range(n_s1)]
    s1_addr = [f"{rng.integers(1, 999)} {rng.choice(streets)}, {rng.choice(cities)}, Region {rng.integers(9)}"
               for _ in range(n_s1)]
    n = n_s1 * k
    idx = np.repeat(np.arange(n_s1), k)
    other = rng.integers(0, n_s1, n)
    rec_name = [s1_name[o].upper() if i % 3 else s1_name[o] + " (France)" for i, o in enumerate(other)]
    rec_addr = [(s1_addr[o].replace("Rue", "R.").upper() if i % 4 == 0 else
                 f"{rng.integers(1, 9999)} {streets[i % len(streets)]}, {cities[i % len(cities)]}")
                for i, o in enumerate(other)]
    country = np.where(idx % 3 == 0, "US", "France")
    return (np.asarray(s1_name, dtype=object)[idx], np.asarray(s1_addr, dtype=object)[idx],
            np.asarray(rec_name, dtype=object), np.asarray(rec_addr, dtype=object), country)


class SpeedTests(unittest.TestCase):
    def test_200k_pairs_per_second(self):
        data = synthetic_pairs()
        self.assertEqual(len(data[0]), 200_000)
        forensic_features(*(d[:2000] for d in data), VOCAB)  # warm caches of imports
        start = time.perf_counter()
        frame = forensic_features(*data, VOCAB)
        elapsed = time.perf_counter() - start
        rate = len(frame) / elapsed
        print(f"\nforensic_features: {len(frame)} pairs in {elapsed:.2f}s = {rate:,.0f} pairs/s", flush=True)
        self.assertEqual(len(frame), 200_000)
        # Measured at 58-64k/s on the bounded two-thread workstation runtime.
        self.assertGreaterEqual(rate, float(os.environ.get("FORENSIC_MIN_PAIRS_PER_SEC", 50_000)))


if __name__ == "__main__":
    unittest.main()
