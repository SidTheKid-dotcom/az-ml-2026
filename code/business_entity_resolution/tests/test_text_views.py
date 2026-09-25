import unittest

import pandas as pd

from ber.text_views import canonicalize, record_text, record_texts


class TextViewTests(unittest.TestCase):
    def test_folds_only_latin_accents_and_preserves_indic_marks(self):
        value = "Àmicale Çlub लक्ष्मी स्टोर क़िला"
        self.assertEqual(canonicalize(value), "amicale club लक्ष्मी स्टोर क़िला")
        self.assertEqual(canonicalize(canonicalize(value)), canonicalize(value))

    def test_removes_only_listed_junk_and_collapses_whitespace(self):
        self.assertEqual(canonicalize("  Foo ## <NULL> >>  << bar  "), "foo bar")
        self.assertEqual(canonicalize("Nullify null business"), "nullify null business")
        self.assertEqual(canonicalize(pd.NA), "")

    def test_retains_explicit_fields_without_ids_or_country(self):
        frame = pd.DataFrame({"entity_id": ["SECRET"], "country": ["US"],
                              "business_name": ["Café"], "business_address": [" 9 Rue  "]})
        self.assertEqual(record_texts(frame), ["[NAME] cafe [ADDRESS] 9 rue"])
        self.assertEqual(record_text("Café", "9 Rue", canonical=False, field_tagged=False),
                         "Café | 9 Rue")


if __name__ == "__main__":
    unittest.main()
