"""One-shot confirmation must preserve whole-S1 alignment and pre-truth sealing."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ber.v2_confirm_decision import _decisions, confirm


class ConfirmationTests(unittest.TestCase):
    def test_decision_rejects_missing_or_reordered_candidate_groups(self):
        models = {"__pooled__": {"x": [.01, .99], "y": [.01, .99], "p_ungated": .001}}
        prior = {"p_true_count": [.1, .8, .1], "gate_true_link_discovery": .98,
                 "pair_prevalence": .1}
        kwargs = (np.array([0, 1, 0]), np.array([3, 4, 5]),
                  np.array([.9, .8, .6]), np.array([.9, .8, .6]),
                  np.array([True, True, True]), np.array(["US", "US"]),
                  np.array(["S2-0", "S2-1"]), models, prior, 2)
        with self.assertRaisesRegex(ValueError, "ordered"):
            _decisions(*kwargs)

    def test_confirmation_cannot_read_truth_without_durable_prediction_commitment(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            truth = root / "labels.tsv"
            truth.write_text("deliberately invalid: must not be read")
            fake = (None, pd.DataFrame({"entity_id": ["S1-1"]}), None, None)
            with patch("ber.v2_confirm_decision._checked_inputs", return_value=fake):
                with self.assertRaisesRegex(ValueError, "pre-truth prediction commitment"):
                    confirm(root, root, root, root, root, root, root, truth)
            self.assertEqual(truth.read_text(), "deliberately invalid: must not be read")


if __name__ == "__main__":
    unittest.main()
