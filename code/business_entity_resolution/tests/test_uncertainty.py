import unittest

import numpy as np
import pandas as pd

from ber.uncertainty import aligned_scores, paired_bootstrap


class BootstrapTests(unittest.TestCase):
    def test_identical_models_have_zero_paired_uncertainty(self):
        scores = np.array([0.0, 0.3, 0.8, 1.0])
        report = paired_bootstrap(scores, scores, n_bootstrap=500)
        self.assertEqual(report["confidence_intervals"]["left_minus_right"], [0.0, 0.0])

    def test_constant_model_difference_is_preserved(self):
        scores = np.array([0.2, 0.4, 0.6, 0.8])
        report = paired_bootstrap(scores, scores - .125, n_bootstrap=500)
        np.testing.assert_allclose(report["confidence_intervals"]["left_minus_right"], [.125, .125])
        self.assertEqual(report, paired_bootstrap(scores, scores - .125, n_bootstrap=500))

    def test_extreme_two_reference_sample_has_expected_interval(self):
        report = paired_bootstrap([0.0, 1.0], n_bootstrap=2000)
        self.assertEqual(report["estimates"]["left_mean"], .5)
        self.assertEqual(report["confidence_intervals"]["left_mean"], [0.0, 1.0])

    def test_id_alignment_never_silently_drops_missing_references(self):
        frame = pd.DataFrame({"source1_entity_id": ["a", "b"], "macro_f05": [.1, .9]})
        left, right, _ = aligned_scores(frame, frame.iloc[::-1])
        np.testing.assert_array_equal(left, right)
        with self.assertRaises(ValueError):
            aligned_scores(frame, frame.iloc[:1])
        with self.assertRaises(ValueError):
            aligned_scores(frame, pd.concat([frame, frame.iloc[:1]]))


if __name__ == "__main__":
    unittest.main()
