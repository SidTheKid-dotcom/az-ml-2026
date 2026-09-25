import unittest
import numpy as np

from ber.rerank_inference import minibatch_order, score_reranker_batched


class LengthBatchTests(unittest.TestCase):
    def test_stable_order_preserves_every_pair_once(self):
        batches = list(minibatch_order([12, 4, 12, 3, 4], 2))
        np.testing.assert_array_equal(np.concatenate(batches), [3, 1, 4, 0, 2])
        self.assertEqual([len(x) for x in batches], [2, 2, 1])

    def test_disabled_bucketing_preserves_original_order(self):
        np.testing.assert_array_equal(np.concatenate(list(minibatch_order([9, 2, 5], 2, False))), [0, 1, 2])

    def test_invalid_lengths_and_empty_scoring(self):
        with self.assertRaises(ValueError):
            list(minibatch_order([0, 1], 2))
        with self.assertRaises(ValueError):
            list(minibatch_order([1], 0))
        result = score_reranker_batched(None, None, None, None, [], [], None)
        self.assertEqual(result.shape, (0,))


if __name__ == '__main__':
    unittest.main()
