import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ber.cross_encoder_experiment import _amp_context, detailed_score, score_reranker, train_reranker
from ber.rerank_inference import (score_reranker_batched, score_reranker_selective_fp32,
                                  selective_fp32_mask, _pad_numpy)
from ber.v2_amp_benchmark import (accept_comparison, accepted_at_threshold, amp_parity_pass, compact_pairs, decisions,
                                  false_links_by_country, saved_band_sweep, select_probe)

try:
    import torch
except ImportError:  # The local CPU-only project venv deliberately has no torch.
    torch = None


class BenchmarkAccountingTests(unittest.TestCase):
    def test_probe_is_deterministic_and_concentrates_on_the_fixed_boundary(self):
        scores = np.linspace(.5, 1., 3000, dtype=np.float32)
        eligible = np.ones(len(scores), dtype=bool)
        chosen = select_probe(eligible, scores, seed=20260925)
        np.testing.assert_array_equal(chosen, select_probe(eligible, scores, seed=20260925))
        self.assertEqual(len(np.unique(chosen)), 2000)
        near = np.argsort(np.abs(scores.astype(np.float64) - .775), kind="stable")[:1000]
        self.assertTrue(set(near) <= set(chosen))
        with self.assertRaisesRegex(ValueError, "eligible"):
            select_probe(np.zeros(3000, bool), scores, seed=1)

    def test_compaction_and_accepted_pair_identity_preserve_unsorted_original_rows(self):
        reference = pd.DataFrame({"entity_id": ["s1", "s2", "s3"]})
        pool = pd.DataFrame({"entity_id": ["t1", "t2", "t3"]})
        pairs = pd.DataFrame({"left_idx": [2, 0, 2], "right_idx": [1, 2, 0]})
        compact = compact_pairs(reference, pool, pairs)
        original_ids = [(reference.entity_id.iloc[l], pool.entity_id.iloc[r])
                        for l, r in pairs.itertuples(index=False, name=None)]
        compact_ids = [(compact[0].entity_id.iloc[l], compact[1].entity_id.iloc[r])
                       for l, r in compact[2].itertuples(index=False, name=None)]
        self.assertEqual(compact_ids, original_ids)
        with self.assertRaisesRegex(ValueError, "aligned"):
            accept_comparison([True, False], [True])
        overlap = accept_comparison([True, False, True], [True, True, False])
        self.assertEqual(overlap["decision_disagreements"], 2)
        self.assertAlmostEqual(overlap["accept_overlap_jaccard"], 1 / 3)

    def test_fixed_threshold_scores_all_references_including_singletons_and_misses(self):
        reference = pd.DataFrame({"entity_id": ["a", "b", "c"], "country": ["India", "US", "US"]})
        pool = pd.DataFrame({"entity_id": ["yes", "wrong"]})
        pairs = pd.DataFrame({"left_idx": [0, 1], "right_idx": [0, 1]})
        truth = {"a": {"yes"}, "b": set(), "c": {"missing_from_retrieval"}}
        scores, accepted = decisions(np.array([.9, .9], np.float32),
                                     np.array([.99, .1], np.float32), np.ones(2, bool))
        self.assertEqual(accepted.tolist(), [True, False])
        report = detailed_score(reference, pool, pairs, scores, truth, .775, [True, True, True])
        self.assertEqual(report["reference_count"], 3)
        self.assertEqual(report["true_links"], 2)
        self.assertAlmostEqual(report["macro_f05"], 2 / 3)
        self.assertEqual(report["false_positive_links"], 0)
        self.assertEqual(report["singleton_accuracy"], 1.)
        by_country = false_links_by_country(reference, pairs, np.array([True, False]), accepted)
        self.assertEqual(by_country, {"India": 0, "US": 0})
        self.assertEqual(false_links_by_country(reference, pairs, np.array([True, False]),
                                                np.array([True, True])), {"India": 0, "US": 1})

    def test_stress_probe_must_pass_even_when_full_tune_overlap_passes(self):
        sample = {"accept_overlap_jaccard": .9980645161290322}
        full = {"accept_overlap_jaccard": .99992391965916}
        self.assertFalse(amp_parity_pass(1.5628978864312693e-05, sample, full))
        self.assertTrue(amp_parity_pass(1.5628978864312693e-05, full))
        self.assertFalse(amp_parity_pass(.000501, full))

    def test_frozen_float64_threshold_does_not_round_down_to_float32(self):
        values = np.array([.775, np.nextafter(np.float32(.775), np.float32(1))], dtype=np.float32)
        np.testing.assert_array_equal(accepted_at_threshold(values), [False, True])

    def test_label_free_band_catches_large_bf16_shift_without_rescoring_off_gate(self):
        base = np.array([.3876, .2694, .8452, -1], np.float32)
        bf16 = np.array([.9019, .94335, .7866, 0], np.float32)
        fp32 = np.array([.90462, .94421, .7459, 0], np.float32)
        selected = np.array([True, True, True, False])
        narrow = selective_fp32_mask(base, bf16, selected, alpha=.75, threshold=.775, band=.002)
        wide = selective_fp32_mask(base, bf16, selected, alpha=.75, threshold=.775, band=.04)
        self.assertEqual(narrow.tolist(), [True, True, False, False])
        self.assertEqual(wide.tolist(), [True, True, True, False])
        sweep = saved_band_sweep(base[:3], bf16[:3], fp32[:3], (.002, .04))
        self.assertEqual([row["rescored_pairs"] for row in sweep], [2, 3])
        self.assertEqual(sweep[-1]["vs_full_fp32"]["decision_disagreements"], 0)
        with self.assertRaisesRegex(ValueError, "aligned"):
            selective_fp32_mask(base[:-1], bf16, selected, alpha=.75, threshold=.775, band=.04)

    def test_selective_scorer_only_runs_fp32_for_band_and_keeps_callers_args(self):
        from ber import rerank_inference

        calls = []
        selected = np.array([True, True, False])
        base = np.array([.3876, .8452, -1], np.float32)

        def scores(model, tokenizer, reference, pool, pairs, mask, options):
            calls.append((options.amp, mask.tolist()))
            return np.array([.9019, .7866, 0] if options.amp == "bf16"
                            else [.90462, .7459, 0], np.float32)

        args = SimpleNamespace(amp="bf16")
        with patch.object(rerank_inference, "score_reranker_batched", side_effect=scores):
            corrected, near, original = score_reranker_selective_fp32(
                None, None, None, None, [None] * 3, selected, args, base,
                alpha=.75, threshold=.775, band=.04)
        self.assertEqual(calls, [("bf16", [True, True, False]), ("fp32", [True, True, False])])
        self.assertEqual(args.amp, "bf16")
        self.assertEqual(near.tolist(), [True, True, False])
        np.testing.assert_array_equal(original, np.array([.9019, .7866, 0], np.float32))
        np.testing.assert_array_equal(corrected, np.array([.90462, .7459, 0], np.float32))
        with self.assertRaisesRegex(ValueError, "BF16 first pass"):
            score_reranker_selective_fp32(None, None, None, None, [None] * 3, selected,
                                          SimpleNamespace(amp="fp32"), base,
                                          alpha=.75, threshold=.775, band=.04)


class AmpContextTests(unittest.TestCase):
    def test_fp32_disables_cuda_autocast_and_unsupported_bf16_fails_instead_of_fp16(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_bf16_supported=lambda: False),
                               autocast=Mock(return_value=nullcontext()), bfloat16=object())
        with _amp_context(fake, "cuda:0", "fp32"):
            pass
        fake.autocast.assert_called_once_with(device_type="cuda", enabled=False)
        with self.assertRaisesRegex(ValueError, "BF16"):
            _amp_context(fake, "cuda:0", "bf16")
        with self.assertRaisesRegex(ValueError, "amp must"):
            _amp_context(fake, "cpu", "fp16")
        with _amp_context(fake, "cpu", "bf16"):
            pass
        self.assertEqual(fake.autocast.call_count, 1)


@unittest.skipIf(torch is None, "Torch tests run in the project's existing GPU runtime")
class TorchDtypeTests(unittest.TestCase):
    class Tokenizer:
        model_input_names = ["input_ids", "attention_mask"]
        pad_token_id = 1
        padding_side = "right"

        def __init__(self):
            self.padded_lengths = []

        def __call__(self, left, right, *, padding, truncation, max_length, return_tensors=None,
                     return_attention_mask=True):
            rows = [[ord(char) % 7 + 1 for char in (a + " " + b)[:max_length]]
                    for a, b in zip(left, right)]
            if padding:
                return self.pad([{"input_ids": row, "attention_mask": [1] * len(row)} for row in rows],
                                padding=True, return_tensors=return_tensors)
            return {"input_ids": rows, "attention_mask": [[1] * len(row) for row in rows]}

        def pad(self, rows, *, padding, pad_to_multiple_of=None, return_tensors):
            length = max(map(lambda row: len(row["input_ids"]), rows))
            if pad_to_multiple_of:
                length = (length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
            self.padded_lengths.append(length)
            return {key: torch.tensor([row[key] + [self.pad_token_id if key == "input_ids" else 0]
                                       * (length - len(row[key])) for row in rows])
                    for key in self.model_input_names}

    def test_fast_padding_matches_tokenizer_pad_and_preserves_pad_token(self):
        tokenizer = self.Tokenizer()
        encoded = {"input_ids": [[3, 4, 5], [6, 7, 8, 9, 10]],
                   "attention_mask": [[1, 1, 1], [1, 1, 1, 1, 1]]}
        positions = np.array([1, 0])
        actual = _pad_numpy(encoded, tokenizer.model_input_names, positions,
                            np.array([3, 5]), tokenizer, 8, torch)
        expected = tokenizer.pad([{key: encoded[key][i] for key in tokenizer.model_input_names}
                                  for i in positions], padding=True, pad_to_multiple_of=8, return_tensors="pt")
        for key in tokenizer.model_input_names:
            self.assertTrue(torch.equal(actual[key], expected[key]))
        self.assertEqual(actual["input_ids"][1, 3:].tolist(), [1] * 5)
        self.assertEqual(actual["attention_mask"][1, 3:].tolist(), [0] * 5)
        tokenizer.padding_side = "left"
        with self.assertRaisesRegex(ValueError, "right-aligned"):
            _pad_numpy(encoded, tokenizer.model_input_names, positions,
                       np.array([3, 5]), tokenizer, 8, torch)

    class Model(torch.nn.Module if torch is not None else object):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.004))

        def forward(self, input_ids, attention_mask):
            sums = (input_ids.float() * attention_mask).sum(dim=1)
            return SimpleNamespace(logits=(sums * self.weight).to(torch.bfloat16).unsqueeze(1))

    def test_inference_promotes_bf16_logits_and_restores_pair_order(self):
        reference = pd.DataFrame({"business_name": ["alpha", "beta"], "business_address": ["1", "24"]})
        pool = pd.DataFrame({"business_name": ["one", "two"], "business_address": ["3", "456"]})
        pairs = pd.DataFrame({"left_idx": [1, 0, 1], "right_idx": [0, 1, 1]})
        selected = np.array([True, False, True])
        args = SimpleNamespace(device="cpu", amp="bf16", inference_batch_size=1, max_length=80,
                               length_bucketing=True, pad_to_multiple_of=8)
        model, tokenizer = self.Model(), self.Tokenizer()
        old = score_reranker(model, tokenizer, reference, pool, pairs, selected, args)
        new = score_reranker_batched(model, tokenizer, reference, pool, pairs, selected, args)
        self.assertEqual(new.dtype, np.float32)
        np.testing.assert_array_equal(new, old)
        self.assertEqual(new[1], 0.)
        self.assertTrue(all(size % 8 == 0 for size in tokenizer.padded_lengths[-2:]))
        args.amp = "fp32"
        np.testing.assert_array_equal(score_reranker_batched(model, tokenizer, reference, pool,
                                                              pairs, selected, args), old)
        args.amp, args.defer_cpu_transfer = "bf16", True
        np.testing.assert_array_equal(score_reranker_batched(model, tokenizer, reference, pool,
                                                              pairs, selected, args), old)
        args.fast_padding, args.pin_memory = True, True
        np.testing.assert_array_equal(score_reranker_batched(model, tokenizer, reference, pool,
                                                              pairs, selected, args), old)
        args.defer_cpu_transfer = "yes"
        with self.assertRaisesRegex(ValueError, "boolean"):
            score_reranker_batched(model, tokenizer, reference, pool, pairs, selected, args)

    def test_training_loss_receives_fp32_logits_even_if_model_emits_bf16(self):
        reference = pd.DataFrame({"business_name": ["a", "b"], "business_address": ["1", "2"]})
        pool = pd.DataFrame({"business_name": ["c", "d"], "business_address": ["3", "4"]})
        pairs = pd.DataFrame({"left_idx": [0, 0, 1, 1], "right_idx": [0, 1, 0, 1]})
        labels = np.array([1., 0., 0., 1.], np.float32)
        args = SimpleNamespace(device="cpu", amp="bf16", seed=5, steps=1, batch_size=2,
                               positive_fraction=.5, learning_rate=.001, weight_decay=0., max_length=80)
        original = torch.nn.functional.binary_cross_entropy_with_logits
        seen = []

        def check_float_loss(logits, target):
            seen.append((logits.dtype, target.dtype))
            return original(logits, target)

        with patch("torch.nn.functional.binary_cross_entropy_with_logits", side_effect=check_float_loss):
            train_reranker(self.Model(), self.Tokenizer(), reference, pool, pairs, labels,
                           np.array([0, 3]), np.array([1, 2]), args)
        self.assertEqual(seen, [(torch.float32, torch.float32)])

    @unittest.skipUnless(torch is not None and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         "CUDA BF16 training is checked in the existing A40 runtime")
    def test_cuda_forward_obeys_amp_switch_but_bce_stays_fp32(self):
        forward, loss = {}, []

        class LinearModel(torch.nn.Module):
            def __init__(self, mode):
                super().__init__()
                self.mode = mode
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, input_ids, attention_mask):
                features = (input_ids.float() * attention_mask).sum(dim=1, keepdim=True)
                logits = self.linear(features)
                forward[self.mode] = logits.dtype
                return SimpleNamespace(logits=logits)

        reference = pd.DataFrame({"business_name": ["a", "b"], "business_address": ["1", "2"]})
        pool = pd.DataFrame({"business_name": ["c", "d"], "business_address": ["3", "4"]})
        pairs = pd.DataFrame({"left_idx": [0, 0, 1, 1], "right_idx": [0, 1, 0, 1]})
        labels = np.array([1., 0., 0., 1.], np.float32)
        original = torch.nn.functional.binary_cross_entropy_with_logits

        def inspect(logits, target):
            loss.append((logits.dtype, target.dtype, torch.is_autocast_enabled("cuda")))
            return original(logits, target)

        for mode in ("fp32", "bf16"):
            args = SimpleNamespace(device="cuda:0", amp=mode, seed=5, steps=1, batch_size=2,
                                   positive_fraction=.5, learning_rate=.001, weight_decay=0., max_length=80)
            with patch("torch.nn.functional.binary_cross_entropy_with_logits", side_effect=inspect):
                train_reranker(LinearModel(mode).cuda(), self.Tokenizer(), reference, pool, pairs, labels,
                               np.array([0, 3]), np.array([1, 2]), args)
        self.assertEqual(forward, {"fp32": torch.float32, "bf16": torch.bfloat16})
        self.assertEqual(loss, [(torch.float32, torch.float32, False)] * 2)


if __name__ == "__main__":
    unittest.main()
