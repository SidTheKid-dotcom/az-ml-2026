"""CPU tests for the Qwen3-Reranker yes/no adapter. No model download and no GPU."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from ber import qwen_reranker as q


REPO = Path(__file__).resolve().parents[3]
ALLOWLIST = REPO / "problem-statement" / "models.yaml"


class YesNoTokenizer:
    """Greedy stand-in: yes and no are single tokens; every other character is its own."""

    def __init__(self, *, yes_id: int = 3, no_id: int = 4, unk: int = 0, pad: int = 1,
                 decode_space: bool = False, convert_id: int | None = None):
        self.yes_id = yes_id
        self.no_id = no_id
        self.unk_token_id = unk
        self.pad_token_id = pad
        self.decode_space = decode_space
        self.convert_id = convert_id

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "yes":
            return [self.yes_id]
        if text == "no":
            return [self.no_id]
        specials = (
            (q.SYSTEM_PREFIX, 10),
            (q.ASSISTANT_SUFFIX, 11),
            ("yes", self.yes_id),
            ("no", self.no_id),
        )
        ids = []
        index = 0
        while index < len(text):
            matched = False
            for phrase, token in specials:
                if text.startswith(phrase, index):
                    ids.append(token)
                    index += len(phrase)
                    matched = True
                    break
            if not matched:
                ids.append(1000 + ord(text[index]))
                index += 1
        return ids

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        pieces = []
        for token in ids:
            if token == self.yes_id:
                pieces.append(" yes" if self.decode_space else "yes")
            elif token == self.no_id:
                pieces.append(" no" if self.decode_space else "no")
            elif token >= 1000:
                pieces.append(chr(token - 1000))
            else:
                pieces.append("?")
        return "".join(pieces)

    def convert_tokens_to_ids(self, token):
        if token == "yes":
            return self.yes_id if self.convert_id is None else self.convert_id
        if token == "no":
            return self.no_id if self.convert_id is None else self.convert_id
        return self.unk_token_id


class CollidingTokenizer(YesNoTokenizer):
    def __init__(self):
        super().__init__(yes_id=5, no_id=5)
        self.last = None

    def encode(self, text, add_special_tokens=False):
        self.last = text
        if text in {"yes", "no"}:
            return [5]
        return super().encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        if list(ids) == [5] and self.last in {"yes", "no"}:
            return self.last
        return super().decode(ids, skip_special_tokens=skip_special_tokens,
                              clean_up_tokenization_spaces=clean_up_tokenization_spaces)

    def convert_tokens_to_ids(self, token):
        if token in {"yes", "no"}:
            return 5
        return self.unk_token_id


class FakeBackend:
    def __init__(self):
        self.weight = np.zeros(1, dtype=np.float64)
        self.steps = 0
        self.micros = []

    def prepare(self, *, total_steps, learning_rate, weight_decay):
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def micro_backward(self, input_ids, attention, labels, weights, normalizer, yes_id, no_id):
        del attention
        loss = q.weighted_bce_with_logits(np.zeros(len(labels)), labels, weights, normalizer)
        self.micros.append({"labels": np.array(labels), "weights": np.array(weights),
                            "normalizer": normalizer, "loss": loss, "yes_id": yes_id, "no_id": no_id,
                            "width": input_ids.shape[1]})
        return loss

    def optimizer_step(self, step, total_steps):
        del total_steps
        self.steps += 1
        if self.steps != step:
            raise AssertionError(f"optimizer step {step} but backend counted {self.steps}")
        self.weight = self.weight + step

    def save(self, directory: Path):
        np.save(directory / "weights.npy", self.weight)
        (directory / "backend.json").write_text(json.dumps({"steps": self.steps}), encoding="utf-8")

    def load(self, directory: Path):
        self.weight = np.load(directory / "weights.npy")
        self.steps = json.loads((directory / "backend.json").read_text(encoding="utf-8"))["steps"]


class StopAfter(FakeBackend):
    def optimizer_step(self, step, total_steps):
        if step == 3:
            raise RuntimeError("planned stop before step 3")
        super().optimizer_step(step, total_steps)


def arena_frame(extra: dict | None = None) -> pd.DataFrame:
    rows = []
    for index in range(4):
        rows.append({
            "left_business_name": f"Acme {index}",
            "left_business_address": f"{index} Main Street",
            "right_business_name": f"Acme {index}",
            "right_business_address": f"{index} Main St",
            "label": 1,
            "true_count": 1 + (index % 3),
            "negative_type": "",
            "partition": "A",
            "entity_id": f"S1-{900000 + index}",
            "country": "France",
        })
    kinds = list(q.TYPED_NEGATIVE_TYPES) + [q.RANDOM_NEGATIVE_TYPE]
    for index, kind in enumerate(kinds * 2):
        rows.append({
            "left_business_name": f"Owner {index}",
            "left_business_address": f"{index} Side Road",
            "right_business_name": f"Other {index}",
            "right_business_address": f"{index} Side Rd",
            "label": 0,
            "true_count": 0 if index % 4 == 0 else 2,
            "negative_type": kind,
            "partition": "A",
            "entity_id": f"S1-{800000 + index}",
            "country": "France",
        })
    frame = pd.DataFrame(rows)
    frame.insert(0, "pair_row", np.arange(len(frame)))
    if extra:
        for key, value in extra.items():
            frame[key] = value
    return frame.sample(frac=1, random_state=3).reset_index(drop=True)


def training_kwargs(frame: pd.DataFrame, root: Path, *, steps: int = 4) -> dict:
    config = q.training_config(
        frame, steps=steps, positives_per_step=1, grad_accum=2, seed=11, max_length=q.MAX_LENGTH,
        learning_rate=q.DEFAULT_LR, weight_decay=q.DEFAULT_WEIGHT_DECAY, checkpoint_every=1,
        input_sha256=None)
    return {"steps": steps, "positives_per_step": 1, "grad_accum": 2, "seed": 11,
            "checkpoint_every": 1, "checkpoint_root": root, "config": config}


class PromptAndTokenTests(unittest.TestCase):
    def test_prompt_uses_the_documented_chat_template(self):
        query = q.field_tagged_business("Acme", "1 Main")
        document = q.field_tagged_business("Acme", "1 Main Street")
        rendered = q.render_prompt(query, document)
        self.assertTrue(rendered.startswith(q.SYSTEM_PREFIX))
        self.assertTrue(rendered.endswith(q.ASSISTANT_SUFFIX))
        self.assertEqual(
            q.SYSTEM_PREFIX,
            '<|im_start|>system\nJudge whether the Document meets the requirements based on the '
            'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
            '<|im_end|>\n<|im_start|>user\n')
        self.assertEqual(q.ASSISTANT_SUFFIX, '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
        self.assertIn(f"<Instruct>: {q.DEFAULT_INSTRUCTION}\n<Query>: {query}\n<Document>: {document}", rendered)
        self.assertLess(rendered.index("<Query>:"), rendered.index("<Document>:"))

    def test_canonical_field_text_drops_ids_and_keeps_indic(self):
        self.assertEqual(q.canonical_field("  Café <NULL>  SHOP## "), "cafe shop")
        self.assertEqual(q.canonical_field("Àmicale >>Centre<<"), "amicale centre")
        self.assertEqual(q.canonical_field("e\u0301"), "e")
        self.assertEqual(q.canonical_field("\u0915\u093c"), "\u0915\u093c")
        self.assertEqual(q.canonical_field("दुकान"), "दुकान")
        self.assertEqual(
            q.field_tagged_business("Acme", ""),
            "business name: acme ; business address: ")
        frame = arena_frame().iloc[:1].copy()
        frame["left_business_name"] = ["Acme"]
        frame["left_business_address"] = ["1 Main"]
        query, document = q.pair_field_texts(frame)
        rendered = query[0] + document[0]
        self.assertEqual(query[0], "business name: acme ; business address: 1 main")
        self.assertNotIn(str(frame["entity_id"].iloc[0]), rendered)
        self.assertNotIn("S1-", rendered)
        self.assertNotIn("France", rendered)
        self.assertNotIn("entity_id", rendered)

    def test_yes_no_tokens_must_be_exact(self):
        yes_id, no_id = q.verify_yes_no_tokens(YesNoTokenizer())
        self.assertEqual((yes_id, no_id), (3, 4))
        with self.assertRaises(ValueError):
            q.verify_answer_token(YesNoTokenizer(decode_space=True), "yes")
        with self.assertRaises(ValueError):
            q.verify_answer_token(YesNoTokenizer(convert_id=9), "yes")
        with self.assertRaises(ValueError):
            q.verify_answer_token(YesNoTokenizer(yes_id=0, unk=0), "yes")
        with self.assertRaises(ValueError):
            q.verify_yes_no_tokens(CollidingTokenizer())

    def test_multi_token_answer_is_rejected(self):
        class SplitYes(YesNoTokenizer):
            def encode(self, text, add_special_tokens=False):
                if text == "yes":
                    return [3, 3]
                return super().encode(text, add_special_tokens=add_special_tokens)

        with self.assertRaisesRegex(ValueError, "single tokenizer token"):
            q.verify_answer_token(SplitYes(), "yes")

    def test_truncation_keeps_the_assistant_suffix(self):
        tokenizer = YesNoTokenizer()
        prefix, _, suffix = q.prompt_parts("acme", "z" * 4000)
        prefix_ids = q.encode_text(tokenizer, prefix)
        suffix_ids = q.encode_text(tokenizer, suffix)
        max_length = len(prefix_ids) + len(suffix_ids) + 8
        ids = q.assemble_token_ids(tokenizer, "acme", "z" * 4000, max_length=max_length)
        self.assertEqual(ids[-len(suffix_ids):], suffix_ids)
        self.assertEqual(len(ids), max_length)
        with self.assertRaises(ValueError):
            q.assemble_token_ids(tokenizer, "acme", "z", max_length=2)


class ScoreMathTests(unittest.TestCase):
    def test_score_sign_and_sigmoid(self):
        logits = np.zeros((3, 6), dtype=np.float64)
        logits[0, 3] = 2.0
        logits[0, 4] = -1.0
        logits[1, 3] = -2.0
        logits[1, 4] = 2.0
        scores, probabilities = q.yes_no_from_logits(logits, 3, 4)
        self.assertEqual(scores.dtype, np.float32)
        np.testing.assert_allclose(scores, [3.0, -4.0, 0.0])
        np.testing.assert_allclose(probabilities, q.stable_sigmoid(np.array([3.0, -4.0, 0.0])))
        self.assertGreater(probabilities[0], 0.5)
        self.assertLess(probabilities[1], 0.5)
        self.assertAlmostEqual(float(probabilities[2]), 0.5)
        with self.assertRaises(ValueError):
            q.yes_no_from_logits(np.array([1.0, 2.0]), 3, 4)
        with self.assertRaises(ValueError):
            q.yes_no_from_logits(np.full((1, 2), np.nan), 0, 1)
        with self.assertRaises(ValueError):
            q.yes_no_from_logits(np.zeros((2, 3)), 0, 5)

    def test_padding_and_batch_order_restoration(self):
        sequences = [[4], [7, 7, 7, 8], [3, 5]]
        widths = []

        def scorer(input_ids, attention):
            self.assertTrue(np.all(attention[:, -1] == 1))
            self.assertEqual(input_ids.shape[1] % 8, 0)
            widths.append(int(input_ids.shape[1]))
            logits = np.zeros((len(input_ids), 12), dtype=np.float64)
            logits[:, 3] = input_ids[:, -1]
            return logits

        scores, probabilities = q.score_sequences(
            sequences, scorer, yes_id=3, no_id=4, pad_id=0, batch_size=2)
        np.testing.assert_allclose(scores, [4, 8, 5])
        np.testing.assert_allclose(probabilities, q.stable_sigmoid(np.array([4, 8, 5], dtype=np.float64)))
        self.assertEqual(widths, [8, 8])
        ids, mask = q.pad_left([[1, 2, 3]], 0, 8)
        np.testing.assert_array_equal(ids[0, -3:], [1, 2, 3])
        self.assertEqual(int(mask[0, -1]), 1)
        self.assertEqual(ids.shape[1], 8)
        with self.assertRaises(ValueError):
            q.pad_left([[]], 0, 8)
        with self.assertRaises(ValueError):
            q.pad_left([[1]], 0, 0)

    def test_weighted_loss_and_stake(self):
        self.assertEqual(q.raw_stake_weight(1, 1), 1.0)
        self.assertAlmostEqual(q.raw_stake_weight(1, 2), 1 / 6)
        self.assertEqual(q.raw_stake_weight(0, 0), 1.0)
        self.assertAlmostEqual(q.raw_stake_weight(0, 1), 4 / 9)
        self.assertAlmostEqual(q.raw_stake_weight(0, 2), 2 / 7)
        with self.assertRaises(ValueError):
            q.raw_stake_weight(1, 0)
        labels = np.array([1] + [0] * 21)
        raw = np.array([1.0] + [1.0] * 20 + [10000.0])
        normalized = q.class_normalize_weights(raw, labels)
        self.assertAlmostEqual(float(normalized[0]), 1.0)
        self.assertEqual(float(normalized[-1]), 5.0)
        self.assertLess(float(normalized[1]), 1.0)
        scores = np.array([0.0, 2.0])
        loss = q.weighted_bce_with_logits(scores, np.array([1.0, 0.0]), np.array([1.0, 1.0]))
        manual = 0.5 * (
            (np.log1p(np.exp(0.0)))
            + (2.0 + np.log1p(np.exp(-2.0)))
        )
        # bce(0, 1) = log(2); bce(2, 0) = 2 + log1p(exp(-2)); mean of equal weights.
        expected = 0.5 * (np.log(2) + 2.0 + np.log1p(np.exp(-2.0)))
        self.assertAlmostEqual(loss, float(expected))
        self.assertAlmostEqual(loss, float(manual))

    def test_learning_rate_is_warmup_then_constant(self):
        base = 2e-5
        self.assertAlmostEqual(q.screen_learning_rate(1, 100, base), base / 5)
        self.assertAlmostEqual(q.screen_learning_rate(5, 100, base), base)
        self.assertAlmostEqual(q.screen_learning_rate(6, 100, base), base)
        with self.assertRaises(ValueError):
            q.screen_learning_rate(0, 100, base)


class SamplingCheckpointTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_resumable(self):
        frame = arena_frame()
        positive, typed, random_idx = q.training_index_pools(frame)
        first = q.ArenaSampler(positive, typed, random_idx, 11)
        second = q.ArenaSampler(positive, typed, random_idx, 11)
        taken = [first.take(1) for _ in range(2)]
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(taken, [second.take(1) for _ in range(2)])))
        restored = q.ArenaSampler.restore(positive, typed, random_idx, first.state())
        rest = [restored.take(1) for _ in range(2)]
        fresh = q.ArenaSampler(positive, typed, random_idx, 11)
        whole = [fresh.take(1) for _ in range(4)]
        for actual, expected in zip(taken + rest, whole):
            np.testing.assert_array_equal(actual, expected)
        other = q.ArenaSampler(positive, typed, random_idx, 2)
        self.assertFalse(np.array_equal(other.take(1), whole[0]))
        broken = frame.copy()
        broken["partition"] = "B"
        with self.assertRaisesRegex(ValueError, "arena-A"):
            q.training_index_pools(broken)
        missing = frame.copy()
        missing.loc[missing.negative_type.eq("false_accept"), "negative_type"] = "random_gated"
        with self.assertRaisesRegex(ValueError, "Missing typed"):
            q.training_index_pools(missing)

    def test_gradient_accumulation_resume_and_checkpoint_integrity(self):
        frame = arena_frame()
        tokenizer = YesNoTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted = FakeBackend()
            q.run_training(frame, tokenizer, uninterrupted, **training_kwargs(frame, root / "full"))
            self.assertEqual(uninterrupted.steps, 4)
            np.testing.assert_allclose(uninterrupted.weight, [10.0])
            self.assertTrue(all(item["width"] % 8 == 0 for item in uninterrupted.micros))
            self.assertEqual({item["yes_id"] for item in uninterrupted.micros}, {3})
            for start in range(0, len(uninterrupted.micros), 2):
                group = uninterrupted.micros[start:start + 2]
                labels = np.concatenate([item["labels"] for item in group])
                weights = np.concatenate([item["weights"] for item in group])
                total = sum(item["loss"] for item in group)
                self.assertAlmostEqual(total, q.weighted_bce_with_logits(
                    np.zeros(len(labels)), labels, weights, group[0]["normalizer"]))
                self.assertEqual(sorted(np.unique(labels).tolist()), [0, 1])
            with self.assertRaisesRegex(RuntimeError, "planned stop"):
                q.run_training(frame, tokenizer, StopAfter(), **training_kwargs(frame, root / "stop"))
            checkpoint = root / "stop" / "step_000002"
            resumed = FakeBackend()
            q.run_training(frame, tokenizer, resumed, **training_kwargs(frame, root / "resume"),
                           resume=checkpoint)
            np.testing.assert_allclose(resumed.weight, uninterrupted.weight)
            self.assertEqual(resumed.steps, 4)
            self.assertEqual(len(resumed.micros), 4)
            for actual, expected in zip(resumed.micros, uninterrupted.micros[4:]):
                np.testing.assert_array_equal(actual["labels"], expected["labels"])
                np.testing.assert_allclose(actual["weights"], expected["weights"])
            with self.assertRaises(FileExistsError):
                q.run_training(frame, tokenizer, FakeBackend(), **training_kwargs(frame, root / "full", steps=1))
            marker = json.loads((checkpoint / "complete.json").read_text(encoding="utf-8"))
            self.assertEqual(marker["state"], "complete")
            self.assertEqual(marker["step"], 2)
            self.assertIn("weights.npy", marker["files_sha256"])
            state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["completed_steps"], 2)
            weight_path = checkpoint / "weights.npy"
            weight_path.write_bytes(weight_path.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "files changed"):
                q.load_checkpoint(checkpoint, FakeBackend(), marker["config_sha256"])
            (checkpoint / "complete.json").unlink()
            with self.assertRaisesRegex(ValueError, "completed-step"):
                q.verify_checkpoint(checkpoint)
            made = q.save_checkpoint(root / "partial-parent", step=3, config={"schema": 1}, config_sha="abc",
                                     sampler_state={}, backend=FakeBackend(), train_seconds=0.0, last_loss=0.0)
            self.assertTrue((made / "complete.json").is_file())
            with self.assertRaises(FileExistsError):
                q.save_checkpoint(root / "partial-parent", step=3, config={"schema": 1}, config_sha="abc",
                                  sampler_state={}, backend=FakeBackend(), train_seconds=0.0, last_loss=0.0)
            stage = root / "partial-parent" / ".step_000004.incomplete"
            stage.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                q.save_checkpoint(root / "partial-parent", step=4, config={"schema": 1}, config_sha="abc",
                                  sampler_state={}, backend=FakeBackend(), train_seconds=0.0, last_loss=0.0)


class AlignmentAndIdentityTests(unittest.TestCase):
    def test_scores_follow_pair_row_and_manifest_locks_identity(self):
        frame = arena_frame()
        tokenizer = YesNoTokenizer()
        yes_id, no_id = q.verify_yes_no_tokens(tokenizer)
        instruction = "same business"

        def scorer(input_ids, attention):
            logits = np.zeros((len(input_ids), 12), dtype=np.float64)
            logits[:, yes_id] = (input_ids * attention).sum(axis=1)
            return logits

        scores, probabilities = q.score_pair_frame(
            frame, tokenizer, scorer, yes_id=yes_id, no_id=no_id, pad_id=tokenizer.pad_token_id,
            batch_size=2, instruction=instruction)
        self.assertGreater(len(set(scores.tolist())), 1)
        queries, documents = q.pair_field_texts(frame)
        rows = q.pair_rows(frame)
        for position, row in enumerate(rows):
            sequence = [q.assemble_token_ids(tokenizer, queries[position], documents[position],
                                             instruction=instruction)]
            one_score, one_probability = q.score_sequences(
                sequence, scorer, yes_id=yes_id, no_id=no_id, pad_id=tokenizer.pad_token_id, batch_size=1)
            self.assertAlmostEqual(float(scores[int(row)]), float(one_score[0]))
            self.assertAlmostEqual(float(probabilities[int(row)]), float(one_probability[0]))
        self.assertFalse(np.array_equal(rows, np.arange(len(frame))))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pairs = path / "pairs.parquet"
            frame.to_parquet(pairs, index=False)
            manifest = q.write_score_outputs(
                path / "scores", scores=scores, probabilities=probabilities, frame=frame,
                identity={"model_id": q.MODEL_ID, "license": "apache-2.0", "params_b": 0.596,
                          "allowlist_sha256": "abc"},
                yes_id=yes_id, no_id=no_id, input_path=pairs, batch_size=2)
            reloaded = np.load(path / "scores" / "scores.npy")
            np.testing.assert_array_equal(reloaded, scores)
            self.assertEqual(reloaded.dtype, np.float32)
            self.assertEqual(manifest["alignment"], "scores[i] and probabilities[i] belong to pair_row == i")
            self.assertFalse(manifest["labels_read"])
            self.assertIsNone(manifest["official_f05"])
            self.assertEqual(manifest["input_sha256"], q._sha256_file(pairs))
            self.assertEqual(manifest["pair_row_sha256"], q._sha256_bytes(q.pair_rows(frame).tobytes()))
            self.assertEqual(manifest["scores_sha256"], q._sha256_file(path / "scores" / "scores.npy"))
            with self.assertRaises(FileExistsError):
                q.write_score_outputs(
                    path / "scores", scores=scores, probabilities=probabilities, frame=frame,
                    identity={"model_id": q.MODEL_ID, "license": "apache-2.0", "params_b": 0.596},
                    yes_id=yes_id, no_id=no_id, input_path=pairs, batch_size=2)

    def test_allowlist_and_snapshot_identity(self):
        identity = q.verify_allowlist(ALLOWLIST)
        self.assertEqual(identity["model_id"], "Qwen/Qwen3-Reranker-0.6B")
        self.assertEqual(identity["license"], "apache-2.0")
        self.assertEqual(identity["params_b"], 0.596)
        self.assertLessEqual(identity["params_b"], identity["budget_total_params_b"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = root / "bad.yaml"
            bad.write_text("budget_total_params_b: 8\nmodels:\n"
                           "  - {id: Qwen/Qwen3-Reranker-0.6B, licence: mit, params_b: 0.596, role: reranker}\n",
                           encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Apache"):
                q.verify_allowlist(bad)
            wrong_count = root / "count.yaml"
            wrong_count.write_text(
                "budget_total_params_b: 8\nmodels:\n"
                "  - {id: Qwen/Qwen3-Reranker-0.6B, licence: apache-2.0, params_b: 4.022, role: reranker}\n",
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parameter"):
                q.verify_allowlist(wrong_count)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(json.dumps({
                "model_type": "qwen3",
                "architectures": ["Qwen3ForSequenceClassification"],
                "_name_or_path": "Qwen/Qwen3-Reranker-0.6B",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "causal"):
                q.inspect_snapshot(snapshot)
            (snapshot / "config.json").write_text(json.dumps({
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "_name_or_path": "Qwen/Qwen3-Reranker-8B",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "0.6B"):
                q.inspect_snapshot(snapshot)
            (snapshot / "config.json").write_text(json.dumps({
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "_name_or_path": "Qwen/Qwen3-Reranker-0.6B",
            }), encoding="utf-8")
            (snapshot / "LICENSE").write_text("MIT License\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Apache"):
                q.inspect_snapshot(snapshot)
            (snapshot / "LICENSE").write_text("Apache License Version 2.0\n", encoding="utf-8")
            header = {"w": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]}, "__metadata__": {}}
            payload = json.dumps(header).encode()
            (snapshot / "model.safetensors").write_bytes(struct.pack("<Q", len(payload)) + payload + b"\0" * 12)
            self.assertEqual(q.safetensors_parameter_count(snapshot / "model.safetensors"), 6)
            with self.assertRaisesRegex(ValueError, "parameter count"):
                q.inspect_snapshot(snapshot)
        q.assert_parameter_count(596_000_000)
        with self.assertRaises(ValueError):
            q.assert_parameter_count(752_000_000)

    def test_bounds_holdout_and_missing_cache(self):
        self.assertNotIn("torch", sys.modules)
        estimates = q.resource_estimates()
        self.assertEqual(estimates["kind"], "estimate")
        self.assertFalse(estimates["measured"])
        with self.assertRaisesRegex(ValueError, "HOLDOUT-2"):
            q.refuse_holdout_path("/tmp/v2_holdout2/reserved_ids.txt")
        with self.assertRaisesRegex(ValueError, "HOLDOUT-2"):
            q.load_pair_frame(Path("/tmp/holdout-2/pairs.parquet"), training=False)
        score_args = q.build_parser().parse_args([
            "score", "--pairs", "pairs.parquet", "--output", "out", "--device", "cpu"])
        with self.assertRaisesRegex(ValueError, "BF16"):
            q.validate_score_args(score_args)
        screen_args = q.build_parser().parse_args([
            "screen", "--pairs", "pairs.parquet", "--output", "out", "--steps", "20001", "--device", "cuda:0"])
        with self.assertRaisesRegex(ValueError, "1 to 20000"):
            q.validate_screen_args(screen_args)
        payload = vars(screen_args).copy()
        payload.update(steps=10, device="cpu", positives_per_step=1, grad_accum=1, max_length=256,
                       checkpoint_every=1, learning_rate=2e-5, weight_decay=0.01, seed=1, resume=None,
                       allowlist=ALLOWLIST, checkpoint_root=Path("checkpoints"))
        cpu_screen = Namespace(**payload)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            q.validate_screen_args(cpu_screen)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = arena_frame()
            pairs = root / "pairs.parquet"
            frame.to_parquet(pairs, index=False)
            args = q.build_parser().parse_args([
                "screen", "--pairs", str(pairs), "--output", str(root / "run"), "--steps", "1",
                "--positives-per-step", "1", "--grad-accum", "4", "--device", "cuda:0",
                "--allowlist", str(ALLOWLIST),
            ])
            with self.assertRaisesRegex(FileNotFoundError, "refusing to download"):
                q.run_screen_command(args)
            self.assertFalse((root / "run").exists())


if __name__ == "__main__":
    unittest.main()
