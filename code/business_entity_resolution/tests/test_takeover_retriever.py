"""CPU-only correctness tests; no HF model download, supplied data, or GPU needed.

Run from the repository root:
  PYTHONPATH=code/business_entity_resolution/src python -m unittest discover -s code/business_entity_resolution/tests -p test_takeover_retriever.py -v
"""
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from ber import takeover_retriever as r


class TinyTokenizer:
    """Whitespace tokenizer with real clipping/padding; IDs never enter text."""
    seen: ClassVar[list[str]] = []

    def __call__(self, texts, *, padding, truncation, max_length, pad_to_multiple_of, return_tensors):
        self.seen.extend(texts)
        encoded = [[1] + [3 + sum(word.encode()) % 28 for word in text.split()] + [2] for text in texts]
        encoded = [x[:max_length] for x in encoded]
        width = max(map(len, encoded))
        if pad_to_multiple_of:
            width = ((width + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(encoded):
            ids[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def save_pretrained(self, path):
        (Path(path) / "tokenizer.json").write_text('{"test_tokenizer":true}')


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.projection = torch.nn.Linear(8, 8)
        self.dropout = torch.nn.Dropout(0.2)
        self.config = SimpleNamespace(hidden_size=8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.dropout(self.projection(self.embedding(input_ids))))

    def save_pretrained(self, path, safe_serialization=True):
        torch.save(self.state_dict(), Path(path) / "pytorch_model.bin")
        (Path(path) / "config.json").write_text('{"hidden_size":8,"test_model":true}')


def tiny_load(path, device):
    model = TinyEncoder()
    model.load_state_dict(torch.load(Path(path) / "pytorch_model.bin", weights_only=True, map_location="cpu"))
    return model.to(device), TinyTokenizer()


def make_model(path):
    path.mkdir()
    torch.manual_seed(71)
    TinyEncoder().save_pretrained(path)
    TinyTokenizer().save_pretrained(path)
    return path


def training_rows():
    # A2 is a real A copy presented as B's hard negative. Same text, distinct
    # A1/A2 records must remain distinct positives. S is a true singleton.
    pairs = [("A", "A1", 1), ("A", "A2", 1), ("A", "A3", 1), ("A", "X", 0),
             ("B", "B1", 1), ("B", "A2", 0), ("B", "X", 0),
             ("C", "C1", 1), ("C", "C2", 1), ("C", "Y", 0),
             ("S", "Z", 0)]
    texts = {"A1": "Alpha store", "A2": "Alpha store", "A3": "Alpha shop", "B1": "Beta store",
             "C1": "Gamma", "C2": "Gamma store", "X": "Alpha rival", "Y": "Gamma rival", "Z": "Nobody"}
    records = {key: i for i, key in enumerate(texts)}
    counts = {"A": 4, "B": 1, "C": 2, "S": 0}  # A has an unretrieved truth copy.
    return [{"pair_row": i, "left_idx": ord(g) - 65, "right_idx": records[rec],
             "s1_id": g, "rec_id": rec, "left_text": {"A": "Alpha", "B": "Beta", "C": "Gamma", "S": "Solo"}[g],
             "right_text": texts[rec], "label": label, "true_count": counts[g]} for i, (g, rec, label) in enumerate(pairs)]


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def pack(self, rows=None):
        path = self.root / "train.parquet"
        pq.write_table(pa.Table.from_pylist(training_rows() if rows is None else rows), path, row_group_size=3)
        return path

    def test_known_copy_masks_and_duplicate_positive_columns(self):
        pos, neg = r.build_masks([0, 1], [10, 11, 10, 12, 13], [0, 0, 0, 1, -1])
        np.testing.assert_array_equal(pos[0], [True, True, False, False, False])
        self.assertFalse(neg[0, :3].any())
        self.assertFalse(pos[:, 2].any() or neg[:, 2].any())
        self.assertTrue(neg[0, 3] and neg[0, 4] and neg[1, 0])
        with self.assertRaisesRegex(ValueError, "distinct"):
            r.build_masks([0, 0], [10], [0])
        with self.assertRaisesRegex(ValueError, "conflicting ownership"):
            r.build_masks([0, 1], [10, 10], [0, 1])

    def test_control_equals_old_symmetric_cross_entropy(self):
        torch.manual_seed(5)
        q, p = F.normalize(torch.randn(4, 6), dim=1), F.normalize(torch.randn(4, 6), dim=1)
        pos, neg = r.build_masks(np.arange(4), np.arange(4), np.arange(4))
        scores = q @ p.T / 0.05
        expected = (F.cross_entropy(scores, torch.arange(4)) + F.cross_entropy(scores.T, torch.arange(4))) / 2
        self.assertTrue(torch.allclose(r.contrastive_loss(q, p, pos, neg), expected, atol=1e-6))

    def test_finite_gradients_no_same_business_repulsion_and_duplicate_invariance(self):
        q = torch.eye(2, requires_grad=True)
        p = torch.tensor([[0.9, 0.1], [0.4, 0.6], [0.0, 1.0], [0.5, 0.5]], requires_grad=True)
        pos, neg = r.build_masks([0, 1], [1, 2, 3, 4], [0, 0, 1, -1])
        base = r.contrastive_loss(q, p, pos, neg, 0.2)
        base.backward()
        self.assertTrue(torch.isfinite(q.grad).all() and torch.isfinite(p.grad).all())
        # Increasing either true score helps; a positive is never a negative of its sibling.
        for row in (0, 1):
            changed = p.detach().clone()
            changed[row, 0] += 0.01
            self.assertLess(r.contrastive_loss(q.detach(), changed, pos, neg, 0.2).item(), base.item())
        duplicated = torch.cat((p.detach(), p.detach()[0:1]), dim=0).requires_grad_()
        pm, nm = r.build_masks([0, 1], [1, 2, 3, 4, 1], [0, 0, 1, -1, 0])
        duplicate_loss = r.contrastive_loss(q.detach(), duplicated, pm, nm, 0.2)
        self.assertTrue(torch.allclose(duplicate_loss, base))
        duplicate_loss.backward()
        self.assertTrue(torch.equal(duplicated.grad[-1], torch.zeros(2)))

    def test_no_negative_batch_has_finite_zero_loss_and_gradients(self):
        q, p = torch.randn(1, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        pos, neg = r.build_masks([0], [1, 2], [0, 0])
        loss = r.contrastive_loss(q, p, pos, neg)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(q.grad).all() and torch.isfinite(p.grad).all())
        with self.assertRaisesRegex(ValueError, "genuine positive"):
            r.contrastive_loss(q, p, np.zeros_like(pos), np.ones_like(neg))

    def test_whole_group_loader_sampling_and_matched_plans(self):
        data = r.load_training_data(self.pack())
        self.assertEqual(data.stats["singleton_businesses"], 1)
        self.assertEqual(len(data.eligible), 3)
        plan = r.batch_plan(data.eligible, 5, 3, 1, 31)
        np.testing.assert_array_equal(plan, r.batch_plan(data.eligible, 5, 3, 1, 31))
        for step, groups in enumerate(plan):
            control, _, _ = r.sample_passages(data, groups, step, 31, "control")
            multi, pos, neg = r.sample_passages(data, groups, step, 31, "multi")
            self.assertEqual(len(control), 3)
            self.assertEqual(len(multi), len(np.unique(multi)))
            self.assertTrue(set(control).issubset(set(multi)))
            self.assertTrue(np.array_equal(pos, data.owners[multi][None, :] == groups[:, None]))
            self.assertFalse(np.any(neg & pos))
            for i, group in enumerate(groups):
                self.assertGreaterEqual(pos[i].sum(), min(2, len(data.positive_records[group])))
                for record in data.positive_records[group]:
                    self.assertFalse(neg[i, multi == record].any())

    def test_loader_rejects_duplicate_pairs_conflicts_and_alignment_errors(self):
        rows = training_rows()
        for mutate, message in [
                (lambda x: x.append({**x[0], "pair_row": 99}), "Duplicate training pairs"),
                (lambda x: x[5].update(label=1, true_count=2), "multiple S1"),
                (lambda x: x[1].update(left_text="changed"), "Inconsistent left_text"),
                (lambda x: x[1].update(true_count=0), "Inconsistent true_count")]:
            changed = [dict(x) for x in rows]
            mutate(changed)
            with self.assertRaisesRegex(ValueError, message):
                r.load_training_data(self.pack(changed))

    def test_prefix_token_clipping_dynamic_padding_and_pooling(self):
        tokenizer = TinyTokenizer()
        raw = "Café दुकान | 12 rue de Paris"
        for kind in ("query", "passage"):
            first = r.tokenize(tokenizer, [raw], kind, 128, "cpu")
            second = r.tokenize(tokenizer, [f"{kind}: {raw}"], kind, 128, "cpu")
            self.assertTrue(torch.equal(first["input_ids"], second["input_ids"]))
            self.assertLess(first["input_ids"].shape[1], 128)
        clipped = r.tokenize(tokenizer, ["word " * 400], "query", 13, "cpu")
        self.assertEqual(clipped["input_ids"].shape[1], 13)
        with self.assertRaisesRegex(ValueError, "Wrong E5 prefix"):
            r.with_prefix("passage: store", "query")
        pooled = r.mean_pool(torch.tensor([[[3., 0.], [0., 4.], [999., 999.]]]), torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(pooled, torch.tensor([[0.6, 0.8]]))

    def test_training_checkpoint_resume_reproduces_dropout_weights_optimizer_and_rng(self):
        source, model = self.pack(), make_model(self.root / "model")
        common = ["train", "--train", str(source), "--model", str(model), "--mode", "multi", "--steps", "4",
                  "--batch-size", "3", "--seed", "53", "--save-every", "2", "--device", "cpu", "--threads", "1"]
        with patch.object(r, "load_encoder", side_effect=tiny_load):
            result = r.main(common + ["--output", str(self.root / "full")])
            self.assertEqual(result["state"], "complete")
            full_last = self.root / "full/checkpoints/step_000004"
            original = torch.load(full_last / "pytorch_model.bin", weights_only=True)
            saved_rng = torch.load(full_last / "rng.pt", weights_only=True)
            saved_optimizer = torch.load(full_last / "optimizer.pt", weights_only=True)

            @contextmanager
            def after_two_steps(minutes):
                calls = [0]

                def stop():
                    calls[0] += 1
                    return calls[0] > 2

                yield stop

            with patch.object(r, "stop_requested", side_effect=after_two_steps):
                partial = r.main(common + ["--output", str(self.root / "resumed")])
            self.assertEqual((partial["state"], partial["step"]), ("paused", 2))
            # The checkpoint commit also survives an unpublished latest.json pointer.
            (self.root / "resumed/latest.json").unlink()
            r.main(common + ["--output", str(self.root / "resumed"), "--resume"])
            resumed_last = self.root / "resumed/checkpoints/step_000004"
            resumed_weights = torch.load(resumed_last / "pytorch_model.bin", weights_only=True)
            resumed_rng = torch.load(resumed_last / "rng.pt", weights_only=True)
            resumed_optimizer = torch.load(resumed_last / "optimizer.pt", weights_only=True)
            for name in original:
                self.assertTrue(torch.equal(original[name], resumed_weights[name]), name)
            self.assertTrue(torch.equal(saved_rng["torch"], resumed_rng["torch"]))
            for key, state in saved_optimizer["state"].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, resumed_optimizer["state"][key][name]))
            with self.assertRaisesRegex(ValueError, "configuration"):
                r.main(common + ["--output", str(self.root / "full"), "--steps", "5", "--resume"])
            with self.assertRaisesRegex(ValueError, "rewind"):
                r.main(common + ["--output", str(self.root / "full"), "--resume",
                                 str(self.root / "full/checkpoints/step_000002")])
        (full_last / "optimizer.pt").write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "changed"):
            r.verify_checkpoint(full_last)

    def test_wall_limit_saves_complete_resumable_checkpoint_without_claiming_finished(self):
        source, model = self.pack(), make_model(self.root / "model")
        command = ["train", "--train", str(source), "--model", str(model), "--mode", "control", "--steps", "2",
                   "--batch-size", "3", "--output", str(self.root / "limited"), "--device", "cpu", "--threads", "1"]
        with patch.object(r, "load_encoder", side_effect=tiny_load):
            result = r.main(command + ["--max-minutes", "0.0000000001"])
            self.assertEqual((result["state"], result["step"]), ("paused", 0))
            self.assertFalse((self.root / "limited/complete.json").exists())
            r.verify_checkpoint(self.root / "limited" / result["checkpoint"])
            result = r.main(command + ["--resume"])
            self.assertEqual((result["state"], result["step"]), ("complete", 2))

    def test_encode_stream_resume_order_prefix_and_corruption_detection(self):
        model = make_model(self.root / "model")
        source = self.root / "texts.parquet"
        texts = ["query: Alpha", "Beta store", "query: Gamma", "long " * 40, "Other", "query: Last", "Final"]
        pq.write_table(pa.table({"entity_id": [f"q{i}" for i in range(7)], "text": texts,
                                 "cc": np.array([0, 1, 0, 1, 2, 1, 0], np.int8)}), source, row_group_size=3)
        command = ["encode", "--input", str(source), "--model", str(model), "--kind", "query", "--batch-size", "2",
                   "--chunk-rows", "2", "--max-length", "16", "--device", "cpu", "--threads", "1"]

        @contextmanager
        def one_chunk(minutes):
            calls = [0]

            def stop():
                calls[0] += 1
                return calls[0] > 1

            yield stop

        with patch.object(r, "load_encoder", side_effect=tiny_load):
            r.main(command + ["--output", str(self.root / "full")])
            with patch.object(r, "stop_requested", side_effect=one_chunk):
                result = r.main(command + ["--output", str(self.root / "resumed")])
            self.assertEqual((result["state"], result["rows_done"]), ("paused", 2))
            self.assertFalse((self.root / "resumed/complete.json").exists())
            TinyTokenizer.seen = []
            r.main(command + ["--output", str(self.root / "resumed"), "--resume"])
            self.assertNotIn("query: Alpha", TinyTokenizer.seen)
            self.assertNotIn("query: Beta store", TinyTokenizer.seen)
            np.testing.assert_array_equal(np.load(self.root / "full/vectors.npy"), np.load(self.root / "resumed/vectors.npy"))
            manifest = r.read_json(self.root / "resumed/manifest.json")
            ids = pa.concat_tables([pq.read_table(self.root / "resumed" / c["ids"]) for c in manifest["chunks"]])
            self.assertEqual(ids["entity_id"].to_pylist(), [f"q{i}" for i in range(7)])
            self.assertEqual(ids["row_idx"].to_pylist(), list(range(7)))
            values = np.load(self.root / "resumed/vectors.npy", mmap_mode="r+")
            values[0, 0] += 0.1
            values.flush()
            with self.assertRaisesRegex(ValueError, "chunk changed"):
                r.main(command + ["--output", str(self.root / "resumed"), "--resume"])

    def test_encode_rejects_duplicate_ids_across_chunks(self):
        model = make_model(self.root / "model")
        source = self.root / "dup.parquet"
        pq.write_table(pa.table({"entity_id": ["x", "y", "x"], "text": ["a", "b", "c"], "cc": [0, 0, 0]}), source)
        with patch.object(r, "load_encoder", side_effect=tiny_load), self.assertRaisesRegex(ValueError, "Duplicate encoding"):
            r.main(["encode", "--input", str(source), "--model", str(model), "--output", str(self.root / "encoded"),
                    "--kind", "passage", "--chunk-rows", "1", "--device", "cpu", "--threads", "1"])
        self.assertFalse((self.root / "encoded/complete.json").exists())

    def test_exact_tiled_country_search_matches_brute_force_with_ties_missing_and_subset(self):
        rng = np.random.default_rng(4)
        q = rng.normal(size=(7, 5)).astype(np.float16)
        p = rng.normal(size=(11, 5)).astype(np.float16)
        p[7] = p[0]  # tie across pool tiles must prefer original pool row 0.
        qc = np.array([0, 1, 2, 0, 1, 0, 2])
        pc = np.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1])
        rows = np.array([4, 2, 0, 6, 3])
        expected_ix = np.full((len(rows), 8), -1, np.int64)
        expected_sc = np.full((len(rows), 8), -np.inf, np.float32)
        qn, pn = q.astype(np.float32), p.astype(np.float32)
        qn /= np.linalg.norm(qn, axis=1, keepdims=True)
        pn /= np.linalg.norm(pn, axis=1, keepdims=True)
        for out, row in enumerate(rows):
            eligible = np.flatnonzero(pc == qc[row])
            sc = qn[row] @ pn[eligible].T
            order = np.lexsort((eligible, -sc))[:8]
            expected_ix[out, :len(order)], expected_sc[out, :len(order)] = eligible[order], sc[order]
        for qt, pt in [(1, 1), (2, 3), (8, 20)]:
            actual_ix, actual_sc = np.empty_like(expected_ix), np.empty_like(expected_sc)
            for positions, ix, sc in r.iter_search_tiles(q, p, qc, pc, query_rows=rows,
                                                       top_k=8, query_tile=qt, pool_tile=pt, device="cpu"):
                actual_ix[positions], actual_sc[positions] = ix, sc
            np.testing.assert_array_equal(actual_ix, expected_ix)
            np.testing.assert_allclose(actual_sc, expected_sc, atol=3e-7)
        with self.assertRaisesRegex(ValueError, "unique valid"):
            list(r.iter_search_tiles(q, p, qc, pc, query_rows=np.array([0, 0]), device="cpu"))

    def test_search_cli_outputs_original_indices_and_complete_marker(self):
        values = {"queries": np.array([[2., 0.], [0., 3.], [1., 1.]], np.float16),
                  "pool": np.array([[0., 4.], [5., 0.]], np.float16),
                  "query_countries": np.array([0, 1, 2], np.int8),
                  "pool_countries": np.array([1, 0], np.int8), "query_rows": np.array([2, 0, 1])}
        command = ["search", "--out", str(self.root / "search"), "--top-k", "3", "--device", "cpu", "--threads", "1"]
        for name, array in values.items():
            path = self.root / f"{name}.npy"
            np.save(path, array)
            command += ["--" + name.replace("_", "-"), str(path)]
        result = r.main(command)
        self.assertEqual(result["state"], "complete")
        np.testing.assert_array_equal(np.load(self.root / "search/query_indices.npy"), [2, 0, 1])
        np.testing.assert_array_equal(np.load(self.root / "search/pool_indices.npy"), [[-1, -1, -1], [1, -1, -1], [0, -1, -1]])
        with self.assertRaises(FileExistsError):
            r.main(command)

    def test_search_refuses_unfinished_encoding_artifacts(self):
        q, p = self.root / "q", self.root / "p"
        q.mkdir()
        p.mkdir()
        for folder in (q, p):
            np.save(folder / "vectors.npy", np.ones((2, 4), np.float16))
            np.save(folder / "countries.npy", np.zeros(2, np.int8))
            r.write_json(folder / "run_config.json", {"command": "encode"})
        with self.assertRaisesRegex(ValueError, "Unfinished encoding"):
            r.main(["search", "--queries", str(q / "vectors.npy"), "--pool", str(p / "vectors.npy"),
                    "--query-countries", str(q / "countries.npy"), "--pool-countries", str(p / "countries.npy"),
                    "--out", str(self.root / "search"), "--device", "cpu", "--threads", "1"])


if __name__ == "__main__":
    unittest.main()
