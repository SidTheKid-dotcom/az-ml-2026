"""CPU tests for ber.qwen_train with a tiny random Qwen3 (no downloads)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ber import qwen_train as qt  # noqa: E402

YES, NO, PAD = 5, 6, 0


class CharTokenizer:
    """Tiny stand-in: one token per character (ord % 97 + 7)."""

    pad_token_id = PAD
    eos_token_id = 1
    unk_token_id = 2

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            return {"input_ids": [ord(c) % 97 + 7 for c in texts]}
        return {"input_ids": [[ord(c) % 97 + 7 for c in text] for text in texts]}

    def convert_tokens_to_ids(self, word):
        return {"yes": YES, "no": NO}[word]


def tiny_model_dir(tmp_path: Path) -> Path:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512,
                         tie_word_embeddings=True)
    path = tmp_path / "tiny_qwen3"
    Qwen3ForCausalLM(config).save_pretrained(path)
    return path


def test_assemble_matches_manual_concatenation(monkeypatch):
    monkeypatch.setattr(qt, "MAX_LENGTH", 1024)  # the character tokenizer makes the prompt long
    tok = CharTokenizer()
    left_text, right_text = ["ab | c", "x" * 150, ""], ["d | e", "fgh", "y" * 3]
    tokens, offsets = qt.tokenize_pairs(tok, left_text, right_text, np.array([0, 1, 2, 0]), np.array([1, 0, 2, 2]))
    pre, mid, suf = (np.asarray(tok(t)["input_ids"]) for t in (qt.PROMPT_PREFIX, qt.PROMPT_MIDDLE, qt.PROMPT_SUFFIX))
    for k, (i, j) in enumerate([(0, 1), (1, 0), (2, 2), (0, 2)]):
        expect = np.concatenate([pre, np.asarray(tok(left_text[i])["input_ids"][:qt.RECORD_CAP], dtype=np.int64), mid,
                                 np.asarray(tok(right_text[j])["input_ids"][:qt.RECORD_CAP], dtype=np.int64), suf])
        assert np.array_equal(tokens[offsets[k]:offsets[k + 1]], expect)


def test_pad_right_and_microbatches():
    tokens = np.arange(1, 31, dtype=np.int32)
    offsets = np.array([0, 3, 10, 12, 30])
    ids, last = qt.pad_right(tokens, offsets, np.array([2, 0, 3]), PAD)
    assert ids.shape == (3, 48) and list(last) == [1, 2, 17]
    assert list(ids[1, :4]) == [1, 2, 3, PAD]
    lengths = np.array([5, 1, 9, 3, 7, 2, 8, 4])
    offs = np.r_[0, np.cumsum(lengths)]
    parts = qt.step_microbatches(offs, 1, 4, 2)
    assert sorted(np.concatenate(parts).tolist()) == [4, 5, 6, 7]
    assert [lengths[p].tolist() for p in parts] == [[2, 4], [7, 8]]
    assert qt.bucket_width(49) == 64 and qt.bucket_width(256) == 256


def test_unmasked_right_padding_matches_single_sequence_and_lm_logits(tmp_path):
    model = qt.load_model(tiny_model_dir(tmp_path), "cpu", torch)
    model.eval()
    scorer = qt.YesNoScorer(model, YES, NO)
    seqs = [np.arange(3, 3 + n) % 120 + 3 for n in (5, 17, 9)]
    tokens = np.concatenate(seqs).astype(np.int32)
    offsets = np.r_[0, np.cumsum([len(s) for s in seqs])]
    ids, last = qt.pad_right(tokens, offsets, np.arange(3), PAD)
    with torch.no_grad():
        batch = scorer(torch, torch.from_numpy(ids), torch.from_numpy(last)).numpy()
        for k, seq in enumerate(seqs):
            one = torch.from_numpy(np.asarray(seq, dtype=np.int64))[None]
            alone = scorer(torch, one, torch.tensor([len(seq) - 1])).numpy()[0]
            logits = model(input_ids=one).logits[0, -1]
            assert batch[k] == pytest.approx(alone, abs=1e-5)
            assert batch[k] == pytest.approx(float(logits[YES] - logits[NO]), abs=1e-5)


def write_train_data(path: Path, steps: int = 4, batch: int = 8) -> None:
    rng = np.random.default_rng(1)
    lengths = rng.integers(6, 30, size=steps * batch)
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    tokens = rng.integers(3, 120, size=int(offsets[-1])).astype(np.int32)
    labels = np.tile(np.array([1, 0, 0, 0], dtype=np.int8), steps * batch // 4)
    weights = rng.uniform(.5, 2, size=steps * batch).astype(np.float32)
    path.mkdir(parents=True)
    for name, values in (("tokens.npy", tokens), ("offsets.npy", offsets), ("labels.npy", labels),
                         ("weights.npy", weights)):
        np.save(path / name, values)
    manifest = {"kind": "train", "state": "complete", "batch_size": batch, "yes_id": YES, "no_id": NO,
                "pad_id": PAD, "prompt_sha256": "p", "files": {"tokens.npy": "x"}}
    (path / "manifest.json").write_text(json.dumps(manifest))


def train_args(data, model, output, **extra):
    base = dict(data=str(data), model=str(model), output=str(output), device="cpu", amp="fp32", steps=None,
                time_budget_min=None, reserve_min=0.0, calibrate_steps=220, microbatches=2, learning_rate=1e-3,
                weight_decay=.01, warmup=1, schedule="cosine", lr_floor=.1, checkpoint_every=2, log_every=1,
                resume=False, compile=False, threads=1, seed=7)
    base.update(extra)
    return SimpleNamespace(**base)


def weights_of(checkpoint: Path) -> dict:
    from safetensors.torch import load_file

    return load_file(checkpoint / "model.safetensors")


def test_training_resume_is_exact(tmp_path, monkeypatch):
    model_dir = tiny_model_dir(tmp_path)
    data = tmp_path / "data"
    write_train_data(data)
    full = qt.train(train_args(data, model_dir, tmp_path / "full"))
    assert full["completed_steps"] == 4 and full["stop_reason"] == "max_steps"
    checkpoints = sorted((tmp_path / "full" / "checkpoints").glob("step_*"))
    assert [p.name for p in checkpoints] == ["step_000002", "step_000004"]
    assert not (checkpoints[0] / "optimizer.pt").exists() and (checkpoints[1] / "optimizer.pt").exists()

    real_lr = qt.lr_at

    def crash(step, **kwargs):
        if step == 3:
            raise RuntimeError("simulated preemption")
        return real_lr(step, **kwargs)

    monkeypatch.setattr(qt, "lr_at", crash)
    with pytest.raises(RuntimeError):
        qt.train(train_args(data, model_dir, tmp_path / "resumed"))
    monkeypatch.setattr(qt, "lr_at", real_lr)
    resumed = qt.train(train_args(data, model_dir, tmp_path / "resumed", resume=True))
    assert resumed["completed_steps"] == 4
    a = weights_of(tmp_path / "full" / "checkpoints" / "step_000004")
    b = weights_of(tmp_path / "resumed" / "checkpoints" / "step_000004")
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    first = weights_of(tmp_path / "full" / "checkpoints" / "step_000002")
    assert any(not torch.equal(first[k], a[k]) for k in a)
    status = json.loads((tmp_path / "full" / "status.json").read_text())
    assert status["step"] == 4 and "pairs_per_s" in status


def test_time_budget_sizes_total_steps(tmp_path):
    model_dir = tiny_model_dir(tmp_path)
    data = tmp_path / "data"
    write_train_data(data, steps=40, batch=4)
    result = qt.train(train_args(data, model_dir, tmp_path / "budget", time_budget_min=5.0, calibrate_steps=22,
                                 checkpoint_every=1000, log_every=1000))
    assert result["total_steps"] == 40 and result["completed_steps"] == 40  # capped by the materialized steps
    assert result["stop_reason"] == "max_steps"
    stopped = qt.train(train_args(data, model_dir, tmp_path / "over", time_budget_min=1e-9, checkpoint_every=1000))
    assert stopped["stop_reason"] == "time_budget_exceeded" and stopped["completed_steps"] == 1
    assert (tmp_path / "over" / "checkpoints" / "step_000001" / "complete.json").exists()


def test_score_writes_c2_layout(tmp_path):
    model_dir = tiny_model_dir(tmp_path)
    data = tmp_path / "train"
    write_train_data(data, steps=2, batch=4)
    qt.train(train_args(data, model_dir, tmp_path / "run"))
    checkpoint = tmp_path / "run" / "checkpoints" / "step_000002"
    score_data = tmp_path / "score"
    score_data.mkdir()
    rng = np.random.default_rng(3)
    lengths = rng.integers(5, 40, size=7)
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    tokens = rng.integers(3, 120, size=int(offsets[-1])).astype(np.int32)
    pair_index = np.array([1, 4, 5, 9, 10, 15, 19])
    for name, values in (("tokens.npy", tokens), ("offsets.npy", offsets), ("pair_index.npy", pair_index)):
        np.save(score_data / name, values)
    (score_data / "manifest.json").write_text(json.dumps({
        "kind": "test", "state": "complete", "yes_id": YES, "no_id": NO, "pad_id": PAD, "prompt_sha256": "p",
        "total_pairs": 20, "gate": .01}))
    args = SimpleNamespace(data=str(score_data), checkpoint=str(checkpoint), output=str(tmp_path / "out"),
                           device="cpu", amp="fp32", batch_size=3, token_budget=10_000, flush_every=1, threads=1,
                           c2_scores=None, min_c2=None)
    result = qt.score(args)
    scores = np.load(tmp_path / "out" / "scores.npy")
    mask = np.load(tmp_path / "out" / "scored_mask.npy")
    assert result["scored_pairs"] == 7 and scores.shape == (20,) and mask.sum() == 7
    assert np.array_equal(np.flatnonzero(mask), pair_index) and np.all(scores[~mask] == 0)
    model = qt.load_model(checkpoint, "cpu", torch)
    model.eval()
    scorer = qt.YesNoScorer(model, YES, NO)
    with torch.no_grad():
        for k in range(7):
            seq = torch.from_numpy(tokens[offsets[k]:offsets[k + 1]].astype(np.int64))[None]
            value = float(scorer(torch, seq, torch.tensor([lengths[k] - 1]))[0])
            assert scores[pair_index[k]] == pytest.approx(1 / (1 + np.exp(-value)), abs=1e-5)
    assert json.loads((tmp_path / "out" / "complete.json").read_text())["state"] == "complete"

    # optional C2 filter: only gated rows with C2 >= min_c2 are scored, the rest stay unscored
    c2_dir = tmp_path / "c2"
    c2_dir.mkdir()
    c2 = np.zeros(20, dtype=np.float32)
    c2[pair_index] = [.5, .001, .2, .0, .9, .05, .3]
    np.save(c2_dir / "scores.npy", c2)
    np.save(c2_dir / "scored_mask.npy", np.isin(np.arange(20), pair_index))
    (c2_dir / "complete.json").write_text("{}")
    filtered = SimpleNamespace(**{**vars(args), "output": str(tmp_path / "out_filtered"), "c2_scores": str(c2_dir),
                                  "min_c2": .05})
    result = qt.score(filtered)
    mask_f = np.load(tmp_path / "out_filtered" / "scored_mask.npy")
    scores_f = np.load(tmp_path / "out_filtered" / "scores.npy")
    assert np.array_equal(np.flatnonzero(mask_f), pair_index[[0, 2, 4, 5, 6]])
    assert np.allclose(scores_f[mask_f], scores[mask_f], atol=1e-6) and np.all(scores_f[~mask_f] == 0)
    assert result["c2_filter"]["kept"] == 5 and result["scored_pairs"] == 5
