"""Offline CPU tests: real tiny T5 forwards/backwards, parquet and exact resume."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from transformers import ByT5Tokenizer, T5Config, T5ForConditionalGeneration

from ber import takeover_byte as byte


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny(tmp_path):
    torch.manual_seed(61)
    tokenizer = ByT5Tokenizer()
    model = T5ForConditionalGeneration(T5Config(
        vocab_size=len(tokenizer), d_model=24, d_kv=8, d_ff=48,
        num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0.15,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    ))
    location = tmp_path / "tiny"
    model.save_pretrained(location)
    tokenizer.save_pretrained(location)
    return model, tokenizer, location


def pair_rows(n=12):
    names = ["Café à Paris", "दुकान १२", "東京店", "A <pad> shop"]
    return [dict(pair_row=100 + (n - i) * 7, s1_id=f"S1-{i // 2}", rec_id=f"S{2 + i % 2}-{i}",
                 left_text=names[i % 4] + " | 12 Rue", right_text=names[(i + i % 2) % 4] + f" | {12 + i} Rue",
                 source=f"S{2 + i % 2}", label=i % 2) for i in range(n)]


def write_pairs(path, rows):
    table = pa.Table.from_pylist(rows)
    if "label" in table.column_names and pa.types.is_integer(table["label"].type):
        table = table.set_column(table.schema.get_field_index("label"), "label", table["label"].cast(pa.int8()))
    pq.write_table(table, path)


def train_args(path, model, output, steps=3):
    return ["train", "--train", str(path), "--model", str(model), "--output", str(output),
            "--steps", str(steps), "--batch-size", "4", "--microbatch", "2", "--seed", "17",
            "--max-pairs", "10", "--max-input-length", "40", "--max-target-length", "32",
            "--mle-warmup-steps", "1", "--discriminative-weight", "0.3", "--save-every", "2",
            "--log-every", "1", "--device", "cpu", "--amp", "fp32", "--threads", "1", "--local-files-only"]


def score_args(pairs, model, output):
    return ["score", "--pairs", str(pairs), "--model", str(model), "--output", str(output),
            "--batch-size", "3", "--chunk-size", "5", "--device", "cpu", "--amp", "fp32", "--threads", "1"]


def test_masked_nll_and_gradients_ignore_padding():
    logits = torch.tensor([[[0., 1., 2.], [3., 1., -2.], [7., -1., 5.]],
                           [[1., 0., 2.], [0., 0., 0.], [-1., 2., 1.]]], requires_grad=True)
    # Token ID zero is deliberately valid in the first sequence.
    ids = torch.tensor([[2, 0, 1], [1, 2, 0]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    nll, count = byte.masked_nll(logits, ids, mask)
    logp = logits.log_softmax(-1)
    torch.testing.assert_close(nll, torch.stack([-logp[0, 0, 2] - logp[0, 1, 0], -logp[1, 0, 1]]))
    assert count.tolist() == [2, 1]
    nll.sum().backward()
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    with pytest.raises(ValueError, match="Expected logits"):
        byte.masked_nll(logits, ids[:, :2], mask)
    with pytest.raises(ValueError, match="at least EOS"):
        byte.masked_nll(logits, ids, torch.zeros_like(mask))


def test_utf8_common_context_literal_special_tokens_and_eos(tiny):
    _, tokenizer, _ = tiny
    rows = [{"source": "S2", "left_text": "é東京🙂" * 20, "right_text": "दुकान" * 20}]
    batch, stat = byte.collate_pairs(tokenizer, rows, max_input_length=34, max_target_length=18, pad_to_multiple_of=1)
    text = [tokenizer.decode(ids[mask.bool()], skip_special_tokens=True)
            for ids, mask in zip(batch["input_ids"], batch["attention_mask"])]
    assert text[0].removeprefix("COPY S2: ") == text[1].removeprefix("NONMATCH S2: ")
    assert all("�" not in t for t in text)
    assert stat.report()["left_clip_rate"] == stat.report()["right_clip_rate"] == 1
    for ids, mask in zip(batch["target_ids"], batch["target_mask"]):
        assert ids[mask][-1] == tokenizer.eos_token_id
        assert len(ids[mask]) <= 18
        assert "�" not in tokenizer.decode(ids[mask], skip_special_tokens=True)
    literal = [{"source": "S3", "left_text": "A <pad>", "right_text": "<pad> </s> café"}]
    batch, _ = byte.collate_pairs(tokenizer, literal, pad_to_multiple_of=1)
    valid = batch["target_ids"][0][batch["target_mask"][0]]
    assert tokenizer.pad_token_id not in valid.tolist()
    assert tokenizer.decode(valid, skip_special_tokens=True) == literal[0]["right_text"]


def test_eval_padding_width_invariance_and_conditions_differ(tiny):
    model, tokenizer, _ = tiny
    model.eval()
    rows = pair_rows(3)
    with torch.no_grad():
        narrow, _ = byte.collate_pairs(tokenizer, rows, pad_to_multiple_of=1)
        wide, _ = byte.collate_pairs(tokenizer, rows, pad_to_multiple_of=64)
        a, ac = byte.teacher_forced_nll(model, narrow)
        b, bc = byte.teacher_forced_nll(model, wide)
    torch.testing.assert_close(ac, bc, rtol=0, atol=0)
    torch.testing.assert_close(a, b, rtol=1e-6, atol=3e-5)
    assert not torch.allclose(a[:3], a[3:], atol=1e-4, rtol=0)
    assert narrow["input_ids"].shape[0] == 6


def test_discriminative_loss_has_correct_sign_and_both_gradients():
    copy = torch.tensor([2., 9.], requires_grad=True)
    nonmatch = torch.tensor([9., 2.], requires_grad=True)
    counts, labels = torch.tensor([2, 2]), torch.tensor([1, 0])
    mle, disc = byte.discriminative_objectives(copy, nonmatch, counts, labels, temperature=1)
    assert mle.item() == 1
    assert disc.item() < 0.04
    disc.backward()
    assert copy.grad[0] > 0 and nonmatch.grad[0] < 0
    assert copy.grad[1] < 0 and nonmatch.grad[1] > 0
    _, wrong = byte.discriminative_objectives(copy, nonmatch, counts, 1 - labels, temperature=1)
    assert wrong.item() > 3


def test_selection_balance_seed_hash_and_a_guard(tmp_path):
    path = tmp_path / "train.parquet"
    write_pairs(path, pair_rows(20))
    a, positions, identity = byte.select_training(path, 10, 9)
    b, positions_b, identity_b = byte.select_training(path, 10, 9)
    assert a == b and identity == identity_b
    np.testing.assert_array_equal(positions, positions_b)
    assert sum(r["label"] for r in a) == 5
    assert len({r["pair_row"] for r in a}) == 10
    assert byte.select_training(path, 10, 10)[2]["selected_data_sha256"] != identity["selected_data_sha256"]
    manifest = dict(state="complete", partition="A", holdout2_rows_read=0, train_sha256=byte.sha256(path))
    (tmp_path / "train_complete.json").write_text(json.dumps(manifest))
    assert byte.select_training(path, 10, 9)[2]["partition_check"] == "train_complete.json"
    manifest["holdout2_rows_read"] = 1
    (tmp_path / "train_complete.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="holdout-excluded"):
        byte.select_training(path, 10, 9)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_nested_equal(left, right)
    else:
        assert a == b


def test_train_resume_bitwise_optimizer_rng_and_guards(tiny, tmp_path, capsys):
    original, _, model_path = tiny
    path = tmp_path / "train.parquet"
    write_pairs(path, pair_rows())
    full, split = tmp_path / "full", tmp_path / "split"
    byte.main(train_args(path, model_path, full, 4))
    byte.main(train_args(path, model_path, split, 2))
    # An unfinished save must not supersede the latest completed checkpoint.
    (split / ".checkpoint-00000003-interrupted").mkdir()
    report = byte.main(train_args(path, model_path, split, 4) + ["--resume"])
    assert report["step"] == 4
    checkpoints = [byte.checkpoint_path(p) for p in (full, split)]
    states = [torch.load(p / "training_state.pt", map_location="cpu", weights_only=False) for p in checkpoints]
    assert_nested_equal(states[0], states[1])
    fitted = [byte.load_model(p, torch.device("cpu"), True)[0] for p in checkpoints]
    assert_nested_equal(fitted[0].state_dict(), fitted[1].state_dict())
    assert any(not torch.equal(v, original.state_dict()[k]) for k, v in fitted[0].state_dict().items())
    with np.load(checkpoints[1] / "selected_rows.npz") as selected:
        assert len(selected["pair_row"]) == 10 and len(np.unique(selected["pair_row"])) == 10
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{"')]
    losses = [e for e in events if e.get("event") == "loss"]
    assert any(e["discriminative_bce"] is None for e in losses)
    assert any(e["discriminative_active"] and e["discriminative_bce"] > 0 for e in losses)
    with pytest.raises(FileExistsError):
        byte.main(train_args(path, model_path, split, 4))
    changed_args = train_args(path, model_path, split, 5) + ["--resume", "--microbatch", "1"]
    with pytest.raises(ValueError, match="configuration differs"):
        byte.main(changed_args)
    changed = pair_rows()
    changed[0]["left_text"] += " modified"
    write_pairs(path, changed)
    with pytest.raises(ValueError, match="identity differs"):
        byte.main(train_args(path, model_path, split, 5) + ["--resume"])
    (checkpoints[1] / "selected_rows.npz").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="integrity failure"):
        byte.verify_checkpoint(checkpoints[1])


def test_score_preserves_every_row_order_and_never_reads_labels(tiny, tmp_path, monkeypatch):
    _, _, model_path = tiny
    train_path = tmp_path / "train.parquet"
    write_pairs(train_path, pair_rows())
    run = tmp_path / "run"
    byte.main(train_args(train_path, model_path, run, 2))
    rows = pair_rows(11)
    paths = [tmp_path / name for name in ("normal.parquet", "poison.parquet", "absent.parquet")]
    write_pairs(paths[0], rows)
    poison = [{**r, "label": "THIS MUST NOT BE READ", "true_count": None, "v1_tree": float("nan")} for r in rows]
    write_pairs(paths[1], poison)
    write_pairs(paths[2], [{k: v for k, v in r.items() if k != "label"} for r in rows])
    real_pf = pq.ParquetFile
    class LabelGuard:
        def __init__(self, *args, **kwargs):
            self.inner = real_pf(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def iter_batches(self, *args, **kwargs):
            assert kwargs["columns"] == list(byte.PAIR_COLUMNS)
            return self.inner.iter_batches(*args, **kwargs)
    monkeypatch.setattr(byte.pq, "ParquetFile", LabelGuard)
    results = []
    for i, path in enumerate(paths):
        output = tmp_path / f"scores{i}.parquet"
        report = byte.main(score_args(path, run, output))
        assert report["rows"] == len(rows) and report["labels_used"] is False
        result = pq.read_table(output)
        assert result["pair_row"].to_pylist() == [r["pair_row"] for r in rows]
        assert result["s1_id"].to_pylist() == [r["s1_id"] for r in rows]
        assert result["rec_id"].to_pylist() == [r["rec_id"] for r in rows]
        arrays = {k: result[k].to_numpy() for k in byte.SCORE_NAMES}
        np.testing.assert_allclose(arrays["log_ratio"], arrays["nonmatch_nll"] - arrays["copy_nll"], rtol=0, atol=0)
        np.testing.assert_allclose(arrays["log_ratio_normalized"], arrays["log_ratio"] / arrays["token_count"], rtol=1e-6)
        assert np.all(arrays["token_count"] > 0)
        results.append(result)
    assert results[0].equals(results[1]) and results[1].equals(results[2])
    with pytest.raises(FileExistsError):
        byte.main(score_args(paths[0], run, tmp_path / "scores0.parquet"))
    empty = tmp_path / "empty.parquet"
    pq.write_table(pq.read_table(paths[2]).slice(0, 0), empty)
    output = tmp_path / "empty_scores.parquet"
    assert byte.main(score_args(empty, run, output))["rows"] == 0
    assert pq.read_table(output).column_names == [*byte.PAIR_COLUMNS[:3], *byte.SCORE_NAMES]
    duplicate = tmp_path / "duplicate.parquet"
    write_pairs(duplicate, rows + [rows[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        byte.main(score_args(duplicate, run, tmp_path / "bad_scores.parquet"))
    assert not (tmp_path / "bad_scores.parquet").exists()


def test_time_budget_saves_complete_step_zero_and_can_resume(tiny, tmp_path):
    _, _, model_path = tiny
    path = tmp_path / "train.parquet"
    write_pairs(path, pair_rows())
    output = tmp_path / "budget"
    report = byte.main(train_args(path, model_path, output, 1) + ["--max-minutes", "0.000000001"])
    assert report["reason"] == "max_minutes" and report["step"] == 0
    assert byte.verify_checkpoint(Path(report["checkpoint"]))["step"] == 0
    report = byte.main(train_args(path, model_path, output, 1) + ["--resume"])
    assert report["step"] == 1
