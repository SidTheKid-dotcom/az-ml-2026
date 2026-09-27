"""Offline CPU tests with a tiny, random XLM-R and a locally built tokenizer."""
import ast
import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForSequenceClassification

from ber import takeover_residual as r


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def tiny(tmp_path):
    vocabulary = ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "Cafe", "Blue", "Shop", "Road",
                  "12", "13", "|", "Street", "West", "दुकान", "Café", "A", "B", "C"]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>",
        unk_token="<unk>", pad_token="<pad>", mask_token="<mask>", model_input_names=["input_ids", "attention_mask"])
    torch.manual_seed(615)
    model = XLMRobertaForSequenceClassification(XLMRobertaConfig(
        vocab_size=len(vocabulary), hidden_size=24, intermediate_size=40, num_hidden_layers=1,
        num_attention_heads=2, num_labels=1, max_position_embeddings=96, type_vocab_size=1,
        pad_token_id=1, bos_token_id=0, eos_token_id=2, hidden_dropout_prob=0.15,
        attention_probs_dropout_prob=0.1, classifier_dropout=0.2))
    path = tmp_path / "tiny"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return model, tokenizer, path


def rows(n=96):
    result = []
    for i in range(n):
        group = i // 8
        name = ["Cafe", "दुकान", "Café", "Blue Shop"][group % 4]
        address = "" if group % 3 == 0 else "12 Road " + "West " * (group % 5)
        source = "S2" if i % 2 == 0 else "S3"
        label = int(i % 8 in (0, 5) and group % 4 != 3)
        right = name + (" " + "Street " * (i % 11) if i % 3 else "")
        right += " | " + ("" if i % 5 == 0 else "13 Road " + "West " * (i % 7))
        result.append(dict(pair_row=5000 + (n - i) * 7, left_idx=group * 3, right_idx=i * 2,
            s1_id=f"S1-{group}", rec_id=f"{source}-{i}", source=source, left_text=name + " | " + address,
            right_text=right, label=label, true_count=0 if group % 4 == 3 else 2 + group % 3,
            v1_tree=(i % 17) / 16, v1_blend=((i * 7) % 19) / 18))
    return result


def write_pack(path, contents, sealed=True):
    table = pa.Table.from_pylist(contents)
    pq.write_table(table, path)
    if sealed:
        seal = {"state": "complete", "partition": "A", "holdout2_rows_read": 0,
                "train_sha256": r.sha256(path), "pairs": len(contents),
                "positives": sum(row["label"] for row in contents), "train_s1": len({row["s1_id"] for row in contents})}
        (path.parent / "train_complete.json").write_text(json.dumps(seal))
    return table


def runtime_args():
    return ["--device", "cpu", "--amp", "fp32", "--threads", "1", "--seed", "17", "--max-length", "32"]


def mine_args(pack, model, output):
    return ["mine", "--train", str(pack), "--model", str(model), "--output", str(output),
            "--max-pairs", "32", "--batch-size", "7", "--tokenize-batch-size", "9", *runtime_args()]


def train_args(pack, model, mined, output, mode="residual"):
    return ["train", "--train", str(pack), "--model", str(model), "--mined", str(mined),
            "--output", str(output), "--mode", mode, "--steps", "4", "--batch-size", "8", "--microbatch", "3",
            "--learning-rate", ".001", "--mining-batch-size", "7", "--length-buckets", "8,16,32",
            "--log-every", "1", *runtime_args()]


@pytest.fixture
def prepared(tiny, tmp_path):
    pack, mined = tmp_path / "train.parquet", tmp_path / "mined"
    write_pack(pack, rows())
    r.main(mine_args(pack, tiny[2], mined))
    return pack, mined


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("mode", ["control", "residual"])
def test_exact_class_and_hard_quotas_no_duplicates_seed_and_replay(mode):
    labels = np.r_[np.ones(16, bool), np.zeros(48, bool)]
    hard = {0: np.arange(16, 40), 1: np.arange(8)}
    args = dict(mode=mode, batch_size=16, seed=9)
    sampler = r.BatchSampler(labels, hard, **args)
    twin = r.BatchSampler(labels, hard, **args)
    for _ in range(20):  # Includes positive-cycle wrap and uniform/hard overlap.
        ids, kinds = sampler.take()
        np.testing.assert_array_equal(twin.take()[0], ids)
        assert len(np.unique(ids)) == 16
        assert labels[ids].sum() == 4
        assert kinds[~labels[ids]].sum() == 6
        assert kinds[labels[ids]].sum() == (2 if mode == "residual" else 0)
        assert set(ids[kinds.astype(bool) & ~labels[ids]]) <= set(hard[0])
    state = json.loads(json.dumps(sampler.state()))
    resumed = r.BatchSampler(labels, hard, **args, state=state)
    for _ in range(5):
        assert_nested_equal(sampler.take(), resumed.take())
    changed = r.BatchSampler(labels, hard, **{**args, "seed": 10})
    assert not np.array_equal(changed.take()[0], r.BatchSampler(labels, hard, **args).take()[0])
    with pytest.raises(ValueError, match="resume seed"):
        r.BatchSampler(labels, hard, **{**args, "seed": 10}, state=state)
    with pytest.raises(ValueError, match="hard_sha256"):
        r.BatchSampler(labels, {0: np.arange(20, 44), 1: hard[1]}, **args, state=state)


def test_positive_cycle_covers_all_before_repeating_and_rejects_small_pools():
    cycle = r.PositiveCycle(np.arange(9), 8)
    first = np.concatenate([cycle.take(3) for _ in range(3)])
    assert len(np.unique(first)) == 9
    assert len(np.unique(cycle.take(8, [3]))) == 8
    with pytest.raises(ValueError, match="too small"):
        cycle.take(9, [3])
    with pytest.raises(ValueError, match="multiple of eight"):
        r.BatchSampler([1, 0], {0: [1]}, mode="control", batch_size=4, seed=3)
    with pytest.raises(ValueError, match="too small"):
        r.BatchSampler([1, 1, 0, 0], {0: [2, 3]}, mode="residual", batch_size=8, seed=3)


def test_stakes_exactly_match_existing_fast_trainer_function():
    # Load only the existing function's AST: no optional tree/retrieval imports.
    source = Path(r.__file__).with_name("fast_trainer.py").read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "stake_weights")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "fast_trainer.stake_weights", "exec"), namespace)
    rng = np.random.default_rng(77)
    labels = rng.random(1000) < .25
    counts = rng.integers(0, 15, 1000)
    counts[labels] = np.maximum(counts[labels], 1)
    for normalize in (False, True):
        for cap in (1.0, 5.0, 20.0):
            np.testing.assert_array_equal(r.stake_weights(labels, counts, cap=cap, normalize=normalize),
                namespace["stake_weights"](labels, counts, cap=cap, normalize=normalize))
    with pytest.raises(ValueError, match="no-match"):
        r.stake_weights([1], [0])


def test_seeded_mining_subset_balanced_unique_and_preserves_physical_indices():
    labels = np.asarray([row["label"] for row in rows()])
    a = r.mining_indices(labels, 32, 7)
    np.testing.assert_array_equal(a, r.mining_indices(labels, 32, 7))
    assert len(a) == 32 and labels[a].sum() == 16
    assert (np.diff(a) > 0).all()
    assert not np.array_equal(a, r.mining_indices(labels, 32, 8))
    all_balanced = r.mining_indices(labels, 400000, 7)
    assert len(all_balanced) == 2 * labels.sum()
    for bad in (1, 3, -2):
        with pytest.raises(ValueError):
            r.mining_indices(labels, bad, 0)


def test_training_requires_hash_seal_and_rejects_non_a_before_reading_labels(tmp_path, monkeypatch):
    path = tmp_path / "train.parquet"
    write_pack(path, rows(), sealed=False)
    with pytest.raises(ValueError, match="train_complete.json"):
        r.load_training(path)
    write_pack(path, rows())
    assert r.load_training(path).identity["partition"] == "A"
    seal_path = tmp_path / "train_complete.json"
    seal = json.loads(seal_path.read_text())
    for field, value in (("partition", "B_REST"), ("holdout2_rows_read", 1), ("train_sha256", "wrong")):
        seal_path.write_text(json.dumps({**seal, field: value}))
        with pytest.raises(ValueError, match="seal"):
            r.load_training(path)
    bad = [{**row, "partition": "B_REST" if i == 2 else "A"} for i, row in enumerate(rows())]
    write_pack(path, bad)
    actual_read = pq.ParquetFile.read
    seen = []
    def guarded(self, *args, **kwargs):
        seen.extend(kwargs["columns"])
        assert "label" not in kwargs["columns"]
        return actual_read(self, *args, **kwargs)
    monkeypatch.setattr(pq.ParquetFile, "read", guarded)
    with pytest.raises(ValueError, match="non-A"):
        r.load_training(path)
    assert seen == ["partition"]


@pytest.mark.parametrize("bad_kind", ["pair_row", "pair", "counts", "v1", "index"])
def test_training_rejects_duplicate_or_inconsistent_records(tmp_path, bad_kind):
    values = rows()
    if bad_kind == "pair_row":
        values[1]["pair_row"] = values[0]["pair_row"]
    elif bad_kind == "pair":
        values[1].update(s1_id=values[0]["s1_id"], rec_id=values[0]["rec_id"])
    elif bad_kind == "counts":
        values[1]["true_count"] = 99
    elif bad_kind == "v1":
        values[1]["v1_tree"] = float("nan")
    else:
        values[1]["left_idx"] = 99
    path = tmp_path / "train.parquet"
    write_pack(path, values)
    with pytest.raises(ValueError):
        r.load_training(path)


def test_cache_tokenizes_once_and_predict_restores_index_alignment(tiny):
    model, tokenizer, _ = tiny
    model.eval()
    texts = pa.Table.from_pylist(rows()).select(r.TEXT_COLUMNS)
    class Counted:
        calls = 0
        def __getattr__(self, key):
            return getattr(tokenizer, key)
        def __call__(self, left, right, **kwargs):
            self.calls += len(left)
            return tokenizer(left, right, **kwargs)
    counted = Counted()
    cache = r.TokenCache(texts, counted, max_length=32, tokenize_batch_size=3)
    ids = np.array([13, 4, 8, 0, 5, 29, 42])
    values = r.predict(model, cache, ids, batch_size=3, device=torch.device("cpu"), amp="fp32")
    repeated = r.predict(model, cache, ids[::-1], batch_size=3, device=torch.device("cpu"), amp="fp32")
    np.testing.assert_allclose(repeated[::-1], values, rtol=1e-5, atol=1e-7)
    assert counted.calls == cache.tokenized_rows == len(ids)
    for position, value in zip(ids, values):
        with torch.inference_mode():
            expected = model(**cache.batch([position], torch.device("cpu"))).logits.item()
        assert value == pytest.approx(expected, abs=1e-7)
    assert counted.calls == len(ids)
    with pytest.raises(ValueError, match="only left_text"):
        r.TokenCache(pa.Table.from_pylist(rows()), tokenizer)
    parts = list(r.bucket_positions(cache.lengths[ids], 2, [8, 16, 32]))
    assert sorted(np.concatenate(parts).tolist()) == list(range(len(ids)))
    assert all(len(part) <= 2 for part in parts)


def test_bucketed_microbatch_gradients_equal_whole_batch_stake_objective(tiny, prepared):
    data = r.load_training(prepared[0])
    ids = np.r_[np.flatnonzero(data.labels)[[0, 3]], np.flatnonzero(~data.labels)[[0, 1, 8, 15, 20, 25]]]
    reference, bucketed = copy.deepcopy(tiny[0]), copy.deepcopy(tiny[0])
    for model in (reference, bucketed):
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
    cache = r.TokenCache(data.table.select(r.TEXT_COLUMNS), tiny[1], 32)
    weights = torch.from_numpy(r.stake_weights(data.labels[ids], data.counts[ids]))
    target = torch.tensor(data.labels[ids], dtype=torch.float32)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
    logits = reference(**cache.batch(ids, torch.device("cpu"))).logits[:, 0]
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).sum() / weights.sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    reference_optimizer.step()
    actual, _ = r.train_step(bucketed, bucketed, torch.optim.SGD(bucketed.parameters(), lr=.01), cache, data, ids,
        microbatch=2, boundaries=[8, 16, 32], device=torch.device("cpu"), amp="fp32", grad_clip=1.0)
    assert actual == pytest.approx(loss.item(), abs=1e-7)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, bucketed.state_dict()[key], rtol=1e-5, atol=1e-8)


def test_mining_artifact_has_logits_losses_reference_and_checked_alignment(prepared):
    pack, directory = prepared
    data = r.load_training(pack)
    mined = r.load_mining(directory, data)
    assert mined["manifest"]["labels_used_for_forward"] is False
    assert mined["manifest"]["candidate_injection"] is False
    table = pq.read_table(mined["directory"] / "scores.parquet")
    ids = table["input_position"].to_numpy()
    np.testing.assert_array_equal(table["pair_row"].to_numpy(), data.pair_rows[ids])
    assert not np.array_equal(ids, data.pair_rows[ids])
    logits = torch.tensor(table["logit"].to_numpy().copy())
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor(data.labels[ids], dtype=torch.float32), reduction="none")
    np.testing.assert_allclose(table["loss"].to_numpy(), expected.numpy(), rtol=1e-6)
    pools = r.high_loss_pools(data, mined, 0.25, {0: 3, 1: 1})
    assert all((data.labels[pool] == c).all() and set(pool) <= set(ids) for c, pool in pools.items())
    # Even a self-consistently rehashed but misjoined score table is refused.
    column = table.schema.get_field_index("pair_row")
    pq.write_table(table.set_column(column, "pair_row", pa.array(table["pair_row"].to_numpy()[::-1])),
                   mined["directory"] / "scores.parquet")
    r.commit_inventory(mined["directory"], "mining")
    with pytest.raises(ValueError, match="pair_row is misaligned"):
        r.load_mining(directory, data)


def test_mining_rejects_changed_dataset_even_with_a_new_valid_seal(prepared):
    pack, directory = prepared
    changed = rows()
    changed[0]["left_text"] += " changed"
    write_pack(pack, changed)
    with pytest.raises(ValueError, match="dataset identity"):
        r.load_mining(directory, r.load_training(pack))


def test_high_loss_and_v1_controls_choose_the_documented_ranking(prepared):
    pack, directory = prepared
    data = r.load_training(pack)
    mined = r.load_mining(directory, data)
    losses = np.arange(len(mined["indices"]), dtype=np.float32)
    synthetic = {**mined, "losses": losses, "strata": np.zeros(len(losses), np.uint8)}
    pools = r.high_loss_pools(data, synthetic, 0.25, {0: 3, 1: 1})
    for label in (0, 1):
        positions = np.flatnonzero(data.labels[mined["indices"]] == label)
        assert set(pools[label]) == set(mined["indices"][positions[-4:]])
    hard = r.v1_hard_pool(data, .25, 3)
    negative = np.flatnonzero(~data.labels)
    ranked = sorted(negative, key=lambda i: (-data.table["v1_blend"][int(i)].as_py(),
                                            -data.table["v1_tree"][int(i)].as_py(), data.pair_rows[i]))
    assert set(hard) == set(ranked[:int(np.ceil(len(negative) * .25))])


@pytest.mark.parametrize("mode,stop_step", [("control", 2), ("residual", 2), ("residual", 3)])
def test_resume_exact_weights_optimizer_rng_indices_and_one_retained_checkpoint(
        tiny, prepared, tmp_path, monkeypatch, mode, stop_step):
    pack, mined = prepared
    full, split = tmp_path / "full", tmp_path / "split"
    full_report = r.main(train_args(pack, tiny[2], mined, full, mode) + ["--save-every", "1"])
    counter = [0]
    real_step, real_reason = r.train_step, r.Deadline.reason
    def counted(*args, **kwargs):
        result = real_step(*args, **kwargs)
        counter[0] += 1
        return result
    with monkeypatch.context() as m:
        m.setattr(r, "train_step", counted)
        m.setattr(r.Deadline, "reason", lambda self: "test_pause" if counter[0] >= stop_step else real_reason(self))
        report = r.main(train_args(pack, tiny[2], mined, split, mode))
    assert report["state"] == "paused" and report["step"] == stop_step
    assert report["refresh_completed"] is (mode == "residual" and stop_step > 2)
    paused = r.checkpoint_path(split)
    # Simulate a killed staging save; only the run-tagged staging directory is reclaimed.
    stage = split / ".checkpoint-interrupted"
    stage.mkdir()
    (stage / "owner.json").write_text((paused / "owner.json").read_text())
    (stage / "partial.bin").write_bytes(b"partial")
    report = r.main(train_args(pack, tiny[2], mined, split, mode) + ["--resume"])
    assert report["state"] == "complete" and report["step"] == 4
    assert not stage.exists()
    assert report["comparison_sha256"] == full_report["comparison_sha256"]
    checkpoints = [r.checkpoint_path(root) for root in (full, split)]
    for path in checkpoints:
        assert r.verify_inventory(path, "checkpoint")["step"] == 4
        assert len([p for p in path.parent.glob("checkpoint-*") if p.is_dir()]) == 1
        assert not list(path.parent.glob(".checkpoint-*"))
    for name in ("optimizer.pt", "rng.pt"):
        states = [torch.load(path / name, map_location="cpu", weights_only=False) for path in checkpoints]
        assert_nested_equal(*states)
    assert r.read_json(checkpoints[0] / "sampler.json") == r.read_json(checkpoints[1] / "sampler.json")
    fitted = [r.load_model(path, torch.device("cpu"))[0] for path in checkpoints]
    assert_nested_equal(fitted[0].state_dict(), fitted[1].state_dict())
    assert any(not torch.equal(value, tiny[0].state_dict()[key]) for key, value in fitted[0].state_dict().items())
    with np.load(checkpoints[0] / "sampled_indices.npz") as a, np.load(checkpoints[1] / "sampled_indices.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        assert all(len(np.unique(batch)) == 8 for batch in a["input_position"])
        np.testing.assert_array_equal(a["generation"], [0, 0, 1, 1] if mode == "residual" else [0, 0, 0, 0])
    if mode == "residual":
        initial = pq.read_table(mined / "snapshot" / "scores.parquet")
        refresh = pq.read_table(split / "refresh" / "scores.parquet")
        assert refresh["input_position"].equals(initial["input_position"])
        assert not np.array_equal(refresh["logit"].to_numpy(), initial["logit"].to_numpy())
        assert r.read_json(split / "refresh" / "sampler_before.json")["step"] == 2
        assert (split / "refresh" / "rng_before.pt").is_file()
    else:
        assert not (split / "refresh").exists()


def test_arms_share_comparison_config_and_resume_rejects_changes(tiny, prepared, tmp_path):
    pack, mined = prepared
    control, residual = tmp_path / "control", tmp_path / "residual"
    # Deadline stops after setup; a complete step-zero state remains resumable.
    reports = [r.main(train_args(pack, tiny[2], mined, out, mode) + ["--max-minutes", "0.000000001"])
               for out, mode in ((control, "control"), (residual, "residual"))]
    assert all(report["step"] == 0 and report["state"] == "paused" for report in reports)
    assert reports[0]["comparison_sha256"] == reports[1]["comparison_sha256"]
    with pytest.raises(FileExistsError):
        r.main(train_args(pack, tiny[2], mined, residual))
    for changed in (["--steps", "6"], ["--seed", "18"], ["--microbatch", "4"], ["--learning-rate", ".002"]):
        with pytest.raises(ValueError, match="configuration differs"):
            r.main(train_args(pack, tiny[2], mined, residual) + ["--resume", *changed])
    report = r.main(train_args(pack, tiny[2], mined, residual) + ["--resume"])
    assert report["step"] == 4 and report["refresh_completed"]
    scoring = ["score", "--pairs", str(pack), "--model", str(residual),
               "--output", str(residual / "scores.parquet"), *runtime_args()]
    assert r.main(scoring)["rows"] == 96
    with pytest.raises(ValueError, match="token length"):
        r.main([*scoring, "--output", str(residual / "wrong_length.parquet"), "--max-length", "64"])
    path = r.checkpoint_path(residual)
    (path / "optimizer.pt").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="integrity failure"):
        r.main(train_args(pack, tiny[2], mined, residual) + ["--resume"])


def test_interrupted_refresh_resumes_from_committed_training_boundary(tiny, prepared, tmp_path, monkeypatch):
    pack, mined = prepared
    root = tmp_path / "interrupt_refresh"
    real_predict = r.predict
    def interrupted(*args, **kwargs):
        raise r.DeadlineExceeded("test_refresh_pause")
    with monkeypatch.context() as m:
        m.setattr(r, "predict", interrupted)
        report = r.main(train_args(pack, tiny[2], mined, root))
    assert report["step"] == 2 and report["reason"] == "test_refresh_pause"
    assert not report["refresh_completed"] and not (root / "refresh").exists()
    assert r.read_json(r.checkpoint_path(root) / "sampler.json")["step"] == 2
    assert r.predict is real_predict
    report = r.main(train_args(pack, tiny[2], mined, root) + ["--resume"])
    assert report["step"] == 4 and report["refresh_completed"]


def test_score_projects_only_text_and_ids_ignores_poisoned_or_missing_labels(tiny, tmp_path, monkeypatch):
    values = rows(24)
    paths = [tmp_path / name for name in ("normal.parquet", "poison.parquet", "absent.parquet")]
    write_pack(paths[0], values, sealed=False)
    write_pack(paths[1], [{**row, "label": "DO NOT READ", "true_count": None,
                          "v1_tree": float("nan"), "v1_blend": "POISON"} for row in values], sealed=False)
    write_pack(paths[2], [{k: v for k, v in row.items() if k in r.PAIR_COLUMNS} for row in values], sealed=False)
    actual_read = pq.ParquetFile.read
    calls = []
    def guarded(self, *args, **kwargs):
        calls.append(kwargs["columns"])
        assert kwargs["columns"] == r.PAIR_COLUMNS
        return actual_read(self, *args, **kwargs)
    monkeypatch.setattr(pq.ParquetFile, "read", guarded)
    results = []
    for i, source in enumerate(paths):
        output = tmp_path / f"scores{i}.parquet"
        report = r.main(["score", "--pairs", str(source), "--model", str(tiny[2]), "--output", str(output),
                         "--batch-size", "7", *runtime_args()])
        assert report["rows"] == len(values) and report["labels_read"] is False
        result = pq.read_table(output)
        assert result.column_names == r.SCORE_COLUMNS
        assert result["pair_row"].to_pylist() == [row["pair_row"] for row in values]
        assert result["s1_id"].to_pylist() == [row["s1_id"] for row in values]
        assert result["rec_id"].to_pylist() == [row["rec_id"] for row in values]
        assert np.isfinite(result["logit"].to_numpy()).all()
        results.append(result)
    assert len(calls) == 3 and results[0].equals(results[1]) and results[1].equals(results[2])
    with pytest.raises(FileExistsError):
        r.main(["score", "--pairs", str(paths[0]), "--model", str(tiny[2]),
                "--output", str(tmp_path / "scores0.parquet"), *runtime_args()])


def test_score_empty_and_duplicate_ids(tiny, tmp_path):
    path, output = tmp_path / "pairs.parquet", tmp_path / "scores.parquet"
    table = pa.Table.from_pylist(rows(8)).select(r.PAIR_COLUMNS)
    pq.write_table(table.slice(0, 0), path)
    args = ["score", "--pairs", str(path), "--model", str(tiny[2]), "--output", str(output), *runtime_args()]
    assert r.main(args)["rows"] == 0
    assert pq.read_table(output).column_names == r.SCORE_COLUMNS
    pq.write_table(pa.concat_tables([table, table]), path)
    bad = tmp_path / "bad.parquet"
    with pytest.raises(ValueError, match="Duplicate"):
        r.main([*args, "--output", str(bad)])
    assert not bad.exists()
