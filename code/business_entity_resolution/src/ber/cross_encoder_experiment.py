"""Train a small multilingual reranker on TRAIN pairs and compare on TUNE.

No Trainer dependency and no default holdout scoring. Tuning inference is gated
only by the saved base probability, never by tuning labels. The base model and
every blend are evaluated on the exact same candidate rows.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from .experiment import json_write, load_data
from .hf_experiment import frame_digest, record_texts
from .metrics import candidate_report, pair_labels, score_pairs, tune_threshold
from .models import predict_proba
from .prepare import log


MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
SUPPORTED_MODEL_IDS = (MODEL_ID, "BAAI/bge-reranker-v2-m3")
BLEND_ALPHAS = (0.25, 0.5, 0.75, 1.0)


def load_base_bundle(base_dir: Path, base_format: str = "auto"):
    """Load a completed HF or tabular sweep without regenerating candidate edges.

The tabular sweep writes pairs.parquet last, after its selected checkpoint and
aligned base/numeric feature tables. Recompute its scores from that checkpoint
so an intermediate/stale best_probabilities.npy cannot silently select a model.
"""
    if base_format == "auto":
        base_format = "tabular" if (base_dir / "best.joblib").exists() else "hf"
    if base_format not in {"hf", "tabular"}:
        raise ValueError("base_format must be auto, hf, or tabular")
    pair_path = base_dir / "pairs.parquet"
    if not pair_path.is_file():
        raise FileNotFoundError(f"Base bundle is not complete: {pair_path} is required")
    pairs = pd.read_parquet(pair_path)
    if not {"left_idx", "right_idx"} <= set(pairs):
        raise ValueError("Base candidates need positional left_idx/right_idx columns")
    features = None
    if base_format == "tabular":
        package = joblib.load(base_dir / "best.joblib")
        base_features = pd.read_parquet(base_dir / "base_features.parquet")
        numeric = pd.read_parquet(base_dir / "numeric_features.parquet")
        if len(base_features) != len(pairs) or len(numeric) != len(pairs):
            raise ValueError("Tabular feature rows do not align with completed pairs.parquet")
        if set(base_features).intersection(numeric):
            raise ValueError("Base and numeric feature tables have overlapping column names")
        features = pd.concat([base_features.reset_index(drop=True), numeric.reset_index(drop=True)], axis=1)
        missing = set(package["feature_columns"]) - set(features)
        if missing:
            raise ValueError(f"Selected tabular checkpoint needs missing features: {sorted(missing)}")
        probabilities = predict_proba(package["model"], features[package["feature_columns"]])
        provenance = {"format": "tabular", "checkpoint": "best.joblib",
                      "probability_source": "recomputed_from_selected_checkpoint_and_aligned_features",
                      "feature_count": len(package["feature_columns"])}
    else:
        package = joblib.load(base_dir / "model.joblib")
        probabilities = np.load(base_dir / "probabilities.npy")
        provenance = {"format": "hf", "checkpoint": "model.joblib",
                      "probability_source": "probabilities.npy"}
    if "threshold" not in package:
        raise ValueError("Selected base checkpoint has no saved operating threshold")
    if probabilities.shape != (len(pairs),):
        raise ValueError("Base probabilities must align exactly with pairs.parquet")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Base probabilities must be finite and in [0,1]")
    return package, pairs, probabilities, features, provenance


def conflict_priorities(features: pd.DataFrame) -> np.ndarray:
    """Text/numeric evidence only; labels and split membership are not inputs."""
    def values(name, default=0.0):
        if name not in features:
            return np.full(len(features), default, dtype=np.float32)
        return np.nan_to_num(features[name].to_numpy(dtype=np.float32))

    numeric = np.maximum.reduce([values(name) for name in (
        "num_unit_conflict", "num_alpha_prefix_conflict_same_digits",
        "num_alpha_suffix_conflict_same_digits", "num_name_conflict",
        "num_short_small_offset", "num_shared_only_long", "num_postcode_conflict",
    )])
    address_support = np.maximum(values("address_token_sort"), values("retrieval_address")) >= .70
    word_mismatch = np.minimum(values("name_token_left_coverage", 1),
                               values("name_token_right_coverage", 1)) < .999
    name_conflict = address_support & word_mismatch & (values("core_exact") < .5) & (values("core_token_sort") >= .30)
    return (numeric > 0).astype(np.float32) + name_conflict.astype(np.float32)


def _save_array(path: Path, values: np.ndarray) -> None:
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values)
    temporary.replace(path)


def verify_model(args, base_package: dict) -> tuple[str, dict]:
    import yaml

    manifest = yaml.safe_load(Path(args.allowlist).read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in manifest["models"]}
    # Both bounded screens use standard single-logit sequence classifiers.
    if args.model_id not in SUPPORTED_MODEL_IDS:
        raise ValueError(f"This experiment supports only {SUPPORTED_MODEL_IDS}")
    row = entries.get(args.model_id)
    if row is None or row.get("role") != "reranker" or row.get("licence") not in {"mit", "apache-2.0"}:
        raise ValueError("Reranker is absent from the MIT/Apache model allowlist")
    base_id = base_package.get("embedding_config", {}).get("model_id")
    total = float(row["params_b"])
    if base_id:
        if base_id not in entries:
            raise ValueError(f"Base embedding model is absent from allowlist: {base_id}")
        total += float(entries[base_id]["params_b"])
    if total > float(manifest["budget_total_params_b"]):
        raise ValueError(f"Combined pretrained model parameter budget exceeded: {total}B")
    return args.model_id, {**row, "combined_params_b": total, "base_embedding_id": base_id}


def verify_full_pool_data(directory: Path, metadata: dict, reference: pd.DataFrame) -> str | None:
    """Accept the original hard pool or its completed, traceable dense union."""
    if not reference.split.isin(["train", "tune"]).all():
        raise ValueError("Hard-pool reranker experiments accept TRAIN/TUNE only")
    kind = metadata.get("dataset_kind")
    if kind == "full_pool_hard_candidates":
        return None
    if kind is None and "mining_full_pool_records" in metadata:
        completed = json.loads((directory / "complete.json").read_text())
        if completed.get("state") != "complete" or completed.get("metadata") != metadata:
            raise ValueError("Dense dataset is not committed with matching metadata")
        identity = completed["config"]["pool_embedding_identity"]
        if (identity.get("scope") != "full_pool" or metadata.get("positive_candidate_injection") is not False
                or completed["config"].get("labels_used_for_mining") is not False):
            raise ValueError("Dense candidates need label-free completed full-pool mining")
        reference_hash = hashlib.sha256("\n".join(sorted(reference.entity_id)).encode()).hexdigest()
        splits = {f"{split}/{country}": int(count) for (split, country), count
                  in reference.groupby(["split", "country"]).size().items()}
        if metadata.get("reference_id_sha256") != reference_hash or metadata.get("split_counts") != splits:
            raise ValueError("Dense metadata does not match actual reference IDs/splits")
        if not metadata["mining_full_pool_records"] == identity["target_rows"] == identity["full_dataset_rows"]:
            raise ValueError("Dense candidates were not mined against the complete original pool")
        encoder_report = Path(identity["checkpoint"]["path"]).parent / "results.json"
        return json.loads(encoder_report.read_text())["model_id"]
    if kind != "full_pool_lexical_phonetic_dense_union":
        raise ValueError("Unrecognized full-pool reranker dataset provenance")
    completed = json.loads((directory / "complete.json").read_text())
    if completed.get("state") != "complete" or completed.get("metadata") != metadata:
        raise ValueError("Union dataset is not committed with matching metadata")
    if metadata.get("positive_candidate_injection") is not False or metadata.get("original_splits_preserved") is not True:
        raise ValueError("Union must preserve original splits without injected positive edges")
    reference_hash = hashlib.sha256("\n".join(sorted(reference.entity_id)).encode()).hexdigest()
    splits = {f"{split}/{country}": int(count) for (split, country), count
              in reference.groupby(["split", "country"]).size().items()}
    lexical = json.loads((Path(metadata["parent_lexical"]) / "metadata.json").read_text())
    dense_complete = json.loads((Path(metadata["parent_dense"]) / "complete.json").read_text())
    dense = dense_complete["metadata"]
    identity = dense_complete["config"]["pool_embedding_identity"]
    if dense_complete.get("state") != "complete" or identity.get("scope") != "full_pool":
        raise ValueError("Dense parent was not completed against the full pool")
    for parent in (metadata, lexical, dense):
        if parent.get("reference_id_sha256") != reference_hash or parent.get("split_counts") != splits:
            raise ValueError("Full-pool parents do not share the actual reference IDs and splits")
    total = metadata["full_pool_records"]
    if not total == dense["mining_full_pool_records"] == identity["target_rows"] == sum(lexical["full_pool_records_by_country"].values()):
        raise ValueError("Full-pool parent target counts disagree")
    if metadata["checkpoint_sha256"] != identity["checkpoint"]["sha256"]:
        raise ValueError("Union embedding checkpoint identity disagrees with its dense parent")
    encoder_report = Path(identity["checkpoint"]["path"]).parent / "results.json"
    return json.loads(encoder_report.read_text())["model_id"]


def training_pool(pairs: pd.DataFrame, reference: pd.DataFrame, labels: np.ndarray,
                  base_scores: np.ndarray, negative_ratio: float, *,
                  conflicts: np.ndarray | None = None,
                  conflict_fraction: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """TRAIN positives plus high-score TRAIN negatives, optionally conflict-focused."""
    if not 0 <= conflict_fraction <= 1 or negative_ratio <= 0:
        raise ValueError("conflict_fraction must lie in [0,1] and negative_ratio must be positive")
    if conflicts is not None and np.asarray(conflicts).shape != (len(pairs),):
        raise ValueError("Conflict priorities must align with candidate rows")
    train = reference.split.to_numpy()[pairs.left_idx.to_numpy(dtype=np.int64)] == "train"
    positive = np.flatnonzero(train & (labels == 1))
    negative = np.flatnonzero(train & (labels == 0))
    if not len(positive) or not len(negative):
        raise ValueError("Reranker training needs both positive and negative TRAIN pairs")
    order = np.lexsort((negative, -base_scores[negative]))
    keep = min(len(negative), max(1, int(np.ceil(len(positive) * negative_ratio))))
    ranked = negative[order]
    if conflicts is not None and conflict_fraction > 0:
        focused = ranked[np.asarray(conflicts)[ranked] > 0]
        focused = focused[:min(len(focused), int(np.ceil(keep * conflict_fraction)))]
        remaining = ranked[~np.isin(ranked, focused)]
        chosen = np.r_[focused, remaining[:keep-len(focused)]]
        return positive, chosen
    return positive, ranked[:keep]


def rerank_mask(pairs: pd.DataFrame, reference: pd.DataFrame,
                base_scores: np.ndarray, cutoff: float) -> np.ndarray:
    """Label-free gate; deliberately does not accept labels or ground truth."""
    if len(pairs) != len(base_scores):
        raise ValueError("Candidate rows and saved base probabilities are misaligned")
    return np.asarray(base_scores >= cutoff, dtype=bool)


class _ShuffledCycle:
    """Consume a shuffled class pool before repeating, for balanced batches."""
    def __init__(self, values: np.ndarray, rng: np.random.Generator):
        self.values, self.rng = np.asarray(values), rng
        self.order = rng.permutation(self.values)
        self.position = 0

    def take(self, size: int) -> np.ndarray:
        chunks, remaining = [], size
        while remaining:
            if self.position == len(self.order):
                self.order = self.rng.permutation(self.values)
                self.position = 0
            count = min(remaining, len(self.order) - self.position)
            chunks.append(self.order[self.position:self.position + count])
            self.position += count
            remaining -= count
        return np.concatenate(chunks)


def load_reranker(path: str, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
    )
    if model.config.num_labels != 1:
        raise ValueError(f"Expected one positive-match logit; checkpoint has {model.config.num_labels} labels")
    return model.to(device), tokenizer


def _amp_context(torch, device: str, amp: str):
    """Keep the historical CUDA BF16 default; make FP32 an explicit control."""
    if amp not in {"fp32", "bf16"}:
        raise ValueError("amp must be fp32 or bf16")
    if not device.startswith("cuda"):
        return nullcontext()  # CPU inference/training has always run in FP32.
    if amp == "fp32":
        return torch.autocast(device_type="cuda", enabled=False)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 AMP requires a CUDA device with BF16 support")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _tokens(tokenizer, left_texts: list[str], right_texts: list[str],
            left: np.ndarray, right: np.ndarray, ids: np.ndarray, args):
    batch = tokenizer([left_texts[int(left[i])] for i in ids],
                      [right_texts[int(right[i])] for i in ids],
                      padding=True, truncation=True, max_length=args.max_length,
                      return_tensors="pt")
    return {key: value.to(args.device) if hasattr(value, "to") else value for key, value in batch.items()}


def train_reranker(model, tokenizer, reference, pool, pairs, labels,
                   positive: np.ndarray, negative: np.ndarray, args) -> dict:
    import torch
    from torch.nn import functional as F

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    positive_cycle, negative_cycle = _ShuffledCycle(positive, rng), _ShuffledCycle(negative, rng)
    positive_batch = max(1, min(args.batch_size - 1, round(args.batch_size * args.positive_fraction)))
    negative_batch = args.batch_size - positive_batch
    left, right = pairs.left_idx.to_numpy(dtype=np.int64), pairs.right_idx.to_numpy(dtype=np.int64)
    left_texts, right_texts = record_texts(reference), record_texts(pool)
    amp = getattr(args, "amp", "bf16")
    _amp_context(torch, args.device, amp)  # Fail before the first optimizer update.
    seen_positive, seen_negative, trace = set(), set(), []
    for step in range(args.steps):
        pids, nids = positive_cycle.take(positive_batch), negative_cycle.take(negative_batch)
        seen_positive.update(pids.tolist())
        seen_negative.update(nids.tolist())
        ids = rng.permutation(np.r_[pids, nids])
        tokens = _tokens(tokenizer, left_texts, right_texts, left, right, ids, args)
        target = torch.as_tensor(labels[ids], dtype=torch.float32, device=args.device)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(torch, args.device, amp):
            logits = model(**tokens).logits
        if logits.shape != (len(ids), 1):
            raise ValueError(f"Expected (batch, 1) positive-match logits, got {tuple(logits.shape)}")
        # Loss and its target remain FP32 even when the forward uses BF16.
        loss = F.binary_cross_entropy_with_logits(logits[:, 0].float(), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        warmup = max(1, args.steps // 10)
        factor = min((step + 1) / warmup, (args.steps - step) / max(1, args.steps - warmup))
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * factor
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.steps:
            item = {"step": step + 1, "bce": float(loss.detach().cpu()),
                    "unique_train_positive_seen": len(seen_positive),
                    "unique_train_negative_seen": len(seen_negative)}
            trace.append(item)
            log(f"Cross-encoder train {item}")
    model.eval()
    return {"positive_pool": len(positive), "hard_negative_pool": len(negative),
            "positive_per_batch": positive_batch, "negative_per_batch": negative_batch,
            "unique_train_positive_seen": len(seen_positive), "unique_train_negative_seen": len(seen_negative),
            "all_retrieved_train_positives_available": True, "trace": trace}


def score_reranker(model, tokenizer, reference, pool, pairs, selected, args) -> np.ndarray:
    import torch

    model.eval()
    ids = np.flatnonzero(selected)
    probabilities = np.zeros(len(pairs), dtype=np.float32)
    left, right = pairs.left_idx.to_numpy(dtype=np.int64), pairs.right_idx.to_numpy(dtype=np.int64)
    left_texts, right_texts = record_texts(reference), record_texts(pool)
    amp = getattr(args, "amp", "bf16")
    with torch.inference_mode():
        for start in range(0, len(ids), args.inference_batch_size):
            batch_ids = ids[start:start + args.inference_batch_size]
            tokens = _tokens(tokenizer, left_texts, right_texts, left, right, batch_ids, args)
            with _amp_context(torch, args.device, amp):
                logits = model(**tokens).logits
            if logits.shape != (len(batch_ids), 1):
                raise ValueError("Reranker inference must emit exactly one positive-match logit")
            probabilities[batch_ids] = torch.sigmoid(logits[:, 0].float()).cpu().numpy()
            if start == 0 or (start // args.inference_batch_size + 1) % 100 == 0:
                log(f"Reranked {min(start + len(batch_ids), len(ids)):,}/{len(ids):,} pairs")
    if not np.isfinite(probabilities).all():
        raise ValueError("Reranker emitted non-finite probabilities")
    return probabilities


def blend_probabilities(base: np.ndarray, cross: np.ndarray, selected: np.ndarray,
                        alpha: float) -> np.ndarray:
    values = np.zeros(len(base), dtype=np.float32)
    values[selected] = (1 - alpha) * base[selected] + alpha * cross[selected]
    return values


def _thresholds(scores: np.ndarray) -> np.ndarray:
    # Unscored rows are zero. A strictly positive grid makes their rejection
    # unconditional, including for the pure cross-encoder experiment.
    maximum = float(np.max(scores)) if len(scores) else 1.0
    return np.unique(np.r_[np.linspace(0.05, 0.9, 35), .925, .95, .975, .99, .995,
                           np.nextafter(max(maximum, .995), float("inf"))])


def detailed_score(reference, pool, pairs, probabilities, truth, threshold, mask) -> dict:
    report = score_pairs(reference, pool, pairs, probabilities, truth, threshold, mask)
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    labels = pair_labels(reference, pool, pairs, truth).astype(bool)
    selected = np.asarray(mask, dtype=bool)
    keep = (np.asarray(probabilities, dtype=np.float64) >= threshold) & selected[left]
    false_positive = keep & ~labels
    true_counts = np.asarray([len(truth.get(entity, set())) for entity in reference.entity_id])
    fp_entities = np.unique(left[false_positive])
    report.update(false_positive_links=int(false_positive.sum()),
                  entities_with_false_positive=int(len(fp_entities)),
                  singleton_false_merges=int(np.sum(true_counts[fp_entities] == 0)),
                  retrieved_true_links_rejected=int(np.sum(labels & selected[left] & ~keep)))
    return report


def compare_blends(reference, pool, pairs, base, cross, selected, truth,
                   saved_threshold: float, evaluate: bool) -> tuple[dict, np.ndarray, dict]:
    tuning_mask = reference.split.eq("tune")
    summaries, traces, vectors = [], {}, {}
    gated = blend_probabilities(base, cross, selected, 0.0)
    options = [("baseline", 0.0, base), ("gated_baseline", 0.0, gated)]
    options += [(f"blend_{alpha:g}", alpha, blend_probabilities(base, cross, selected, alpha))
                for alpha in BLEND_ALPHAS]
    for label, alpha, values in options:
        threshold, trace = tune_threshold(reference, pool, pairs, values, truth, tuning_mask,
                                           thresholds=_thresholds(values[np.asarray(tuning_mask)[pairs.left_idx]]))
        record = {"variant": label, "cross_encoder_weight": alpha, "threshold": threshold,
                  "tune": detailed_score(reference, pool, pairs, values, truth, threshold, tuning_mask)}
        if evaluate:
            record["eval"] = detailed_score(reference, pool, pairs, values, truth, threshold, reference.split.eq("eval"))
        summaries.append(record)
        traces[label], vectors[label] = trace, values
    # A tie keeps the original base model; extra complexity must earn its place.
    best = max(summaries, key=lambda row: (row["tune"]["macro_f05"], row["variant"] == "baseline"))
    report = {"baseline_at_saved_threshold": detailed_score(reference, pool, pairs, base, truth,
                                                              saved_threshold, tuning_mask),
              "comparisons": summaries, "selected_by_tune": best}
    return report, vectors[best["variant"]], traces


def run(args):
    import torch

    started = time.time()
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and args.gpu_memory_fraction is not None:
        device_index = torch.device(args.device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=device_index)
    output, base_dir = Path(args.output), Path(args.base_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference, pool, truth = load_data(args.data)
    metadata_path = Path(args.data) / "metadata.json"
    data_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    retrieval_model_id = verify_full_pool_data(Path(args.data), data_metadata, reference) if args.require_hard_pool else None
    dataset_kind = data_metadata.get("dataset_kind")
    retrieval_checkpoint_hash = data_metadata.get("checkpoint_sha256")
    if retrieval_model_id and dataset_kind is None:
        dataset_kind = "full_pool_dense_candidates"
        retrieval_checkpoint_hash = json.loads((Path(args.data) / "complete.json").read_text())["config"]["checkpoint_sha256"]
    base_package, original_pairs, original_probabilities, base_features, base_provenance = load_base_bundle(
        base_dir, args.base_format,
    )
    if retrieval_model_id:
        base_package = {**base_package, "embedding_config": {"model_id": retrieval_model_id}}
    model_id, license_info = verify_model(args, base_package)
    if original_pairs.duplicated(["left_idx", "right_idx"]).any():
        raise ValueError("Base candidates contain duplicate pairs")
    if len(original_pairs) and (original_pairs.left_idx.min() < 0 or original_pairs.left_idx.max() >= len(reference)
                               or original_pairs.right_idx.min() < 0 or original_pairs.right_idx.max() >= len(pool)):
        raise ValueError("Base candidate indices are outside the supplied hard dataset")
    splits = ("train", "tune", "eval") if args.evaluate else ("train", "tune")
    reference_mask = reference.split.isin(splits).to_numpy()
    keep = reference_mask[original_pairs.left_idx.to_numpy(dtype=np.int64)]
    positions = np.flatnonzero(keep)
    pairs = original_pairs.loc[keep].reset_index(drop=True)
    probabilities = original_probabilities[keep].astype(np.float32)
    if args.evaluate and not np.any(reference.split.to_numpy()[pairs.left_idx.to_numpy(dtype=np.int64)] == "eval"):
        raise ValueError("Base artifact has no eval candidates; explicit evaluation requires a base artifact that includes eval")
    if not reference.split.eq("train").any() or not reference.split.eq("tune").any():
        raise ValueError("Train and tuning reference splits are both required")
    allowed_ids = set(reference.loc[reference_mask, "entity_id"])
    truth = {key: value for key, value in truth.items() if key in allowed_ids}
    labels = pair_labels(reference, pool, pairs, truth)
    conflicts = None
    if args.conflict_fraction > 0:
        if base_features is None:
            from .features import build_features
            from .numeric_features import build_numeric_features
            focus_features = pd.concat([build_features(reference, pool, pairs),
                                        build_numeric_features(reference, pool, pairs)], axis=1)
        else:
            focus_features = base_features.loc[keep].reset_index(drop=True)
        conflicts = conflict_priorities(focus_features)
        del focus_features
    del base_features
    positive, negative = training_pool(pairs, reference, labels, probabilities, args.negative_ratio,
                                       conflicts=conflicts, conflict_fraction=args.conflict_fraction)
    selected = rerank_mask(pairs, reference, probabilities, args.rerank_cutoff)
    # Include low-base-score positives for training diagnostics only. No tuning
    # record is added by label, and the tuning gate above remains label-free.
    selected[positive] = True
    if args.tune_only_inference:
        inference_splits = ("tune", "eval") if args.evaluate else ("tune",)
        selected &= reference.split.isin(inference_splits).to_numpy()[pairs.left_idx.to_numpy(dtype=np.int64)]
    train_ids = reference.loc[reference.split.eq("train"), "entity_id"]
    config = {"schema": 2, "model_id": model_id, "model_source": args.model, "seed": args.seed,
              "steps": args.steps, "batch_size": args.batch_size, "max_length": args.max_length,
              "device": args.device, "amp": getattr(args, "amp", "bf16"),
              "inference_batch_size": args.inference_batch_size,
              "threads": args.threads, "gpu_memory_fraction": args.gpu_memory_fraction,
              "tune_only_inference": args.tune_only_inference,
              "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
              "negative_ratio": args.negative_ratio, "positive_fraction": args.positive_fraction,
              "conflict_fraction": args.conflict_fraction,
              "conflict_negative_pool": int(np.sum(conflicts[negative] > 0)) if conflicts is not None else 0,
              "base_bundle": base_provenance,
              "dataset_kind": dataset_kind,
              "full_pool_records_by_country": data_metadata.get("full_pool_records_by_country"),
              **({"full_pool_records": data_metadata.get("full_pool_records", data_metadata.get("mining_full_pool_records")),
                  "retrieval_checkpoint_sha256": retrieval_checkpoint_hash}
                 if retrieval_model_id else {}),
              "require_hard_pool": args.require_hard_pool,
              "rerank_cutoff": args.rerank_cutoff, "blend_alphas": list(BLEND_ALPHAS),
              "evaluation_used": args.evaluate, "base_dir": str(base_dir.resolve()),
              "base_saved_threshold": float(base_package["threshold"]),
              "reference_digest": frame_digest(reference, ["entity_id", "business_name", "business_address", "country", "split"]),
              "pool_digest": frame_digest(pool, ["entity_id", "business_name", "business_address", "country"]),
              "pair_digest": frame_digest(pairs, ["left_idx", "right_idx"]),
              "base_probability_digest": hashlib.sha256(probabilities.tobytes()).hexdigest(),
              "training_truth_digest": hashlib.sha256(json.dumps(
                  {key: sorted(truth.get(key, set())) for key in train_ids}, sort_keys=True).encode()).hexdigest(),
              "license": license_info}
    config_path = output / "experiment_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output already belongs to a different experiment; choose another --output")
    json_write(config_path, config)
    pairs.to_parquet(output / "pairs.parquet", index=False)
    _save_array(output / "base_pair_positions.npy", positions)
    _save_array(output / "base_probabilities.npy", probabilities)
    _save_array(output / "reranked_mask.npy", selected)
    checkpoint, marker = output / "checkpoint", output / "checkpoint_complete.json"
    stage = time.time()
    if marker.exists():
        log("Loading completed cross-encoder checkpoint")
        model, tokenizer = load_reranker(str(checkpoint), args.device)
        training = json.loads(marker.read_text())
    else:
        log(f"Loading offline {model_id}; train pool {len(positive):,} positives + {len(negative):,} hard negatives")
        model, tokenizer = load_reranker(args.model, args.device)
        training = train_reranker(model, tokenizer, reference, pool, pairs, labels, positive, negative, args)
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        json_write(marker, training)
    training_seconds = time.time() - stage
    stage = time.time()
    cross_path = output / "cross_probabilities.npy"
    if cross_path.exists():
        cross = np.load(cross_path)
        if cross.shape != probabilities.shape:
            raise ValueError("Cached cross probabilities have an unexpected shape")
        if not np.isfinite(cross).all() or np.any((cross < 0) | (cross > 1)):
            raise ValueError("Cached cross probabilities must be finite and in [0,1]")
    else:
        cross = score_reranker(model, tokenizer, reference, pool, pairs, selected, args)
        _save_array(cross_path, cross)
    inference_seconds = time.time() - stage
    comparison, chosen_probabilities, traces = compare_blends(
        reference, pool, pairs, probabilities, cross, selected, truth,
        float(base_package["threshold"]), args.evaluate,
    )
    retrieval = {"all_candidates": {split: candidate_report(reference, pool, pairs, truth, reference.split.eq(split))
                                     for split in splits},
                 "reranked_candidates": {split: candidate_report(reference, pool, pairs.loc[selected], truth,
                                                                  reference.split.eq(split)) for split in splits}}
    tune_pairs = reference.split.to_numpy()[pairs.left_idx.to_numpy(dtype=np.int64)] == "tune"
    report = {"model_id": model_id, "amp": config["amp"], "licence": license_info["licence"],
              "combined_params_b": license_info["combined_params_b"], "evaluation_used": args.evaluate,
              "base_bundle": base_provenance, "dataset_kind": dataset_kind,
              "conflict_fraction": args.conflict_fraction,
              "conflict_negative_pool": config["conflict_negative_pool"],
              "tune_only_inference": args.tune_only_inference,
              "training": training, "tune_candidates": int(tune_pairs.sum()),
              "tune_candidates_reranked": int((tune_pairs & selected).sum()),
              "retrieval": retrieval, **comparison,
              "timings": {"training_seconds": training_seconds, "inference_seconds": inference_seconds,
                          "total_seconds": time.time() - started}}
    _save_array(output / "probabilities.npy", chosen_probabilities)
    json_write(output / "thresholds.json", traces)
    json_write(output / "results.json", report)
    json_write(output / "selected_model.json", {"base_dir": str(base_dir.resolve()),
                "cross_encoder_checkpoint": str(checkpoint.resolve()),
                "cross_encoder_weight": comparison["selected_by_tune"]["cross_encoder_weight"],
                "variant": comparison["selected_by_tune"]["variant"],
                "threshold": comparison["selected_by_tune"]["threshold"],
                "rerank_cutoff": args.rerank_cutoff, "config": config})
    log(json.dumps(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/quick")
    parser.add_argument("--base-dir", required=True, help="Completed HF artifact or tabular_sweep bundle directory")
    parser.add_argument("--base-format", choices=["auto", "hf", "tabular"], default="auto")
    parser.add_argument("--require-hard-pool", action="store_true", help="Reject data without full-pool mining provenance")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MODEL_ID, help="Offline HF ID or local checkpoint/cache directory")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--allowlist", default=str(Path(__file__).resolve().parents[4] / "problem-statement/models.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("fp32", "bf16"), default="bf16",
                        help="CUDA forward precision; BF16 matches the existing A40 runtime (CPU stays FP32)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float,
                        help="Optional PyTorch allocator share on the selected GPU")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--tune-only-inference", action="store_true",
                        help="Skip unnecessary TRAIN prediction after fitting; score TUNE (and EVAL only if explicitly requested)")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--conflict-fraction", type=float, default=0.0,
                        help="Reserve this fraction of TRAIN negative pool for high-score numeric/name conflicts")
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--rerank-cutoff", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--evaluate", action="store_true", help="Explicitly evaluate the separate eval split once")
    args = parser.parse_args()
    if min(args.steps, args.inference_batch_size, args.max_length) <= 0 or args.batch_size < 2:
        parser.error("Steps, inference batch size and length must be positive; train batch size must be >=2")
    if args.negative_ratio <= 0 or not 0 < args.positive_fraction < 1:
        parser.error("Negative ratio must be positive; positive fraction must be between zero and one")
    if not 0 <= args.conflict_fraction <= 1:
        parser.error("Conflict fraction must lie in [0,1]")
    if not 0 <= args.rerank_cutoff <= 1 or args.learning_rate <= 0:
        parser.error("Rerank cutoff must be in [0,1] and learning rate positive")
    if args.threads < 1 or (args.gpu_memory_fraction is not None and not 0 < args.gpu_memory_fraction <= 1):
        parser.error("Threads must be positive and GPU memory fraction must lie in (0,1]")
    run(args)


if __name__ == "__main__":
    main()
