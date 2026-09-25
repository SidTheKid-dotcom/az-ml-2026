"""Compare an allowlisted multilingual retriever against a fixed lexical pool.

The default run uses train/tune entities only. Holdout labels are never scored
unless --evaluate is explicitly supplied. GPU dependencies are imported lazily
so data/pair utilities and --help also work in the lightweight local environment.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from .experiment import json_write, load_data
from .features import build_features
from .metrics import candidate_report, pair_labels, score_pairs, tune_threshold
from .models import fit_model, predict_proba
from .normalization import normalize_text
from .prepare import log


def frame_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def resolve_model_id(model: str, label: str | None, explicit_id: str | None,
                     allowlist: Path) -> tuple[str, dict]:
    """Validate provenance before loading local weights; never trust remote code."""
    import yaml

    document = yaml.safe_load(allowlist.read_text(encoding="utf-8"))
    entries = {row["id"]: row for row in document["models"]}
    choices = [explicit_id, model, label]
    for component in Path(model).parts:
        if component.startswith("models--"):
            choices.append(component[len("models--"):].replace("--", "/", 1))
    selected = next((value for value in choices if value in entries), None)
    if selected is None:
        raise ValueError("Supply an allowlisted Hugging Face ID, or --model-id for a local cache directory")
    row = entries[selected]
    if row.get("role") != "embedding" or row.get("licence") not in {"mit", "apache-2.0"}:
        raise ValueError(f"Model is not an allowed MIT/Apache embedding model: {selected}")
    if float(row["params_b"]) > float(document["budget_total_params_b"]):
        raise ValueError("Embedding model exceeds the model parameter budget")
    return selected, row


def record_texts(frame: pd.DataFrame, prefix: str = "") -> list[str]:
    names = frame.business_name.fillna("").astype(str)
    addresses = frame.business_address.fillna("").astype(str)
    # Keep original scripts: the multilingual encoder, unlike lexical retrieval,
    # must see the native spelling. E5 query/passage prefixes are explicit.
    return (prefix + names + " | " + addresses).tolist()


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values, dtype=np.float32))
    temporary.replace(path)


def finetune_contrastive(model, reference: pd.DataFrame, pool: pd.DataFrame,
                         truth: dict[str, set[str]], query_prefix: str,
                         passage_prefix: str, args) -> list[dict]:
    """Train only on distinct TRAIN businesses, with one positive per business."""
    import torch
    from torch.nn import functional as F

    lookup = {entity: i for i, entity in enumerate(pool.entity_id)}
    eligible, positives = [], {}
    owner = {}
    for left in np.flatnonzero(reference.split.eq("train")):
        entity = reference.iloc[left].entity_id
        positions = [lookup[value] for value in sorted(truth.get(entity, set())) if value in lookup]
        if not positions:
            continue
        for right in positions:
            if right in owner and owner[right] != left:
                raise ValueError("Contrastive batches require disjoint positive business groups")
            owner[right] = left
        eligible.append(left)
        positives[left] = positions
    if len(eligible) < 2:
        raise ValueError("Contrastive fine-tuning needs at least two training businesses with matches")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_learning_rate, weight_decay=0.01)
    batch_size = min(args.finetune_batch_size, len(eligible))
    queries, passages = record_texts(reference, query_prefix), record_texts(pool, passage_prefix)
    trace = []
    for step in range(args.finetune_steps):
        selected = rng.choice(eligible, size=batch_size, replace=False)
        selected_right = [int(rng.choice(positives[int(left)])) for left in selected]
        qtokens = {k: value.to(args.device) if hasattr(value, 'to') else value
                   for k, value in model.tokenize([queries[i] for i in selected]).items()}
        ptokens = {k: value.to(args.device) if hasattr(value, 'to') else value
                   for k, value in model.tokenize([passages[i] for i in selected_right]).items()}
        optimizer.zero_grad(set_to_none=True)
        use_bfloat = args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bfloat else nullcontext()
        with amp:
            query_vectors = F.normalize(model(qtokens)["sentence_embedding"], p=2, dim=1)
            passage_vectors = F.normalize(model(ptokens)["sentence_embedding"], p=2, dim=1)
            scores = query_vectors @ passage_vectors.T / args.temperature
            target = torch.arange(batch_size, device=scores.device)
            loss = 0.5 * (F.cross_entropy(scores.float(), target) + F.cross_entropy(scores.T.float(), target))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Linear warmup then decay; no validation labels enter the optimizer.
        warmup = max(1, args.finetune_steps // 10)
        factor = min((step + 1) / warmup, (args.finetune_steps - step) / max(1, args.finetune_steps - warmup))
        for group in optimizer.param_groups:
            group["lr"] = args.finetune_learning_rate * factor
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.finetune_steps:
            row = {"step": step + 1, "loss": float(loss.detach().cpu()), "distinct_train_businesses": batch_size}
            trace.append(row)
            log(f"Contrastive {row}")
    model.eval()
    return trace


def encode_cached(reference: pd.DataFrame, pool: pd.DataFrame, truth: dict,
                  args, model_id: str, signature: dict) -> tuple[np.ndarray, np.ndarray]:
    output = Path(args.output)
    manifest_path = output / "embedding_manifest.json"
    ref_path, pool_path = output / "reference_embeddings.npy", output / "pool_embeddings.npy"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous != signature:
            raise ValueError("Output embedding cache belongs to a different model/data/config; use another --output")
        if ref_path.exists() and pool_path.exists():
            log("Reusing matching embedding cache")
            return np.load(ref_path, mmap_mode="r"), np.load(pool_path, mmap_mode="r")
    from sentence_transformers import SentenceTransformer
    import torch

    log(f"Loading offline model {model_id} from {args.model}")
    model = SentenceTransformer(args.model, device=args.device, local_files_only=True, trust_remote_code=False)
    model.max_seq_length = args.max_length
    query_prefix = "query: " if "multilingual-e5" in model_id else ""
    passage_prefix = "passage: " if "multilingual-e5" in model_id else ""
    if args.finetune_steps:
        trace = finetune_contrastive(model, reference, pool, truth, query_prefix, passage_prefix, args)
        model.save(str(output / "finetuned_model"))
        json_write(output / "contrastive_training.json", trace)
    model.eval()
    if args.device.startswith("cuda"):
        model.half()
    for frame, path, prefix in ((reference, ref_path, query_prefix), (pool, pool_path, passage_prefix)):
        log(f"Encoding {len(frame):,} rows to {path.name}")
        vectors = model.encode(record_texts(frame, prefix), batch_size=args.batch_size,
                               show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(frame) or not np.isfinite(vectors).all():
            raise ValueError("Encoder returned invalid embedding shape or values")
        _atomic_npy(path, vectors)
        del vectors
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    json_write(manifest_path, signature)
    return np.load(ref_path, mmap_mode="r"), np.load(pool_path, mmap_mode="r")


def _torch_topk(queries: np.ndarray, passages: np.ndarray, top_k: int,
                device: str, query_batch_size: int, pool_batch_size: int):
    import torch

    all_scores, all_positions = [], []
    with torch.inference_mode():
        for start in range(0, len(queries), query_batch_size):
            query = torch.as_tensor(np.array(queries[start:start + query_batch_size]), device=device)
            best_scores = torch.empty((len(query), 0), device=device)
            best_positions = torch.empty((len(query), 0), dtype=torch.long, device=device)
            for pstart in range(0, len(passages), pool_batch_size):
                passage = torch.as_tensor(np.array(passages[pstart:pstart + pool_batch_size]), device=device)
                scores = query @ passage.T
                block_k = min(top_k, len(passage))
                block_scores, block_positions = torch.topk(scores, block_k, dim=1)
                merged_scores = torch.cat((best_scores, block_scores), dim=1)
                merged_positions = torch.cat((best_positions, block_positions + pstart), dim=1)
                keep = min(top_k, merged_scores.shape[1])
                best_scores, order = torch.topk(merged_scores, keep, dim=1)
                best_positions = torch.gather(merged_positions, 1, order)
                del passage, scores
            all_scores.append(best_scores.cpu().numpy())
            all_positions.append(best_positions.cpu().numpy())
    return np.concatenate(all_scores), np.concatenate(all_positions)


def dense_candidates(reference: pd.DataFrame, pool: pd.DataFrame,
                     ref_vectors: np.ndarray, pool_vectors: np.ndarray,
                     query_mask: np.ndarray, top_k: int, device: str = "cuda",
                     backend: str = "auto", query_batch_size: int = 256,
                     pool_batch_size: int = 131072) -> pd.DataFrame:
    """Exact dense top-K with country blocking and bounded similarity matrices."""
    if backend == "auto":
        backend = "torch" if device.startswith("cuda") else "faiss"
        if backend == "faiss" and importlib.util.find_spec("faiss") is None:
            backend = "torch"
    ref_countries = reference.country.map(normalize_text).to_numpy()
    pool_countries = pool.country.map(normalize_text).to_numpy()
    chunks = []
    for country in sorted(set(ref_countries[query_mask]) & set(pool_countries)):
        left = np.flatnonzero(query_mask & (ref_countries == country))
        right = np.flatnonzero(pool_countries == country)
        k = min(top_k, len(right))
        if not len(left) or not k:
            continue
        queries = np.asarray(ref_vectors[left], dtype=np.float32)
        passages = np.asarray(pool_vectors[right], dtype=np.float32)
        log(f"Dense {country}: {len(left):,} queries x {len(right):,} pool, top{k}, {backend}")
        if backend == "faiss":
            import faiss

            index = faiss.IndexFlatIP(passages.shape[1])
            index.add(np.ascontiguousarray(passages))
            scores, positions = index.search(np.ascontiguousarray(queries), k)
        elif backend == "torch":
            scores, positions = _torch_topk(queries, passages, k, device, query_batch_size, pool_batch_size)
        elif backend == "numpy":
            # Small CPU smoke checks only; the same query batch bound applies.
            score_chunks, position_chunks = [], []
            for start in range(0, len(queries), query_batch_size):
                similarity = queries[start:start + query_batch_size] @ passages.T
                positions_part = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
                score_chunks.append(np.take_along_axis(similarity, positions_part, axis=1))
                position_chunks.append(positions_part)
            scores, positions = np.concatenate(score_chunks), np.concatenate(position_chunks)
        else:
            raise ValueError(f"Unknown dense retrieval backend {backend}")
        chunks.append(pd.DataFrame({"left_idx": np.repeat(left, k), "right_idx": right[positions.ravel()],
                                    "dense_score": scores.ravel().astype(np.float32),
                                    "dense_rank": np.tile(np.arange(1, k + 1), len(left)).astype(np.int32)}))
    if not chunks:
        return pd.DataFrame({"left_idx": pd.Series(dtype="int64"), "right_idx": pd.Series(dtype="int64"),
                             "dense_score": pd.Series(dtype="float32"), "dense_rank": pd.Series(dtype="int32")})
    return pd.concat(chunks, ignore_index=True)


def union_candidates(base: pd.DataFrame, dense: pd.DataFrame,
                     ref_vectors: np.ndarray, pool_vectors: np.ndarray) -> pd.DataFrame:
    """Preserve lexical scores; explicitly flag absent scores on dense-only pairs."""
    base = base.copy()
    if base.duplicated(["left_idx", "right_idx"]).any():
        raise ValueError("Lexical candidate file contains duplicate pairs")
    base["_lexical"] = True
    result = base.merge(dense, on=["left_idx", "right_idx"], how="outer", validate="one_to_one")
    result["dense_only"] = result["_lexical"].isna().astype(np.float32)
    result.drop(columns="_lexical", inplace=True)
    for field in ("name_score", "address_score", "retrieval_score"):
        if field not in result:
            result[field] = np.float32(0)
        result[field] = result[field].fillna(0).astype(np.float32)
    maximum_rank = base.groupby("left_idx").retrieval_rank.max()
    fallback_rank = result.left_idx.map(maximum_rank).fillna(0) + result.dense_rank.fillna(1)
    result["retrieval_rank"] = result.retrieval_rank.fillna(fallback_rank).astype(np.int32)
    # Every lexical candidate receives its actual dense cosine, even when it was
    # not in dense top-K. This avoids confusing missing scores with low similarity.
    left = result.left_idx.to_numpy(dtype=np.int64)
    right = result.right_idx.to_numpy(dtype=np.int64)
    scores = np.empty(len(result), dtype=np.float32)
    for start in range(0, len(result), 20000):
        stop = start + 20000
        scores[start:stop] = np.einsum("ij,ij->i", ref_vectors[left[start:stop]], pool_vectors[right[start:stop]])
    result["dense_score"] = np.clip(scores, -1, 1)
    result["dense_rank"] = result.dense_rank.fillna(0).astype(np.int32)
    result.sort_values(["left_idx", "retrieval_rank", "right_idx"], kind="stable", inplace=True, ignore_index=True)
    return result


def run(args):
    started = time.time()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model_id, license_row = resolve_model_id(args.model, args.model_label, args.model_id, Path(args.allowlist))
    reference, pool, truth = load_data(args.data)
    splits = ("train", "tune", "eval") if args.evaluate else ("train", "tune")
    query_mask = reference.split.isin(splits).to_numpy()
    # Labels outside the requested splits are removed before any experiment work.
    permitted_ids = set(reference.loc[query_mask, "entity_id"])
    truth = {key: value for key, value in truth.items() if key in permitted_ids}
    columns = ["entity_id", "business_name", "business_address", "country"]
    signature = {"schema": 1, "model_id": model_id, "model_path": args.model,
                 "max_length": args.max_length, "reference_digest": frame_digest(reference, columns),
                 "pool_digest": frame_digest(pool, columns), "seed": args.seed,
                 "finetune_steps": args.finetune_steps, "finetune_batch_size": args.finetune_batch_size,
                 "finetune_learning_rate": args.finetune_learning_rate, "temperature": args.temperature}
    if args.finetune_steps:
        train_ids = reference.loc[reference.split.eq("train"), "entity_id"]
        signature["train_truth_digest"] = hashlib.sha256(json.dumps(
            {key: sorted(truth.get(key, set())) for key in train_ids}, sort_keys=True).encode()).hexdigest()
    timings = {}
    stage = time.time()
    rv, pv = encode_cached(reference, pool, truth, args, model_id, signature)
    timings["embedding_seconds"] = time.time() - stage
    stage = time.time()
    base = pd.read_parquet(args.base_pairs)
    base = base.loc[query_mask[base.left_idx.to_numpy(dtype=np.int64)]].reset_index(drop=True)
    dense = dense_candidates(reference, pool, rv, pv, query_mask, args.top_k, args.device,
                             args.retrieval_backend, args.query_batch_size, args.pool_batch_size)
    pairs = union_candidates(base, dense, rv, pv)
    pairs.to_parquet(output / "pairs.parquet", index=False)
    timings["retrieval_seconds"] = time.time() - stage
    retrieval = {name: {split: candidate_report(reference, pool, candidate, truth, reference.split.eq(split))
                        for split in splits}
                 for name, candidate in (("lexical", base), ("dense", dense), ("union", pairs))}
    json_write(output / "retrieval.json", retrieval)
    log(f"Tuning retrieval: {retrieval['union']['tune']}")
    stage = time.time()
    X = build_features(reference, pool, pairs)
    X["dense_score"] = pairs.dense_score.to_numpy(dtype=np.float32)
    X["dense_only"] = pairs.dense_only.to_numpy(dtype=np.float32)
    X.to_parquet(output / "features.parquet", index=False)
    timings["features_seconds"] = time.time() - stage
    y = pair_labels(reference, pool, pairs, truth)
    split_values = reference.split.to_numpy()[pairs.left_idx.to_numpy(dtype=np.int64)]
    train_idx, tune_idx = np.flatnonzero(split_values == "train"), np.flatnonzero(split_values == "tune")
    if len(train_idx) > args.max_train_pairs:
        rng = np.random.default_rng(args.seed)
        positives, negatives = train_idx[y[train_idx] == 1], train_idx[y[train_idx] == 0]
        keep = max(1, args.max_train_pairs - len(positives))
        train_idx = np.sort(np.r_[positives, rng.choice(negatives, min(keep, len(negatives)), replace=False)])
    stage = time.time()
    classifier_device = args.classifier_device or ("GPU" if args.device.startswith("cuda") else "CPU")
    model = fit_model(X.iloc[train_idx], y[train_idx], kind="catboost", device=classifier_device,
                      seed=args.seed, iterations=args.iterations, depth=args.depth,
                      thread_count=args.threads, eval_set=(X.iloc[tune_idx], y[tune_idx]), verbose=100)
    timings["fit_seconds"] = time.time() - stage
    probabilities = predict_proba(model, X)
    threshold, trace = tune_threshold(reference, pool, pairs, probabilities, truth, reference.split.eq("tune"))
    scores = {split: score_pairs(reference, pool, pairs, probabilities, truth, threshold, reference.split.eq(split))
              for split in splits if split != "train"}
    report = {"model": args.model_label or model_id, "model_id": model_id, "licence": license_row["licence"],
              "params_b": license_row["params_b"], "evaluation_used": args.evaluate,
              "finetune_steps": args.finetune_steps, "train_pairs": len(train_idx),
              "feature_count": len(X.columns), "dense_only_pairs": int(pairs.dense_only.sum()),
              "retrieval": retrieval, **scores, "timings": timings, "elapsed_seconds": time.time() - started}
    joblib.dump({"model": model, "threshold": threshold, "feature_columns": list(X.columns),
                 "embedding_config": signature, "dense_top_k": args.top_k, "record": report},
                output / "model.joblib")
    _atomic_npy(output / "probabilities.npy", probabilities)
    json_write(output / "thresholds.json", trace)
    json_write(output / "results.json", report)
    log(json.dumps(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/quick")
    parser.add_argument("--base-pairs", required=True)
    parser.add_argument("--model", required=True, help="Allowlisted HF ID or offline local cache directory")
    parser.add_argument("--model-label")
    parser.add_argument("--model-id", help="Canonical allowlisted HF ID when --model is a custom local directory")
    parser.add_argument("--allowlist", default=str(Path(__file__).resolve().parents[4] / "problem-statement/models.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--retrieval-backend", choices=["auto", "torch", "faiss", "numpy"], default="auto")
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--pool-batch-size", type=int, default=131072)
    parser.add_argument("--classifier-device", choices=["CPU", "GPU"])
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--max-train-pairs", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--evaluate", action="store_true", help="Explicitly score the separate eval split once")
    parser.add_argument("--finetune-steps", type=int, default=0)
    parser.add_argument("--finetune-batch-size", type=int, default=32)
    parser.add_argument("--finetune-learning-rate", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    args = parser.parse_args()
    if min(args.batch_size, args.max_length, args.top_k, args.query_batch_size, args.pool_batch_size) <= 0:
        parser.error("Batch sizes, max-length, and top-k must be positive")
    if args.finetune_steps < 0 or args.temperature <= 0 or args.finetune_batch_size < 2:
        parser.error("Fine-tune steps must be nonnegative, batch size >=2 and temperature positive")
    run(args)


if __name__ == "__main__":
    main()
