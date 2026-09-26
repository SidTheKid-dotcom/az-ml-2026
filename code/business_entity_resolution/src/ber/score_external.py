"""Score any pair set with a fast-trainer checkpoint, gated by the v1 tree.

Uses the checkpoint's own text view, maximum length and microbatch tokenizer
(`fast_trainer._score_chunk`), so arena B_REST and test pairs are scored exactly
as training saw text. Only rows with tree >= gate are scored (others stay 0),
in pair order, resumable chunk by chunk. No labels are read.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Rust batch encoding across cores

import numpy as np
import pandas as pd

from .cross_encoder_experiment import load_reranker
from .fast_trainer import verified_checkpoint
from .text_views import record_texts


def _encode(tokenizer, left_text, right_text, left, right, ids, max_length):
    """Same pair encoding as training (truncation, max length), without padding."""
    encoded = tokenizer([left_text[i] for i in left[ids]], [right_text[j] for j in right[ids]],
                        padding=False, truncation=True, max_length=max_length,
                        return_attention_mask=False)
    sequences = encoded["input_ids"]
    lengths = np.fromiter(map(len, sequences), dtype=np.int32, count=len(sequences))
    return ids, sequences, lengths


def _batches(sequences, lengths, batch, pad_id, multiple=8):
    """Length-sorted microbatches padded with NumPy (attention mask from lengths)."""
    order = np.argsort(lengths, kind="stable")
    for start in range(0, len(order), batch):
        positions = order[start:start + batch]
        size = lengths[positions]
        width = int(-(-int(size.max()) // multiple) * multiple)
        matrix = np.full((len(positions), width), pad_id, dtype=np.int64)
        for row, position in enumerate(positions):
            sequence = sequences[position]
            matrix[row, :len(sequence)] = sequence
        mask = (np.arange(width)[None, :] < size[:, None]).astype(np.int64)
        yield positions, matrix, mask


def _indices(args) -> tuple[np.ndarray, np.ndarray]:
    if args.left and args.right:
        return np.load(args.left, mmap_mode="r"), np.load(args.right, mmap_mode="r")
    import pyarrow.parquet as pq
    table = pq.read_table(args.pairs, columns=["left_idx", "right_idx"])
    return table.column("left_idx").to_numpy(), table.column("right_idx").to_numpy()


def run(args) -> dict:
    import torch

    torch.set_num_threads(args.threads)
    config, checkpoint = verified_checkpoint(Path(args.checkpoint))
    profile = config["profile"]
    left, right = _indices(args)
    tree = np.load(args.tree, mmap_mode="r")
    if not (len(left) == len(right) == len(tree)):
        raise ValueError("Pair indices and tree scores are not aligned")
    selected = np.asarray(tree >= args.gate)
    if args.population:
        population = np.load(args.population)
        selected &= population[np.asarray(left)]
    ids_all = np.flatnonzero(selected)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.json"
    done = json.loads(progress_path.read_text())["done"] if progress_path.exists() else 0
    score_path = out / "scores.npy"
    if done == 0 and not score_path.exists():
        memmap = np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float32, shape=(len(left),))
        memmap[:] = 0
        memmap.flush()
        del memmap
    scores = np.load(score_path, mmap_mode="r+")
    reference = pd.read_parquet(args.reference, columns=["business_name", "business_address"]).fillna("")
    pool = pd.read_parquet(args.pool, columns=["business_name", "business_address"]).fillna("")
    left_text = record_texts(reference, canonical=profile["canonical_text"], field_tagged=profile["field_tagged"])
    right_text = record_texts(pool, canonical=profile["canonical_text"], field_tagged=profile["field_tagged"])
    model, tokenizer = load_reranker(str(args.checkpoint), args.device)
    model.eval()
    pad_id = tokenizer.pad_token_id
    cuda = args.device.startswith("cuda")
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float32
    left64, right64 = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
    started, begin_done = time.perf_counter(), done
    chunks = [ids_all[b:b + args.score_chunk] for b in range(done, len(ids_all), args.score_chunk)]
    pool = ThreadPoolExecutor(max_workers=args.encode_workers)
    futures = [pool.submit(_encode, tokenizer, left_text, right_text, left64, right64, chunk,
                           config["max_length"]) for chunk in chunks[:args.encode_workers + 1]]
    submitted = len(futures)
    for index in range(len(chunks)):
        ids, sequences, lengths = futures[index].result()
        futures[index] = None
        if submitted < len(chunks):
            futures.append(pool.submit(_encode, tokenizer, left_text, right_text, left64, right64,
                                       chunks[submitted], config["max_length"]))
            submitted += 1
        output = np.empty(len(ids), dtype=np.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=cuda and args.amp == "bf16"):
            for positions, matrix, mask in _batches(sequences, lengths, args.inference_batch, pad_id):
                tensors = {"input_ids": torch.from_numpy(matrix), "attention_mask": torch.from_numpy(mask)}
                if cuda:
                    tensors = {k: v.pin_memory().to(args.device, non_blocking=True) for k, v in tensors.items()}
                logits = model(**tensors).logits
                if logits.shape != (len(positions), 1):
                    raise ValueError("Checkpoint scorer requires one match logit per pair")
                output[positions] = torch.sigmoid(logits[:, 0].float()).cpu().numpy()
        scores[ids] = output
        scores.flush()
        done += len(ids)
        rate = (done - begin_done) / max(time.perf_counter() - started, 1e-9)
        progress_path.write_text(json.dumps({"done": done, "total": int(len(ids_all)),
                                             "pairs_per_second": rate}) + "\n")
        print(json.dumps({"done": done, "total": int(len(ids_all)), "pairs_per_second": round(rate, 1)}), flush=True)
    np.save(out / "scored_mask.npy", selected)
    manifest = {"state": "complete", "checkpoint": checkpoint, "profile": profile,
                "max_length": config["max_length"], "gate": args.gate, "pairs": int(len(left)),
                "scored_pairs": int(len(ids_all)), "labels_read": False,
                "inputs": {"reference": str(args.reference), "pool": str(args.pool),
                           "left": str(args.left or args.pairs), "right": str(args.right or args.pairs),
                           "tree": str(args.tree), "population": str(args.population)},
                "runtime": {"amp": args.amp, "inference_batch": args.inference_batch,
                            "seconds": time.perf_counter() - started}}
    (out / "complete.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--pairs", help="Parquet with left_idx/right_idx (if no --left/--right)")
    parser.add_argument("--left")
    parser.add_argument("--right")
    parser.add_argument("--tree", required=True, help="Aligned v1 tree scores (gate)")
    parser.add_argument("--gate", type=float, default=.01)
    parser.add_argument("--population", help="Optional per-S1 boolean .npy restricting scored S1s")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--inference-batch", type=int, default=512)
    parser.add_argument("--score-chunk", type=int, default=200_000)
    parser.add_argument("--encode-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    print(json.dumps(run(parser.parse_args()), indent=2, default=str))


if __name__ == "__main__":
    main()
