"""Score any pair set with a fast-trainer checkpoint, gated by the v1 tree.

Uses the checkpoint's own text view, maximum length and microbatch tokenizer
(`fast_trainer._score_chunk`), so arena B_REST and test pairs are scored exactly
as training saw text. Only rows with tree >= gate are scored (others stay 0),
in pair order, resumable chunk by chunk. No labels are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .cross_encoder_experiment import load_reranker
from .fast_trainer import _score_chunk, verified_checkpoint
from .text_views import record_texts


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
    options = SimpleNamespace(max_length=config["max_length"], inference_batch=args.inference_batch,
                              pin_memory=args.device.startswith("cuda"), device=args.device, amp=args.amp)
    left64, right64 = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
    started, begin_done = time.perf_counter(), done
    for begin in range(done, len(ids_all), args.score_chunk):
        ids = ids_all[begin:begin + args.score_chunk]
        scores[ids] = _score_chunk(model, tokenizer, left64, right64, ids, left_text, right_text, options, torch)
        scores.flush()
        done = begin + len(ids)
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
    parser.add_argument("--score-chunk", type=int, default=500_000)
    parser.add_argument("--threads", type=int, default=8)
    print(json.dumps(run(parser.parse_args()), indent=2, default=str))


if __name__ == "__main__":
    main()
