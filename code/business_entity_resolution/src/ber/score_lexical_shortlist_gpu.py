"""Standalone, label-free GPU scoring of a frozen lexical rescue shortlist.

Copy this file and the small Parquet/manifest into an isolated GPU directory.
The preserved v1 BGE checkpoint is read-only; no production files are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    started = time.monotonic()
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or args.device != "cuda:0":
        raise RuntimeError("This bounded experiment requires the explicitly free A40")
    shortlist, output, checkpoint = map(Path, (args.shortlist, args.output, args.checkpoint))
    manifest = json.loads(Path(args.manifest).read_text())
    if sha256(shortlist) != manifest["shortlist_sha256"]:
        raise ValueError("Shortlist text/pair rows changed after TRAIN-CAL gate selection")
    frame = pd.read_parquet(shortlist)
    expected = ["split", "new_pair_position", "source1_entity_id", "target_entity_id",
                "lexical_probability", "left_text", "right_text"]
    counts = {"cal": manifest["calibration_pairs"], "tune": manifest["tune_pairs"]}
    if (list(frame) != expected or len(frame) != sum(counts.values())
            or frame.split.value_counts().to_dict() != counts
            or frame.duplicated(["source1_entity_id", "target_entity_id"]).any()
            or frame[expected].isna().any().any()):
        raise ValueError("Shortlist coverage, row order or identities changed")
    weights = checkpoint / "model.safetensors"
    weights_hash = sha256(weights)
    if weights_hash != args.weights_sha256:
        raise ValueError("Frozen BGE checkpoint weights do not match the preserved v1 identity")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"New experiment output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False).to(args.device).eval()
    if model.config.num_labels != 1:
        raise ValueError("Expected the preserved single-logit BGE reranker")
    encoded = tokenizer(frame.left_text.tolist(), frame.right_text.tolist(),
                        padding=False, truncation=True, max_length=256, return_attention_mask=True)
    columns = [key for key in tokenizer.model_input_names if key in encoded]
    if "input_ids" not in columns or any(len(encoded[key]) != len(frame) for key in columns):
        raise ValueError("Tokenizer did not preserve every candidate pair")
    order = np.argsort([len(ids) for ids in encoded["input_ids"]], kind="stable")
    probabilities = np.empty(len(frame), dtype=np.float32)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        for start in range(0, len(order), 128):
            positions = order[start:start + 128]
            rows = [{key: encoded[key][int(i)] for key in columns} for i in positions]
            tokens = tokenizer.pad(rows, padding=True, pad_to_multiple_of=8, return_tensors="pt")
            tokens = {key: value.to(args.device) for key, value in tokens.items()}
            logits = model(**tokens).logits
            if logits.shape != (len(positions), 1):
                raise ValueError("BGE must produce exactly one positive-match logit per pair")
            probabilities[positions] = torch.sigmoid(logits[:, 0].float()).cpu().numpy()
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Non-finite or invalid cross-encoder probabilities")
    path = output / "probabilities.npy"
    with (output / "probabilities.npy.tmp").open("wb") as stream:
        np.save(stream, probabilities)
    (output / "probabilities.npy.tmp").replace(path)
    report = {"schema": 1, "input_shortlist_sha256": manifest["shortlist_sha256"],
              "input_manifest_sha256": sha256(args.manifest),
              "checkpoint": str(checkpoint), "checkpoint_weights_sha256": weights_hash,
              "checkpoint_identity_sha256": manifest["v1_reranker_checkpoint_sha256"],
              "model_id": "BAAI/bge-reranker-v2-m3", "scored_rows": len(frame),
              "calibration_rows": counts["cal"], "tune_rows": counts["tune"],
              "device": args.device, "autocast_dtype": str(dtype), "batch_size": 128,
              "max_length": 256, "length_sorted": True, "pad_to_multiple_of": 8,
              "labels_read": False, "probabilities_sha256": sha256(path),
              "seconds": time.monotonic() - started}
    temporary = output / "score_report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output / "score_report.json")
    print(json.dumps(report, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
