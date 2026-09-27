"""Agent Y (27 Sep): turn the Laya-multilingual checkpoint into a standard 1-logit ModernBERT cross-encoder.

Laya (convaiinnovations/laya-multilingual, Apache-2.0, 321,908,998 params) is an mmBERT-base encoder
(``encoder.*``, 306,939,648 params, ModernBERT) plus a task-specific decision head (2 transformer layers,
option scorer, act/escalate head, type embedding, temperature). Only the encoder is kept; the decision head
is dropped and a fresh ``ModernBertForSequenceClassification`` head (dense+norm+1 logit) is initialised with a
fixed seed, so the initial model directory is fully determined and hashable. Weights are upcast FP16 -> FP32
(lossless) so AdamW keeps FP32 master weights (transformers 5 loads the stored dtype by default).

    python -m ber.laya_convert --raw DIR/laya-multilingual-raw --output DIR/laya-multilingual-seqcls
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LAYA_ID = "convaiinnovations/laya-multilingual"
LAYA_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
SEED = 20260927


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(raw: Path, output: Path) -> dict:
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer, ModernBertConfig, ModernBertForSequenceClassification

    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output must be new or empty: {output}")
    config = ModernBertConfig.from_pretrained(raw / "encoder")
    config.num_labels = 1
    config.id2label, config.label2id = {0: "LABEL_0"}, {"LABEL_0": 0}
    config.architectures = ["ModernBertForSequenceClassification"]
    torch.manual_seed(SEED)
    model = ModernBertForSequenceClassification(config).float()
    encoder, dropped, total = {}, [], 0
    with safe_open(str(raw / "model.safetensors"), "pt") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            total += tensor.numel()
            if key.startswith("encoder."):
                encoder["model." + key[len("encoder."):]] = tensor.float()
            else:
                dropped.append(key)
    missing, unexpected = model.load_state_dict(encoder, strict=False)
    fresh = sorted(missing)
    if unexpected or any(not name.startswith(("head.", "classifier.")) for name in fresh):
        raise SystemExit(f"encoder mapping failed: unexpected={unexpected[:5]} missing={fresh[:5]}")
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(raw / "tokenizer")
    tokenizer.save_pretrained(output)
    pair = tokenizer("Boulangerie Martin SARL | 12 Rue de la Paix", "Boulangerie Martin | 12 rue de la paix")
    tokens = tokenizer.convert_ids_to_tokens(pair["input_ids"])
    report = {
        "source": {"hf_id": LAYA_ID, "revision": LAYA_REVISION, "licence": "apache-2.0",
                   "params_total_hf_metadata": 321_908_998, "params_total_counted": int(total),
                   "files_sha256": {str(p.relative_to(raw)): sha256(p) for p in sorted(raw.rglob("*"))
                                    if p.is_file() and ".cache" not in p.parts}},
        "kept": {"encoder_tensors": len(encoder),
                 "encoder_params": int(sum(t.numel() for t in encoder.values()))},
        "dropped_decision_head_tensors": dropped,
        "fresh_head": {"tensors": fresh, "params": int(sum(p.numel() for n, p in model.named_parameters() if n in fresh)),
                       "seed": SEED},
        "model_params_after_conversion": int(sum(p.numel() for p in model.parameters())),
        "classifier_pooling": config.classifier_pooling, "dtype_saved": "float32",
        "pair_template_example": tokens,
        "output_files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output.parent / f"{output.name}.provenance.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = convert(args.raw, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "source"}, indent=1))


if __name__ == "__main__":
    main()
