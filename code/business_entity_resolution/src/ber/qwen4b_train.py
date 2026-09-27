"""Qwen3-Reranker-4B LoRA screen in the 0.6B trainer's data world, scored with the same layout.

Commands (GPU train/score; data is the 0.6B materialization, tokenizer byte-identical)::

    python -m ber.qwen4b_train train --data /workspace/az-ml-new/qwen/train_data --model M4B --output RUN \
        [--steps 3250] [--time-budget-min 75] [--stop-after-min 20 --resume]
    python -m ber.qwen4b_train score --data /workspace/az-ml-new/qwen/score_brest \
        --checkpoint RUN/checkpoints/step_XXXXXX --output DIR [--min-c2 .01 --c2-scores C2_DIR]

Everything except the model is ``ber.qwen_train``: same materialized batches (step
``s`` trains rows ``[s*B, (s+1)*B)``), same prompt (S1 in the query slot), same
``logit(yes) - logit(no)`` score at the final real token, same stake-weighted BCE,
same checkpoint/resume/chunk mechanics and the same score outputs (``scores.npy``,
``scored_mask.npy``, ``complete.json``) that ``ber.qwen_ensemble`` reads.

Model: base weights frozen in BF16; LoRA (default r 32, alpha 64, dropout 0.05) on
every attention and MLP projection, adapters in FP32 under BF16 autocast. The tied
embedding / lm_head stays frozen, so the yes-no direction is the base model's.
Checkpoints hold only the adapter (peft layout) + optimizer. Scoring merges the
adapter into the BF16 base. Partition A only; no SLICE or HOLDOUT-2 label is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import qwen_train as qt

MODEL_ID = "Qwen/Qwen3-Reranker-4B"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_base(model_dir, device, torch):
    return qt.load_model(model_dir, device, torch, dtype=torch.bfloat16)


class LoraBackend:
    model_id = MODEL_ID

    def __init__(self, args):
        self.r, self.alpha, self.dropout = args.lora_r, args.lora_alpha, args.lora_dropout
        self.checkpointing = bool(args.gradient_checkpointing)

    def config(self, args) -> dict:
        return {"lora": {"r": self.r, "alpha": self.alpha, "dropout": self.dropout, "targets": list(TARGETS),
                         "adapter_dtype": "fp32"},
                "base_dtype": "bf16", "gradient_checkpointing": self.checkpointing}

    def load(self, args, device, torch, resume):
        from peft import LoraConfig, PeftModel, get_peft_model

        base = load_base(args.model, device, torch)
        if self.checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if resume is None:
            config = LoraConfig(r=self.r, lora_alpha=self.alpha, lora_dropout=self.dropout,
                                target_modules=list(TARGETS), bias="none")
            model = get_peft_model(base, config)
        else:
            model = PeftModel.from_pretrained(base, str(resume), is_trainable=True)
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "lora_" not in name:
                    raise ValueError(f"Unexpected trainable parameter {name}")
                param.data = param.data.float()
                params.append(param)
        trainable = sum(p.numel() for p in params)
        total = sum(p.numel() for p in model.parameters())
        qt.log(f"LoRA r={self.r} alpha={self.alpha}: {trainable:,} trainable of {total:,} parameters; "
               f"gradient checkpointing {self.checkpointing}")
        return model, params

    def init_optimizer_state(self, args, optimizer, torch) -> None:
        return None

    def scorer(self, model, manifest, args, torch):
        return qt.YesNoScorer(model.get_base_model(), manifest["yes_id"], manifest["no_id"])

    def save(self, model, optimizer, torch, **kwargs):
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        adapter = {key: value.detach().to("cpu", copy=True).contiguous()
                   for key, value in get_peft_model_state_dict(model).items()}
        peft_config = model.peft_config["default"]

        def write_model(stage):
            save_file(adapter, str(stage / "adapter_model.safetensors"), metadata={"format": "pt"})
            peft_config.save_pretrained(str(stage))

        return qt.save_checkpoint(model, optimizer, torch, write_model=write_model, **kwargs)


def lora_loader(base_dir, merge: bool = True):
    """Scoring model: BF16 base with the adapter merged in (or kept as separate LoRA layers)."""

    def load(checkpoint, device, torch):
        from peft import PeftModel

        base = load_base(base_dir, device, torch)
        model = PeftModel.from_pretrained(base, str(checkpoint))
        model = model.merge_and_unload() if merge else model.get_base_model()
        model.eval()
        return model

    return load


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True, help="ber.qwen_train prepare-train output (tokenizer-identical)")
    p.add_argument("--model", required=True, help="Local Qwen3-Reranker-4B snapshot")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("bf16",), default="bf16")
    p.add_argument("--steps", type=int, default=3250)
    p.add_argument("--time-budget-min", type=float)
    p.add_argument("--reserve-min", type=float, default=3.0)
    p.add_argument("--calibrate-steps", type=int, default=120)
    p.add_argument("--microbatches", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    p.add_argument("--lr-floor", type=float, default=.1)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-min", type=float, help="Checkpoint and exit after this many wall minutes")
    p.add_argument("--stop-file", help="Checkpoint and exit early once this path exists")
    p.add_argument("--data-offset-steps", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=qt.DEFAULT_SEED)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=.05)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p = sub.add_parser("score")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", help="4B base dir (default: model_dir in the run's train_config.json)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--token-budget", type=int, default=98_304)
    p.add_argument("--flush-every", type=int, default=100)
    p.add_argument("--c2-scores")
    p.add_argument("--min-c2", type=float)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--no-merge", action="store_true", help="Keep LoRA layers unmerged (exact adapter arithmetic)")
    p.add_argument("--keep-mask", help="Boolean .npy over score rows: score only these rows")
    args = parser.parse_args(argv)
    if args.command == "train":
        args.compile = False
        args.init_optimizer = False
        if args.time_budget_min is not None and args.calibrate_steps <= 20:
            parser.error("--calibrate-steps must exceed 20")
        qt.train(args, backend=LoraBackend(args))
    else:
        if (args.min_c2 is None) != (args.c2_scores is None):
            parser.error("--min-c2 and --c2-scores go together")
        base = args.model
        if base is None:
            config_path = Path(args.checkpoint).parent.parent / "train_config.json"
            base = json.loads(config_path.read_text())["model_dir"]
        qt.score(args, loader=lora_loader(base, merge=not args.no_merge), model_id=MODEL_ID)


if __name__ == "__main__":
    main()
