# Track M status (Qwen3-Reranker-4B LoRA screen)

Updated: 2026-09-27 00:45 IST

## Model
- Qwen/Qwen3-Reranker-4B, HF revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5,
  local `/workspace/az-ml-new/models/Qwen3-Reranker-4B`.
- Licence: **apache-2.0** (model card metadata).
- Parameters: **4,021,784,576** (embedding 388,272,640, tied with lm_head; non-embedding 3,633,511,936). Below the 8B cap.
- Tokenizer: tokenizer.json, vocab.json, merges.txt and tokenizer_config.json are byte-identical to the 0.6B files;
  ids identical on 40,005 sample texts; yes/no/pad = 9693/2152/151643, as for the 0.6B.
  So the 0.6B materialized data is reused unchanged: `/workspace/az-ml-new/qwen/{train_data,score_brest,score_test}`.
- peft 0.21.0 + accelerate 1.15.0 installed into /opt/az-venv (imports OK with transformers 5.17 / torch 2.8).

## Plan / ETA (IST)
- 00:45-01:10 write ber/qwen4b_train.py (LoRA r32/a64/d0.05, BF16 base), smoke-test throughput/memory.
- ~01:10-02:25 LoRA screen on the first 3,250 steps (416k pairs, same as 0.6B), 75-min budget.
- ~02:25-03:35 score B_REST (1.85M pairs, C2>=0.01 filter), then evaluate (~20 min CPU).
- Gate: 4B plain grid >= 0.9815 OR pair tree with 4B >= 0.9865.

## Current
- Model downloaded. Writing the trainer.
