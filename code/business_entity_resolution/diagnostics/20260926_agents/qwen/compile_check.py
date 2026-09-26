"""Does train(--compile) run and match eager on a tiny random Qwen3 (CPU)?"""
import sys, tempfile, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
sys.path.insert(0, "/opt/az-ml-new/agents/qwen")
import test_qwen_train as t
from ber import qwen_train as qt
tmp = Path(tempfile.mkdtemp())
model_dir = t.tiny_model_dir(tmp)
data = tmp / "data"; t.write_train_data(data, steps=6, batch=8)
eager = qt.train(t.train_args(data, model_dir, tmp / "eager", checkpoint_every=1000))
comp = qt.train(t.train_args(data, model_dir, tmp / "comp", checkpoint_every=1000, compile=True))
a = t.weights_of(Path(eager["final_checkpoint"])); b = t.weights_of(Path(comp["final_checkpoint"]))
print(json.dumps({"max_abs_diff": max(float((a[k] - b[k]).abs().max()) for k in a), "keys_equal": a.keys() == b.keys()}))
