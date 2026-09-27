"""Agent Y CPU check: Laya/ModernBERT logits must not depend on right-padding width (training pads to length buckets,
scoring to multiples of 8). Usage: y_pad_check.py CHECKPOINT"""
import sys, json
import numpy as np, torch
from ber.cross_encoder_experiment import load_reranker
model, tok = load_reranker(sys.argv[1], "cpu")
model.eval()
pairs = [("Boulangerie Martin SARL | 12 Rue de la Paix, 75002 Paris", "Boulangerie Martin | 12 rue de la paix 75002"),
         ("Tata Motors Ltd | Bombay House, Mumbai", "TATA MOTORS LIMITED | 24 Homi Mody St, Mumbai 400001"),
         ("Joe's Pizza | 7 Carmine St, New York, NY", "Joes Pizza | 9 Carmine Street New York"),
         ("Groupe Durand Holding | 3 Avenue Foch", "Durand | 3 av. Foch")]
out = {}
with torch.inference_mode():
    for width in (None, 64, 256):
        enc = tok([a for a, _ in pairs], [b for _, b in pairs], padding="max_length" if width else True,
                  max_length=width, truncation=True, return_tensors="pt")
        out[str(width)] = model(**enc).logits[:, 0].tolist()
    single = [model(**tok(a, b, return_tensors="pt")).logits[0, 0].item() for a, b in pairs]
out["single_unpadded"] = single
ref = np.asarray(single)
out["max_abs_diff_vs_single"] = {k: float(np.abs(np.asarray(v) - ref).max()) for k, v in out.items() if k != "single_unpadded"}
print(json.dumps(out, indent=1))
