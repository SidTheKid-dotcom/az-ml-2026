"""Run ber.score_external with the un-fine-tuned base bge-reranker-v2-m3 (F7). The text view and max length are
taken from C2 step_010000's saved config (canonical_text False, field_tagged False, max_length 256)."""
import ber.score_external as se
from ber.fast_trainer import verified_checkpoint as vc
from pathlib import Path
REF = Path("/workspace/az-ml-new/c2_plain/ckpt/step_010000")
def fake(path):
    config, _ = vc(REF)
    return config, {"path": str(path), "base_model": True, "text_config_from": str(REF)}
se.verified_checkpoint = fake
se.main()
