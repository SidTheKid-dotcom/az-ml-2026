"""Agent Y (27 Sep): C2-FR recipe with the Laya-multilingual (mmBERT/ModernBERT) backbone.

Runs the unchanged ``ber.fast_trainer`` (train) and ``ber.score_external`` (score). The only adaptation is
the initial-model binding: ``fast_trainer.verified_checkpoint`` accepts exactly one model id
(``fast_trainer.MODEL_ID``, BGE), so this entry point rebinds it to the Laya id before dispatching. The
allowlist passed with ``--allowlist`` must list the Laya id (role reranker, Apache-2.0, 0.322B).

    python -m ber.laya_c2 train --mode c2 ... --model DIR/laya-multilingual-seqcls --allowlist models_laya.yaml
    python -m ber.laya_c2 score --checkpoint CKPT --reference ... (score_external arguments)
"""
from __future__ import annotations

import sys

from ber import fast_trainer

LAYA_ID = "convaiinnovations/laya-multilingual"


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "score"}:
        raise SystemExit("usage: python -m ber.laya_c2 {train|score} ARGS...")
    command, argv = sys.argv[1], sys.argv[2:]
    fast_trainer.MODEL_ID = LAYA_ID  # read at call time by verified_checkpoint (training + scoring)
    if command == "train":
        if "--model-id" not in argv:
            argv = ["--model-id", LAYA_ID, *argv]
        fast_trainer.main(["--command", "train", *argv])
        return
    from ber import score_external

    sys.argv = ["ber.score_external", *argv]
    score_external.main()


if __name__ == "__main__":
    main()
