"""Drop-in train script for Google Colab with BPE + nn.Transformer baseline.

Mirrors `train.py` but pre-sets TOKENIZER = "bpe" and uses the
`BaselineNNTransformer` instead of the hand-rolled model. The resulting
checkpoint is compatible with `evaluate.py` and `translate.py` via the
same `build_model_from_checkpoint`-style loading (handled inside
`evaluate.py` and `translate.py` via a small switch on a CLI flag).

NOTE: `build_model_from_checkpoint` in `translate.py` currently instantiates
the hand-rolled `Transformer` class. To load a baseline checkpoint into
`evaluate.py` / `translate.py` without changing those files, run them with
the `--baseline-nn` flag (added by this commit).

Usage on Colab:
    !git clone <this-repo>
    %cd transformer
    !pip install -r requirements.txt
    !python scripts/train_baseline_nn_colab.py --epochs 20

After training, download:
    checkpoints/baseline_best.pt
    checkpoints/baseline_latest.pt
    checkpoints/baseline_metrics.jsonl
    .data/spm/spm_en.model + spm_en.vocab
    .data/spm/spm_de.model + spm_de.vocab
"""
from __future__ import annotations

import os
import sys

# Ensure the project root is on sys.path so `import config` works when this
# script is invoked from the `scripts/` directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# CRITICAL: must be set BEFORE importing config / dataset anywhere.
import config
config.TOKENIZER = "bpe"

# Now safe to import everything else.
from train import main  # noqa: E402

if __name__ == "__main__":
    # Inject --baseline-nn flag into argv before parsing.
    if "--baseline-nn" not in sys.argv:
        sys.argv.insert(1, "--baseline-nn")
    # Replace argv[0] (the script name) so argparse's help text reads correctly.
    sys.argv[0] = "train.py"
    raise SystemExit(main())
