"""Drop-in train script for Google Colab with BPE.

Mirrors `train.py` but pre-sets TOKENIZER = "bpe" so the same checkpoint
format works with `evaluate.py` and `translate.py`.

Usage on Colab:
    !git clone <this-repo>
    %cd transformer
    !pip install -r requirements.txt
    !python scripts/train_bpe_colab.py --epochs 20

After training, download:
    checkpoints/best.pt
    checkpoints/latest.pt
    checkpoints/metrics.jsonl
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
config.NORM_FIRST = True  # Pre-LN: far more stable, required for convergence on this depth

# Now safe to import everything else.
from train import main  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = sys.argv[0].replace("scripts/train_bpe_colab.py", "train.py")
    raise SystemExit(main())
