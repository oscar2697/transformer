"""Project-wide hyperparameters and constants."""

import os

import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
DETERMINISTIC = False  # Set True to force deterministic cuDNN (slower on GPU)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model (hand-rolled Transformer, sized for CPU training)
# ---------------------------------------------------------------------------
# Compact configuration (~12 M params, ~9 h / 15 epochs on CPU for the
# Multi30k run). Tatoeba EN->DE has ~10x more pairs so we keep the same
# architecture and let it train for proportionally fewer epochs.
D_MODEL = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
D_FF = 1024
DROPOUT = 0.1
MAX_SEQ_LENGTH = 100  # Tatoeba median is 6; p99 ~= 25

# Weight tying between target embedding and output projection (Press & Wolf, 2017).
# Saves parameters; usually neutral or slightly positive for translation.
TIE_EMBEDDINGS = False

# LayerNorm placement. False = Post-LN (original Vaswani et al., 2017,
# faithful but less stable for deep stacks). True = Pre-LN (modern default:
# LayerNorm before each sublayer + final norm; much more stable, the
# baseline nn.Transformer uses this via norm_first=True). Checkpoints
# trained with Post-LN remain loadable when this flag is False.
NORM_FIRST = False

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
BETAS = (0.9, 0.98)
EPS = 1e-9
NUM_EPOCHS = 8
CLIP_GRAD = 1.0

# Noam scheduler (lr = factor * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5))
WARMUP_STEPS = 2000
LR_FACTOR = 1.0  # Multiplier on top of the Noam base schedule

# Label smoothing epsilon (0.0 = disabled; 0.1 is the usual default).
LABEL_SMOOTHING = 0.1

# Logging cadence (in optimizer steps).
LOG_INTERVAL = 100
VAL_INTERVAL = 1000  # Steps between validation runs; 0 = only at end of each epoch.

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
# Tokenization mode:
#   "word" — regex word-level + word vocab (min_freq-based; default).
#   "bpe"  — SentencePiece BPE; vocab size fixed by SPM_VOCAB_SIZE.
TOKENIZER = "bpe"

# Only used when TOKENIZER == "bpe".
SPM_VOCAB_SIZE = 8000
SPM_MODEL_DIR = os.environ.get("TRANSFORMER_SPM_DIR", "./.data/spm")
SPM_SRC_PREFIX = "spm_en"   # produces spm_en.model / spm_en.vocab
SPM_TGT_PREFIX = "spm_de"   # produces spm_de.model / spm_de.vocab
SPM_CHAR_COVERAGE = 1.0     # 0.9995 is the default; 1.0 keeps all bytes (works for DE umlauts)

# Word-level vocab parameters (only used when TOKENIZER == "word").
SRC_MIN_FREQ = 2
TGT_MIN_FREQ = 2

# Special token indices. PAD/UNK/SOS/EOS are appended in this exact order
# by `Vocab.build`, so the indices here are contractual — do not reorder.
PAD_IDX = 1
UNK_IDX = 0
SOS_IDX = 2
EOS_IDX = 3

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("TRANSFORMER_DATA_DIR", "./.data")

# ---------------------------------------------------------------------------
# Checkpointing / early stopping
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = os.environ.get("TRANSFORMER_CKPT_DIR", "./checkpoints")
PATIENCE = 10  # Epochs without val_loss improvement before stopping (0 = disabled)
BEST_CKPT_NAME = "best.pt"
LATEST_CKPT_NAME = "latest.pt"

# ---------------------------------------------------------------------------
# Inference / evaluation
# ---------------------------------------------------------------------------
MAX_DECODE_LEN = 100
BEAM_SIZE = 1  # 1 = greedy. Increase for beam search (not yet implemented).
