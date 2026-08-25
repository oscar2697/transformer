"""Generate the three remaining paper figures from the actual Tatoeba data.

Outputs (under figures/):
  vocab_distribution.png/.pdf   - BPE piece rank-frequency (Zipf) plot, EN/DE training split
  tatoeba_sample.png/.pdf       - sample source/reference pairs from the deterministic test split
  transformer_architecture.png/.pdf - schematic of the hand-rolled encoder-decoder

Run from the repo root:  python scripts/make_paper_figures.py
"""

import os
import random
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from dataset import _read_pairs, _ensure_spm  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DATA_TXT = os.path.join(config.DATA_DIR, "deu.txt")


# ---------------------------------------------------------------------------
# Deterministic split (mirrors dataset._load_data_bpe)
# ---------------------------------------------------------------------------
def raw_splits():
    src, tgt = _read_pairs(DATA_TXT)
    n_total = len(src)
    rng = random.Random(config.SEED)
    idx = list(range(n_total))
    rng.shuffle(idx)
    n_test = max(1, int(0.01 * n_total))
    n_val = max(1, int(0.01 * n_total))
    test_idx = sorted(idx[:n_test])
    train_idx = sorted(idx[n_test + n_val :])
    return src, tgt, train_idx, test_idx, n_total


# ---------------------------------------------------------------------------
# 1. Vocabulary distribution (BPE piece frequencies on the training split)
# ---------------------------------------------------------------------------
def make_vocab_distribution():
    spm = _ensure_spm()
    src_sp = spm.SentencePieceProcessor()
    src_sp.Load(os.path.join(config.SPM_MODEL_DIR, config.SPM_SRC_PREFIX + ".model"))
    tgt_sp = spm.SentencePieceProcessor()
    tgt_sp.Load(os.path.join(config.SPM_MODEL_DIR, config.SPM_TGT_PREFIX + ".model"))

    src, tgt, train_idx, _, _ = raw_splits()
    en_counter, de_counter = Counter(), Counter()
    for i in train_idx:
        en_counter.update(src_sp.EncodeAsIds(src[i]))
        de_counter.update(tgt_sp.EncodeAsIds(tgt[i]))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    for ax, counter, label, color in (
        (axes[0], en_counter, "English (source)", "#1f77b4"),
        (axes[1], de_counter, "German (target)", "#d62728"),
    ):
        freqs = sorted(counter.values(), reverse=True)
        ax.loglog(range(1, len(freqs) + 1), freqs, lw=1.2, color=color)
        ax.set_xlabel("Piece rank")
        ax.set_ylabel("Frequency in training split" if label.startswith("English") else "")
        ax.set_title(f"{label}\n({len(counter):,} pieces observed)", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(
        f"SentencePiece BPE rank-frequency distribution (vocab {config.SPM_VOCAB_SIZE} per language, seed {config.SEED})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"vocab_distribution.{ext}"), dpi=200)
    plt.close(fig)
    print("vocab_distribution done")


# ---------------------------------------------------------------------------
# 2. Sample test-set pairs
# ---------------------------------------------------------------------------
def make_tatoeba_sample(n_show=8):
    src, tgt, _, test_idx, _ = raw_splits()
    # Evenly spaced pairs across the first quarter of the test split for variety.
    pool = test_idx[: len(test_idx) // 4]
    step = max(1, len(pool) // n_show)
    chosen = pool[::step][:n_show]

    rows = [(src[i], tgt[i]) for i in chosen]

    fig, ax = plt.subplots(figsize=(9, 0.52 * n_show + 0.9))
    ax.axis("off")
    table = ax.table(
        cellText=[(en, de) for en, de in rows],
        colLabels=["English (source)", "German (reference)"],
        cellLoc="left",
        colLoc="left",
        loc="upper center",
        bbox=(0.0, 0.0, 1.0, 0.82),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0:
            cell.set_facecolor("#e8e8f2")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f6f6fa")
        if c == 0:
            cell.set_width(0.42)
        else:
            cell.set_width(0.58)
        cell.PAD = 0.03
    ax.set_title(
        f"Sample pairs from the deterministic Tatoeba EN-DE test split "
        f"(1% of the corpus, seed {config.SEED}; n = {len(test_idx):,})",
        fontsize=11,
        pad=12,
    )
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"tatoeba_sample.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("tatoeba_sample done")


# ---------------------------------------------------------------------------
# 3. Architecture schematic
# ---------------------------------------------------------------------------
def _block(ax, x, y, w, h, label, fc="#dce6f2", fs=8.5, ec="#365a8d"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=10, color="#444444", lw=1.0))


def make_architecture():
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")

    enc_x, enc_w = 0.9, 2.9
    dec_x, dec_w = 6.2, 2.9

    def stack(x, w, blocks, top_label, bottom_caption):
        """Draw a bottom-up stack; blocks = list of (label, color). Returns top y."""
        n = len(blocks)
        h = 0.72
        gap = 0.38
        y = 1.2
        centers = []
        for label, fc in blocks:
            _block(ax, x, y, w, h, label, fc=fc, fs=8)
            centers.append(y + h / 2)
            y += h + gap
        top = y - gap
        # upward arrows between consecutive blocks
        for i in range(n - 1):
            _arrow(ax, x + w / 2, centers[i] + h / 2, x + w / 2, centers[i + 1] - h / 2)
        ax.text(x + w / 2, top + 0.45, top_label, ha="center", fontsize=10, weight="bold")
        ax.text(x + w / 2, 0.55, bottom_caption, ha="center", fontsize=8.5, style="italic")
        return top

    enc_top = stack(
        enc_x, enc_w,
        [
            ("Input Embedding", "#f7e8d0"),
            ("Multi-Head\nSelf-Attention", "#dce6f2"),
            ("Add & Norm", "#f0f0f0"),
            ("Positionwise\nFFN", "#e8f0e0"),
            ("Add & Norm", "#f0f0f0"),
        ],
        f"Encoder x {config.NUM_ENCODER_LAYERS}",
        "source tokens (EN)",
    )

    dec_top = stack(
        dec_x, dec_w,
        [
            ("Output Embedding", "#f7e8d0"),
            ("Multi-Head Self-Attention\n(causal)", "#dce6f2"),
            ("Add & Norm", "#f0f0f0"),
            ("Multi-Head\nCross-Attention", "#fde8d8"),
            ("Add & Norm", "#f0f0f0"),
            ("Positionwise\nFFN", "#e8f0e0"),
            ("Add & Norm", "#f0f0f0"),
            ("Linear + Softmax", "#e0e0f0"),
        ],
        f"Decoder x {config.NUM_DECODER_LAYERS}",
        "target tokens (DE, shifted right) \u2192 next-token probabilities",
    )

    # Encoder output feeds K, V into the decoder cross-attention block.
    cross_y = 1.2 + 3 * (0.72 + 0.38) + 0.36  # center of the Cross-Attention block
    _arrow(ax, enc_x + enc_w, enc_top - 0.2, dec_x, cross_y)
    ax.text((enc_x + enc_w + dec_x) / 2, (enc_top + cross_y) / 2 + 0.25, "K, V",
            ha="center", fontsize=9, color="#365a8d", weight="bold")

    ax.set_title(
        "Hand-rolled Transformer encoder-decoder "
        f"(d_model={config.D_MODEL}, h={config.NUM_HEADS}, Pre-LN; residual connections omitted for clarity)",
        fontsize=10.5,
    )
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"transformer_architecture.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("transformer_architecture done")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    make_vocab_distribution()
    make_tatoeba_sample()
    make_architecture()
