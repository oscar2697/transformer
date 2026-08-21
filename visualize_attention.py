"""Attention-map visualizations for the paper.

Renders (default — averaged over heads):
  - figures/attn_grid_layer{N}.{pdf,png}             (Nx3 grid: enc-self / dec-self / dec-cross)
  - figures/attn_sentence{K}_layer{N}.pdf            (3-panel figure per sentence)

Renders additionally when --per-head is set:
  - figures/per_head/attn_grid_layer{N}_per_head.pdf (Nx(3*num_heads) grid, one col per head)
  - figures/per_head/attn_grid_layer{N}_per_head.png
  - figures/per_head/attn_sentence{K}_layer{N}_per_head.pdf

Usage:
    python visualize_attention.py
    python visualize_attention.py --num-sentences 6 --layer -1
    python visualize_attention.py --per-head --num-sentences 4 --layer -1
"""
from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config
from dataset import load_data_dispatched as load_data, tokenize
from translate import build_model_from_checkpoint


# Same rcParams style as visualize.py for visual consistency.
_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
for k, v in _RC.items():
    matplotlib.rcParams[k] = v


def _truncate_pad(tokens: List[str], pad_idx: int) -> List[str]:
    """Strip trailing <pad> tokens from a token list (if any leaked through)."""
    # We can't recover the pad string from indices; rely on the encoder lengths
    # to chop the visualization cleanly.
    return tokens


def _draw_heatmap(ax, weights: torch.Tensor, row_labels: List[str], col_labels: List[str], title: str):
    """Draw a single attention heatmap with token labels on both axes."""
    im = ax.imshow(weights.cpu().numpy(), aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    # Colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _draw_heatmap_no_labels(ax, weights: torch.Tensor, title: str):
    """Compact heatmap without token tick labels (used for per-head grids)."""
    im = ax.imshow(weights.cpu().numpy(), aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _grid_for_sentence(
    src_tokens: List[str],
    enc_attn: torch.Tensor,    # (src_len, src_len)
    dec_self: torch.Tensor,    # (tgt_len, tgt_len)
    dec_cross: torch.Tensor,   # (tgt_len, src_len)
    tgt_tokens: List[str],
    layer: int,
    out_path: str,
) -> None:
    """3-panel figure: encoder self-attn, decoder self-attn, decoder cross-attn for one sentence."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    _draw_heatmap(axes[0], enc_attn, src_tokens, src_tokens, f"Encoder self-attn (L{layer+1})")
    _draw_heatmap(axes[1], dec_self, tgt_tokens, tgt_tokens, f"Decoder self-attn (L{layer+1})")
    _draw_heatmap(axes[2], dec_cross, tgt_tokens, src_tokens, f"Decoder cross-attn (L{layer+1})")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _attn_grid(
    sentences_src: List[List[str]],
    sentences_tgt: List[List[str]],
    enc_for_layer: List[torch.Tensor],   # list of per-sentence (src_len, src_len) tensors
    self_for_layer: List[torch.Tensor],  # list of per-sentence (tgt_len, tgt_len) tensors
    cross_for_layer: List[torch.Tensor], # list of per-sentence (tgt_len, src_len) tensors
    out_path: str,
) -> None:
    """Render a 4x3 grid: rows = sentences, cols = encoder self / decoder self / decoder cross."""
    n = len(sentences_src)
    fig, axes = plt.subplots(n, 3, figsize=(13, 2.6 * n))
    if n == 1:
        axes = [axes]
    for i in range(n):
        _draw_heatmap(
            axes[i][0],
            enc_for_layer[i][: len(sentences_src[i]), : len(sentences_src[i])],
            sentences_src[i], sentences_src[i],
            f"S{i+1}: encoder self-attn",
        )
        _draw_heatmap(
            axes[i][1],
            self_for_layer[i][: len(sentences_tgt[i]), : len(sentences_tgt[i])],
            sentences_tgt[i], sentences_tgt[i],
            f"S{i+1}: decoder self-attn",
        )
        _draw_heatmap(
            axes[i][2],
            cross_for_layer[i][: len(sentences_tgt[i]), : len(sentences_src[i])],
            sentences_tgt[i], sentences_src[i],
            f"S{i+1}: decoder cross-attn",
        )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Visualize Transformer attention maps")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data-dir", default=config.DATA_DIR)
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--num-sentences", type=int, default=4,
                   help="How many test sentences to visualize (1-8 keep the grid readable)")
    p.add_argument("--layer", type=int, default=-1,
                   help="Which layer to show in the grid (0-indexed; -1 = last)")
    p.add_argument("--per-head", action="store_true",
                   help="Render one column per attention head (Nx(3*num_heads) grid).")
    p.add_argument("--head", type=int, default=None,
                   help="When set with --per-head, render only this single head (0-indexed).")
    p.add_argument("--min-len", type=int, default=2,
                   help="Skip test sentences with fewer than this many non-special source tokens. "
                        "Increase to focus attention figures on structurally richer sentences.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading vocabs ...")
    _, _, test_loader, src_vocab, tgt_vocab = load_data(batch_size=1, data_dir=args.data_dir)
    print(f"  src vocab: {len(src_vocab)}, tgt vocab: {len(tgt_vocab)}")

    print(f"Loading checkpoint {args.checkpoint} ...")
    model = build_model_from_checkpoint(args.checkpoint, src_vocab, tgt_vocab, device)
    model.eval()

    # Early-out for the baseline: PyTorch's nn.Transformer does not expose
    # attention weights through a public API in PyTorch >= 2.10 (the
    # `output_attentions` flag was removed). We keep this script focused on
    # the hand-rolled model, which is the one we use for the §5 attention
    # analysis anyway. The baseline row of Table 5 is supported by its
    # BLEU/chrF2 numbers; attention-viz comparison is left as future work.
    if hasattr(model, "transformer") and not hasattr(model, "encoder_layers"):
        print(
            "This checkpoint uses the BaselineNNTransformer (nn.Transformer) "
            "wrapper, which does not expose attention weights in PyTorch >= 2.10.\n"
            "Skipping attention visualization. To visualize attention, run "
            "against the hand-rolled checkpoint (checkpoints/best.pt) instead."
        )
        return 0

    # Pull a few short-ish test examples so the heatmaps stay readable.
    examples = []
    seen_lengths = []
    for batch in test_loader:
        src = batch["src"][0]   # (src_len,)
        if src.eq(config.PAD_IDX).all():
            continue
        # Decode tokens back to strings for labels.
        src_ids = src.tolist()
        src_tokens = [t for t in src_vocab.decode(src_ids, skip_specials=False).split(" ") if t]
        # Filter by minimum content length (excluding specials) so that attention
        # has something to attend over. Default 2 keeps the original behaviour;
        # raise to 6-8 for sentence-level head specialisation to emerge.
        n_content = sum(1 for tid in src_ids if tid not in (config.PAD_IDX, config.SOS_IDX, config.EOS_IDX, config.UNK_IDX))
        if n_content < args.min_len:
            continue
        if len(examples) >= args.num_sentences:
            break
        # Reference translation (target), stripped of specials.
        tgt = batch["tgt"][0]
        tgt_ids = tgt.tolist()
        tgt_tokens = [t for t in tgt_vocab.decode(tgt_ids, skip_specials=False).split(" ") if t]
        examples.append((src, tgt, src_tokens, tgt_tokens, batch["src_pad_mask"][0]))
        seen_lengths.append(n_content)

    print(f"Collected {len(examples)} sentences (min content tokens = {args.min_len}).")
    if seen_lengths:
        print(f"  content-token lengths: {seen_lengths}")

    # Run a forward pass per sentence so we can capture attention weights.
    # Store as list-of-lists keyed by layer so each sentence keeps its own shape;
    # the plotting helpers slice with [:len(src_t)] / [:len(tgt_t)].
    # When --per-head is set, each stored tensor has shape (heads, seq, seq)
    # instead of (seq, seq).
    enc_per_layer: List[List[torch.Tensor]] = []
    self_per_layer: List[List[torch.Tensor]] = []
    cross_per_layer: List[List[torch.Tensor]] = []

    with torch.no_grad():
        for src, tgt, _, _, src_pad_mask in examples:
            # Build a decoder input (teacher-forced from the gold target minus last).
            tgt_input = tgt[:-1].unsqueeze(0).to(device)
            tgt_pad_mask = tgt_input.eq(config.PAD_IDX)
            _, enc_a, self_a, cross_a = model(
                src.unsqueeze(0).to(device),
                tgt_input,
                src_pad_mask=src_pad_mask.unsqueeze(0).to(device),
                tgt_pad_mask=tgt_pad_mask,
                return_attention=True,
                return_per_head=args.per_head,
            )
            if not enc_per_layer:
                # enc_a[j]: (1, heads, src, src) if per-head else (1, src, src)
                enc_per_layer = [[a[0].cpu().squeeze(0)] for a in enc_a]
                self_per_layer = [[a[0].cpu().squeeze(0)] for a in self_a]
                cross_per_layer = [[a[0].cpu().squeeze(0)] for a in cross_a]
            else:
                for j, a in enumerate(enc_a):
                    enc_per_layer[j].append(a[0].cpu().squeeze(0))
                for j, a in enumerate(self_a):
                    self_per_layer[j].append(a[0].cpu().squeeze(0))
                for j, a in enumerate(cross_a):
                    cross_per_layer[j].append(a[0].cpu().squeeze(0))

    os.makedirs(args.out_dir, exist_ok=True)

    # Per-layer, per-type average heatmap (over all sentences).
    n_layers = len(enc_per_layer)
    layer = args.layer if args.layer >= 0 else n_layers + args.layer
    print(f"Rendering layer {layer+1} of {n_layers}.")

    # When --per-head is set, the stored tensors have shape (heads, seq, seq).
    # The averaged-head path below still needs 2-D inputs, so project back to
    # the head-mean representation for that path only.
    if args.per_head:
        enc_avg = [t.mean(dim=0) for t in enc_per_layer[layer]]
        self_avg = [t.mean(dim=0) for t in self_per_layer[layer]]
        cross_avg = [t.mean(dim=0) for t in cross_per_layer[layer]]
    else:
        enc_avg = enc_per_layer[layer]
        self_avg = self_per_layer[layer]
        cross_avg = cross_per_layer[layer]

    # Single-figure grid for the chosen layer.
    src_tokens_list = [ex[2] for ex in examples]
    tgt_tokens_list = [ex[3] for ex in examples]
    _attn_grid(
        src_tokens_list, tgt_tokens_list,
        enc_avg, self_avg, cross_avg,
        os.path.join(args.out_dir, f"attn_grid_layer{layer+1}.pdf"),
    )
    _attn_grid(
        src_tokens_list, tgt_tokens_list,
        enc_avg, self_avg, cross_avg,
        os.path.join(args.out_dir, f"attn_grid_layer{layer+1}.png"),
    )

    # One figure per sentence for the chosen layer (cleaner for paper inclusion).
    for i, (_, _, src_t, tgt_t, _) in enumerate(examples):
        _grid_for_sentence(
            src_t,
            enc_avg[i][: len(src_t), : len(src_t)],
            self_avg[i][: len(tgt_t), : len(tgt_t)],
            cross_avg[i][: len(tgt_t), : len(src_t)],
            tgt_t,
            layer,
            os.path.join(args.out_dir, f"attn_sentence{i+1}_layer{layer+1}.pdf"),
        )

    # ------------------------------------------------------------------
    # Per-head renderings (optional)
    # ------------------------------------------------------------------
    if args.per_head:
        # Each tensor in enc_per_layer[layer] has shape (heads, seq, seq).
        # Figure out how many heads the model actually used.
        n_heads = enc_per_layer[layer][0].shape[0]
        if args.head is not None:
            head_idxs = [args.head]
            suffix = f"_head{args.head+1}"
        else:
            head_idxs = list(range(n_heads))
            suffix = "_per_head"
        out_root = os.path.join(args.out_dir, "per_head")
        os.makedirs(out_root, exist_ok=True)

        n = len(examples)
        ncols = 3 * len(head_idxs)
        fig, axes = plt.subplots(
            n, ncols,
            figsize=(2.2 * ncols, 2.4 * n),
            gridspec_kw={"wspace": 0.3, "hspace": 0.4},
        )
        # Defensive: ensure axes is 2-D even for n == 1.
        if n == 1:
            axes = axes.reshape(1, -1)
        for i, (_, _, src_t, tgt_t, _) in enumerate(examples):
            enc_t = enc_per_layer[layer][i]   # (heads, src, src)
            self_t = self_per_layer[layer][i] # (heads, tgt, tgt)
            cross_t = cross_per_layer[layer][i]  # (heads, tgt, src)
            for k, h in enumerate(head_idxs):
                _draw_heatmap_no_labels(
                    axes[i][k],
                    enc_t[h, : len(src_t), : len(src_t)],
                    f"S{i+1} enc L{layer+1} H{h+1}",
                )
                _draw_heatmap_no_labels(
                    axes[i][len(head_idxs) + k],
                    self_t[h, : len(tgt_t), : len(tgt_t)],
                    f"S{i+1} self L{layer+1} H{h+1}",
                )
                _draw_heatmap_no_labels(
                    axes[i][2 * len(head_idxs) + k],
                    cross_t[h, : len(tgt_t), : len(src_t)],
                    f"S{i+1} cross L{layer+1} H{h+1}",
                )
        for ext in ("pdf", "png"):
            out = os.path.join(out_root, f"attn_grid_layer{layer+1}{suffix}.{ext}")
            fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_root}/attn_grid_layer{layer+1}{suffix}.{{pdf,png}}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
