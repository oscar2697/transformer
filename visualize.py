"""Genera figuras para el paper: training/val loss y ppl curves.

Usage:
    python visualize.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
for k, v in CONFIG.items():
    matplotlib.rcParams[k] = v


def load_metrics(path: str):
    steps, train_loss, val_loss, val_ppl = [], [], [], []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("split") == "epoch":
                steps.append(obj["step"])
                train_loss.append(obj.get("train_loss"))
                val_loss.append(obj.get("val_loss"))
                val_ppl.append(obj.get("val_ppl"))
    return steps, train_loss, val_loss, val_ppl


def plot_loss(steps, train_loss, val_loss, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, train_loss, "o-", color="#1f77b4", label="Train loss", linewidth=1.5, markersize=4)
    ax.plot(steps, val_loss, "s--", color="#d62728", label="Val loss", linewidth=1.5, markersize=4)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(framealpha=0.4)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_ppl(steps, val_ppl, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, val_ppl, "s--", color="#2ca02c", linewidth=1.5, markersize=4)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Validation perplexity")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_combined(steps, train_loss, val_loss, val_ppl, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

    axes[0].plot(steps, train_loss, "o-", color="#1f77b4", label="Train", linewidth=1.5, markersize=4)
    axes[0].plot(steps, val_loss, "s--", color="#d62728", label="Val", linewidth=1.5, markersize=4)
    axes[0].set_xlabel("Optimization step")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(framealpha=0.4)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    axes[1].plot(steps, val_ppl, "s--", color="#2ca02c", linewidth=1.5, markersize=4)
    axes[1].set_xlabel("Optimization step")
    axes[1].set_ylabel("Validation perplexity")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    metrics_path = "checkpoints/metrics.jsonl"
    fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)

    steps, train_loss, val_loss, val_ppl = load_metrics(metrics_path)

    plot_loss(steps, train_loss, val_loss, f"{fig_dir}/loss_curve.pdf")
    plot_ppl(steps, val_ppl, f"{fig_dir}/ppl_curve.pdf")
    plot_combined(steps, train_loss, val_loss, val_ppl, f"{fig_dir}/training_curves.pdf")

    # También guardar PNG para LinkedIn / Medium
    plot_combined(steps, train_loss, val_loss, val_ppl, f"{fig_dir}/training_curves.png")

    print(f"\nEpoch summary:")
    for i, (sl, vl, vp) in enumerate(zip(train_loss, val_loss, val_ppl)):
        print(f"  Epoch {i+1:2d}: train_loss={sl:.4f}  val_loss={vl:.4f}  val_ppl={vp:.2f}")


if __name__ == "__main__":
    main()
