"""Training entry point for the hand-rolled Transformer on Tatoeba (en -> de).

Usage:
    python train.py
    python train.py --epochs 8 --batch-size 32 --label-smoothing 0.1
    python train.py --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from dataset import load_data_dispatched as load_data
from model import Transformer
from utils import AverageMeter, format_seconds, set_seed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer on Tatoeba (en->de)")
    # Data
    p.add_argument("--data-dir", default=config.DATA_DIR)
    p.add_argument("--checkpoint-dir", default=config.CHECKPOINT_DIR)
    # Model
    p.add_argument("--d-model", type=int, default=config.D_MODEL)
    p.add_argument("--num-heads", type=int, default=config.NUM_HEADS)
    p.add_argument("--num-layers", type=int, default=config.NUM_ENCODER_LAYERS)
    p.add_argument("--d-ff", type=int, default=config.D_FF)
    p.add_argument("--dropout", type=float, default=config.DROPOUT)
    p.add_argument("--max-seq-length", type=int, default=config.MAX_SEQ_LENGTH)
    # Training
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr-factor", type=float, default=config.LR_FACTOR)
    p.add_argument("--warmup", type=int, default=config.WARMUP_STEPS)
    p.add_argument("--label-smoothing", type=float, default=config.LABEL_SMOOTHING)
    p.add_argument("--clip-grad", type=float, default=config.CLIP_GRAD)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--log-interval", type=int, default=config.LOG_INTERVAL)
    p.add_argument("--val-interval", type=int, default=config.VAL_INTERVAL)
    # Misc
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--deterministic", action="store_true", default=config.DETERMINISTIC)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--baseline-nn", action="store_true",
                   help="Use PyTorch's nn.Transformer (Opción B baseline) instead of the hand-rolled model.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Noam scheduler
# ---------------------------------------------------------------------------
def make_noam_scheduler(
    optimizer: torch.optim.Optimizer,
    d_model: int,
    warmup: int,
    factor: float = 1.0,
):
    """lr(step) = factor * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)."""
    d_model = float(d_model)

    def lr_lambda(step: int) -> float:
        step = max(int(step), 1)
        return factor * (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def _config_snapshot(args: argparse.Namespace) -> Dict:
    """Pickled config snapshot for reproducibility of a checkpoint."""
    return {
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "lr_factor": args.lr_factor,
        "warmup": args.warmup,
        "label_smoothing": args.label_smoothing,
        "clip_grad": args.clip_grad,
        "seed": args.seed,
        "model_type": "baseline_nn" if args.baseline_nn else "hand_rolled",
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    val_loss: float,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "config": _config_snapshot(args),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


# ---------------------------------------------------------------------------
# Training / validation step
# ---------------------------------------------------------------------------
def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_one_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    clip_grad: float,
) -> float:
    model.train()
    batch = move_batch(batch, device)

    # Teacher forcing: decoder input is tgt[:, :-1], target is tgt[:, 1:].
    tgt_input = batch["tgt"][:, :-1]
    tgt_output = batch["tgt"][:, 1:]
    src_pad_mask = batch["src_pad_mask"]
    tgt_input_pad_mask = batch["tgt_pad_mask"][:, :-1]

    logits = model(
        batch["src"], tgt_input,
        src_pad_mask=src_pad_mask,
        tgt_pad_mask=tgt_input_pad_mask,
    )
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        tgt_output.reshape(-1),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    scheduler.step()

    return float(loss.item())


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        tgt_input = batch["tgt"][:, :-1]
        tgt_output = batch["tgt"][:, 1:]
        logits = model(
            batch["src"], tgt_input,
            src_pad_mask=batch["src_pad_mask"],
            tgt_pad_mask=batch["tgt_pad_mask"][:, :-1],
        )
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
        )
        n = tgt_output.numel()
        total_loss += float(loss.item()) * n
        total_count += n
    return total_loss / max(total_count, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, "metrics.jsonl")
    log_f = open(log_path, "a", encoding="utf-8")

    # Data
    train_loader, val_loader, _, src_vocab, tgt_vocab = load_data(
        batch_size=args.batch_size, data_dir=args.data_dir, seed=args.seed
    )

    # Model
    if args.baseline_nn:
        from baseline_nn import BaselineNNTransformer
        ModelCls = BaselineNNTransformer
    else:
        ModelCls = Transformer
    model = ModelCls(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
    ).to(device)
    print(f"Model parameters ({'baseline nn.Transformer' if args.baseline_nn else 'hand-rolled'}): {model.count_parameters():,}")

    # Optimizer (base lr is 1.0; the Noam scheduler returns the absolute lr).
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=config.BETAS, eps=config.EPS
    )
    scheduler = make_noam_scheduler(
        optimizer, d_model=args.d_model, warmup=args.warmup, factor=args.lr_factor
    )

    # Loss
    criterion = nn.CrossEntropyLoss(
        ignore_index=config.PAD_IDX,
        label_smoothing=args.label_smoothing,
    )

    # Resume
    start_epoch = 0
    best_val_loss = math.inf
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", math.inf)
        print(f"  -> epoch {start_epoch}, best_val_loss {best_val_loss:.4f}")

    # Train
    global_step = 0
    patience_counter = 0
    train_loss_meter = AverageMeter()
    overall_start = time.time()
    best_ckpt = os.path.join(args.checkpoint_dir, config.BEST_CKPT_NAME)
    latest_ckpt = os.path.join(args.checkpoint_dir, config.LATEST_CKPT_NAME)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_loss_meter.reset()
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        for batch in progress:
            loss = train_one_step(
                model, batch, optimizer, scheduler,
                criterion, device, args.clip_grad,
            )
            train_loss_meter.update(loss, n=batch["tgt"].size(0))
            global_step += 1

            if global_step % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                progress.set_postfix(
                    loss=f"{train_loss_meter.avg:.4f}",
                    ppl=f"{math.exp(min(train_loss_meter.avg, 20)):.2f}",
                    lr=f"{lr:.2e}",
                )
                log_f.write(json.dumps({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "split": "train",
                    "loss": train_loss_meter.avg,
                    "ppl": math.exp(min(train_loss_meter.avg, 20)),
                    "lr": lr,
                }) + "\n")
                log_f.flush()

            if args.val_interval > 0 and global_step % args.val_interval == 0:
                val_loss = validate(model, val_loader, criterion, device)
                log_f.write(json.dumps({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "split": "val",
                    "loss": val_loss,
                    "ppl": math.exp(min(val_loss, 20)),
                }) + "\n")
                log_f.flush()
                # Back to train mode
                model.train()

        # End-of-epoch validation
        val_loss = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        epoch_ppl = math.exp(min(val_loss, 20))
        elapsed = time.time() - overall_start

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                best_ckpt, model, optimizer, scheduler,
                epoch=epoch, global_step=global_step,
                val_loss=val_loss, best_val_loss=best_val_loss, args=args,
            )
        else:
            patience_counter += 1

        save_checkpoint(
            latest_ckpt, model, optimizer, scheduler,
            epoch=epoch, global_step=global_step,
            val_loss=val_loss, best_val_loss=best_val_loss, args=args,
        )

        log_f.write(json.dumps({
            "step": global_step,
            "epoch": epoch + 1,
            "split": "epoch",
            "train_loss": train_loss_meter.avg,
            "val_loss": val_loss,
            "val_ppl": epoch_ppl,
            "epoch_time": epoch_time,
            "elapsed": elapsed,
            "improved": improved,
        }) + "\n")
        log_f.flush()

        print(
            f"Epoch {epoch+1:>3} | "
            f"train_loss {train_loss_meter.avg:.4f} | "
            f"val_loss {val_loss:.4f} | val_ppl {epoch_ppl:.2f} | "
            f"best {best_val_loss:.4f} | "
            f"{format_seconds(epoch_time)} (elapsed {format_seconds(elapsed)})"
        )

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    log_f.close()
    print(f"Best val_loss: {best_val_loss:.4f}. Done in {format_seconds(time.time() - overall_start)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
