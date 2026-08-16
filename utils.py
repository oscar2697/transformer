"""Shared helpers: seeding, metrics, formatting."""

from __future__ import annotations

import random
import time
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG we can find.

    Setting `cudnn.deterministic` slows training noticeably on GPU; off by default.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Running average for any scalar (loss, accuracy, ...)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class Timer:
    """Context manager that records elapsed wall-clock time."""

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.time() - self.start

    def __str__(self) -> str:
        return f"{self.elapsed:.1f}s"


def format_seconds(seconds: float) -> str:
    """Format seconds as `Hh Mm Ss` or `Mm Ss` or `Ss`."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def count_tokens(token_lists: Iterable[list[int]]) -> int:
    """Sum of lengths — handy for token/s throughput metrics."""
    return sum(len(t) for t in token_lists)
