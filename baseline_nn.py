"""Baseline: PyTorch's official nn.Transformer for fair comparison.

This module wraps `torch.nn.Transformer` in the same interface as our
hand-rolled `model.Transformer` so it can be trained with `train.py`,
evaluated with `evaluate.py`, and used with `translate.py` and
`visualize_attention.py` without changes — provided we set
`config.TOKENIZER = "bpe"` (BPE is needed because nn.Transformer expects
`batch_first=True` embeddings; the encoder/decoder embedding layers
handle subword IDs natively without an external Vocab object).

Differences from the hand-rolled model that matter for the comparison:

- `nn.Transformer` uses **Pre-LayerNorm** by default (PyTorch's choice),
  not the original post-LN that our hand-rolled model implements. This
  is documented as a design choice in Section 6.4 of the paper.
- `nn.Transformer` uses an optimized, fused multi-head attention
  implementation (cuDNN-accelerated on GPU, MKL/oneDNN on CPU), so any
  BLEU gap on identical hyperparameters + identical seed is attributable
  to either (a) the LN convention difference, or (b) subtle numerical
  differences (mask convention, dropout ordering, attention scaling).

Usage:
    # Training (same flags as train.py)
    python train.py --baseline-nn --epochs 8

    # Evaluation
    python evaluate.py --checkpoint checkpoints/baseline_best.pt
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

import config


class BaselineNNTransformer(nn.Module):
    """PyTorch `nn.Transformer` wrapped for the same (batch, seq) interface.

    Embedding layers (shared src/tgt embeddings are NOT used here for fair
    comparison with the hand-rolled model, which has separate embeddings).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = config.D_MODEL,
        num_heads: int = config.NUM_HEADS,
        num_layers: int = config.NUM_ENCODER_LAYERS,
        d_ff: int = config.D_FF,
        dropout: float = config.DROPOUT,
        max_seq_length: int = config.MAX_SEQ_LENGTH,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=config.PAD_IDX)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=config.PAD_IDX)
        self.pos_embed = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

        # nn.Transformer expects (S, N, E) by default; we use batch_first=True.
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN — PyTorch's default and the modern convention.
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def _add_positional(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_embed(positions))

    def encode(
        self, src: torch.Tensor, src_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        src_emb = self._add_positional(self.src_embed(src) * (self.d_model ** 0.5))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_pad_mask: Optional[torch.Tensor] = None,
        memory_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Causal mask (float -inf where masked) of shape (tgt_len, tgt_len).
        T = tgt.size(1)
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=tgt.device), diagonal=1
        )
        tgt_emb = self._add_positional(self.tgt_embed(tgt) * (self.d_model ** 0.5))
        out = self.transformer.decoder(
            tgt_emb, memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        return self.fc_out(out)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns logits (batch, tgt_len, tgt_vocab).

        Note: PyTorch >= 2.10 removed the public ``output_attentions`` API from
        ``nn.TransformerEncoder``/``Decoder``. Capturing attention weights now
        requires either ``torch.nn.modules.activation`` hooks on the internal
        ``MultiheadAttention`` modules or a custom layer subclass. We keep the
        baseline lightweight and do not extract attention here; for the
        attention analysis see the hand-rolled model in ``model.py`` and the
        figures generated against ``checkpoints/best.pt``.
        """
        memory = self.encode(src, src_pad_mask=src_pad_mask)
        return self.decode(tgt, memory, tgt_pad_mask=tgt_pad_mask, memory_pad_mask=src_pad_mask)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_pad_mask: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding. Mirrors model.Transformer.greedy_decode."""
        device = src.device
        B = src.size(0)
        memory = self.encode(src.to(device), src_pad_mask=src_pad_mask.to(device))
        ys = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            tgt_pad = ys.eq(config.PAD_IDX)
            logits = self.decode(ys, memory, tgt_pad_mask=tgt_pad, memory_pad_mask=src_pad_mask.to(device))
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, config.PAD_IDX), next_token)
            ys = torch.cat([ys, next_token], dim=1)
            finished = finished | next_token.squeeze(1).eq(eos_idx)
            if finished.all():
                break
        return ys
