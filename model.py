"""Hand-rolled Transformer (post-LN, batch_first=True).

Based on "Attention is All You Need" (Vaswani et al., 2017).
- MultiHeadAttention, PositionwiseFeedForward and PositionalEncoding are
  implemented from scratch (no nn.MultiheadAttention).
- Encoder/Decoder layers use Post-LayerNorm (the original formulation).
- All tensors inside the model use shape (batch, seq_len, d_model).
- Mask convention: True = masked out (PyTorch-style).
    * key_padding_mask: (batch, seq_k)   True at pad positions.
    * attn_mask:        (seq_q, seq_k)   True at positions to mask out.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

import config


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention, hand-rolled.

    Q, K, V are expected in (batch, seq, d_model) layout.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,        # (seq_q, seq_k) bool
        key_padding_mask: Optional[torch.Tensor] = None,  # (batch, seq_k) bool
        return_attention: bool = False,
        return_per_head: bool = False,
    ):
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split heads: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, heads, seq_q, seq_k)

        if key_padding_mask is not None:
            # (batch, seq_k) -> (batch, 1, 1, seq_k)
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), value=-1e9
            )
        if attn_mask is not None:
            # (seq_q, seq_k) -> (1, 1, seq_q, seq_k)
            scores = scores.masked_fill(
                attn_mask.unsqueeze(0).unsqueeze(0), value=-1e9
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)  # (batch, heads, seq_q, d_k)

        # Combine heads: (batch, heads, seq, d_k) -> (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        out = self.W_o(out)

        if return_attention:
            if return_per_head:
                # Keep the head dimension for per-head visualizations.
                # attn: (batch, heads, seq_q, seq_k)
                return out, attn
            # Average across heads — (batch, seq_q, seq_k) — easier to plot.
            return out, attn.mean(dim=1)
        return out


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward with ReLU."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,  # (batch, src_len) bool, True=pad
        return_attention: bool = False,
        return_per_head: bool = False,
    ):
        attn_out, attn = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            return_attention=True,
            return_per_head=return_per_head,
        )
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        if return_attention:
            return x, attn
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,               # (tgt_len, tgt_len) bool
        tgt_key_padding_mask: Optional[torch.Tensor] = None,   # (batch, tgt_len) bool
        memory_key_padding_mask: Optional[torch.Tensor] = None,  # (batch, src_len) bool
        return_attention: bool = False,
        return_per_head: bool = False,
    ):
        # Masked self-attention
        attn_out, self_attn = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            return_attention=True,
            return_per_head=return_per_head,
        )
        x = self.norm1(x + self.dropout(attn_out))
        # Cross-attention
        attn_out, cross_attn = self.cross_attn(
            x, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            return_attention=True,
            return_per_head=return_per_head,
        )
        x = self.norm2(x + self.dropout(attn_out))
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        if return_attention:
            return x, self_attn, cross_attn
        return x


class Transformer(nn.Module):
    """Hand-rolled Transformer (post-LN, batch_first=True)."""

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

        self.src_tok_emb = nn.Embedding(
            src_vocab_size, d_model, padding_idx=config.PAD_IDX
        )
        self.tgt_tok_emb = nn.Embedding(
            tgt_vocab_size, d_model, padding_idx=config.PAD_IDX
        )
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=max_seq_length, dropout=dropout
        )

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Weight tying is a common micro-optimisation (decoder embedding = output projection).
        # Disabled by default to keep the model faithful to the paper; flip TIE_EMBEDDINGS
        # in config.py to enable.
        self.fc_out = nn.Linear(d_model, tgt_vocab_size, bias=False)
        if getattr(config, "TIE_EMBEDDINGS", False):
            self.fc_out.weight = self.tgt_tok_emb.weight

        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self) -> None:
        # Embeddings use a small normal init (the scale is compensated by
        # the * sqrt(d_model) factor in forward); all other matrices keep
        # Xavier uniform. Biases and LayerNorm params remain at their
        # PyTorch defaults (0 / 1).
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if "tok_emb" in name:
                    nn.init.normal_(p, mean=0.0, std=0.02)
                else:
                    nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Returns (seq_len, seq_len) bool mask. True = mask out (future positions)."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        src: torch.Tensor,                            # (batch, src_len)
        tgt: torch.Tensor,                            # (batch, tgt_len)
        src_pad_mask: Optional[torch.Tensor] = None,  # (batch, src_len) bool, True=pad
        tgt_pad_mask: Optional[torch.Tensor] = None,  # (batch, tgt_len) bool, True=pad
        return_attention: bool = False,
        return_per_head: bool = False,
    ):
        src_emb = self.dropout(self.pos_encoder(self.src_tok_emb(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_encoder(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)))

        causal_mask = self.generate_causal_mask(tgt.size(1), src.device)

        enc_attns: List = []
        memory = src_emb
        for layer in self.encoder_layers:
            if return_attention:
                memory, attn = layer(
                    memory, key_padding_mask=src_pad_mask,
                    return_attention=True, return_per_head=return_per_head,
                )
                enc_attns.append(attn)
            else:
                memory = layer(memory, key_padding_mask=src_pad_mask)

        dec_self_attns: List = []
        dec_cross_attns: List = []
        output = tgt_emb
        for layer in self.decoder_layers:
            if return_attention:
                output, self_a, cross_a = layer(
                    output, memory,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    memory_key_padding_mask=src_pad_mask,
                    return_attention=True,
                    return_per_head=return_per_head,
                )
                dec_self_attns.append(self_a)
                dec_cross_attns.append(cross_a)
            else:
                output = layer(
                    output, memory,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    memory_key_padding_mask=src_pad_mask,
                )

        logits = self.fc_out(output)  # (batch, tgt_len, tgt_vocab_size)
        if return_attention:
            return logits, enc_attns, dec_self_attns, dec_cross_attns
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,                  # (batch, src_len)
        src_pad_mask: torch.Tensor,         # (batch, src_len) bool, True=pad
        sos_idx: int,
        eos_idx: int,
        max_len: int,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding. Returns (batch, decoded_len).

        `repetition_penalty` follows the HuggingFace convention: logits of
        already-generated tokens are divided by the penalty if positive and
        multiplied by it if negative, breaking the repetition loops that
        greedy decoding can fall into for out-of-distribution or very short
        inputs. 1.0 disables it; typical useful values are 1.1--1.3.

        `no_repeat_ngram_size` blocks any n-gram that has already appeared
        in the generated prefix (0 disables). Complements the soft
        repetition penalty with a hard constraint.
        """
        self.eval()
        batch_size, _ = src.shape
        device = src.device

        # Encode source once
        src_emb = self.dropout(
            self.pos_encoder(self.src_tok_emb(src) * math.sqrt(self.d_model))
        )
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, key_padding_mask=src_pad_mask)

        decoded = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_pad_mask = decoded.eq(config.PAD_IDX)
            causal_mask = self.generate_causal_mask(decoded.size(1), device)

            tgt_emb = self.dropout(
                self.pos_encoder(self.tgt_tok_emb(decoded) * math.sqrt(self.d_model))
            )
            output = tgt_emb
            for layer in self.decoder_layers:
                output = layer(
                    output, memory,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    memory_key_padding_mask=src_pad_mask,
                )

            logits = self.fc_out(output[:, -1, :])  # (batch, vocab)

            # Repetition penalty (HuggingFace convention) — exclude special
            # tokens (PAD/SOS/EOS/UNK) so termination and padding are not
            # suppressed.
            if repetition_penalty != 1.0:
                specials = {config.PAD_IDX, config.UNK_IDX, config.SOS_IDX, config.EOS_IDX}
                mask = torch.ones_like(decoded, dtype=torch.bool)
                for s in specials:
                    mask &= decoded.ne(s)
                score = torch.gather(logits, 1, decoded)
                penalised = torch.where(
                    score < 0,
                    score * repetition_penalty,
                    score / repetition_penalty,
                )
                score = torch.where(mask, penalised, score)
                logits.scatter_(1, decoded, score)

            # No-repeat n-gram blocking (hard constraint).
            if no_repeat_ngram_size > 0 and decoded.size(1) >= no_repeat_ngram_size - 1:
                n = no_repeat_ngram_size
                for b in range(batch_size):
                    seq = decoded[b].tolist()
                    if len(seq) < n - 1:
                        continue
                    seen = set()
                    for i in range(len(seq) - n + 1):
                        seen.add(tuple(seq[i : i + n]))
                    prefix = tuple(seq[-(n - 1) :]) if n > 1 else ()
                    for tok in range(logits.size(-1)):
                        if prefix + (tok,) in seen:
                            logits[b, tok] = -1e9

            next_token = logits.argmax(dim=-1, keepdim=True)  # (batch, 1)

            # If a sequence already produced EOS, force PAD to keep batches aligned.
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, config.PAD_IDX),
                next_token,
            )
            finished = finished | next_token.squeeze(-1).eq(eos_idx)

            decoded = torch.cat([decoded, next_token], dim=1)
            if finished.all():
                break

        return decoded

    @torch.no_grad()
    def beam_search_decode(
        self,
        src: torch.Tensor,                  # (batch, src_len)
        src_pad_mask: torch.Tensor,         # (batch, src_len) bool, True=pad
        sos_idx: int,
        eos_idx: int,
        max_len: int,
        beam_size: int = 4,
        length_penalty_alpha: float = 0.6,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        """Batched beam-search decoding. Returns (batch, decoded_len).

        For each batch element, maintains `beam_size` candidate hypotheses
        in parallel and returns the best one (selected with a Wu-style
        length penalty). beam_size = 1 reduces to greedy decoding
        behaviour.

        `length_penalty_alpha` follows Wu et al. (2016):
            lp(y) = (5 + |y|)^alpha / (5 + 1)^alpha
        Set alpha = 0 to disable length normalisation. The penalty is
        applied only at the final beam selection.

        `repetition_penalty` applies the same HuggingFace convention as in
        `greedy_decode` (1.0 disables it).

        `no_repeat_ngram_size` blocks repeated n-grams (0 disables).
        """
        self.eval()
        batch_size, src_len = src.shape
        device = src.device
        vocab_size = self.tgt_tok_emb.num_embeddings

        # Encode source once (shared across all beams of the same input).
        src_emb = self.dropout(
            self.pos_encoder(self.src_tok_emb(src) * math.sqrt(self.d_model))
        )
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, key_padding_mask=src_pad_mask)

        # Expand encoder memory and source pad mask across beam positions.
        memory_expanded = memory.unsqueeze(1).expand(
            batch_size, beam_size, src_len, self.d_model
        ).reshape(batch_size * beam_size, src_len, self.d_model)
        src_pad_mask_expanded = src_pad_mask.unsqueeze(1).expand(
            batch_size, beam_size, src_len
        ).reshape(batch_size * beam_size, src_len)

        # Initialise beam state. Only the first beam of each batch starts
        # with score 0; the others begin at -inf so the top-k selection at
        # step 1 picks beam_size distinct first tokens.
        decoded = torch.full(
            (batch_size * beam_size, 1), sos_idx, dtype=torch.long, device=device
        )
        beam_scores = torch.zeros(batch_size * beam_size, device=device)
        beam_scores.view(batch_size, beam_size)[:, 1:] = -1e9  # every beam except the first of each group
        finished = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_pad_mask = decoded.eq(config.PAD_IDX)
            causal_mask = self.generate_causal_mask(decoded.size(1), device)

            tgt_emb = self.dropout(
                self.pos_encoder(self.tgt_tok_emb(decoded) * math.sqrt(self.d_model))
            )
            output = tgt_emb
            for layer in self.decoder_layers:
                output = layer(
                    output, memory_expanded,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    memory_key_padding_mask=src_pad_mask_expanded,
                )

            logits = self.fc_out(output[:, -1, :])  # (batch * beam, vocab)

            if repetition_penalty != 1.0:
                specials = {config.PAD_IDX, config.UNK_IDX, config.SOS_IDX, config.EOS_IDX}
                mask = torch.ones_like(decoded, dtype=torch.bool)
                for s in specials:
                    mask &= decoded.ne(s)
                score = torch.gather(logits, 1, decoded)
                penalised = torch.where(
                    score < 0,
                    score * repetition_penalty,
                    score / repetition_penalty,
                )
                score = torch.where(mask, penalised, score)
                logits.scatter_(1, decoded, score)

            if no_repeat_ngram_size > 0 and decoded.size(1) >= no_repeat_ngram_size - 1:
                n = no_repeat_ngram_size
                for b in range(decoded.size(0)):
                    seq = decoded[b].tolist()
                    if len(seq) < n - 1:
                        continue
                    seen = set()
                    for i in range(len(seq) - n + 1):
                        seen.add(tuple(seq[i : i + n]))
                    prefix = tuple(seq[-(n - 1) :]) if n > 1 else ()
                    for tok in range(logits.size(-1)):
                        if prefix + (tok,) in seen:
                            logits[b, tok] = -1e9

            log_probs = torch.log_softmax(logits, dim=-1)  # (batch * beam, vocab)

            # Finished beams stay alive by emitting PAD with zero log-prob
            # contribution, so their cumulative score doesn't change.
            if finished.any():
                log_probs = log_probs.masked_fill(
                    finished.unsqueeze(-1), -1e9
                )
                log_probs[finished, config.PAD_IDX] = 0.0

            # Cumulative log-prob of every (beam, next_token) extension.
            cumulative = log_probs + beam_scores.unsqueeze(-1)
            # Reshape to (batch, beam * vocab) and pick top beam_size.
            cumulative = cumulative.view(batch_size, beam_size * vocab_size)
            top_scores, top_indices = cumulative.topk(beam_size, dim=-1)
            source_beam = top_indices // vocab_size
            next_token = top_indices % vocab_size

            # Gather the parent sequences from the selected source beams
            # and append the new tokens.
            decoded = decoded.view(batch_size, beam_size, -1)
            decoded = decoded.gather(
                1, source_beam.unsqueeze(-1).expand(-1, -1, decoded.size(-1))
            )
            decoded = torch.cat([decoded, next_token.unsqueeze(-1)], dim=-1)
            decoded = decoded.view(batch_size * beam_size, -1)

            # A beam is finished if it just emitted EOS.
            newly_finished = next_token.eq(eos_idx)  # (batch, beam)
            finished = finished.view(batch_size, beam_size).gather(
                1, source_beam
            ) | newly_finished
            finished = finished.view(batch_size * beam_size)

            beam_scores = top_scores.view(-1)

            if finished.view(batch_size, beam_size).all():
                break

        # Final selection: best beam per batch, optionally length-penalised.
        final_scores = beam_scores.view(batch_size, beam_size).clone()
        if length_penalty_alpha > 0:
            seq_lens = decoded.view(batch_size, beam_size, -1).ne(
                config.PAD_IDX
            ).sum(dim=-1).float()
            lp = ((5.0 + seq_lens) ** length_penalty_alpha) / (
                (5.0 + 1.0) ** length_penalty_alpha
            )
            final_scores = final_scores / lp
        best_beam = final_scores.argmax(dim=-1)

        decoded = decoded.view(batch_size, beam_size, -1)
        best = decoded.gather(
            1, best_beam.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, decoded.size(-1))
        ).squeeze(1)

        return best
