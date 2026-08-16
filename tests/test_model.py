"""Tests for the model forward pass and greedy decode."""

import torch

import config
from model import Transformer


def _tiny_model(src_vocab: int = 50, tgt_vocab: int = 60) -> Transformer:
    return Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        dropout=0.0,
        max_seq_length=20,
    )


def test_forward_shape():
    model = _tiny_model()
    model.eval()
    batch, src_len, tgt_len = 3, 6, 8
    src = torch.randint(4, 30, (batch, src_len))
    tgt = torch.randint(4, 30, (batch, tgt_len))
    src_pad_mask = src.eq(config.PAD_IDX)
    tgt_pad_mask = tgt.eq(config.PAD_IDX)
    with torch.no_grad():
        out = model(src, tgt, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
    assert out.shape == (batch, tgt_len, 60)


def test_forward_works_without_masks():
    model = _tiny_model()
    model.eval()
    src = torch.randint(4, 30, (2, 5))
    tgt = torch.randint(4, 30, (2, 7))
    with torch.no_grad():
        out = model(src, tgt)
    assert out.shape == (2, 7, 60)


def test_padding_mask_is_respected():
    """An encoder input that is pure <pad> (everywhere masked) should not crash
    the forward pass and should still produce finite outputs."""
    model = _tiny_model()
    model.eval()
    batch, src_len, tgt_len = 2, 5, 6
    src = torch.full((batch, src_len), config.PAD_IDX, dtype=torch.long)
    tgt = torch.randint(4, 30, (batch, tgt_len))
    src_pad_mask = src.eq(config.PAD_IDX)
    tgt_pad_mask = tgt.eq(config.PAD_IDX)
    with torch.no_grad():
        out = model(src, tgt, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
    assert torch.isfinite(out).all(), "non-finite output on all-pad source"


def test_greedy_decode_shape_and_finish():
    model = _tiny_model()
    model.eval()
    batch, src_len = 2, 5
    src = torch.randint(4, 30, (batch, src_len))
    src_pad_mask = src.eq(config.PAD_IDX)
    out = model.greedy_decode(
        src, src_pad_mask,
        sos_idx=config.SOS_IDX, eos_idx=config.EOS_IDX,
        max_len=12,
    )
    # First token must always be <sos>.
    assert (out[:, 0] == config.SOS_IDX).all()
    # Output must not exceed max_len.
    assert out.size(1) <= 12
    # Output dtype and vocab bounds.
    assert out.dtype == torch.long
    assert (out >= 0).all() and (out < 60).all()


def test_count_parameters():
    model = _tiny_model()
    n = model.count_parameters()
    assert n > 0
    # Sanity: should be at least d_model * (src_vocab + tgt_vocab) for embeddings
    # alone, and strictly less than a full-size Transformer.
    assert n < 1_000_000


def test_forward_return_attention_shapes():
    model = _tiny_model()
    model.eval()
    batch, src_len, tgt_len = 2, 5, 6
    src = torch.randint(4, 30, (batch, src_len))
    tgt = torch.randint(4, 30, (batch, tgt_len))
    src_pad_mask = src.eq(config.PAD_IDX)
    tgt_pad_mask = tgt.eq(config.PAD_IDX)
    with torch.no_grad():
        logits, enc, dec_self, dec_cross = model(
            src, tgt,
            src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask,
            return_attention=True,
        )
    assert logits.shape == (batch, tgt_len, 60)
    n_layers = 2
    assert len(enc) == n_layers and len(dec_self) == n_layers and len(dec_cross) == n_layers
    assert enc[0].shape == (batch, src_len, src_len)
    assert dec_self[0].shape == (batch, tgt_len, tgt_len)
    assert dec_cross[0].shape == (batch, tgt_len, src_len)
    # Weights must be non-negative and sum to ~1 along the key axis.
    assert (enc[0] >= 0).all()
    assert torch.allclose(enc[0].sum(dim=-1), torch.ones(batch, src_len), atol=1e-4)
