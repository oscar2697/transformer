"""Tests for the mask utilities."""

import torch

from model import Transformer


def test_causal_mask_shape():
    seq_len = 7
    mask = Transformer.generate_causal_mask(seq_len, torch.device("cpu"))
    assert mask.shape == (seq_len, seq_len)
    assert mask.dtype == torch.bool


def test_causal_mask_values():
    seq_len = 4
    mask = Transformer.generate_causal_mask(seq_len, torch.device("cpu"))
    expected = torch.tensor(
        [
            [False, True,  True,  True],
            [False, False, True,  True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    assert torch.equal(mask, expected), f"\n{mask}\n!=\n{expected}"


def test_causal_mask_no_self_attend_to_future():
    """Row i must not be able to attend to any column j > i."""
    seq_len = 5
    mask = Transformer.generate_causal_mask(seq_len, torch.device("cpu"))
    for i in range(seq_len):
        assert mask[i, i + 1 :].all(), f"row {i} allows future attention"
        assert not mask[i, : i + 1].any(), f"row {i} blocks past/present attention"
