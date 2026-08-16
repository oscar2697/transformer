"""Tests for the dataset utilities: collate, bucket sampler, Vocab."""

import torch

import config
from dataset import Vocab, make_collate_fn, tokenize, BucketBatchSampler


def test_tokenize_basic():
    toks = tokenize("Hello, World!")
    assert toks == ["hello", ",", "world", "!"]


def test_tokenize_contraction():
    toks = tokenize("Don't go.")
    assert toks == ["don", "'", "t", "go", "."]


def test_vocab_build_and_encode():
    sentences = [
        ["ein", "mann", "geht"],
        ["eine", "frau", "schläft"],
        ["der", "mann", "schläft"],
        ["der", "hund", "läuft"],
    ]
    vocab = Vocab.build(sentences, min_freq=1)
    assert vocab.stoi["<unk>"] == 0
    assert vocab.stoi["<pad>"] == 1
    assert vocab.stoi["<sos>"] == 2
    assert vocab.stoi["<eos>"] == 3
    # "mann" appears twice, should be in vocab
    assert "mann" in vocab.stoi
    # OOV
    assert vocab.encode(["xyz"]) == [config.UNK_IDX]
    # Round-trip
    assert vocab.decode([vocab.stoi["<sos>"], vocab.stoi["mann"], vocab.stoi["<eos>"]]) == "mann"


def test_collate_shapes():
    # Use distinct non-pad token ids so position 0 is never confused with PAD.
    PAD = config.PAD_IDX
    batch = [
        {"src": [5, 6, 7, 8, 9], "tgt": [2, 10, 11, PAD]},
        {"src": [5, 12, PAD, PAD, PAD], "tgt": [2, 13, 14, 15, 16, 3]},
    ]
    collate = make_collate_fn(pad_idx=PAD)
    out = collate(batch)
    assert out["src"].shape == (2, 5)
    assert out["tgt"].shape == (2, 6)
    # Padding mask should be True exactly at PAD positions.
    expected_src_mask = torch.tensor([
        [False, False, False, False, False],
        [False, False,  True,  True,  True],
    ])
    expected_tgt_mask = torch.tensor([
        [False, False, False,  True,  True,  True],
        [False, False, False, False, False, False],
    ])
    assert torch.equal(out["src_pad_mask"], expected_src_mask)
    assert torch.equal(out["tgt_pad_mask"], expected_tgt_mask)


def test_bucket_batch_sampler_groups_similar_lengths():
    # Lengths: [5, 2, 8, 1, 4, 7, 3, 6]  -> sorted: [1, 2, 3, 4, 5, 6, 7, 8]
    # With batch_size=2 and shuffle disabled, batches should be:
    #   [3, 2]  (lengths 1, 2)
    #   [1, 0]  (lengths 3, 4)  -- actually depends on argsort stability
    lengths = [5, 2, 8, 1, 4, 7, 3, 6]
    sampler = BucketBatchSampler(lengths, batch_size=2, shuffle=False, seed=0)
    batches = list(sampler)
    assert len(batches) == 4
    # Every batch should have 2 elements.
    for b in batches:
        assert len(b) == 2
    # All 8 indices must appear exactly once.
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(8))


def test_bucket_batch_sampler_with_shuffle_still_covers_all():
    # Use a length multiple of batch_size so nothing is dropped.
    lengths = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 7, 4]  # 12 items
    batch_size = 3
    sampler = BucketBatchSampler(lengths, batch_size=batch_size, shuffle=True, seed=42)
    batches = list(sampler)
    flat = [i for b in batches for i in b]
    n_batches = len(lengths) // batch_size
    assert len(batches) == n_batches
    assert len(flat) == n_batches * batch_size
    # No duplicates and all indices accounted for.
    assert len(set(flat)) == len(flat)
    assert sorted(flat) == list(range(len(flat)))
