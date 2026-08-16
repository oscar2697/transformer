"""End-to-end smoke test: one training step + one validation step with a tiny model.

This exercises the full data -> model -> loss -> optimizer -> scheduler pipeline
without any real dataset download.
"""

import math

import torch
import torch.nn as nn

import config
from model import Transformer
from dataset import TranslationDataset, make_collate_fn, BucketBatchSampler, tokenize, Vocab
from torch.utils.data import DataLoader
from train import make_noam_scheduler, train_one_step, validate, move_batch


def _make_fake_loader(batch_size: int = 4, n: int = 8, src_vocab_size: int = 30, tgt_vocab_size: int = 40):
    # Build vocabularies from fake "sentences" (so that the indices line up).
    src_sents = [
        ["ein", "mann", "geht"],
        ["eine", "frau", "schläft", "im", "park"],
        ["der", "hund", "läuft", "schnell"],
        ["katzen", "sind", "nett"],
    ]
    tgt_sents = [
        ["a", "man", "walks"],
        ["a", "woman", "sleeps", "in", "the", "park"],
        ["the", "dog", "runs", "fast"],
        ["cats", "are", "nice"],
    ]
    src_vocab = Vocab.build(src_sents, min_freq=1)
    tgt_vocab = Vocab.build(tgt_sents, min_freq=1)

    src_ids = [src_vocab.encode(s) for s in src_sents]
    tgt_ids = [
        [tgt_vocab.stoi["<sos>"]] + tgt_vocab.encode(t) + [tgt_vocab.stoi["<eos>"]]
        for t in tgt_sents
    ]

    ds = TranslationDataset(src_ids, tgt_ids)
    loader = DataLoader(
        ds,
        batch_sampler=BucketBatchSampler(
            [len(s) for s in src_ids], batch_size, shuffle=False, seed=0
        ),
        collate_fn=make_collate_fn(),
    )
    return loader, src_vocab, tgt_vocab


def test_one_training_step():
    torch.manual_seed(0)
    loader, src_vocab, tgt_vocab = _make_fake_loader(batch_size=4, n=4)
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=32, num_heads=4, num_layers=2, d_ff=64,
        dropout=0.0, max_seq_length=20,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=config.BETAS, eps=config.EPS)
    scheduler = make_noam_scheduler(optimizer, d_model=32, warmup=4, factor=1.0)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)

    device = torch.device("cpu")
    model.to(device)

    batch = next(iter(loader))
    loss = train_one_step(model, batch, optimizer, scheduler, criterion, device, clip_grad=1.0)
    assert math.isfinite(loss)
    assert loss > 0
    # LR should have stepped.
    assert scheduler.get_last_lr()[0] > 0


def test_validation_loss_finite():
    torch.manual_seed(0)
    loader, src_vocab, tgt_vocab = _make_fake_loader(batch_size=2, n=4)
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=32, num_heads=4, num_layers=2, d_ff=64,
        dropout=0.0, max_seq_length=20,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
    device = torch.device("cpu")
    model.to(device)
    val_loss = validate(model, loader, criterion, device)
    assert math.isfinite(val_loss)
    assert val_loss > 0
