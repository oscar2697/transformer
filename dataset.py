"""Tatoeba (en -> de) data pipeline.

Downloads the Tatoeba EN-DE sentence pair collection from
https://www.manythings.org/anki/ (CC-BY licensed), tokenizes with a regex-based
word-level tokenizer, builds per-side vocabularies, and returns length-bucketed
DataLoaders with dynamic padding.

The Tatoeba collection has ~330k sentence pairs. We split it deterministically
into train/val/test (98/1/1) since Tatoeba ships a single file.

No torchtext dependency (torchtext 0.6 is dead; 0.15+ is heavy).
"""

from __future__ import annotations

import os
import random
import re
import tempfile
import urllib.request
import zipfile
from collections import Counter
from io import BytesIO
from typing import Dict, List, Sequence, Tuple

import requests
import torch
from torch.utils.data import DataLoader, Dataset

import config


# ---------------------------------------------------------------------------
# Tokenization (regex-based, language-agnostic)
# ---------------------------------------------------------------------------
# Splits on word characters and individual punctuation marks. Handles
# contractions ("don't" -> "don", "'", "t") and lowercases everything.
_TOKEN_RE = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lowercase, then split on word runs and individual punctuation marks."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]


class Vocab:
    """Minimal word-level vocabulary.

    Indices of special tokens are contractual — see `config.SPECIAL_*` — and
    MUST match the order in `SPECIAL_TOKENS` (UNK=0, PAD=1, SOS=2, EOS=3).
    """

    def __init__(self, stoi: Dict[str, int], itos: List[str]) -> None:
        self.stoi = stoi
        self.itos = itos

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
        special_ids = {self.stoi[s] for s in SPECIAL_TOKENS}
        out: List[str] = []
        for i in ids:
            if skip_specials and i in special_ids:
                continue
            out.append(self.itos[i])
        return " ".join(out)

    @classmethod
    def build(
        cls, sentences: Sequence[Sequence[str]], min_freq: int
    ) -> "Vocab":
        counter: Counter = Counter()
        for sent in sentences:
            counter.update(sent)

        # Specials first, in the canonical order, so PAD/UNK/SOS/EOS get their
        # hard-coded indices. The Vocab assumes a one-to-one mapping with
        # config.{UNK,PAD,SOS,EOS}_IDX — never reorder SPECIAL_TOKENS.
        stoi: Dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        for tok, cnt in counter.most_common():
            if cnt >= min_freq and tok not in stoi:
                stoi[tok] = len(stoi)

        itos = [""] * len(stoi)
        for tok, idx in stoi.items():
            itos[idx] = tok
        return cls(stoi, itos)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
_TATOEBA_URL = "http://www.manythings.org/anki/deu-eng.zip"
_TATOEBA_ZIP_NAME = "deu-eng.zip"
_TATOEBA_TXT_NAME = "deu.txt"
_TATOEBA_MAX_BYTES = 60 * 1024 * 1024  # 60 MB safety cap; actual file is ~12 MB


def _download_file(url: str, target: str, chunk: int = 8192) -> None:
    """Stream-download with a 60 MB cap and a permissive User-Agent header.

    The www.manythings.org server occasionally returns HTTP 406 to default
    Python user agents; setting an explicit browser UA makes it reliable.
    """
    print(f"  downloading {os.path.basename(target)} ...", end=" ", flush=True)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, stream=True, timeout=60, headers=headers)
    response.raise_for_status()
    with open(target, "wb") as f:
        n = 0
        for piece in response.iter_content(chunk_size=chunk):
            if piece:
                n += len(piece)
                if n > _TATOEBA_MAX_BYTES:
                    raise RuntimeError(
                        f"Download exceeded {_TATOEBA_MAX_BYTES // (1024*1024)} MB safety cap"
                    )
                f.write(piece)
    print("done")


def download_tatoeba(data_dir: str = config.DATA_DIR) -> str:
    """Downloads and extracts Tatoeba EN-DE to `data_dir`. Idempotent.

    Returns the absolute path to the extracted `deu.txt` file.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, _TATOEBA_ZIP_NAME)
    txt_path = os.path.join(data_dir, _TATOEBA_TXT_NAME)

    if not os.path.exists(txt_path):
        if not os.path.exists(zip_path):
            _download_file(_TATOEBA_URL, zip_path)
        # Extract just the .txt (zipfile also includes a small _about.txt).
        with zipfile.ZipFile(zip_path) as z:
            for member in z.namelist():
                if member == _TATOEBA_TXT_NAME:
                    with z.open(member) as f_in, open(txt_path, "wb") as f_out:
                        f_out.write(f_in.read())
                    break
    return txt_path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# Tatoeba lines look like:  "EN\tDE\tCC-BY 2.0 (France) Attribution: tatoeba.org #..."
# We discard the attribution column. Some rows have empty cells; skip them.
def _read_pairs(path: str) -> Tuple[List[str], List[str]]:
    src, tgt = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            en, de = parts[0].strip(), parts[1].strip()
            if not en or not de:
                continue
            # Drop pairs containing replacement characters (U+FFFD) which
            # appear in rows with mixed latin-1 / UTF-8 encoding in the
            # raw Tatoeba file.
            if "�" in en or "�" in de:
                continue
            src.append(en)   # English (source)
            tgt.append(de)   # German  (target)
    return src, tgt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TranslationDataset(Dataset):
    """Pre-tokenized, integer-encoded translation pairs.

    Tokens are stored as Python lists of ints; the collate function takes
    care of padding to a per-batch max length.
    """

    def __init__(
        self,
        src_ids: Sequence[Sequence[int]],
        tgt_ids: Sequence[Sequence[int]],
    ) -> None:
        assert len(src_ids) == len(tgt_ids), "src/tgt length mismatch"
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return {"src": list(self.src_ids[idx]), "tgt": list(self.tgt_ids[idx])}


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def make_collate_fn(pad_idx: int = config.PAD_IDX):
    def collate(batch: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        src_lens = [len(item["src"]) for item in batch]
        tgt_lens = [len(item["tgt"]) for item in batch]
        max_src = max(src_lens)
        max_tgt = max(tgt_lens)

        src = torch.full((len(batch), max_src), pad_idx, dtype=torch.long)
        tgt = torch.full((len(batch), max_tgt), pad_idx, dtype=torch.long)
        for i, item in enumerate(batch):
            src[i, : len(item["src"])] = torch.tensor(item["src"], dtype=torch.long)
            tgt[i, : len(item["tgt"])] = torch.tensor(item["tgt"], dtype=torch.long)

        return {
            "src": src,
            "tgt": tgt,
            "src_pad_mask": src.eq(pad_idx),
            "tgt_pad_mask": tgt.eq(pad_idx),
            "src_lens": torch.tensor(src_lens),
            "tgt_lens": torch.tensor(tgt_lens),
        }

    return collate


# ---------------------------------------------------------------------------
# Length-bucketed batch sampler
# ---------------------------------------------------------------------------
class BucketBatchSampler:
    """Length-bucketed batch sampler.

    Sorts all example indices by source length, then groups them into
    batches of `batch_size`. The order of batches is shuffled each epoch so
    we still see variety; examples inside a batch share similar length,
    so padding overhead is low.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.lengths)
        order = sorted(range(n), key=lambda i: self.lengths[i])
        n_batches = n // self.batch_size
        usable = n_batches * self.batch_size
        order = order[:usable]
        order = [order[i : i + self.batch_size] for i in range(0, usable, self.batch_size)]
        if self.shuffle:
            self._rng.shuffle(order)
        for batch in order:
            yield batch


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def load_data(
    batch_size: int = config.BATCH_SIZE,
    data_dir: str = config.DATA_DIR,
    seed: int = config.SEED,
    max_seq_length: int = config.MAX_SEQ_LENGTH,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocab, Vocab]:
    """Downloads (if needed), tokenizes, builds vocab and returns loaders.

    Splits the Tatoeba file deterministically into train/val/test. Pairs whose
    source OR target exceeds `max_seq_length` tokens (counted after `<sos>`/
    `<eos>` insertion on the target) are dropped to keep batches within the
    model's positional budget.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
        where src_vocab is English (source side) and tgt_vocab is German.
    """
    print("Preparing Tatoeba (en->de) ...")
    txt_path = download_tatoeba(data_dir)

    raw_src, raw_tgt = _read_pairs(txt_path)
    assert len(raw_src) == len(raw_tgt)
    n_total = len(raw_src)
    print(f"  total: {n_total} pairs (raw)")

    # Tokenize first so we can filter by length BEFORE building the vocab
    # (saves ~10x memory on the 330k corpus vs filtering after vocab).
    print("  tokenizing + filtering by length ...", flush=True)
    pairs: List[Tuple[List[str], List[str]]] = []
    for en, de in zip(raw_src, raw_tgt):
        en_tok = tokenize(en)
        de_tok = tokenize(de)
        # Budget: source can be up to max_seq_length tokens; target adds
        # <sos>/<eos> so leave 2 tokens of headroom.
        if 1 <= len(en_tok) <= max_seq_length and 1 <= len(de_tok) <= max_seq_length - 2:
            pairs.append((en_tok, de_tok))
    print(f"  kept:  {len(pairs)} pairs (after length filter)")

    # Deterministic train/val/test split
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_test = max(1, int(test_ratio * len(pairs)))
    n_val = max(1, int(val_ratio * len(pairs)))
    test_idx = sorted(idx[:n_test])
    val_idx = sorted(idx[n_test : n_test + n_val])
    train_idx = sorted(idx[n_test + n_val :])
    print(f"  train: {len(train_idx):>6} pairs")
    print(f"  val:   {len(val_idx):>6} pairs")
    print(f"  test:  {len(test_idx):>6} pairs")

    # Build vocabularies from the training split only.
    src_vocab = Vocab.build([pairs[i][0] for i in train_idx], min_freq=config.SRC_MIN_FREQ)
    tgt_vocab = Vocab.build([pairs[i][1] for i in train_idx], min_freq=config.TGT_MIN_FREQ)
    print(f"  src (en) vocab: {len(src_vocab):>5} tokens")
    print(f"  tgt (de) vocab: {len(tgt_vocab):>5} tokens")

    # Encode each split. Source is read raw; target is wrapped in <sos>/<eos>.
    def encode_split(sel_idx):
        src_ids = [src_vocab.encode(pairs[i][0]) for i in sel_idx]
        tgt_ids = [
            [tgt_vocab.stoi["<sos>"]] + tgt_vocab.encode(pairs[i][1]) + [tgt_vocab.stoi["<eos>"]]
            for i in sel_idx
        ]
        return src_ids, tgt_ids

    train_src, train_tgt = encode_split(train_idx)
    val_src, val_tgt = encode_split(val_idx)
    test_src, test_tgt = encode_split(test_idx)

    # Datasets
    train_ds = TranslationDataset(train_src, train_tgt)
    val_ds = TranslationDataset(val_src, val_tgt)
    test_ds = TranslationDataset(test_src, test_tgt)

    collate = make_collate_fn()

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_sampler=BucketBatchSampler(
            [len(s) for s in train_src], batch_size, shuffle=True, seed=seed
        ),
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=BucketBatchSampler(
            [len(s) for s in val_src], batch_size, shuffle=False, drop_last=False, seed=seed
        ),
        collate_fn=collate,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=BucketBatchSampler(
            [len(s) for s in test_src], batch_size, shuffle=False, drop_last=False, seed=seed
        ),
        collate_fn=collate,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


# ===========================================================================
# SentencePiece BPE tokenizer (Opción C)
# ===========================================================================
# This section is only used when config.TOKENIZER == "bpe". The Vocab class
# above remains the canonical word-level implementation; BpeVocab subclasses
# it to add the SPM encode/decode bridge while keeping the same SPECIAL_TOKENS
# contract (UNK=0, PAD=1, SOS=2, EOS=3).
#
# Training a SentencePiece model is a one-time cost (~30-90 s on CPU for the
# 324k-pair Tatoeba corpus at vocab_size=8000). The trained model is cached
# under config.SPM_MODEL_DIR and reused on subsequent runs.
# ===========================================================================

def _ensure_spm() -> "sentencepiece.SentencePieceProcessor":  # type: ignore[name-defined]
    """Late import so word-only users don't pay the import cost."""
    import sentencepiece as spm
    return spm


def _spm_train(
    sentences: Sequence[str],
    model_prefix: str,
    vocab_size: int,
    char_coverage: float,
) -> None:
    """Train a SentencePiece BPE model from a list of raw sentences.

    Writes `<model_prefix>.model` and `<model_prefix>.vocab` to disk. SPM is
    fed one sentence per line via a temporary file because the API doesn't
    accept an in-memory list directly.
    """
    spm = _ensure_spm()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", encoding="utf-8", delete=False
    ) as f:
        for s in sentences:
            f.write(s.strip() + "\n")
        tmp_path = f.name
    try:
        spm.SentencePieceTrainer.Train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=char_coverage,
            # Match the SPECIAL_TOKENS contract by reserving 4 IDs for the
            # user-defined symbols; SPM fills the rest with byte-level
            # fallback pieces for OOV robustness.
            user_defined_symbols=["<pad>", "<sos>", "<eos>"],
            unk_id=0,
            pad_id=1,
            bos_id=2,
            eos_id=3,
            # Keep input case-sensitive to preserve German capitalization
            # for proper nouns; the word-level tokenizer lowercased.
            normalization_rule_name="identity",
            # Don't split on whitespace aggressively — let BPE handle it.
            split_by_unicode_script=True,
            split_by_whitespace=True,
            split_digits=True,
        )
    finally:
        os.remove(tmp_path)


class BpeVocab(Vocab):
    """SentencePiece-backed vocabulary. Same interface as Vocab.

    The 4 special tokens occupy the same indices as Vocab (UNK=0, PAD=1,
    SOS=2, EOS=3) so the rest of the pipeline (model.py, train.py,
    translate.py) needs no changes.
    """

    def __init__(self, sp_processor) -> None:
        self.sp = sp_processor
        # Build itos/stoi in the SAME order Vocab would: 4 specials + SPM pieces.
        itos = ["<unk>", "<pad>", "<sos>", "<eos>"]
        for i in range(4, sp_processor.GetPieceSize()):
            itos.append(sp_processor.IdToPiece(i))
        stoi = {tok: idx for idx, tok in enumerate(itos)}
        # Make sure the specials map correctly even if SPM renumbered them.
        stoi["<unk>"] = sp_processor.PieceToId("<unk>")
        stoi["<pad>"] = sp_processor.PieceToId("<pad>")
        stoi["<sos>"] = sp_processor.PieceToId("<sos>")
        stoi["<eos>"] = sp_processor.PieceToId("<eos>")
        # Rebuild itos to honour the canonical indices chosen by SPM.
        size = sp_processor.GetPieceSize()
        itos = [sp_processor.IdToPiece(i) for i in range(size)]
        super().__init__(stoi, itos)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        # Accept a list of pre-tokenized strings OR a single raw string.
        if isinstance(tokens, str) or (
            isinstance(tokens, list) and len(tokens) == 1 and isinstance(tokens[0], str)
        ):
            text = tokens if isinstance(tokens, str) else tokens[0]
            return self.sp.EncodeAsIds(text)
        text = " ".join(tokens)
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
        # SPM's IdToPiece returns the raw subword (with the SentencePiece
        # "▁" marker for word boundaries). DecodeAsPieces gives the joined
        # pieces; DecodeAsIds applies both steps and turns "▁" back into
        # spaces, producing the natural target string.
        if skip_specials:
            special_ids = {self.stoi[s] for s in SPECIAL_TOKENS}
            ids = [i for i in ids if i not in special_ids]
        return self.sp.DecodeIds(list(ids))


def _train_or_load_spm(
    raw_sentences: Sequence[str],
    model_dir: str,
    model_prefix: str,
    vocab_size: int,
    char_coverage: float,
) -> "spm.SentencePieceProcessor":  # type: ignore[name-defined]
    """Train SPM if no model file exists, then load and return it."""
    spm = _ensure_spm()
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_prefix}.model")
    if not os.path.exists(model_path):
        print(f"  training SentencePiece BPE ({model_prefix}, vocab={vocab_size}) ...", flush=True)
        prefix = os.path.join(model_dir, model_prefix)
        _spm_train(raw_sentences, prefix, vocab_size, char_coverage)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


def _load_data_bpe(
    batch_size: int = config.BATCH_SIZE,
    data_dir: str = config.DATA_DIR,
    seed: int = config.SEED,
    max_seq_length: int = config.MAX_SEQ_LENGTH,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
) -> Tuple[DataLoader, DataLoader, DataLoader, BpeVocab, BpeVocab]:
    """BPE-mode equivalent of load_data(). Same return contract."""
    print("Preparing Tatoeba (en->de) [BPE mode] ...")
    txt_path = download_tatoeba(data_dir)
    raw_src, raw_tgt = _read_pairs(txt_path)
    assert len(raw_src) == len(raw_tgt)
    n_total = len(raw_src)
    print(f"  total: {n_total} pairs (raw)")

    # Deterministic train/val/test split BEFORE filtering length, so the same
    # pair indices are used as in word mode (debug + reproducibility).
    rng = random.Random(seed)
    idx = list(range(n_total))
    rng.shuffle(idx)
    n_test = max(1, int(test_ratio * n_total))
    n_val = max(1, int(val_ratio * n_total))
    test_idx = sorted(idx[:n_test])
    val_idx = sorted(idx[n_test : n_test + n_val])
    train_idx = sorted(idx[n_test + n_val :])
    raw_train_src = [raw_src[i] for i in train_idx]
    raw_train_tgt = [raw_tgt[i] for i in train_idx]

    # Train SPM models on the training split only.
    src_sp = _train_or_load_spm(
        raw_train_src, config.SPM_MODEL_DIR, config.SPM_SRC_PREFIX,
        config.SPM_VOCAB_SIZE, config.SPM_CHAR_COVERAGE,
    )
    tgt_sp = _train_or_load_spm(
        raw_train_tgt, config.SPM_MODEL_DIR, config.SPM_TGT_PREFIX,
        config.SPM_VOCAB_SIZE, config.SPM_CHAR_COVERAGE,
    )
    src_vocab = BpeVocab(src_sp)
    tgt_vocab = BpeVocab(tgt_sp)
    print(f"  src (en) SPM vocab: {len(src_vocab):>5} pieces")
    print(f"  tgt (de) SPM vocab: {len(tgt_vocab):>5} pieces")

    # Length-filter using SPM tokenization, then encode.
    print("  tokenizing + filtering by length ...", flush=True)
    pairs: List[Tuple[List[int], List[int]]] = []
    for i in train_idx + val_idx + test_idx:
        en_ids = src_sp.EncodeAsIds(raw_src[i])[: max_seq_length]
        de_ids = tgt_sp.EncodeAsIds(raw_tgt[i])[: max_seq_length - 2]
        if 1 <= len(en_ids) <= max_seq_length and 1 <= len(de_ids) <= max_seq_length - 2:
            pairs.append((i, en_ids, de_ids))
    print(f"  kept:  {len(pairs)} pairs (after length filter)")

    # Re-derive split indices on the filtered set.
    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    f_train, f_val, f_test = [], [], []
    for new_i, (orig_i, en_ids, de_ids) in enumerate(pairs):
        if orig_i in train_set: f_train.append(new_i)
        elif orig_i in val_set: f_val.append(new_i)
        else: f_test.append(new_i)
    print(f"  train: {len(f_train):>6} pairs")
    print(f"  val:   {len(f_val):>6} pairs")
    print(f"  test:  {len(f_test):>6} pairs")

    def wrap_tgt(de_ids: List[int]) -> List[int]:
        return [tgt_vocab.stoi["<sos>"]] + de_ids + [tgt_vocab.stoi["<eos>"]]

    train_src = [pairs[i][1] for i in f_train]
    train_tgt = [wrap_tgt(pairs[i][2]) for i in f_train]
    val_src   = [pairs[i][1] for i in f_val]
    val_tgt   = [wrap_tgt(pairs[i][2]) for i in f_val]
    test_src  = [pairs[i][1] for i in f_test]
    test_tgt  = [wrap_tgt(pairs[i][2]) for i in f_test]

    train_ds = TranslationDataset(train_src, train_tgt)
    val_ds   = TranslationDataset(val_src, val_tgt)
    test_ds  = TranslationDataset(test_src, test_tgt)

    collate = make_collate_fn()

    train_loader = DataLoader(
        train_ds,
        batch_sampler=BucketBatchSampler(
            [len(s) for s in train_src], batch_size, shuffle=True, seed=seed
        ),
        collate_fn=collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=BucketBatchSampler(
            [len(s) for s in val_src], batch_size, shuffle=False, drop_last=False, seed=seed
        ),
        collate_fn=collate, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=BucketBatchSampler(
            [len(s) for s in test_src], batch_size, shuffle=False, drop_last=False, seed=seed
        ),
        collate_fn=collate, num_workers=0,
    )
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


# Top-level dispatcher: route to BPE or word based on config.TOKENIZER.
def load_data_dispatched(*args, **kwargs):
    if getattr(config, "TOKENIZER", "word") == "bpe":
        return _load_data_bpe(*args, **kwargs)
    return load_data(*args, **kwargs)
