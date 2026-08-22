"""Translate sentences with a trained Transformer checkpoint.

Usage:
    python translate.py --checkpoint checkpoints/best.pt --text "ein mann geht die straße entlang"
    python translate.py --checkpoint checkpoints/best.pt --interactive
    python translate.py --checkpoint checkpoints/best.pt --src-file input.en --out-file output.de
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import torch

import config
from dataset import Vocab, BpeVocab, load_data_dispatched as load_data, tokenize
from model import Transformer
from utils import set_seed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate with a trained Transformer")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--data-dir", default=config.DATA_DIR, help="Used to rebuild the vocabs")
    p.add_argument("--text", default=None, help="Single sentence to translate")
    p.add_argument("--src-file", default=None, help="File with one source sentence per line")
    p.add_argument("--out-file", default=None, help="Where to write translations")
    p.add_argument("--interactive", action="store_true", help="Read sentences from stdin")
    p.add_argument("--max-len", type=int, default=config.MAX_DECODE_LEN)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--print-source", action="store_true", help="Also print the source")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="Penalty for already-generated tokens (HuggingFace convention). "
                        "1.0 disables; 1.1-1.3 typically breaks repetition loops on short inputs.")
    p.add_argument("--beam-size", type=int, default=1,
                   help="Beam size for decoding. 1 = greedy. Beam search explores "
                        "multiple hypotheses in parallel and usually improves quality "
                        "for short inputs; it is ~beam_size times slower than greedy.")
    p.add_argument("--length-penalty", type=float, default=0.6,
                   help="Wu et al. (2016) length-penalty exponent applied at final "
                        "beam selection. 0 disables. Ignored when --beam-size 1.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def build_model_from_checkpoint(
    checkpoint_path: str, src_vocab: Vocab, tgt_vocab: Vocab, device: torch.device
):
    """Load a checkpoint into either the hand-rolled Transformer or the
    nn.Transformer baseline, depending on the `model_type` field saved in
    the checkpoint config. Returns the loaded model.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_type = cfg.get("model_type", "hand_rolled")
    if model_type == "baseline_nn":
        from baseline_nn import BaselineNNTransformer as ModelCls
    else:
        ModelCls = Transformer
    model = ModelCls(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.get("d_model", config.D_MODEL),
        num_heads=cfg.get("num_heads", config.NUM_HEADS),
        num_layers=cfg.get("num_layers", config.NUM_ENCODER_LAYERS),
        d_ff=cfg.get("d_ff", config.D_FF),
        dropout=cfg.get("dropout", config.DROPOUT),
        max_seq_length=cfg.get("max_seq_length", config.MAX_SEQ_LENGTH),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def encode_source(sentences: List[str], src_vocab: Vocab, max_len: int) -> torch.Tensor:
    """Encode a list of source sentences to a padded (batch, max_src_len) tensor.

    The encoding path must mirror training exactly:

    - For BPE checkpoints (``src_vocab`` is a ``BpeVocab``), call SentencePiece
      directly on the raw text. Going through the word-level ``tokenize()``
      helper and then re-joining with spaces produces a different ID sequence
      (SPM inserts an extra leading-whitespace piece ``▁`` per word boundary)
      and the model sees a token distribution it was never trained on.
    - For word-level checkpoints, keep the original tokenize-then-encode path.
    - In neither mode do we wrap the source with ``<sos>``/``<eos>``: training
      feeds the bare source sequence to the encoder (see ``load_data`` and
      ``_load_data_bpe`` in ``dataset.py``).
    """
    is_bpe = isinstance(src_vocab, BpeVocab)
    ids: List[List[int]] = []
    for s in sentences:
        if is_bpe:
            ids.append(src_vocab.sp.EncodeAsIds(s)[: max_len])
        else:
            toks = tokenize(s)[: max_len]
            ids.append(src_vocab.encode(toks))
    max_src = max(len(x) for x in ids)
    out = torch.full((len(ids), max_src), config.PAD_IDX, dtype=torch.long)
    for i, seq in enumerate(ids):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def translate_batch(
    model: Transformer,
    src: torch.Tensor,
    src_pad_mask: torch.Tensor,
    sos_idx: int,
    eos_idx: int,
    max_len: int,
    repetition_penalty: float = 1.0,
    beam_size: int = 1,
    length_penalty_alpha: float = 0.6,
) -> torch.Tensor:
    if beam_size <= 1:
        return model.greedy_decode(
            src, src_pad_mask,
            sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len,
            repetition_penalty=repetition_penalty,
        )
    return model.beam_search_decode(
        src, src_pad_mask,
        sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len,
        beam_size=beam_size,
        length_penalty_alpha=length_penalty_alpha,
        repetition_penalty=repetition_penalty,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # We need the vocabs. Easiest way: run load_data() — it will not re-download
    # the dataset if the txt files are already on disk. We discard the loaders.
    print("Loading vocabs ...")
    _, _, _, src_vocab, tgt_vocab = load_data(batch_size=args.batch_size, data_dir=args.data_dir)
    print(f"  src vocab: {len(src_vocab)}, tgt vocab: {len(tgt_vocab)}")

    model = build_model_from_checkpoint(args.checkpoint, src_vocab, tgt_vocab, device)
    print(f"Loaded checkpoint {args.checkpoint}")

    sos_idx = tgt_vocab.stoi["<sos>"]
    eos_idx = tgt_vocab.stoi["<eos>"]

    sentences: List[str] = []

    if args.text:
        sentences.append(args.text)
    elif args.src_file:
        with open(args.src_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
    elif args.interactive:
        print("Interactive mode — type an English sentence and press Enter (Ctrl-D / Ctrl-Z to quit).")
        for line in sys.stdin:
            line = line.strip()
            if line:
                sentences.append(line)
    else:
        # Default: read stdin
        for line in sys.stdin:
            line = line.strip()
            if line:
                sentences.append(line)

    if not sentences:
        print("No input sentences.", file=sys.stderr)
        return 1

    out_lines: List[str] = []
    with torch.no_grad():
        for start in range(0, len(sentences), args.batch_size):
            batch = sentences[start : start + args.batch_size]
            src = encode_source(batch, src_vocab, args.max_len).to(device)
            src_pad_mask = src.eq(config.PAD_IDX)
            ids = translate_batch(
                model, src, src_pad_mask,
                sos_idx=sos_idx, eos_idx=eos_idx, max_len=args.max_len,
                repetition_penalty=args.repetition_penalty,
                beam_size=args.beam_size,
                length_penalty_alpha=args.length_penalty,
            )
            for i in range(ids.size(0)):
                translation = tgt_vocab.decode(ids[i].tolist(), skip_specials=True)
                if args.print_source:
                    out_lines.append(f"{batch[i]}\t{translation}")
                else:
                    out_lines.append(translation)

    output = "\n".join(out_lines) + "\n"
    if args.out_file:
        with open(args.out_file, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Wrote {len(out_lines)} translations to {args.out_file}")
    else:
        print(output, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
