"""Evaluate a trained Transformer on the Tatoeba test set.

Computes BLEU and chrF with sacrebleu. Honors config.TOKENIZER so the
same script works for both word-level and BPE-trained checkpoints.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --checkpoint checkpoints/best.pt --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List

import sacrebleu
import torch
from tqdm import tqdm

import config
from dataset import load_data_dispatched as load_data, tokenize
from model import Transformer
from translate import build_model_from_checkpoint, encode_source, translate_batch
from utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Transformer")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default=config.DATA_DIR)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=config.MAX_DECODE_LEN)
    p.add_argument("--max-samples", type=int, default=None, help="Cap on test set size")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--out-json", default=None, help="Optional path to write the score JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    print("Loading data ...")
    _, _, test_loader, src_vocab, tgt_vocab = load_data(
        batch_size=args.batch_size, data_dir=args.data_dir
    )

    model = build_model_from_checkpoint(args.checkpoint, src_vocab, tgt_vocab, device)
    print(f"Loaded {args.checkpoint}")

    sos_idx = tgt_vocab.stoi["<sos>"]
    eos_idx = tgt_vocab.stoi["<eos>"]

    hypotheses: List[str] = []
    references: List[str] = []

    start = time.time()
    n_done = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Translating"):
            src = batch["src"]
            tgt = batch["tgt"]
            src_pad_mask = batch["src_pad_mask"]

            ids = translate_batch(
                model, src.to(device), src_pad_mask.to(device),
                sos_idx=sos_idx, eos_idx=eos_idx, max_len=args.max_len,
            )

            # Decode predictions and gold targets back to strings.
            for i in range(ids.size(0)):
                hypotheses.append(tgt_vocab.decode(ids[i].tolist(), skip_specials=True))
                references.append(tgt_vocab.decode(tgt[i].tolist(), skip_specials=True))

            n_done += src.size(0)
            if args.max_samples and n_done >= args.max_samples:
                break

    if args.max_samples:
        hypotheses = hypotheses[: args.max_samples]
        references = references[: args.max_samples]

    elapsed = time.time() - start
    print(f"Translated {len(hypotheses)} sentences in {elapsed:.1f}s "
          f"({len(hypotheses)/max(elapsed, 1):.1f} sent/s)")

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])

    results = {
        "n": len(hypotheses),
        "bleu": bleu.score,
        "bleu_signature": str(bleu),
        "chrf": chrf.score,
        "chrf_signature": str(chrf),
        "elapsed_sec": elapsed,
    }
    print(json.dumps(results, indent=2))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote results to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
