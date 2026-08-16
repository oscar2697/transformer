# Transformer (en -> de) — hand-rolled on Tatoeba

A clean, from-scratch PyTorch implementation of the original Transformer
(Vaswani et al., 2017) for **English → German** translation on the
[Tatoeba](https://www.manythings.org/anki/) sentence-pair collection
(CC-BY licensed, ~330k pairs).

Everything is hand-rolled:

- `MultiHeadAttention`, `PositionwiseFeedForward`, `PositionalEncoding`,
  `EncoderLayer`, `DecoderLayer`, `Transformer` (no `nn.MultiheadAttention`,
  no `nn.Transformer`, no `torchtext`).
- Post-LayerNorm (the formulation from the original paper).
- Word-level tokenization with a regex-based tokenizer; vocabulary built
  from the training split with `min_freq`.
- Length-bucketed batching with dynamic padding.
- Noam learning-rate schedule, gradient clipping, label smoothing,
  early stopping, checkpoint resume.
- Greedy decoding exposed as a CLI for translation.

## Project layout

```
transformer/
├── config.py            # all hyperparameters
├── model.py             # hand-rolled Transformer
├── dataset.py           # Tatoeba loader + Vocab + BucketBatchSampler
├── train.py             # training entry point
├── translate.py         # greedy decoding CLI
├── evaluate.py          # BLEU/chrF on the test set
├── visualize.py         # training-curve plots
├── visualize_attention.py # attention heatmaps
├── utils.py             # seeding, AverageMeter, formatters
├── tests/               # pytest test suite
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# from the project root
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

The first time you run anything, `dataset.py` downloads Tatoeba
(~12 MB) from `https://www.manythings.org/anki/deu-eng.zip` into `.data/`.

## Train

```bash
# default config (compact: d_model=256, 4+4 layers, 8 epochs, batch 32)
python train.py

# shorter smoke run
python train.py --epochs 2 --batch-size 32 --val-interval 0

# resume from a checkpoint
python train.py --resume checkpoints/latest.pt
```

Useful flags: `--d-model`, `--num-layers`, `--num-heads`, `--d-ff`,
`--dropout`, `--label-smoothing 0.1`, `--warmup`, `--lr-factor`,
`--patience` (set to 0 to disable early stopping).

Outputs:

- `checkpoints/best.pt` — best model by validation loss.
- `checkpoints/latest.pt` — most recent epoch (for resume).
- `checkpoints/metrics.jsonl` — training metrics (one JSON per line).

## Translate

```bash
# one sentence
python translate.py --checkpoint checkpoints/best.pt --text "hello, how are you?"

# interactive
python translate.py --checkpoint checkpoints/best.pt --interactive

# batch from a file (one sentence per line, UTF-8)
python translate.py --checkpoint checkpoints/best.pt --src-file input.en --out-file output.de
```

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pt
python evaluate.py --checkpoint checkpoints/best.pt --max-samples 500 --out-json results.json
```

Reports BLEU and chrF on the test split (sacrebleu).

## Visualize

```bash
# Training curves (loss / perplexity)
python visualize.py

# Attention heatmaps from the best checkpoint
python visualize_attention.py --checkpoint checkpoints/best.pt --out-dir figures/attention
```

## Tests

```bash
pytest -q
```

The test suite covers:

- Mask shapes and values (causal, padding).
- Model forward / greedy-decode shapes.
- BucketBatchSampler / collate round-trip.
- One end-to-end training step (smoke test).

## Hyperparameters

See `config.py`. Defaults are tuned for CPU training:

| Param | Value |
|---|---|
| d_model | 256 |
| num_heads | 8 |
| num_layers | 4 (encoder & decoder) |
| d_ff | 1024 |
| dropout | 0.1 |
| batch_size | 32 |
| warmup | 2000 steps |
| label_smoothing | 0.1 |
| epochs | 8 |
| patience | 10 |

> **Note:** on CPU the default configuration will be slow (~hours per
> epoch). For a quick smoke test, run
> `python train.py --d-model 128 --num-layers 2 --d-ff 256 --batch-size 32 --epochs 2`.

## Troubleshooting

- **`ModuleNotFoundError: __path__ attribute not found on 'train' ...`**
  You typed `python -m train.py`. With `-m`, pass the module name without
  `.py`: `python -m train` (or just `python train.py`).
- **Tatoeba download fails with HTTP 406**
  The server occasionally rejects default Python user agents. `dataset.py`
  sets an explicit browser-style User-Agent header and a 60 MB download
  cap; if it still fails, manually fetch
  `http://www.manythings.org/anki/deu-eng.zip` and place it under
  `.data/`. The script is idempotent — re-running resumes where it
  stopped.

## License

MIT.
