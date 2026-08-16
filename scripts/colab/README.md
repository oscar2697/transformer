# Colab Notebooks

Two notebooks to retrain the Transformer on Google Colab, surviving the
~12 h session cap by **chunking epochs and resuming from `latest.pt`** at
each new session.

## Files

| Notebook | Purpose | Produces |
|----------|---------|----------|
| `train_bpe_colab.ipynb` | Hand-rolled model + SentencePiece BPE | `checkpoints/{best,latest}.pt`, `checkpoints/metrics.jsonl` |
| `train_baseline_nn_colab.ipynb` | `nn.Transformer` baseline + BPE | `checkpoints/baseline_{best,latest}.pt`, `checkpoints/baseline_metrics.jsonl` |

Both write checkpoints to a **persistent Google Drive** folder so they
survive across sessions. The first cell of each notebook mounts Drive and
clones the repo (or pulls if already cloned).

## Usage on Colab

1. Open the notebook in Colab (Runtime → Change runtime type → **T4 GPU**).
2. Run cells top-to-bottom on the **first** session.
3. Colab will disconnect after ~12 h or 90 min of inactivity. When it
   does, reopen the same notebook, **rerun all cells** — the resume logic
   picks up from `checkpoints/latest.pt` automatically.
4. After the full epoch budget is reached, the last cell packages the
   artifacts into a `.tar.gz` in Drive ready for download.

## Why chunked training?

The previous single-shot 8-epoch run was killed mid-epoch-2 (see
`checkpoints/train.log`, stuck at step 511/10146 after 6 h). The Tatoeba
dataset has ~190 k pairs and the compact 12 M-param model trains at
roughly 1.7 s/step on a T4, which means **~9 h per epoch** for the full
set. Any single session longer than ~5 epochs will hit the cap.

## What "done" looks like

- `metrics.jsonl` (or `baseline_metrics.jsonl`) contains N records with
  `split == "epoch"` for the full target epoch count.
- The final epoch record shows a `val_ppl` consistent with the rest of
  the curve (no NaN, no explosion).
- `best.pt` is dated after the final epoch.

Then download from Drive: the checkpoint, the metrics log, and the SPM
models under `.data/spm/` (only needed once — copy from an existing run
if you already have them).
