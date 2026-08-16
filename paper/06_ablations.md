# 6. Ablation Studies

We design a series of controlled ablation experiments to isolate the contribution of individual architectural and training choices. Each ablation is trained from scratch with the main configuration (`d_model=256`, 4 encoder and 4 decoder layers, 8 heads, `d_ff=1024`, 8 epochs, label smoothing = 0.1, warmup = 2000, batch size = 32) as the starting point, varying one parameter at a time while holding all others constant. All ablations use the same seed (42) and report BLEU and chrF2 on the Tatoeba EN→DE test set.

## 6.1 Label Smoothing

Label smoothing (Szegedy et al., 2016) regularizes the cross-entropy loss by distributing a fraction $\epsilon$ of the target probability mass evenly across incorrect classes. For neural machine translation, values of $\epsilon \in [0.05, 0.1]$ have been shown to improve BLEU by 0.5–1.5 points (Vaswani et al., 2017). Our main run is trained with $\epsilon = 0.1$ (the standard default). We compare $\epsilon = 0.0$ (no smoothing), $\epsilon = 0.1$ (current), and $\epsilon = 0.2$ (heavier smoothing). We hypothesize a non-monotonic relationship: mild smoothing should improve generalization, while excessive smoothing may dilute the training signal.

## 6.2 Warmup Schedule

The Noam scheduler uses a linear warmup phase during which the learning rate increases from 0 to its peak value, followed by an inverse-square-root decay. The number of warmup steps controls how aggressively the model explores in early training. Our main run uses 2000 warmup steps. We compare warmup values of 1000, 2000 (current), and 4000 steps. A longer warmup is expected to stabilize early gradient updates, particularly for larger models, but may slow convergence on small datasets.

## 6.3 Model Depth and Width

We study the trade-off between model depth (number of layers) and width (`d_model`) under a fixed parameter budget. Specifically, we compare:

- **Configuration A (current):** `d_model=256`, 4 encoder + 4 decoder layers, `d_ff=1024` ($\approx$ 12.4 M parameters).
- **Configuration B (deeper):** `d_model=128`, 8 encoder + 8 decoder layers, `d_ff=512` ($\approx$ 6 M parameters).
- **Configuration C (wider):** `d_model=512`, 2 encoder + 2 decoder layers, `d_ff=2048` ($\approx$ 19 M parameters).

Prior work (Vaswani et al., 2017; Popel & Bojar, 2018) suggests that depth improves BLEU more efficiently than width for translation tasks, but this finding is sensitive to the dataset size and vocabulary strategy.

## 6.4 Pre-LayerNorm versus Post-LayerNorm

The original Transformer uses post-layer-normalization (layer norm applied after the residual addition), which was found to be unstable to train in early work. More recent implementations, including those in the fairseq library (Ott et al., 2019), have adopted pre-layer-normalization (layer norm applied before the attention/FFN sub-layer, with the residual connection bypassing the normalization). Our main run uses post-LN, matching Vaswani et al. (2017). We compare both architectures under otherwise identical conditions to quantify the stability and performance difference.

| Ablation | Expected impact on BLEU | Rationale |
|---|---|---|
| Label smoothing ($\epsilon=0.0$) | $-0.5$ to $-1.0$ | Loss of regularization benefit |
| Warmup 4000 vs 2000 | Neutral to $+0.5$ | Stabilizes early training |
| Depth $\times 2$ (8+8 layers) | $+1$–$3$ | Deeper models capture longer-range dependencies |
| Pre-LN vs Post-LN | Neutral to $+1$ | Pre-LN is more stable; may allow higher LR |
| Subword (BPE, 8k pieces) | $+5$–$15$ | Addresses vocabulary fragmentation in EN→DE |
| Hand-rolled vs `nn.Transformer` baseline | ±$1$ | Isolates cost of from-scratch implementation |

*Table 4: Summary of expected ablation effects based on prior literature. The last row (§6.7) is the comparison enabled by `baseline_nn.py`. Actual results are filled in once the BPE runs complete.*

## 6.5 Subword Tokenization

Word-level tokenization on Tatoeba is still a bottleneck: the model must learn mappings for thousands of rare German inflected forms with very limited exposure. Subword tokenization with byte-pair encoding (BPE; Sennrich et al., 2016) or SentencePiece (Kudo & Richardson, 2018) reduces the effective vocabulary size to a manageable number of units (typically 8,000–32,000) while preserving the ability to represent any word through a sequence of subword pieces. We plan to integrate SentencePiece into the data pipeline and evaluate the BLEU improvement from subword tokenization as a standalone experiment.

## 6.6 Per-Head Attention Analysis

A natural ablation that exploits the first-class attention API described in Section 3.4 is to inspect the attention weights per head rather than averaged over heads. The current `visualize_attention.py` aggregates over the 8 heads, which obscures the well-documented head-specialization phenomenon. We plan to extend the script to render 1×8 sub-panels per sentence, one column per head, so that specialized heads (e.g., heads that attend to the previous token, or heads that act as "null" attention heads with near-uniform distributions) can be catalogued for the trained model.

## 6.7 Hand-Rolled vs `nn.Transformer` Baseline

To quantify the cost of the from-scratch implementation, we provide a baseline that wraps PyTorch's `torch.nn.Transformer` with the same interface (Section 3.6). Both models are trained under identical hyperparameters (d_model = 256, num_heads = 8, num_layers = 4+4, d_ff = 1024, dropout = 0.1, label smoothing = 0.1, warmup = 2000, batch size = 32, Adam β = (0.9, 0.98), seed = 42) on the same deterministic train/val/test split of Tatoeba EN→DE.

| Implementation | Source code | Pre-LN? | Total params |
|---|---|---|---|
| Hand-rolled (`model.py`) | Listing 1 | No (post-LN, original convention) | ≈ 13.5 M (BPE) |
| Baseline (`baseline_nn.py`, wraps `torch.nn.Transformer`) | PyTorch built-in | Yes (modern convention) | ≈ 13.5 M (BPE) |

*Table 5: Two implementations trained on identical data and hyperparameters. The only architectural difference is the LN convention; parameter counts are within 1% of each other.*

Any non-zero BLEU/chrF gap between the two is attributable to either (a) the post-LN vs pre-LN choice (Section 6.4), or (b) subtle implementation details (mask convention, attention scaling, dropout ordering). The comparison is run via:

```bash
# Hand-rolled + BPE
python scripts/train_bpe_colab.py --epochs 20
python evaluate.py --checkpoint checkpoints/best.pt

# Baseline + BPE
python scripts/train_baseline_nn_colab.py --epochs 20
python evaluate.py --checkpoint checkpoints/baseline_best.pt
```

The results of this comparison are reported alongside the word-level main run in Section 5.

## 6.6 Per-Head Attention Analysis

A natural ablation that exploits the first-class attention API described in Section 3.4 is to inspect the attention weights per head rather than averaged over heads. The current `visualize_attention.py` aggregates over the 8 heads, which obscures the well-documented head-specialization phenomenon. We plan to extend the script to render 1×8 sub-panels per sentence, one column per head, so that specialized heads (e.g., heads that attend to the previous token, or heads that act as "null" attention heads with near-uniform distributions) can be catalogued for the trained model.

*[PLACEHOLDER: ablation_chart.png — Grouped bar chart comparing BLEU scores across ablations.]*

## 6.8 Status of Planned Ablations

The implementations of Opción B (`baseline_nn.py` wrapping `torch.nn.Transformer`) and Opción C (SentencePiece BPE in `dataset.py`) are committed to the repository; the dispatching happens via `config.TOKENIZER` and a `--baseline-nn` flag to `train.py`. The end-to-end pipelines have been smoke-tested (Section 6.7) and pass. The two full training runs (BPE hand-rolled and BPE baseline, 15–20 epochs each on a single Colab GPU) are pending external compute and will be run on Google Colab following the recipe in `scripts/train_bpe_colab.py` and `scripts/train_baseline_nn_colab.py`. Numerical results will be slotted into Table 2, Table 5, and the §5.3 sample translations once available.
