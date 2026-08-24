# 6. Ablation Studies

We design a series of controlled ablation experiments to isolate the contribution of individual architectural and training choices. Each ablation varies one factor at a time while holding all others constant, using as reference either the word-level Post-LN diagnostic configuration (`d_model=256`, 4 encoder and 4 decoder layers, 8 heads, `d_ff=1024`, 8 epochs, label smoothing = 0.1, warmup = 2000, batch size = 32) or—for comparisons involving subword tokenization—the final main configuration of Table 1. All ablations use seed 42 and report BLEU and chrF2 on the Tatoeba EN→DE test set. Of the comparisons below, the Pre-LN versus Post-LN stabilisation and the hand-rolled versus `nn.Transformer` comparison have been completed (Sections 5.1 and 6.7); the remaining rows of Table 4 are specified hypotheses left as future work.

## 6.1 Label Smoothing

Label smoothing (Szegedy et al., 2016) regularizes the cross-entropy loss by distributing a fraction $\epsilon$ of the target probability mass evenly across incorrect classes. For neural machine translation, values of $\epsilon \in [0.05, 0.1]$ have been shown to improve BLEU by 0.5–1.5 points (Vaswani et al., 2017). Our main run is trained with $\epsilon = 0.1$ (the standard default). We compare $\epsilon = 0.0$ (no smoothing), $\epsilon = 0.1$ (current), and $\epsilon = 0.2$ (heavier smoothing). We hypothesize a non-monotonic relationship: mild smoothing should improve generalization, while excessive smoothing may dilute the training signal.

## 6.2 Warmup Schedule

The Noam scheduler uses a linear warmup phase during which the learning rate increases from 0 to its peak value, followed by an inverse-square-root decay. The number of warmup steps controls how aggressively the model explores in early training. Our main run uses 2000 warmup steps. We compare warmup values of 1000, 2000 (current), and 4000 steps. A longer warmup is expected to stabilize early gradient updates, particularly for larger models, but may slow convergence on small datasets.

## 6.3 Model Depth and Width

We study the trade-off between model depth (number of layers) and width (`d_model`) under a fixed parameter budget. Specifically, we compare:

- **Configuration A (word-level reference):** `d_model=256`, 4 encoder + 4 decoder layers, `d_ff=1024` (≈ 12.4 M parameters at the word-level vocabulary).
- **Configuration B (deeper):** `d_model=128`, 8 encoder + 8 decoder layers, `d_ff=512` ($\approx$ 6 M parameters).
- **Configuration C (wider):** `d_model=512`, 2 encoder + 2 decoder layers, `d_ff=2048` ($\approx$ 19 M parameters).

Prior work (Vaswani et al., 2017; Popel & Bojar, 2018) suggests that depth improves BLEU more efficiently than width for translation tasks, but this finding is sensitive to the dataset size and vocabulary strategy.

## 6.4 Pre-LayerNorm versus Post-LayerNorm

The original Transformer uses post-layer-normalization (layer norm applied after the residual addition), which was found to be unstable to train in early work. More recent implementations, including those in the fairseq library (Ott et al., 2018), have adopted pre-layer-normalization (layer norm applied before the attention/FFN sub-layer, with the residual connection bypassing the normalization). The word-level diagnostic run of this paper uses post-LN, matching Vaswani et al. (2017), whereas the final main run adopts pre-LN; the observed stability difference between the two runs is consistent with this literature, although the two runs also differ in tokenization and schedule, so a fully controlled same-schedule comparison remains future work.

| Ablation | Expected impact on BLEU | Rationale |
|---|---|---|
| Label smoothing ($\epsilon=0.0$) | $-0.5$ to $-1.0$ | Loss of regularization benefit |
| Warmup 4000 vs 2000 | Neutral to $+0.5$ | Stabilizes early training |
| Depth $\times 2$ (8+8 layers) | $+1$–$3$ | Deeper models capture longer-range dependencies |
| Pre-LN vs Post-LN | Neutral to $+1$ | Pre-LN is more stable; may allow higher LR |
| Subword (BPE, 8k pieces) | $+5$–$15$ | Addresses vocabulary fragmentation in EN→DE |
| Hand-rolled vs `nn.Transformer` baseline | ±$1$ | Isolates cost of from-scratch implementation |

*Table 4: Summary of expected ablation effects based on prior literature. The last row (§6.7) is the comparison enabled by `baseline_nn.py`; its results are reported in Table 6. The subword and LN-convention effects are partially evidenced by the completed runs of Sections 5.1 and 6.7, though not yet isolated by controlled same-schedule ablations; the remaining rows are planned future work.*

## 6.5 Subword Tokenization

Word-level tokenization on Tatoeba is a bottleneck: the model must learn mappings for thousands of rare German inflected forms with very limited exposure. Subword tokenization with byte-pair encoding (BPE; Sennrich et al., 2016) or SentencePiece (Kudo & Richardson, 2018) reduces the effective vocabulary size to a manageable number of units (typically 8,000–32,000) while preserving the ability to represent any word through a sequence of subword pieces. SentencePiece BPE (8k pieces per language) has been integrated into the data pipeline (`config.TOKENIZER = "bpe"`) and is used by both the final main run and the baseline. The quality gap between the word-level diagnostic and the BPE configurations (Tables 2 and 6) is consistent with a large contribution from subword tokenization, although that gap also reflects longer training and the Pre-LN stabilisation, so the standalone effect of BPE is not yet isolated by a controlled ablation.

## 6.6 Per-Head Attention Analysis

A natural ablation that exploits the first-class attention API described in Section 3.4 is to inspect the attention weights per head rather than averaged over heads. The current `visualize_attention.py` aggregates over the 8 heads, which obscures the well-documented head-specialization phenomenon. We plan to extend the script to render 1×8 sub-panels per sentence, one column per head, so that specialized heads (e.g., heads that attend to the previous token, or heads that act as "null" attention heads with near-uniform distributions) can be catalogued for the trained model.

## 6.7 Hand-Rolled vs `nn.Transformer` Baseline

To quantify the cost of the from-scratch implementation, we provide a baseline that wraps PyTorch's `torch.nn.Transformer` with the same interface (Section 3.6). Both models share the architecture (d_model = 256, num_heads = 8, num_layers = 4+4, d_ff = 1024, dropout = 0.1, label smoothing = 0.1, Adam β = (0.9, 0.98), seed = 42), the same deterministic train/val/test split of Tatoeba EN→DE, and the SentencePiece BPE pipeline, and both are trained for 25 epochs. The training schedules differ: the baseline uses batch size 32 with warmup 2,000, whereas the hand-rolled run uses batch size 64 with warmup 4,000. The comparison therefore measures the joint effect of implementation and schedule rather than the implementation alone.

| Implementation | Source code | Pre-LN? | Total params |
|---|---|---|---|
| Hand-rolled (`model.py`) | Listing 1 | Configurable; main run: Pre-LN | ≈ 13.5 M (BPE) |
| Baseline (`baseline_nn.py`, wraps `torch.nn.Transformer`) | PyTorch built-in | Yes (Pre-LN, `norm_first=True`) | ≈ 13.5 M (BPE) |

*Table 5: Two implementations trained on identical data and architecture for 25 epochs each. The hand-rolled model supports both LN conventions (the word-level diagnostic used post-LN; the main run uses pre-LN); the baseline is pre-LN only. Parameter counts are within 1% of each other. Training schedules differ: batch 32 / warmup 2,000 (baseline) vs batch 64 / warmup 4,000 (hand-rolled).*

Any non-zero BLEU/chrF gap between the two is attributable to the combination of (a) the training-schedule difference (batch and warmup), and (b) implementation details such as mask convention, attention scaling, or dropout ordering. The comparison is run via:

```bash
# Hand-rolled + BPE
python scripts/train_bpe_colab.py --epochs 25
python evaluate.py --checkpoint checkpoints/best.pt

# Baseline + BPE
python scripts/train_baseline_nn_colab.py --epochs 25
python evaluate.py --checkpoint checkpoints/baseline_best.pt
```

The results of this comparison are reported in Section 6.8 and Table 6.

## 6.6 Per-Head Attention Analysis

A natural ablation that exploits the first-class attention API described in Section 3.4 is to inspect the attention weights per head rather than averaged over heads. The current `visualize_attention.py` aggregates over the 8 heads, which obscures the well-documented head-specialization phenomenon. We plan to extend the script to render 1×8 sub-panels per sentence, one column per head, so that specialized heads (e.g., heads that attend to the previous token, or heads that act as "null" attention heads with near-uniform distributions) can be catalogued for the trained model.

*[PLACEHOLDER: ablation_chart.png — Grouped bar chart comparing BLEU scores across ablations.]*

## 6.8 Status of Ablations and Baseline Comparison

The baseline wrapper (`baseline_nn.py`, wrapping `torch.nn.Transformer`; Section 6.7) and the SentencePiece BPE tokenizer (`dataset.py`; Section 3.5) are committed to the repository; dispatching happens via `config.TOKENIZER` and a `--baseline-nn` flag to `train.py`. The end-to-end pipelines have been smoke-tested (Section 6.7) and pass.

**Baseline + BPE status: *completed*** (Table 6). Pre-LN, BPE 8 k, 25 epochs, batch 32, warmup 2000, seed 42: val_ppl = 12.39, BLEU = 38.30, chrF2 = 56.95. Wall-clock: ≈ 2.3 h on T4.

**Hand-rolled + BPE (Pre-LN) status: *completed*** (Table 6). Pre-LN, BPE 8 k, 25 epochs, batch 64, warmup 4000, seed 42: val_ppl = 9.29, BLEU = 44.00 greedy / 45.30 beam=4, chrF2 = 62.41/63.37. Wall-clock: ≈ 2.5 h on T4. The checkpoint is `checkpoints/best.pt` (Pre-LN, `norm_first=True`) and is fully compatible with `translate.py` / `evaluate.py`.

| Implementation | Tokenization | Epochs | val_ppl | BLEU (test) | chrF2 (test) | BP |
|---|---|---|---|---|---|---|
| Hand-rolled + word-level (Post-LN) | word | 8 | 149.84 | 0.38 | 11.28 | 0.593 |
| Baseline + BPE (Pre-LN) | SentencePiece 8 k | 25 | 12.39 | 38.30 | 56.95 | 1.000 |
| **Hand-rolled + BPE (Pre-LN)** | **SentencePiece 8 k** | **25** | **9.29** | **44.00** | **62.41** | **0.995** |
| | | | | *45.30 (beam=4)* | *63.37* | *0.987* |

*Table 6: Baseline comparison on the Tatoeba EN→DE test set. Rows differ in more than one factor (tokenization, LN convention, epochs, batch, warmup), so the 12× drop in validation perplexity from the word-level diagnostic (149.84) to the BPE configurations (12.39 and 9.29) reflects the joint effect of subword tokenization, Pre-LN stabilisation, corrected warmup, and longer training; it does not isolate any single factor. The 5.7 BLEU margin over the baseline (38.30 → 44.00) additionally mixes implementation and schedule effects (Section 6.7). Beam search (beam = 4) adds a further +1.3 BLEU over greedy decoding on the hand-rolled model.*

Under this protocol the hand-rolled + BPE (Pre-LN) row quantifies the implementation comparison: the from-scratch implementation, with the stabilisation fixes of Sections 3–4, matches or exceeds the built-in baseline; a fully matched-hyperparameter re-run is required before attributing the margin to the implementation itself.
