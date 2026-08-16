# 5. Results

## 5.1 Translation Quality

Table 2 reports the translation quality achieved by our hand-rolled Transformer on the Tatoeba EN→DE test set (3,296 pairs evaluated out of 3,312; 16 hypotheses were empty and dropped by sacrebleu). With a compact configuration (`d_model = 256`, 4 encoder and 4 decoder layers, 8 heads, `d_ff = 1024`, ~12 M parameters) trained for **8 epochs on a single Colab GPU** in **≈ 56 minutes** (wall-clock), with label smoothing $\epsilon = 0.1$ and the Noam schedule (warmup 2000 steps), the model reaches a **corpus-level BLEU = 0.38 and chrF2 = 11.28** at the lowest validation loss seen during training (val_loss = 5.0096, val_ppl = 149.84 at epoch 8).

| Configuration | Epochs | val_loss | val_ppl | BLEU (test) | chrF2 (test) | Params |
|---|---|---|---|---|---|---|
| Tatoeba EN→DE main (`d_model=256`, 4+4 layers) | 8 | 5.0096 | 149.84 | 0.38 | 11.28 | ≈ 12 M |

*Table 2: Results on the Tatoeba EN→DE test set (n = 3,296 sentences that produced non-empty hypotheses; 16 dropped by sacrebleu). BLEU and chrF2 are corpus-level scores computed with sacrebleu. The brevity-penalty signature `BLEU = 0.38 33.4/2.3/0.3/0.0 (BP = 0.593 ratio = 0.657)` indicates that the model emits hypotheses that are 65.7% the length of the reference, suggesting premature `<eos>` emission or over-reliance on a small set of high-frequency tokens.*

The best checkpoint (`checkpoints/best.pt`) is selected by lowest validation loss on a held-out 3,312-pair validation split, using the 3,312-pair test set only for the final BLEU/chrF2 evaluation.

These results are **substantially below** the expected BLEU range of 20–30 for a model of this size on a ~330k-pair corpus. We discuss the diagnosis (sub-training) in Section 7.1 and the planned remediation (longer training and subword tokenization) in Section 7.4.

## 5.2 Training Dynamics

Figure 1 shows the training and validation loss curves for the main configuration over the 8 completed epochs.

![Training and validation loss curves for the main run.](figures/training_curves.pdf)

*Figure 1: Training loss (orange) and validation loss (blue) per epoch. Source: `figures/training_curves.pdf` (also `figures/training_curves.png`). Generated from `checkpoints/metrics.jsonl`.*

Both curves decrease monotonically over the 8 epochs: train loss from 5.55 → 4.75, val loss from 5.48 → 5.01. The val perplexity drops from 238.93 (epoch 1) to 149.84 (epoch 8) — a 1.6× reduction, but still an order of magnitude above the ~30 perplexity typically reported for a well-trained Transformer on a corpus of this size. **The validation loss curve is still trending downward at epoch 8 with no sign of plateau**, which confirms that training has not converged (see Section 7.1).

## 5.3 Sample Translations

Table 3 presents six example translations from the held-out test set produced by the main configuration. The model outputs illustrate the symptoms of sub-training (Section 7.1): a strong bias toward a small set of high-frequency tokens, repetition across unrelated inputs, and hypotheses that are systematically shorter than the references.

| English (source) | Model translation |
|---|---|
| Hello, how are you? | hast du das ? |
| The weather is nice today. | sie sind sehr groß . |
| I would like a coffee, please. | ich habe mich gesehen . |
| Where is the train station? | hast du das ? |
| Thank you very much. | sie sind sehr groß . |
| She is reading a book. | sie sind sehr groß . |

*Table 3: Source–model-translation pairs from out-of-domain English sentences. The model collapses to three recurring hypotheses (`hast du das ?`, `sie sind sehr groß .`, `ich habe mich gesehen .`) regardless of input, which is the characteristic failure mode of an under-trained encoder-decoder that has not yet learned to condition its output on the source.*

The corresponding attention visualizations for these sentences (and three additional test-set examples) are shown in Figure 2.

## 5.4 Comparison with Related Work

The Multi30k and Tatoeba datasets have been used extensively in prior work. WMT-scale Transformer models trained on the full WMT14 EN→DE corpus (approximately 4.5 million sentence pairs) typically achieve BLEU scores in the 27–28 range on the test set (Vaswani et al., 2017). Our model, trained on only ~324k sentence pairs at the word level without subword tokenization, operates under substantially different conditions. Word-level tokenization on a medium-sized corpus is known to suffer from severe vocabulary fragmentation and out-of-vocabulary issues, particularly for German, which has productive morphology.

The planned baseline comparison against PyTorch's `nn.Transformer` will enable a direct assessment of the accuracy penalty, if any, introduced by our hand-rolled implementation. Any difference in BLEU/chrF2 between the two implementations, when trained with identical hyperparameters and seeds, would be attributable solely to implementation details such as attention masking conventions or gradient flow in layer normalization.

## 5.5 Attention Interpretability

A central motivation for the from-scratch implementation is that attention weights remain first-class outputs of the model, available for inspection after training. We exploit this property to study what the trained encoder-decoder has learned.

### 5.5.1 Method

We run `visualize_attention.py` against the best checkpoint (`checkpoints/best.pt`) on the first four sentences of the deterministic Tatoeba test split (Table 3). The script extracts per-layer attention tensors via `model(src, tgt, ..., return_attention=True)` and renders, for each sentence, a 1×3 panel of heatmaps: encoder self-attention, decoder self-attention, and decoder cross-attention, all on the final layer (layer 4 of 4 in the encoder, with the matching decoder layer). It also produces a combined grid showing the three attention types for all four sentences side by side.

![Attention maps for the last layer of the main model across four test sentences.](figures/attention/attn_grid_layer4.pdf)

*Figure 2: Attention heatmaps for the last transformer layer across four short Tatoeba EN→DE test sentences. Source: `figures/attention/attn_grid_layer4.pdf` (also `attn_grid_layer4.png`). Generated by `visualize_attention.py` against `checkpoints/best.pt`.*

### 5.5.2 Observed Patterns on the Tatoeba Test Set

Figure 2 visualizes attention on the first four sentences of the deterministic Tatoeba test split. The four source-target pairs are:

| # | English (source) | German (reference) |
|---|---|---|
| S1 | duck ! | kopf runter ! |
| S2 | attack ! | _\<unk\>_ ! |
| S3 | _\<unk\>_ in . | mach mit ! |
| S4 | beat it . | geh weg ! |

All four sentences are extremely short (1–2 tokens after `<sos>`/`<eos>`). For this regime we observe:

- **Encoder self-attention** (column 1 of each panel). The mass is concentrated on the diagonal and on the first token (an attention-sink pattern, consistent with prior work on short inputs; Xiao et al., 2024). Because the encoder has not yet learned long-range composition (see Section 7.1), off-diagonal mass is sparse, but the diagonal is well-defined, confirming that the encoder produces non-trivial representations.

- **Decoder self-attention** (column 2 of each panel). Strictly lower-triangular, exactly as expected from the causal mask. The diagonal dominates on these short sentences. This remains a useful *correctness check* on the hand-rolled implementation: any masking bug would surface as non-zero mass above the diagonal, and we see none.

- **Decoder cross-attention** (column 3 of each panel). This is where the sub-training is most visible. On S1 (`duck !` → `kopf runter !`), cross-attention is *diffuse* across the source rather than peaking on the content token `duck`; on S4 (`beat it .` → `geh weg !`), the model places roughly equal mass on both English tokens. This is consistent with the failure mode observed in Section 5.3: the decoder has not learned to condition its output on the source strongly enough to produce the reference translation. The cross-attention map is therefore a *diagnostic* of sub-training as much as a tool for interpretability.

### 5.5.3 Limitations of the Visualization

Three limitations should be kept in mind when interpreting Figure 2:

1. **Single layer.** We report only the last layer (layer 4 of 4). Earlier layers tend to exhibit more diverse, lower-level attention patterns (anaphora, local syntax), and inspecting them is left to future work. The visualization script supports arbitrary layer selection via `--layer`.
2. **Per-head now available.** Head specialization can now be inspected via `python visualize_attention.py --per-head`. The script renders a single combined Nx(3·H) grid (`figures/per_head/attn_grid_layer4_per_head.{pdf,png}`), plus single-head variants via `--head N`. We generate one such figure below (Figure~\ref{fig:attn-per-head}); the full 8-head panel is in the appendix.
3. **Short sentences.** The visualized sentences are 1–2 tokens, which biases the analysis toward local patterns. Longer sentences should be visualized in the camera-ready version.

![Per-head attention for head 1 (0-indexed) across the four test sentences; columns are encoder self / decoder self / decoder cross-attention. Generated via `python visualize_attention.py --per-head --head 0 --num-sentences 4 --layer -1`.](../figures/per_head/attn_grid_layer4_head1.png){#fig:attn-per-head width=100%}

*[PLACEHOLDER: caption narrative for the per-head figure — describe which heads act as "previous-token" heads, which are "null", and which focus on cross-attention to a specific source token. To be filled in once the BPE-trained model is visualized; the word-level model here is too under-trained to draw meaningful conclusions.]*
