# 3. Implementation

## 3.1 Design Philosophy

The primary goal of this implementation is transparency over performance. Every matrix multiplication, every softmax, and every mask operation is expressed explicitly in PyTorch tensor operations. We deliberately avoid `nn.MultiheadAttention` and `nn.Transformer` so that each computational step can be inspected, unit-tested, and modified without navigating library internals.

A secondary goal is reproducibility. All random seeds are managed through a centralized seeding utility (`SEED = 42` in `config.py`). Hyperparameters are declared in a single `config.py` file, and the full training log (per-step losses, validation metrics, wall-clock per epoch, and a flag indicating whether validation improved) is persisted to a JSON Lines file (`metrics.jsonl`) alongside the best model checkpoint. The training script also accepts optional flags for label smoothing, weight tying, and the warmup schedule, so that ablation studies can be run without code changes.

A third goal is observability of internal representations. Because the implementation exposes attention weights as first-class outputs (`return_attention=True` in the forward pass), it is possible to extract encoder self-attention, decoder self-attention, and decoder cross-attention tensors from any layer and any head for post-hoc analysis. Section 5.5 exploits this capability to study the interpretability of the learned attention maps.

## 3.2 Module Descriptions

### 3.2.1 MultiHeadAttention

The `MultiHeadAttention` module computes scaled dot-product attention across $h$ heads. For each head $i$, we project the input to $d_k$ dimensions, compute attention with the query-key-value triplet, and concatenate the head outputs before a final linear projection. A dropout layer is applied to the attention weights before the weighted sum over values. Listing 1 provides pseudocode for the forward pass.

```pseudocode
# Input: query, key, value (batch, seq_len, d_model)
#        mask (batch, seq_len, seq_len) or (seq_len, seq_len)
Q = linear_q(query)   # (batch, seq_len, d_k)
K = linear_k(key)     # (batch, seq_len, d_k)
V = linear_v(value)   # (batch, seq_len, d_v)

# Reshape for multi-head: (batch, heads, seq_len, d_k)
Q = Q.view(batch, seq_len, h, d_k).transpose(1, 2)
K = K.view(batch, seq_len, h, d_k).transpose(1, 2)
V = V.view(batch, seq_len, h, d_v).transpose(1, 2)

# Scaled dot-product attention
scores = (Q @ K.transpose(-2, -1)) / sqrt(d_k)
scores = scores.masked_fill(mask == True, -1e9)
attn_weights = softmax(scores, dim=-1)
attn_weights = dropout(attn_weights, p=dropout)

# Weighted sum over values
context = attn_weights @ V
context = context.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
output = linear_o(context)
return output, attn_weights
```

*Listing 1: Pseudocode for the multi-head attention forward pass. The mask convention follows PyTorch's standard: `True` indicates positions to be masked.*

When `return_attention=True`, the module returns the per-head attention weight tensor of shape `(batch, heads, seq_len, seq_len)`, which is consumed by `visualize_attention.py` to render per-sentence heatmaps.

### 3.2.2 PositionwiseFeedForward

The feed-forward sub-layer applies two linear transformations with a GELU activation (we use GELU as the default, with ReLU as a configurable alternative) and dropout:

```pseudocode
# Input: x (batch, seq_len, d_model)
x = dropout(gelu(linear_ff1(x)))  # (batch, seq_len, d_ff)
output = linear_ff2(x)            # (batch, seq_len, d_model)
return output
```

### 3.2.3 PositionalEncoding

Positional encodings are generated analytically using the sinusoidal functions defined in Equation (4). They are added directly to the token embeddings at the bottom of both the encoder and decoder stacks. Because the encoding is fixed (not learned), it is generated once at initialization and reused across all forward passes.

### 3.2.4 EncoderLayer and DecoderLayer

Each encoder layer wraps a `MultiHeadAttention` sub-layer and a `PositionwiseFeedForward` sub-layer, each preceded by a dropout and followed by a residual addition and layer normalization. The decoder layer adds a cross-attention sub-layer between the self-attention and feed-forward sub-layers. All sub-layers produce outputs of shape `(batch, seq_len, d_model)`.

### 3.2.5 Full Transformer

The `Transformer` module assembles the encoder and decoder stacks with shared embedding layers (configurable, disabled by default), positional encodings, and a final linear projection head over the target vocabulary. Both layer-normalization conventions are supported: the initial word-level diagnostic run used **post-layer normalization** (matching the original Vaswani et al., 2017 architecture), whereas the final configuration adopts **pre-layer normalization** (`norm_first=True`), with a final layer norm applied to the encoder and decoder outputs. The stability motivation for this choice is analyzed in Sections 5.1 and 6.4.

## 3.3 Training Details

The model is trained with the Adam optimizer ($\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$), a Noam learning rate schedule, and gradient clipping at a maximum norm of 1.0. Label smoothing is configurable via `LABEL_SMOOTHING` and was set to 0.1 for all reported runs. No weight tying is used in the default configuration. The final main run uses 4,000 warmup steps and an effective batch size of 64; the earlier word-level diagnostic run used 2,000 warmup steps and a batch size of 32 (Section 5.1). Table 1 summarizes the hyperparameters and results of the main run.

| Parameter | Value |
|---|---|
| Task / direction | Tatoeba EN→DE |
| Tokenization | SentencePiece BPE, 8k pieces per language |
| Layer norm convention | Pre-LN (`norm_first=True`) |
| $d_{\text{model}}$ | 256 |
| $d_{\text{ff}}$ | 1024 |
| Attention heads ($h$) | 8 |
| Encoder layers ($N$) | 4 |
| Decoder layers ($M$) | 4 |
| Dropout | 0.1 |
| Label smoothing $\epsilon$ | 0.1 |
| Warmup steps | 4000 |
| Batch size | 64 |
| Max sequence length | 100 |
| Epochs trained | 25 |
| Optimizer | Adam ($\beta_1=0.9$, $\beta_2=0.98$) |
| LR schedule | Noam |
| Gradient clip | $\lVert g \rVert \leq 1.0$ |
| Random seed | 42 |
| Total parameters | $\approx$ 13.5 M |
| Training device | Google Colab GPU (NVIDIA T4) |
| Training time | $\approx$ 2.5 h |
| Best epoch (val_loss) | epoch 22 of 25 |
| Best val_loss | 2.2288 |
| Best val_ppl | 9.29 |
| BLEU (test, greedy) | 44.00 (71.3/48.9/37.3/29.3, BP = 0.995) |
| BLEU (test, beam = 4) | 45.30 (72.4/50.7/39.0/30.9, BP = 0.987) |
| chrF2 (test, greedy / beam) | 62.41 / 63.37 |
| Test set size (n) | 3,312 |

*Table 1: Hyperparameters and final results for the main run: hand-rolled Transformer + SentencePiece BPE (Pre-LN). Validation perplexity is computed as $\exp(\text{val\_loss})$. BLEU and chrF2 are corpus-level sacrebleu scores on the Tatoeba EN→DE test set, computed with sacrebleu (see Section 4.2); the final epoch (25) reached val_loss = 2.2393 (val_ppl 9.39). The word-level Post-LN diagnostic run referenced throughout the paper used batch 32, warmup 2000, and 8 epochs (Section 4.3).*

## 3.4 Attention as a First-Class Output

A distinctive feature of this implementation—rare in standard PyTorch boilerplate—is that attention weights are returned explicitly by the model forward pass rather than being computed inside an opaque submodule. Concretely, `model(src, tgt, src_pad_mask, tgt_pad_mask, return_attention=True)` returns a 4-tuple `(logits, enc_attn, self_attn, cross_attn)`, where each `*_attn` value is a Python list of length $N$ (the number of layers), and each element is a tensor of shape `(batch, heads, seq_len, seq_len)`:

- `enc_attn[i]` — encoder self-attention for layer $i$, shape `(batch, h, src_len, src_len)`;
- `self_attn[i]` — decoder self-attention for layer $i$, shape `(batch, h, tgt_len, tgt_len)`, with the upper-triangular causal mask applied before softmax;
- `cross_attn[i]` — decoder cross-attention for layer $i$, shape `(batch, h, tgt_len, src_len)`.

The masks themselves are also accessible via the public API (`MultiHeadAttention._make_mask` for causal masks; `dataset.build_padding_mask` for source and target padding masks), which makes it possible to verify that masking is applied at the correct positions before the softmax.

This design choice makes the implementation suitable for interpretability analysis. The script `visualize_attention.py` (described in Section 5.5) uses this API to render per-sentence attention heatmaps for any combination of layer index, head index, and attention type (encoder self, decoder self, decoder cross) without modifying the model source.

## 3.5 Data Pipeline

The dataset is downloaded automatically from the official Tatoeba mirror at `https://www.manythings.org/anki/deu-eng.zip` (~12 MB). Sentences are tokenized using regex-based splitting and converted to lowercase. Two tokenization modes are supported, selected by `config.TOKENIZER`:

- **`"word"` (default)** — a vocabulary is built from the training split with a minimum frequency threshold of 2; tokens below this threshold are mapped to the UNK token. For the Tatoeba EN→DE corpus this yields 12,843 English and 23,568 German tokens.
- **`"bpe"`** — SentencePiece byte-pair encoding (Kudo & Richardson, 2018; used by the final main run) is trained on the source and target training splits separately, with a fixed vocab size of 8,000 subwords per side. The 4 special tokens (`<unk>`, `<pad>`, `<sos>`, `<eos>`) keep the same indices as in the word-level pipeline (`UNK=0`, `PAD=1`, `SOS=2`, `EOS=3`), so `model.py`, `train.py`, `evaluate.py`, and `translate.py` need no changes when switching modes.

Sentences are bucketed by length to minimize padding overhead within each batch. Dynamic padding is applied per batch, with a padding token index of 1 (`PAD_IDX`). The maximum sequence length is 100 tokens.

*[PLACEHOLDER: vocab_distribution.png — Distribution of token frequencies in the Tatoeba vocabulary.]*

## 3.6 Baseline Module: `nn.Transformer`

For fair comparison with PyTorch's optimized implementation (Section 6.7), we provide a baseline wrapper around `torch.nn.Transformer` in `baseline_nn.py`. The wrapper exposes the same interface as the hand-rolled model (`forward`, `greedy_decode`, `count_parameters`) and uses the same `return_attention`-style contract where possible (PyTorch's `nn.Transformer` does not return attention weights by default; we therefore do not include the hand-rolled interpretability analysis for the baseline).

The baseline uses `nn.Transformer(batch_first=True, norm_first=True)` — that is, **pre-LayerNorm**, matching the LN convention of the final hand-rolled configuration (post-LN is the original Vaswani et al., 2017 convention used by the initial word-level diagnostic run). Both implementations expose the LN convention as a configurable option, and Section 6.4 discusses the stability difference between them. The two implementations share the architecture, BPE data pipeline, and seed; their training schedules differ (Section 6.7), so any BLEU gap between them is attributable to the combination of implementation details (mask convention, dropout ordering, attention scaling) and schedule differences rather than to the implementation alone.
