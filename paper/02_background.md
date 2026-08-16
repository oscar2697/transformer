# 2. Background

The Transformer architecture (Vaswani et al., 2017) replaces recurrent recurrence with self-attention mechanisms, enabling parallelized training and improved handling of long-range dependencies. We briefly review the mathematical formulation of each component that we reimplement, adopting the notation of the original paper.

## 2.1 Scaled Dot-Product Attention

Given queries $\mathbf{Q} \in \mathbb{R}^{T \times d_k}$, keys $\mathbf{K} \in \mathbb{R}^{T \times d_k}$, and values $\mathbf{V} \in \mathbb{R}^{T \times d_v}$, the scaled dot-product attention is defined as:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V} \tag{1}$$

The scaling factor $\sqrt{d_k}$ mitigates the vanishing gradient problem that arises when the dot products grow large in magnitude. A causal mask $\mathbf{M} \in \{\text{True}, \text{False}\}^{T \times T}$ is applied to the attention scores before the softmax, setting masked positions to $-\infty$ so that they receive zero weight.

## 2.2 Multi-Head Attention

Rather than performing a single attention function, the model linearly projects the queries, keys, and values into $h$ subspaces (heads) and applies attention in parallel:

$$\begin{aligned}
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}^O \\
\text{head}_i &= \text{Attention}(\mathbf{Q} \mathbf{W}_i^Q, \mathbf{K} \mathbf{W}_i^K, \mathbf{V} \mathbf{W}_i^V)
\end{aligned} \tag{2}$$

where $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $\mathbf{W}^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$. In our implementation, we set $d_k = d_v = d_{\text{model}} / h$.

## 2.3 Positionwise Feed-Forward Networks

Each layer includes a positionwise fully connected feed-forward network applied independently to every token position:

$$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2 \tag{3}$$

The inner dimension $d_{\text{ff}}$ is typically $4 \times d_{\text{model}}$. Dropout is applied to the output of the ReLU activation.

## 2.4 Positional Encoding

Since self-attention is permutation-invariant, the architecture injects positional information via sinusoidal positional encodings added to the input embeddings:

$$\begin{aligned}
PE_{(pos, 2i)} &= \sin\!\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\!\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
\end{aligned} \tag{4}$$

Alternative formulations (relative positional encodings, rotary embeddings) are left to future work.

## 2.5 Encoder and Decoder Architecture

The encoder consists of $N$ identical layers, each comprising a multi-head self-attention sub-layer followed by a positionwise feed-forward sub-layer, with residual connections and layer normalization applied around each sub-layer. The decoder mirrors this structure but adds a cross-attention layer that attends over the encoder output, with an additional causal mask to prevent attending to future positions.

Our implementation follows the post-layer-normalization convention of Vaswani et al. (2017), where layer normalization is applied after the residual addition. All weight tensors are initialized with Xavier uniform initialization.

*[PLACEHOLDER: transformer_architecture.png — High-level diagram of the Transformer encoder-decoder architecture, adapted from Vaswani et al. (2017).]*
