# 4. Experimental Setup

## 4.1 Dataset

We evaluate on the **Tatoeba English → German** sentence-pair collection, a community-curated corpus distributed under CC-BY through the ManyThings/Anki distribution (the same source data family as Tatoeba.org). The raw file `deu.txt` contains ~331,266 English–German sentence pairs in TSV format (`EN\tDE\tattribution`). After dropping pairs with empty cells or replacement characters (U+FFFD introduced by mixed latin-1 / UTF-8 encoding in the raw file) and filtering by source/target token length (max 100 tokens after target `<sos>`/`<eos>` insertion), the corpus is reduced to ~331,265 pairs, which we split deterministically into **train / val / test = 98% / 1% / 1%** (seed = 42), giving 324,641 / 3,312 / 3,312 pairs respectively.

All data is downloaded automatically at runtime from `http://www.manythings.org/anki/deu-eng.zip` (~12 MB). The downloader sets an explicit browser-style User-Agent header (the server occasionally returns HTTP 406 to default Python agents) and a 60 MB safety cap. No manual preprocessing or external tokenizers are used. Tokenization is performed with a regex-based pattern that splits on whitespace and punctuation, followed by lowercasing. The resulting vocabulary sizes are **12,843 tokens for English** and **23,568 tokens for German** at the word level, with `min_freq = 2` and tokens below that threshold mapped to UNK.

The Tatoeba distribution is a substantial upgrade over the Multi30k captions we previously reported on (Elliott et al., 2016), both in size (~11x more pairs) and in domain coverage (general-purpose sentences rather than image captions). This means the model is exposed to a much broader vocabulary, including technical and conversational German, and is therefore usable for translation of sentences outside the Flickr-image-caption domain.

## 4.2 Evaluation Metrics

We report two standardized metrics for translation quality:

- **BLEU** (Papineni et al., 2002): computed at the corpus level using sacrebleu (Post, 2018) with its default tokenization, which is compatible with standard WMT evaluation protocols. BLEU measures n-gram precision with a brevity penalty.
- **chrF2** (Popović, 2015): also computed via sacrebleu. chrF2 evaluates character n-gram overlap with a beta parameter of 2, providing a more robust measure for morphologically rich languages such as German.

Perplexity on the validation set is reported as a sanity check during training, computed as $\exp(\text{val\_loss})$.

## 4.3 Training Protocol

Models are trained on a single Google Colab GPU (NVIDIA T4), with automatic device placement via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. All experiments use a fixed random seed of 42 for reproducibility. The final main configuration (hand-rolled + BPE, Pre-LN) uses an effective batch size of 64 sequences per step and a Noam learning rate schedule with a warmup of 4,000 steps following Vaswani et al. (2017); it is trained for the full 25 epochs without early stopping, and the best model is selected post-hoc by lowest validation loss (epoch 22). Gradient clipping is applied at a maximum norm of 1.0. An earlier word-level Post-LN control run—batch size 32, warmup 2,000—was stopped after 8 epochs and is retained throughout this paper as a *diagnostic* of sub-training rather than as a competitive result (Sections 5.1 and 7.3).

## 4.4 Baseline

As a reference baseline, we train a Transformer using PyTorch's native `nn.Transformer` module (`norm_first=True`, i.e., Pre-LN) with the architecture matched to our main configuration (`d_model=256`, 8 heads, 4 encoder layers, 4 decoder layers, `d_ff=1024`, label smoothing 0.1, seed 42) and the same SentencePiece BPE data pipeline. The baseline is trained for 25 epochs with batch size 32 and warmup 2,000, whereas the hand-rolled main run uses batch size 64 and warmup 4,000; the comparison therefore measures the joint effect of implementation and training schedule. Full results are reported in Section 6.7.

## 4.5 Hardware and Software

All trained experiments reported in this paper were executed on a single Google Colab GPU (NVIDIA T4). Wall-clock times: the word-level Post-LN diagnostic run completed in **≈ 56 minutes** for 8 epochs; the baseline + BPE configuration required ≈ 2.3 h and the hand-rolled + BPE (Pre-LN) main run ≈ 2.5 h, both for 25 epochs. The code has no dependencies beyond the standard scientific Python stack plus `sacrebleu` for evaluation. PyTorch version is 2.0 or higher; the implementation does not depend on any feature specific to a particular PyTorch release.

*[PLACEHOLDER: tatoeba_sample.png — Sample source (English) and reference (German) sentence pairs from the test set.]*
