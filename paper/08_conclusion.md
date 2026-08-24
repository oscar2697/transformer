# 8. Conclusion

We presented a complete, hand-rolled PyTorch implementation of the Transformer architecture, validated on the **Tatoeba English → German** sentence-pair collection (~324k training pairs; test set n = 3,312). Every component—MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding, EncoderLayer, DecoderLayer, and the full Transformer—is implemented using only primitive tensor operations, without `nn.MultiheadAttention`, `nn.Transformer`, or `torchtext`. The implementation is accompanied by a self-contained data pipeline with word-level and SentencePiece BPE tokenization, length-bucketed batching, and a suite of 17 unit tests that serve as a formal correctness contract.

The experimental narrative of this paper follows a diagnosis-and-correction arc. An initial word-level Post-LN control run, stopped after 8 epochs, collapsed to BLEU = 0.38 at validation perplexity 149.84; a structured analysis showed this failure to be one of sub-training rather than architecture—the validation loss was still descending, all masking contracts held, and all unit tests passed—while its outputs collapsed to a handful of recurring hypotheses and its cross-attention maps remained diffuse. Three corrections motivated by this diagnosis were adopted in the final configuration: pre-layer normalization, a corrected learning-rate warmup schedule (4,000 steps), and subword tokenization (SentencePiece BPE, 8k pieces per language). With these changes and a 25-epoch budget at batch size 64 (≈ 2.5 h on a single NVIDIA T4), the same hand-rolled implementation reaches a corpus-level **BLEU = 44.00 greedy / 45.30 with beam search (b=4)** and **chrF2 = 62.41 / 63.37** at validation perplexity 9.29, competitive with published results on this corpus scale; it also outperforms an `nn.Transformer` baseline trained on identical data and architecture (BLEU 38.30), although that margin mixes implementation effects with training-schedule differences and requires a matched-schedule comparison for a clean attribution.

Because the implementation exposes attention weights as first-class outputs of the forward pass, we rendered per-sentence heatmaps of encoder self-attention, decoder self-attention, and decoder cross-attention. The structural checks hold throughout—decoder self-attention is strictly lower-triangular as required by the causal mask—and the contrast between checkpoints is itself informative: the under-trained control produces diffuse cross-attention maps, whereas the final checkpoint exhibits sharp, roughly monotonic English-to-German alignment peaks on short test sentences (e.g., mass concentrating on *draw* when generating *zeichnen*).

The remaining limitations are acknowledged rather than hidden: results are single-seed, the corpus is small, the baseline comparison is not fully schedule-matched, and most controlled ablations remain future work, alongside multi-seed replication, longer-sequence attention analysis, and additional language pairs. We emphasize that the primary contribution of this work is not a new architecture or a superior translation system, but a transparent, auditable reference implementation whose development process—including its failures—is documented end to end, making it suitable as a teaching resource in graduate-level NLP courses, as a correctness baseline, and as a platform for interpretability research.

## Declaration of AI Assistance

In the interest of transparency, we disclose that the codebase was developed with the assistance of large language models (MiniMax, Muse Spark 1.2, and ox-alpha) under continuous human direction: the models contributed code drafting, documentation, and debugging support, while the author designed the experiments, specified all architectural decisions, executed every training and evaluation run on Google Colab, verified the reported results, and wrote the manuscript. The author takes full responsibility for the final content of this paper.

---

# References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30.

[2] Elliott, D., Frank, S., Sima'an, K., & Specia, L. (2016). Multi30k: Multilingual English-German Image Descriptions. *Proceedings of the 5th Workshop on Vision and Language*, 70–74.

[3] Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). Bleu: a Method for Automatic Evaluation of Machine Translation. *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, 311–318.

[4] Popović, M. (2015). chrF: Character N-gram F-Score for Automatic MT Evaluation. *Proceedings of the 10th Workshop on Statistical Machine Translation*, 392–395.

[5] Post, M. (2018). A Call for Clarity in Reporting BLEU Scores. *Proceedings of the 3rd Conference on Machine Translation*, 186–191.

[6] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception Architecture for Computer Vision. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2818–2826.

[7] Popel, M., & Bojar, O. (2018). Training Tips for the Transformer Model. *The Prague Bulletin of Mathematical Linguistics*, 110, 63–87.

[8] Ott, M., Edunov, S., Grangier, D., & Auli, M. (2018). Scaling Neural Machine Translation. *Proceedings of the 3rd Conference on Machine Translation*, 1–9.

[9] Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1715–1725.

[10] Kudo, T., & Richardson, J. (2018). SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing. *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, 66–71.

[11] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics*, 4171–4186.

[12] Bojar, O., Chatterjee, R., Federmann, C., Graham, Y., Haddow, B., Huck, M., ... & Zampieri, M. (2017). Findings of the 2017 Conference on Machine Translation (WMT17). *Proceedings of the 2nd Conference on Machine Translation*, 169–214.

[13] Press, O., & Smith, N. A. (2018). You May Not Need Attention. *arXiv preprint* arXiv:1810.04805.

[14] Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). Generating Long Sequences with Sparse Transformers. *arXiv preprint* arXiv:1904.10509.

[15] Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). Reformer: The Efficient Transformer. *International Conference on Learning Representations*.

[16] Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q. V., & Salakhutdinov, R. (2019). Transformer-XL: Attentive Language Models beyond a Fixed-Length Context. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, 2978–2988.

[17] Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). Efficient Streaming Language Models with Attention Sinks. *International Conference on Learning Representations*.
