# Research Memory & Notebook

*This document serves as a persistent scratchpad and knowledge base for ongoing AGI/ASI research. Record important findings, mathematical insights, open questions, and summaries of experimental results here.*

---

## Current Focus

* Phase 1: Foundations and Mathematical Modeling. Currently investigating core AGI architecture components, focusing on mathematical formulation and testing hypotheses on small-scale/synthetic datasets. Specifically targeting non-linear transformation capabilities of basic Feed-Forward Networks (FFNs).

## Key Insights

* *(Date: 2024-05-24)* - Verified that a simple 2-layer FFN trained with standard backpropagation and a Mean Squared Error loss can effectively learn non-linear reasoning boundaries (such as the XOR problem). Confirms basic matrix algebra is sufficient for this non-linear component of our eventual architecture.

## Mathematical Notebook

* **Feed-Forward Network (2-Layer)**
  Let $X$ be the input matrix, $W_1, W_2$ be the weight matrices, and $b_1, b_2$ be the bias vectors.
  We define the activation function (sigmoid) as: $\sigma(x) = \frac{1}{1 + e^{-x}}$
  Forward Pass:
  $Z_1 = X W_1 + b_1$
  $A_1 = \sigma(Z_1)$
  $Z_2 = A_1 W_2 + b_2$
  $A_2 = \sigma(Z_2)$

  Loss (Mean Squared Error):
  $\mathcal{L} = \frac{1}{2N} \sum (A_2 - Y)^2$

## Experimental Summaries

* **Experiment `0001_train_tokenizer` (Success):** Learned a BPE vocabulary of 144 tokens on the baseline AGI dataset. Proves baseline tokenization functionality.
* **Experiment `0002_train_ffn_component` (Success):** Trained a small 2-layer FFN on a synthetic XOR dataset. The model successfully converged (Loss < 0.001) over 50k epochs using a learning rate of 1.0, proving manual backpropagation formulation is mathematically sound.

## Open Questions & Hypotheses

1. *(e.g., "Does scaling the depth of the network linearly correlate with reasoning capability on dataset Y?")*
2. *(e.g., "Can we formulate a strict mathematical bound on the error introduced by quantization technique Z?")*
