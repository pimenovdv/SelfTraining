# Project SelfTraining: Path to AGI & ASI

## Overview

This repository serves as a research sandbox dedicated to the systematic exploration, mathematical modeling, and eventual training of an Artificial General Intelligence (AGI) and Artificial Superintelligence (ASI).

## Global Objective

To design and train a foundational model capable of human-level general intelligence across diverse domains (AGI), with the architectural capacity for recursive self-improvement leading to superintelligence (ASI).

## Current Focus

We are currently in **Phase 1: Foundations and Mathematical Modeling**.
The immediate goal is not to train massive language models blindly. Instead, the focus is strictly on:

* Developing rigorous mathematical models for intelligence components.
* Formulating and testing specific architectural hypotheses through controlled, small-scale experiments.
* Understanding and documenting scaling laws for fundamental AI building blocks.
* Building a robust, mathematically justified foundation before attempting any large-scale integration.

## Repository Architecture

* **`README.md`**: High-level overview, goals, and current status of the project.
* **`AGENTS.md`**: Strict operational guidelines, rules, and required workflows for AI researchers/agents operating within this repository.
* **`memory.md`**: A living notebook for recording theoretical insights, mathematical derivations, open questions, and important observational data.
* **`todo.md`**: The strategic, phased roadmap charting the course from current foundational research to the ultimate ASI goal.
* **`docs/`** *(to be created during experiments)*: Directory for detailed, structured reports on individual experiments, documenting successes, failures, and necessary iterations.

## Guiding Philosophy

"Measure twice, cut once." Every major architectural decision must be preceded by theoretical justification and small-scale empirical validation.

## Quick Start: Training the Tokenizer

To demonstrate the foundational approach, this repository includes a simple script to train a Byte-Pair Encoding (BPE) tokenizer from scratch.

1.  **Ensure you have Python 3 installed.**
2.  **Run the training script:**
    ```bash
    python train_bpe_component.py
    ```
    This script will read the sample text data from `data/sample_text.txt`, perform BPE merges, and save the resulting vocabulary and merges to `models/tokenizer/`.
3.  **Adjusting hyperparameters:**
    You can customize the tokenizer training using command-line arguments:
    ```bash
    python train_bpe_component.py --data_path "data/sample_text.txt" --num_merges 200 --output_dir "models/my_tokenizer"
    ```

## Component Testing: AdaLN (Adaptive Layer Normalization)

Building upon Layer Normalization and generative models, we have implemented Adaptive Layer Normalization (AdaLN). This component tests the hypothesis that normalization parameters (gamma and beta) can be dynamically predicted from a conditioning input (e.g., timestep or class embedding) using linear projections. It is tested here with pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the AdaLN component test:**
    ```bash
    python train_adaln_component.py
    ```
    This script tests learning the dynamic generation of scale and shift parameters based on a conditioning input to match a target output over a synthetic dataset, utilizing manual backpropagation.

    You can adjust hyperparameters such as epochs and learning rate:
    ```bash
    python train_adaln_component.py --epochs 10000 --lr 0.1
    ```

## Component Testing: Feed-Forward Network (FFN)

As part of our **Phase 1: Foundations and Mathematical Modeling**, we are exploring individual components of potential AGI architectures. We have implemented a 2-layer Feed-Forward Network to test non-linear transformation hypotheses using purely mathematical operations (via NumPy).

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the FFN component test:**
    ```bash
    python train_ffn_component.py
    ```
    This script trains a simple FFN on the synthetic XOR dataset. It validates our hypothesis that a 2-layer network can successfully model non-linear boundaries. The script demonstrates manual forward and backward passes to rigorously verify the underlying mathematics.

    You can adjust hyperparameters such as hidden size and learning rate:
    ```bash
    python train_ffn_component.py --hidden_size 8 --epochs 20000 --lr 0.5
    ```

## Component Testing: Self-Attention

Building upon our mathematical models, we have implemented a Self-Attention mechanism to test its ability to learn relationships within sequences using pure matrix operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Self-Attention component test:**
    ```bash
    python train_attention_component.py
    ```
    This script trains a simple Self-Attention layer on a synthetic dataset, manually computing forward and backward passes to verify the underlying mathematics.

    You can adjust hyperparameters such as dimension key and learning rate:
    ```bash
    python train_attention_component.py --d_k 4 --epochs 10000 --lr 0.1
    ```

## Component Testing: Multi-Head Attention

Building upon our initial single-head Self-Attention, we have implemented Multi-Head Attention. This allows the model to jointly attend to information from different representation subspaces at different positions, verified using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Multi-Head Attention component test:**
    ```bash
    python train_multihead_attention_component.py
    ```
    This script trains a Multi-Head Attention layer on a synthetic dataset, manually computing forward and backward passes for multiple heads.

    You can adjust hyperparameters such as dimension model, number of heads, epochs, and learning rate:
    ```bash
    python train_multihead_attention_component.py --d_model 4 --num_heads 2 --epochs 10000 --lr 0.1
    ```

## Component Testing: Layer Normalization

Building towards a complete architecture, we have implemented Layer Normalization. This component helps stabilize training and is crucial for deep neural networks, tested here with pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Layer Normalization component test:**
    ```bash
    python train_layernorm_component.py
    ```
    This script tests learning the gamma and beta parameters of layer normalization on a synthetic dataset using manual backpropagation.

    You can adjust hyperparameters such as epochs and learning rate:
    ```bash
    python train_layernorm_component.py --epochs 10000 --lr 0.1
    ```

## Component Testing: Transformer Block

Building upon individual components, we have integrated Self-Attention, Feed-Forward Networks, and Layer Normalization into a single-layer Transformer Block. This tests the interaction of these components and residual connections using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Transformer Block component test:**
    ```bash
    python train_transformer_block_component.py
    ```
    This script tests learning the transformation over a synthetic sequence dataset using manual backpropagation through the entire block.

    You can adjust hyperparameters such as dimension model, keys, ffn, epochs, and learning rate:
    ```bash
    python train_transformer_block_component.py --d_model 4 --d_k 2 --d_ff 8 --epochs 20000 --lr 0.1
    ```

## Component Testing: Positional Encoding

As sequence models like Transformers lack recurrent or convolutional structures, they require explicit information about the order of sequence elements. We have implemented a mathematical formulation of Positional Encoding (using sine and cosine functions) to test its capabilities.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Positional Encoding component test:**
    ```bash
    python train_positional_encoding_component.py
    ```
    This script generates synthetic positional encodings and trains a simple linear layer via manual backpropagation to extract normalized absolute position indices, validating that the encodings contain linearly separable order information.

    You can adjust hyperparameters such as dimension model, sequence length, epochs, and learning rate:
    ```bash
    python train_positional_encoding_component.py --d_model 16 --seq_len 10 --epochs 5000 --lr 0.1
    ```

## Component Testing: Multi-Head Transformer Block

Building upon individual components, we have integrated Multi-Head Attention, Feed-Forward Networks, and Layer Normalization into a single-layer Transformer Block. This tests the interaction of these components and residual connections using pure mathematical operations, replacing the single-head attention with a more robust multi-head variant.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Multi-Head Transformer Block component test:**
    ```bash
    python train_multihead_transformer_block_component.py
    ```
    This script tests learning the transformation over a synthetic sequence dataset using manual backpropagation through the entire block.

    You can adjust hyperparameters such as dimension model, number of heads, ffn, epochs, and learning rate:
    ```bash
    python train_multihead_transformer_block_component.py --d_model 4 --num_heads 2 --d_ff 8 --epochs 20000 --lr 0.1
    ```

## Component Testing: Masked Self-Attention

Building upon the self-attention mechanism, we have implemented Masked Self-Attention. This is a crucial foundational building block for autoregressive models (like GPT). It restricts the attention mechanism from "looking ahead" at future tokens using a causal mask, verified using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Masked Self-Attention component test:**
    ```bash
    python train_masked_attention_component.py
    ```
    This script trains a simple Masked Self-Attention layer on a synthetic dataset, manually computing forward and backward passes.

    You can adjust hyperparameters such as dimension key, epochs, and learning rate:
    ```bash
    python train_masked_attention_component.py --d_k 4 --epochs 10000 --lr 0.1
    ```

## Component Testing: Cross-Attention

Building upon the self-attention mechanism, we have implemented Cross-Attention. This is a crucial foundational building block for encoder-decoder architectures (like the original Transformer for translation). It allows a target sequence to attend to a source sequence, verified using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Cross-Attention component test:**
    ```bash
    python train_cross_attention_component.py
    ```
    This script trains a simple Cross-Attention layer on synthetic target and source datasets, manually computing forward and backward passes.

    You can adjust hyperparameters such as dimension key, epochs, and learning rate:
    ```bash
    python train_cross_attention_component.py --d_k 2 --epochs 10000 --lr 0.1
    ```

## Component Testing: Decoder Block

Building upon Masked Self-Attention and Cross-Attention, we have integrated them with Feed-Forward Networks and Layer Normalization to form a complete Decoder Block. This is the core component of autoregressive sequence generation models.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Decoder Block component test:**
    ```bash
    python train_decoder_block_component.py
    ```
    This script tests learning the transformation over synthetic target and source sequences using manual backpropagation through the entire block.

    You can adjust hyperparameters such as dimension model, keys, ffn, epochs, and learning rate:
    ```bash
    python train_decoder_block_component.py --d_model 4 --d_k 2 --d_ff 8 --epochs 10000 --lr 0.1
    ```

## Component Testing: Full Encoder-Decoder Transformer

Building upon all previous components (Encoder blocks and Decoder blocks), we have integrated them into a full end-to-end Encoder-Decoder Transformer architecture. This tests the complete sequence-to-sequence mapping pipeline using pure mathematical operations, verifying that gradients flow correctly through both networks.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Full Encoder-Decoder Transformer component test:**
    ```bash
    python train_full_encoder_decoder_component.py
    ```
    This script tests learning the transformation from source sequences to target sequences via an end-to-end architecture using manual backpropagation.

    You can adjust hyperparameters such as dimension model, keys, ffn, epochs, and learning rate:
    ```bash
    python train_full_encoder_decoder_component.py --d_model 4 --d_k 2 --d_ff 8 --epochs 20000 --lr 0.1
    ```

## Component Testing: RMSNorm (Root Mean Square Normalization)

Building upon the basic component research, we have implemented RMSNorm. This component tests the hypothesis that removing mean-centering (compared to LayerNorm) still allows the model to learn a stable scale parameter (gamma), while being computationally simpler. It is tested here with pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the RMSNorm component test:**
    ```bash
    python train_rmsnorm_component.py
    ```
    This script tests learning the gamma parameter of RMS normalization on a synthetic dataset using manual backpropagation.

    You can adjust hyperparameters such as epochs and learning rate:
    ```bash
    python train_rmsnorm_component.py --epochs 10000 --lr 0.1
    ```

## Component Testing: SwiGLU (Swish-Gated Linear Unit)

Building upon our initial components, we have implemented the SwiGLU activation. This component tests the hypothesis that advanced gating mechanisms with non-linear activation functions provide richer representational capacity than standard ReLUs or Sigmoids. It is commonly used in state-of-the-art LLMs.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the SwiGLU component test:**
    ```bash
    python train_swiglu_component.py
    ```
    This script tests learning the transformation over a synthetic reasoning dataset (XOR) using pure mathematical operations, verifying manual forward and backward passes.

    You can adjust hyperparameters such as hidden size, epochs, and learning rate:
    ```bash
    python train_swiglu_component.py --hidden_size 8 --epochs 50000 --lr 1.0
    ```

## Component Testing: RoPE (Rotary Positional Embeddings)

Building upon our initial basic Positional Encoding, we have implemented Rotary Positional Embeddings. This component tests the hypothesis that injecting positional information into the query and key representations via rotations allows the attention mechanism to better learn relative distances. It is tested here with pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the RoPE component test:**
    ```bash
    python train_rope_component.py
    ```
    This script tests injecting positional information into attention score queries and keys using pure mathematical operations and manual backpropagation.

    You can adjust hyperparameters such as sequence length, dimension model, epochs, and learning rate:
    ```bash
    python train_rope_component.py --seq_len 10 --d_model 16 --epochs 5000 --lr 0.1
    ```

## Component Testing: Mixture of Experts (MoE)

Building upon the basic FFN component research, we have implemented Mixture of Experts (MoE). This component tests the hypothesis that a router network can successfully learn to distribute inputs across multiple specialized sub-networks (experts) using basic matrix operations and manual backpropagation.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the MoE component test:**
    ```bash
    python train_moe_component.py
    ```
    This script trains a Mixture of Experts block on a synthetic dataset using pure mathematical operations, verifying manual forward and backward passes.

    You can adjust hyperparameters such as number of experts, hidden size, epochs, and learning rate:
    ```bash
    python train_moe_component.py --num_experts 4 --hidden_size 8 --epochs 10000 --lr 0.1
    ```

## Component Testing: Grouped-Query Attention (GQA)

Building upon Multi-Head Attention, we have implemented Grouped-Query Attention. This component tests the hypothesis that sharing key and value heads across multiple query heads significantly reduces computational and memory overhead during inference while maintaining competitive performance. It is tested here with pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the GQA component test:**
    ```bash
    python train_gqa_component.py
    ```
    This script trains a Grouped-Query Attention layer on a synthetic dataset using pure mathematical operations, verifying manual forward and backward passes, including the gradient aggregation for shared heads.

    You can adjust hyperparameters such as dimension model, number of query heads, number of kv heads, epochs, and learning rate:
    ```bash
    python train_gqa_component.py --d_model 4 --num_heads 4 --num_kv_heads 2 --epochs 10000 --lr 0.1
    ```

## Component Testing: LoRA (Low-Rank Adaptation)

Building upon our component research, we have implemented Low-Rank Adaptation (LoRA). This component tests the hypothesis that fine-tuning can be made highly parameter-efficient by freezing base weights and learning only small, low-rank matrices that are injected into the computation. It is tested here with pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the LoRA component test:**
    ```bash
    python train_lora_component.py
    ```
    This script tests learning the low-rank adaptation matrices (A and B) while keeping the base weight matrix frozen, over a synthetic dataset, utilizing manual backpropagation to ensure gradients only flow into the adapter matrices.

    You can adjust hyperparameters such as rank, alpha, epochs, and learning rate:
    ```bash
    python train_lora_component.py --r 2 --alpha 1.0 --epochs 5000 --lr 0.1
    ```

## Component Testing: AdamW Optimizer

Building upon our basic component research using SGD, we have implemented the AdamW Optimizer. This component tests the hypothesis that adaptive moment estimation with decoupled weight decay accelerates convergence on non-linear datasets compared to standard SGD. It is tested here with pure mathematical operations on the XOR problem.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the AdamW Optimizer component test:**
    ```bash
    python train_adamw_component.py
    ```
    This script tests parameter updates with adaptive learning rates, moment tracking, bias correction, and decoupled weight decay using pure mathematical operations.

    You can adjust hyperparameters such as hidden size, epochs, learning rate, and weight decay:
    ```bash
    python train_adamw_component.py --hidden_size 8 --epochs 5000 --lr 0.01 --weight_decay 0.01
    ```

## Component Testing: GELU (Gaussian Error Linear Unit)

Building upon our initial components, we have implemented the GELU activation function. This component tests the hypothesis that advanced activation functions which incorporate stochastic regularization properties provide richer representational capacity and better convergence compared to standard ReLUs or Sigmoids. It is commonly used in state-of-the-art architectures like BERT and GPT.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the GELU component test:**
    ```bash
    python train_gelu_component.py
    ```
    This script tests learning the transformation over a synthetic reasoning dataset (XOR) using pure mathematical operations, verifying manual forward and backward passes.

    You can adjust hyperparameters such as hidden size, epochs, and learning rate:
    ```bash
    python train_gelu_component.py --hidden_size 8 --epochs 50000 --lr 1.0
    ```

## Component Testing: Scaling Laws

Building upon our component research, we have implemented a script to study empirical scaling laws. This tests the hypothesis that model performance (loss) scales predictably with the number of parameters following a power law, providing a foundation for predicting resource requirements.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Scaling Laws component test:**
    ```bash
    python train_scaling_laws_component.py
    ```
    This script trains a Feed-Forward Network on a synthetic dataset across varying hidden layer sizes, performing linear regression to estimate the power-law exponent.

    You can adjust hyperparameters such as epochs and learning rate:
    ```bash
    python train_scaling_laws_component.py --epochs 2000 --lr 0.01
    ```

## Component Testing: Inverted Dropout

Building upon our component research, we have implemented Inverted Dropout. This component tests the hypothesis that randomly dropping neuron activations during training prevents complex co-adaptations and reduces overfitting. It is verified here using pure mathematical operations, including scaling during the forward pass to maintain expected values during inference.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Inverted Dropout component test:**
    ```bash
    python train_dropout_component.py
    ```
    This script tests learning a non-linear dataset while randomly zeroing out hidden layer activations during the forward pass and appropriately masking gradients during the backward pass.

    You can adjust hyperparameters such as hidden size, dropout rate, epochs, and learning rate:
    ```bash
    python train_dropout_component.py --hidden_size 16 --drop_rate 0.2 --epochs 100000 --lr 1.0
    ```

## Component Testing: Direct Preference Optimization (DPO)

Building upon our component research, we have implemented Direct Preference Optimization (DPO). This component tests the hypothesis that a language model policy can be directly aligned to human preferences by optimizing the log-ratio of policy to reference probabilities, bypassing the need for a separate reward model. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the DPO component test:**
    ```bash
    python train_dpo_component.py
    ```
    This script trains a simple policy over a synthetic preference dataset using pure mathematical operations, verifying manual forward and backward passes.

    You can adjust hyperparameters such as dimension model, epochs, learning rate, and beta:
    ```bash
    python train_dpo_component.py --d_model 4 --epochs 5000 --lr 0.1 --beta 0.1
    ```

## Component Testing: Quantization-Aware Training (QAT)

Building upon our component research, we have implemented Quantization-Aware Training (QAT). This component tests the hypothesis that models can adapt to the noise introduced by lower-precision weights (like 8-bit integers) during training, maintaining performance while reducing memory footprint. It relies on absolute maximum (absmax) quantization and the Straight-Through Estimator (STE) for backpropagation.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the QAT component test:**
    ```bash
    python train_quantization_component.py
    ```
    This script tests Quantization-Aware Training on a synthetic dataset by maintaining full-precision weights while simulating 8-bit quantization during the forward pass.

    You can adjust hyperparameters such as hidden size, epochs, and learning rate:
    ```bash
    python train_quantization_component.py --hidden_size 8 --epochs 50000 --lr 1.0
    ```

## Component Testing: GRU (Gated Recurrent Unit)

Building upon our simple RNN component, we have implemented a GRU (Gated Recurrent Unit). This component tests the hypothesis that advanced gating mechanisms (update and reset gates) can successfully control the flow of information over time, effectively mitigating the vanishing gradient problem and allowing for robust sequential memory retention. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the GRU component test:**
    ```bash
    python train_gru_component.py
    ```
    This script tests learning a sequential reasoning dataset (XOR across time steps) using pure mathematical operations, verifying manual forward passes through complex gates and Backpropagation Through Time (BPTT).

    You can adjust hyperparameters such as hidden size, epochs, and learning rate:
    ```bash
    python train_gru_component.py --hidden_size 16 --epochs 50000 --lr 1.0
    ```

## Component Testing: Contrastive Learning (InfoNCE)

Building upon our component research, we have implemented a Contrastive Learning component using the InfoNCE loss. This tests the hypothesis that a dual-encoder (two-tower) model can successfully learn to align representations of paired inputs (different views of the same concept) into a shared continuous vector space while pushing apart un-paired concepts. It utilizes L2 normalized representations and temperature-scaled cosine similarities, all verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Contrastive Learning component test:**
    ```bash
    python train_contrastive_component.py
    ```
    This script tests learning to map paired input domains to a shared representation space via manual forward and backward passes through temperature-scaled cosine similarity logic.

    You can adjust hyperparameters such as output dimension, temperature, epochs, and learning rate:
    ```bash
    python train_contrastive_component.py --out_dim 4 --tau 0.1 --epochs 5000 --lr 0.01
    ```

## Component Testing: Kolmogorov-Arnold Network (KAN)

Building upon our component research, we have implemented a Kolmogorov-Arnold Network (KAN) component. This tests the hypothesis that placing learnable activation functions on edges rather than nodes (verifying the Kolmogorov-Arnold representation theorem) can successfully learn non-linear boundaries. It utilizes Gaussian Radial Basis Functions (RBFs) over grids for the edge functions and is verified using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the KAN component test:**
    ```bash
    python train_kan_component.py
    ```
    This script tests learning a non-linear dataset (XOR) via manual forward and backward passes using Einstein summation (`einsum`) to handle the multi-dimensional tensor gradients of edge basis functions.

    You can adjust hyperparameters such as hidden dimension, grid size, epochs, and learning rate:
    ```bash
    python train_kan_component.py --hidden_dim 4 --grid_size 5 --epochs 50000 --lr 0.1
    ```

## Component Testing: Linear Attention

Building upon our component research, we have implemented Linear Attention. This component tests the hypothesis that by replacing the softmax attention matrix with a kernel feature map (like ELU + 1) to ensure positivity, we can exploit the associativity of matrix multiplication to compute the output as $Q (K^T V)$. This effectively reduces sequence dimension complexity from $O(N^2)$ to $O(N)$, mitigating the bottleneck of standard attention for long sequences.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Linear Attention component test:**
    ```bash
    python train_linear_attention_component.py
    ```
    This script tests learning a sequence dataset using the $O(N)$ factored matrix multiplications and pure mathematical operations, verifying manual forward and backward passes.

    You can adjust hyperparameters such as dimension of keys, epochs, and learning rate:
    ```bash
    python train_linear_attention_component.py --d_k 2 --epochs 10000 --lr 0.1
    ```

## Component Testing: Sparse Autoencoder (SAE)

Building upon our component research, we have implemented a Sparse Autoencoder (SAE). This component tests the hypothesis that learning a sparse, overcomplete representation of data can help in mechanistic interpretability by disentangling complex representations into interpretable features. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Sparse Autoencoder component test:**
    ```bash
    python train_sae_component.py
    ```
    This script tests learning a sparse latent space using an L1 penalty on the hidden activations and Mean Squared Error for reconstruction, utilizing manual backpropagation.

    You can adjust hyperparameters such as hidden dimension, L1 coefficient, epochs, and learning rate:
    ```bash
    python train_sae_component.py --d_hidden 64 --l1_coeff 0.1 --epochs 10000 --lr 0.01
    ```

## Component Testing: Vector Quantized Variational Autoencoder (VQ-VAE)

Building upon our generative model research, we have implemented a Vector Quantized Variational Autoencoder (VQ-VAE). This component tests the hypothesis that a discrete representation of continuous latent spaces through vector quantization using a codebook can effectively model discrete modalities like categorical features, using the Straight-Through Estimator (STE) for backpropagation.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the VQ-VAE component test:**
    ```bash
    python train_vqvae_component.py
    ```
    This script tests a mathematical model of VQ-VAE in pure NumPy, testing discrete representation learning using a codebook and the Straight-Through Estimator (STE) for backpropagation.

## Component Testing: Knowledge Distillation

Building upon our model optimization research, we have implemented Knowledge Distillation (KD). This component tests the hypothesis that a smaller, faster "student" model can achieve comparable performance by learning to match the softened probability distribution of a larger, more capable "teacher" model via KL Divergence, in addition to standard hard-label training. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Knowledge Distillation component test:**
    ```bash
    python train_knowledge_distillation_component.py
    ```
    This script trains a larger teacher model and then transfers its knowledge (softened using temperature) to a smaller student model, verifying manual forward passes and custom gradient combinations for KL Divergence and Cross-Entropy.

    You can adjust hyperparameters such as temperature, distillation weight (alpha), epochs, and learning rate:
    ```bash
    python train_knowledge_distillation_component.py --t 3.0 --alpha 0.5 --epochs 5000 --lr 0.1
    ```

## Component Testing: Batch Normalization

Building upon our component research, we have implemented Batch Normalization. This component tests the hypothesis that normalizing inputs across the batch dimension can stabilize and accelerate deep network training. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Batch Normalization component test:**
    ```bash
    python train_batchnorm_component.py
    ```
    This script tests learning the gamma and beta parameters of batch normalization on a synthetic dataset, utilizing manual backpropagation to ensure gradients flow correctly through both the parameters and the batch statistics (mean and variance).

    You can adjust hyperparameters such as epochs and learning rate:
    ```bash
    python train_batchnorm_component.py --epochs 5000 --lr 0.1
    ```

## Component Testing: Group Normalization

Building upon our component research, we have implemented Group Normalization. This component tests the hypothesis that dividing channels into groups and computing mean and variance within those groups provides stable normalization for small batch sizes (where Batch Normalization struggles). It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Group Normalization component test:**
    ```bash
    python train_groupnorm_component.py
    ```
    This script tests learning the gamma and beta parameters per channel on a synthetic dataset, utilizing manual backpropagation to ensure gradients flow correctly through both the parameters and the reshaped group statistics.

    You can adjust hyperparameters such as number of groups, features, epochs, and learning rate:
    ```bash
    python train_groupnorm_component.py --num_groups 2 --num_features 8 --epochs 5000 --lr 0.1
    ```

## Component Testing: Highway Network

Building upon our component research, we have implemented Highway Networks. This component tests the hypothesis that allowing representations to pass unimpeded through gating mechanisms mitigates the vanishing gradient problem in very deep networks, acting as a precursor to residual connections. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Highway Network component test:**
    ```bash
    python train_highway_component.py
    ```
    This script tests learning the transform and carry gates on a synthetic dataset, utilizing manual backpropagation to ensure gradients flow correctly through both the transformation block and the gating mechanism.

    You can adjust hyperparameters such as dimension, epochs, and learning rate:
    ```bash
    python train_highway_component.py --dim 16 --epochs 10000 --lr 0.05
    ```

## Component Testing: gMLP (Gated MLP)

Building upon our component research into self-attention alternatives, we have implemented the gMLP (Gated MLP) component. This tests the hypothesis that spatial and sequential dependencies can be effectively modeled without attention mechanisms by using a Spatial Gating Unit (SGU) that combines element-wise multiplication with a linear sequence projection. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the gMLP component test:**
    ```bash
    python train_gmlp_component.py
    ```
    This script tests learning a spatial transformation over a sequence using manual backpropagation to ensure gradients flow correctly through the element-wise gating operation and the sequence-wise spatial projection.

    You can adjust hyperparameters such as epochs and learning rate:
    ```bash
    python train_gmlp_component.py --epochs 10000 --lr 0.01
    ```

## Component Testing: Hypernetwork

Building upon our component research, we have implemented a Hypernetwork component. This tests the hypothesis that dynamic weight generation—where a secondary network generates weights for a primary network conditioned on some context—can successfully learn context-dependent functional mappings. It is verified here using pure mathematical operations and batch-wise tensor contractions.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Hypernetwork component test:**
    ```bash
    python train_hypernetwork_component.py
    ```
    This script tests learning a dynamic mapping by explicitly providing context vectors and mapping inputs through dynamically generated weights, verifying manual backpropagation through the context-conditioned matrices.

    You can adjust hyperparameters such as epochs and learning rate:
    ```bash
    python train_hypernetwork_component.py --epochs 5000 --lr 0.01
    ```

## Component Testing: Hopfield Network

Building upon our memory and representation research, we have implemented a Hopfield Network component. This component tests the hypothesis that a fully connected recurrent neural network with symmetric weights can act as an associative memory system, storing and retrieving patterns using energy minimization via pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Hopfield Network component test:**
    ```bash
    python train_hopfield_component.py
    ```
    This script tests associative memory retrieval from noisy patterns using Hebbian learning for weight generation and asynchronous state updates for energy minimization.

    You can adjust hyperparameters such as pattern size, number of patterns, and noise level:
    ```bash
    python train_hopfield_component.py --pattern_size 100 --num_patterns 5 --noise_level 0.2
    ```

## Component Testing: Generative Adversarial Network (GAN)

Building upon our generative model research, we have implemented a Generative Adversarial Network (GAN) component. This component tests the hypothesis that a Generator network can learn to approximate a target distribution by engaging in a minimax game against a Discriminator network, without explicit density estimation. It is verified here using pure mathematical operations on a 1D Gaussian dataset.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the GAN component test:**
    ```bash
    python train_gan_component.py
    ```
    This script tests the adversarial dynamics by co-training a Generator to map random noise to a target distribution (Mean=4.0, Std=1.2) and a Discriminator to distinguish real from fake samples.

    You can adjust hyperparameters such as epochs, learning rate for generator and discriminator, and hidden dimension:
    ```bash
    python train_gan_component.py --epochs 10000 --batch_size 128 --lr_d 0.01 --lr_g 0.01 --hidden_dim 16
    ```

## Component Testing: Graph Convolutional Network (GCN)

Building upon our component research, we have implemented a Graph Convolutional Network (GCN) component. This component tests the hypothesis that graph structures can be effectively integrated into neural networks by propagating node features through a normalized adjacency matrix. It is verified here using pure mathematical operations on a synthetic graph dataset.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the GCN component test:**
    ```bash
    python train_gcn_component.py
    ```
    This script tests the message-passing mechanism by co-training node representations to predict community labels on a synthetic graph.

    You can adjust hyperparameters such as epochs, learning rate, and hidden dimension:
    ```bash
    python train_gcn_component.py --epochs 1000 --lr 0.1 --hidden_dim 16 --num_nodes 100 --num_features 16
    ```

## Component Testing: Restricted Boltzmann Machine (RBM)

Building upon our generative model research, we have implemented a Restricted Boltzmann Machine (RBM) component. This component tests the hypothesis that an energy-based model with bipartite connections can learn to represent the underlying probability distribution of a dataset using Contrastive Divergence (CD-1) for efficient training. It is verified here using pure mathematical operations on a synthetic binary dataset.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the RBM component test:**
    ```bash
    python train_rbm_component.py
    ```
    This script tests the unsupervised learning capabilities by training the RBM to reconstruct synthetic binary patterns, utilizing CD-1 for manual parameter updates.

    You can adjust hyperparameters such as epochs, learning rate, and hidden units:
    ```bash
    python train_rbm_component.py --epochs 1000 --lr 0.1 --num_hidden 4
    ```

## Component Testing: Echo State Network (ESN)

Building upon our recurrent and dynamic state models, we have implemented an Echo State Network (ESN). This component tests Reservoir Computing principles by utilizing a fixed, random recurrent reservoir to project sequential data into a high-dimensional state space, while only the linear readout layer is trained via Ridge Regression. It is verified here using pure mathematical operations on a chaotic time-series prediction task (Mackey-Glass).

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the ESN component test:**
    ```bash
    python train_esn_component.py
    ```
    This script tests the continuous time-series prediction capabilities by training the ESN to forecast the chaotic Mackey-Glass sequence.

    You can adjust hyperparameters such as reservoir size, spectral radius, and sequence length:
    ```bash
    python train_esn_component.py --reservoir_size 500 --spectral_radius 1.25 --seq_len 2000
    ```

## Component Testing: Bayesian Neural Network (BNN)

Building upon our component research into probability and uncertainty estimation, we have implemented a Bayesian Neural Network (BNN). This component tests the hypothesis that weights can be modeled as probability distributions (using the reparameterization trick) and learned via the Bayes by Backprop algorithm to optimize the Evidence Lower Bound (ELBO). This allows the network to estimate uncertainty in its predictions. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the BNN component test:**
    ```bash
    python train_bnn_component.py
    ```
    This script tests learning a non-linear dataset (XOR) while balancing predictive accuracy (NLL) with parameter uncertainty (KL divergence) via manual backpropagation.

    You can adjust hyperparameters such as epochs, learning rate, and KL weight:
    ```bash
    python train_bnn_component.py --epochs 25000 --lr 0.5 --kl_weight 0.001
    ```

## Component Testing: Neural ODE

Building upon our component research into continuous models, we have implemented a Neural Ordinary Differential Equation (Neural ODE) component. This component tests the hypothesis that hidden states can be evolved continuously with depth by parameterized dynamics, solved via numerical integration (e.g., Euler's method). It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Neural ODE component test:**
    ```bash
    python train_neural_ode_component.py
    ```
    This script tests continuous-depth modeling by training a network dynamics function to classify a non-linear dataset, utilizing manual backpropagation through the numerical ODE solver.

    You can adjust hyperparameters such as epochs, learning rate, and solver steps:
    ```bash
    python train_neural_ode_component.py --epochs 5000 --lr 0.1 --steps 10
    ```

## Component Testing: Spiking Neural Network (SNN)

Building upon our component research, we have implemented a Spiking Neural Network (SNN) utilizing Leaky Integrate-and-Fire (LIF) neurons. This component tests the hypothesis that energy-efficient event-based biological processing dynamics can be effectively modeled and trained using Backpropagation Through Time combined with Surrogate Gradients (to address the non-differentiability of the discrete spike function). It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the SNN component test:**
    ```bash
    python train_snn_component.py
    ```
    This script tests sequential spike dynamics and temporal representations on the XOR dataset by encoding it across sequential steps and using a fast sigmoid surrogate function for manual backpropagation.

    You can adjust hyperparameters such as epochs, learning rate, and sequence length (T):
    ```bash
    python train_snn_component.py --epochs 2000 --lr 5.0 --hidden_dim 32 --seq_len 10
    ```

## Component Testing: Graph Attention Network (GAT)

Building upon our graph and attention research, we have implemented a Graph Attention Network (GAT) component. This component tests the hypothesis that node representations can be improved by assigning different importance (attention weights) to different neighbors during the aggregation step, rather than treating all neighbors equally (as in GCN). It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the GAT component test:**
    ```bash
    python train_gat_component.py
    ```
    This script tests the masked attention mechanism by co-training node representations to predict community labels on a synthetic graph exhibiting homophily, utilizing manual backpropagation.

    You can adjust hyperparameters such as epochs, learning rate, and hidden dimension:
    ```bash
    python train_gat_component.py --epochs 2000 --lr 0.05 --hidden_dim 8
    ```

## Component Testing: End-To-End Memory Network (MemN2N)

Building upon our component research, we have implemented an End-To-End Memory Network. This component tests the hypothesis that a network can learn to answer queries by explicitly storing facts in a memory structure, computing soft attention over those memories, and routing the retrieved information to generate an answer. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Memory Network component test:**
    ```bash
    python train_memory_network_component.py
    ```
    This script tests learning to reason over synthetic question-answering facts by manually backpropagating through the memory embeddings (A, C), query embeddings (B), and output transformations.

    You can adjust hyperparameters such as dimension, epochs, and learning rate:
    ```bash
    python train_memory_network_component.py --d 8 --epochs 5000 --lr 0.1
    ```

## Component Testing: REINFORCE (Policy Gradient)

Building upon our foundational modeling research, we have implemented the REINFORCE algorithm component. This component tests the hypothesis that a neural network can learn an optimal policy to maximize expected returns in an environment by performing gradient ascent on the log probabilities of sampled actions scaled by their rewards, using a baseline to reduce variance. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the REINFORCE component test:**
    ```bash
    python train_reinforce_component.py
    ```
    This script tests a simple agent navigating a 1D grid environment, validating the mathematical formulation of the policy gradient objective and manual backpropagation.

    You can adjust hyperparameters such as hidden dimension, epochs, learning rate, and discount factor (gamma):
    ```bash
    python train_reinforce_component.py --hidden_dim 16 --epochs 1000 --lr 0.05 --gamma 0.99
    ```

## Component Testing: Actor-Critic (RL)

Building upon the REINFORCE algorithm, we have implemented an Actor-Critic architecture. This component tests the hypothesis that learning a value function (the Critic) simultaneously with the policy (the Actor) allows the use of Temporal Difference (TD) errors to reduce variance during policy updates. This enables online, step-by-step learning rather than waiting for episode completion. It is verified here using pure mathematical operations through a shared hidden layer.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the Actor-Critic component test:**
    ```bash
    python train_actor_critic_component.py
    ```
    This script trains an agent in a simple 1D grid environment, validating the mathematical formulation of TD errors, policy gradient ascent, and manual backpropagation for multi-head networks.

    You can adjust hyperparameters such as hidden dimension, epochs, learning rate, and discount factor (gamma):
    ```bash
    python train_actor_critic_component.py --hidden_dim 16 --epochs 2000 --lr 0.01 --gamma 0.99
    ```

## Component Testing: Proximal Policy Optimization (PPO)

Building upon the Actor-Critic algorithm, we have implemented Proximal Policy Optimization (PPO). This component tests the hypothesis that clipping the surrogate objective prevents destructively large policy shifts during gradient ascent, thereby stabilizing the training process and allowing for multiple update epochs per rollout. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the PPO component test:**
    ```bash
    python train_ppo_component.py
    ```
    This script trains an agent in a simple 1D grid environment, validating the mathematical formulation of the clipped surrogate objective, multiple epoch updates, and manual backpropagation for multi-head networks.

    You can adjust hyperparameters such as hidden dimension, epochs, learning rate, discount factor (gamma), and clipping epsilon:
    ```bash
    python train_ppo_component.py --hidden_dim 16 --epochs 2000 --lr 0.01 --gamma 0.99 --epsilon 0.2
    ```

## Component Testing: Deep Q-Network (DQN)

Building upon the Q-learning algorithm, we have implemented a Deep Q-Network (DQN). This component tests the hypothesis that Q-learning can be stabilized using deep neural networks by introducing experience replay (to break temporal correlations) and a target network (to provide stable TD targets). It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the DQN component test:**
    ```bash
    python train_dqn_component.py
    ```
    This script trains an agent in a simple 1D grid environment, validating the mathematical formulation of Q-values calculation, experience replay, target networks, and manual backpropagation.

    You can adjust hyperparameters such as hidden dimension, epochs, learning rate, and batch size:
    ```bash
    python train_dqn_component.py --hidden_dim 16 --epochs 1000 --lr 0.01 --batch_size 32
    ```

## Component Testing: Model-Agnostic Meta-Learning (MAML)

Building upon our component research, we have implemented a First-Order Model-Agnostic Meta-Learning (MAML) component. This component tests the hypothesis that a model can learn an internal representation (initialization parameters) that is broadly suitable for many tasks, enabling rapid adaptation to new, unseen tasks with only a few gradient steps (few-shot learning). It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the MAML component test:**
    ```bash
    python train_maml_component.py
    ```
    This script tests learning a meta-initialization for a family of sine wave regression tasks, validating the mathematical formulation of inner and outer loop optimization, and manual backpropagation.

    You can adjust hyperparameters such as meta batch size, epochs, and inner/outer learning rates:
    ```bash
    python train_maml_component.py --epochs 1000 --meta_batch_size 16 --inner_lr 0.01 --outer_lr 0.001
    ```

## Component Testing: Temporal Convolutional Network (TCN)

Building upon our component research, we have implemented a Temporal Convolutional Network (TCN) component. This component tests sequence modeling capabilities using causal dilated convolutions, providing an alternative to RNNs with stable gradients and larger receptive fields. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the TCN component test:**
    ```bash
    python train_tcn_component.py
    ```
    This script tests the model on a synthetic sequence delay task, validating the mathematical formulation of causal dilated convolutions, residual blocks, and manual backpropagation over time.

    You can adjust hyperparameters such as epochs, learning rate, levels (depth), and hidden dimension:
    ```bash
    python train_tcn_component.py --epochs 1000 --lr 0.01 --levels 3 --hidden_dim 8
    ```

## Component Testing: Elastic Weight Consolidation (EWC)

Building upon our component research, we have implemented an Elastic Weight Consolidation (EWC) component. This component tests continual learning capabilities by mitigating catastrophic forgetting across sequential tasks using the Fisher Information Matrix as a proxy for parameter importance. It is verified here using pure mathematical operations and manual backpropagation.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the EWC component test:**
    ```bash
    python train_ewc_component.py
    ```
    This script tests the model on two sequential linear regression tasks, verifying the mathematical formulation of the empirical Fisher Information Matrix and the EWC penalty application to prevent forgetting of the first task.

## Component Testing: Continuous-Time Recurrent Neural Network (CTRNN)

Building upon our recurrent models, we have implemented a Continuous-Time Recurrent Neural Network (CTRNN). This component tests the hypothesis that neural dynamics can be modeled continuously using differential equations governed by time constants, allowing the network to adapt to different timescales. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the CTRNN component test:**
    ```bash
    python train_ctrnn_component.py
    ```
    This script tests continuous-time sequence modeling on a moving average task, validating Euler integration and Backpropagation Through Time.

## Component Testing: Random Feedback Alignment (FA)

Building upon our component research, we have implemented a Random Feedback Alignment component. This component tests the hypothesis that gradients can still provide a useful learning signal even when passing through fixed random matrices, aligning the forward weights to make the random backward weights effective, thereby avoiding the need for symmetric weight transport. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the FA component test:**
    ```bash
    python train_feedback_alignment_component.py
    ```
    This script tests learning non-linear boundaries using fixed random matrices for backward error propagation.

## Component Testing: Direct Feedback Alignment (DFA)

Building upon our component research, we have implemented a Direct Feedback Alignment component. This component explores biologically plausible learning rules by propagating the output error directly to each hidden layer using fixed random matrices, bypassing the backward pass through subsequent hidden layers entirely. This allows for parallel weight updates across layers. It is verified here using pure mathematical operations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the DFA component test:**
    ```bash
    python train_dfa_component.py
    ```
    This script tests learning non-linear boundaries, confirming that directly projecting output errors via random matrices to hidden layers provides a sufficient learning signal.

## Component Testing: Extreme Learning Machine (ELM)

Building upon our component research, we have implemented an Extreme Learning Machine (ELM). This component explores non-iterative learning by fixing random input weights and analytically solving for output weights using the Moore-Penrose pseudoinverse, allowing for exceptionally fast one-shot learning of non-linear representations.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the ELM component test:**
    ```bash
    python train_elm_component.py
    ```
    This script tests rapid one-shot analytical learning, confirming that solving the pseudoinverse of random hidden features provides a sufficient learning mechanism without backpropagation.

## Component Testing: Radial Basis Function (RBF) Network

Building upon our component research, we have implemented a Radial Basis Function (RBF) Network. This component tests the hypothesis that non-linear functions can be effectively approximated using a superposition of localized basis functions (Gaussians), optimizing their centroids, widths, and output weights. It is verified here using pure mathematical operations and manual backpropagation.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the RBF Network component test:**
    ```bash
    python train_rbf_component.py
    ```
    This script tests learning non-linear boundaries, confirming that backpropagation can successfully tune the localized Gaussian basis functions.
## Component Testing: Self-Organizing Map (SOM)

**Script:** `train_som_component.py`
**Description:** Evaluates a Self-Organizing Map (SOM) using competitive learning to map high-dimensional input points into a lower-dimensional topological grid.
## Component Testing: Neural Cellular Automata (NCA)

**Script:** `train_nca_component.py`
**Description:** Evaluates a Neural Cellular Automata (NCA) component to 'grow' a predefined target pattern from a single seed pixel using localized, iterative cell updates via a shared MLP and Sobel filters.
## Component Testing: Liquid Time-Constant (LTC) Network

**Script:** `train_ltc_component.py`
**Description:** Evaluates a Liquid Time-Constant (LTC) Network component dynamically adapting its continuous-time ODE dynamics based on input.
## Component Testing: Neural Turing Machine (NTM)

**Script:** `train_ntm_component.py`
**Description:** Evaluates a Neural Turing Machine (NTM) component, verifying content-based addressing and differentiable read/write memory operations.
## Component Testing: Skip-Gram

**Script:** `train_skipgram_component.py`
**Description:** Evaluates a Skip-Gram component with Negative Sampling, learning semantic word representations by modeling context relationships via manual backpropagation.
## Component Testing: Continuous Bag of Words (CBOW)

**Script:** `train_cbow_component.py`
**Description:** Evaluates a Continuous Bag of Words (CBOW) component, verifying that word representations can be learned by predicting a target word from the average of its context word embeddings, utilizing manual backpropagation.
## Component Testing: Gumbel-Softmax

**Script:** `train_gumbel_softmax_component.py`
**Description:** Evaluates a Gumbel-Softmax component, verifying differentiable discrete sampling from a categorical distribution using the reparameterization trick with Gumbel noise and temperature annealing via manual backpropagation.
## Component Testing: Perceiver Bottleneck

**Script:** `train_perceiver_component.py`
**Description:** Evaluates a Perceiver Bottleneck component, verifying that cross-attention with trainable latents can reduce sequence modeling complexity to $O(N \cdot M)$.
## Component Testing: FNet Block

**Script:** `train_fnet_component.py`
**Description:** Evaluates an FNet block component using pure NumPy. This component tests replacing standard self-attention with a parameter-free 2D Fourier Transform for mixing over sequence and hidden dimensions, maintaining sequence modeling capabilities while avoiding attention overhead.
## Component Testing: Predictive Coding Network (PCN)

**Script:** `train_pcn_component.py`
**Description:** Evaluates a Predictive Coding Network (PCN) component, verifying a biologically plausible alternative to backpropagation that relies on local learning rules and iterative inference to minimize prediction errors without calculating a global backward pass.
## Component Testing: Capsule Network

**Script:** `train_capsule_network_component.py`
**Description:** Evaluates a Capsule Network component, verifying the dynamic routing by agreement mechanism to preserve part-whole relationships between hierarchical vector capsules.
## Component Testing: Deep Sets

**Script:** `train_deepsets_component.py`
**Description:** Evaluates a Deep Sets component, verifying its ability to process unordered sets by applying permutation-invariant transformations using element-wise processing and pooling operations.
## Component Testing: Spectral Normalization

**Script:** `train_spectral_normalization_component.py`
**Description:** Evaluates a Spectral Normalization component, verifying that applying power iteration to constrain the Lipschitz constant allows for stable forward and backward passes.
## Component Testing: Weight Normalization

**Script:** `train_weight_normalization_component.py`
**Description:** Evaluates a Weight Normalization component, verifying that decoupling the weight length from its direction using reparameterization allows for stable gradients and successful convergence.
## Component Testing: Energy-Based Model (EBM)

**Script:** `train_ebm_component.py`
**Description:** Evaluates an Energy-Based Model component, verifying that it can learn an implicit probability distribution over continuous data by minimizing the energy of real samples and maximizing the energy of negative samples generated via Langevin Dynamics.
## Component Testing: Decoupled Neural Interfaces (DNI)

**Script:** `train_dni_component.py`
**Description:** Evaluates Decoupled Neural Interfaces (DNI) using Synthetic Gradients, verifying that layers can be updated asynchronously by predicting their own error gradients, enabling decoupled backward passes without locking.
## Component Testing: Evolution Strategies (ES)

**Script:** `train_es_component.py`
**Description:** Evaluates an Evolution Strategies (ES) component, verifying gradient-free black-box optimization of neural network weights using stochastic parameter perturbations and fitness-based updates, successfully learning a non-linear regression mapping without backpropagation.
## Component Testing: Mixture Density Network (MDN)

**Script:** `train_mdn_component.py`
**Description:** Evaluates a Mixture Density Network (MDN) component, verifying that it can learn multi-modal conditional probability distributions by parameterizing a Gaussian Mixture Model, successfully minimizing Negative Log-Likelihood on a toy inverse kinematics problem.
## Component Testing: RealNVP

**Script:** `train_realnvp_component.py`
**Description:** Evaluates a RealNVP Normalizing Flow component, verifying its ability to model complex distributions by transforming simple base distributions using invertible affine coupling layers.
## Component Testing: Neural Autoregressive Distribution Estimator (NADE)

**Script:** `train_nade_component.py`
**Description:** Evaluates a Neural Autoregressive Distribution Estimator (NADE), verifying its ability to model the joint probability distribution of binary data by factoring it into a product of conditional distributions, optimizing exact likelihood via backpropagation.
## Component Testing: Denoising Autoencoder (DAE)

**Script:** `train_dae_component.py`
**Description:** Evaluates a Denoising Autoencoder (DAE) component, verifying that it can learn robust representations by reconstructing original data from artificially corrupted (noisy) inputs using manual backpropagation.
## Component Testing: Sinusoidal Representation Networks (SIREN)

**Script:** `train_siren_component.py`
**Description:** Evaluates a Sinusoidal Representation Network (SIREN) component, verifying its ability to model complex, high-frequency continuous implicit representations using sine activation functions and a specialized initialization scheme via manual backpropagation.
## Component Testing: Conditional Neural Process (CNP)

**Script:** `train_cnp_component.py`
**Description:** Evaluates a Conditional Neural Process (CNP) component, verifying its ability to model distributions over functions (meta-learning) by processing context points into a global representation and predicting parameters of target distributions for a family of sine waves via manual backpropagation.
## Component Testing: SimSiam (Simple Siamese Networks)

**Script:** `train_simsiam_component.py`
**Description:** Evaluates a SimSiam component for non-contrastive self-supervised representation learning, verifying its ability to prevent representation collapse using a stop-gradient operation and a predictor network without requiring negative pairs, optimized via manual backpropagation.
## Component Testing: Intrinsic Curiosity Module (ICM)

**Script:** `train_icm_component.py`
**Description:** Evaluates an Intrinsic Curiosity Module (ICM) component, verifying its ability to encourage exploration by generating intrinsic reward through predicting the next state feature representation (forward model) and learning action-conditioned representations (inverse model) via manual backpropagation.

## Component Testing: Flow Matching

**Script:** `train_flow_matching_component.py`
**Description:** Evaluates a Flow Matching component for continuous normalizing flows, verifying its ability to model complex target distributions by regressing a vector field that transports a standard normal distribution via straight probability paths, and sampling from the learned flow using Euler integration via manual backpropagation.

## Component Testing: Masked Autoencoder (MAE)

**Script:** `train_mae_component.py`
**Description:** Evaluates a Masked Autoencoder (MAE) component for self-supervised representation learning, verifying its ability to reconstruct original data from heavily masked inputs using an asymmetric encoder-decoder architecture via manual backpropagation.

## Component Testing: Vision Transformer (ViT)

**Script:** `train_vit_component.py`
**Description:** Evaluates a Vision Transformer (ViT) component, verifying its ability to model spatial dependencies by treating images as sequences of non-overlapping patches, prepending a learnable class token, and processing them through multi-head self-attention via manual backpropagation.

## Component Testing: Barlow Twins

**Script:** `train_barlow_twins_component.py`
**Description:** Evaluates a Barlow Twins component for non-contrastive self-supervised representation learning, verifying its ability to prevent representation collapse by driving the cross-correlation matrix between representations of distorted versions of a sample to the identity matrix via manual backpropagation.

## Component Testing: Hebbian Learning (Oja's Rule)

**Script:** `train_hebbian_component.py`
**Description:** Evaluates a Hebbian Learning component using Oja's rule, verifying its ability to extract the first principal component of input data using a biologically plausible, gradient-free learning rule.

## Component Testing: Independent Component Analysis (ICA)

**Script:** `train_ica_component.py`
**Description:** Evaluates an Independent Component Analysis (ICA) component using the FastICA algorithm, verifying its ability to perform blind source separation and recover underlying non-Gaussian signals from linear mixtures via negentropy maximization.

## Component Testing: Wasserstein Generative Adversarial Network (WGAN)

**Script:** `train_wgan_component.py`
**Description:** Evaluates a Wasserstein GAN (WGAN) component, verifying its ability to generate data approximating a target distribution by optimizing the Earth Mover's distance using a critic and weight clipping via manual backpropagation.

## Component Testing: Bootstrap Your Own Latent (BYOL)

**Script:** `train_byol_component.py`
**Description:** Evaluates a Bootstrap Your Own Latent (BYOL) component, verifying its ability to learn self-supervised representations without contrastive negative pairs by minimizing the prediction error between online and target networks using an exponential moving average (EMA) momentum update via manual backpropagation.

### Orthogonal RNN Component (`train_orthogonal_rnn_component.py`)
- **Mathematical Basis**: Orthogonal RNNs parameterize the hidden-to-hidden weight matrix to remain orthogonal, preventing gradients from vanishing or exploding over long sequences. We use the Cayley transform $W = (I - A)(I + A)^{-1}$ with a skew-symmetric matrix $A = V - V^T$ parameterized by unconstrained matrix $V$.
- **Verification**: The component successfully trains on a sequential cumulative sum task, maintaining stable gradient norms during backpropagation.

## Component Testing: Joint Embedding Predictive Architecture (JEPA)

**Script:** `train_jepa_component.py`
**Description:** Evaluates a Joint Embedding Predictive Architecture (JEPA) component, verifying its ability to learn self-supervised representations by predicting the representation of a target (encoded via an EMA stop-gradient network) from a context and condition using manual backpropagation.

## Component Testing: Legendre Memory Unit (LMU)

**Script:** `train_lmu_component.py`
**Description:** Evaluates a Legendre Memory Unit (LMU) component, verifying its ability to model continuous-time representation via analytically derived orthogonal Legendre polynomials, maintaining stable gradients for long-range sequence modeling via manual backpropagation.

## Component Testing: Generative Flow Network (GFlowNet)

**Script:** `train_gflownet_component.py`
**Description:** Evaluates a GFlowNet agent learning to generate compositional objects with probabilities proportional to a reward function, utilizing manual backpropagation on the Trajectory Balance loss.

## Component Testing: Monte Carlo Tree Search (MCTS)

**Script:** `train_mcts_component.py`
**Description:** Evaluates an MCTS agent combined with a neural network evaluating policy and value, simulating core elements of AlphaZero-style planning via manual backpropagation.

## Component Testing: Difference Target Propagation (DTP)

**Script:** `train_target_propagation_component.py`
**Description:** Evaluates a Difference Target Propagation (DTP) component, verifying a biologically plausible alternative to backpropagation that trains neural networks without requiring symmetric weight matrices or continuous gradients, by using autoencoders to propagate target activations rather than gradients.

## Component Testing: Variational Information Bottleneck (VIB)

**Script:** `train_vib_component.py`
**Description:** Evaluates a Deep Variational Information Bottleneck (VIB) component, verifying its ability to learn robust, compressed representations by balancing mutual information and predictive accuracy via manual backpropagation of the ELBO.

## Component Testing: Deep Deterministic Policy Gradient (DDPG)

**Script:** `train_ddpg_component.py`
**Description:** Evaluates a Deep Deterministic Policy Gradient (DDPG) component for continuous action spaces, verifying its ability to learn an actor-critic architecture utilizing the deterministic policy gradient theorem and manual backpropagation for gradient flow.

## Component Testing: Lottery Ticket Hypothesis (IMP)

**Script:** `train_lottery_ticket_component.py`
**Description:** Evaluates the Lottery Ticket Hypothesis, demonstrating that dense randomly-initialized networks contain sparse subnetworks (winning tickets) that, when trained in isolation from their original initializations, can match the test accuracy of the original network.

## Component Testing: Sharpness-Aware Minimization (SAM)

**Script:** `train_sam_component.py`
**Description:** Evaluates a Sharpness-Aware Minimization (SAM) component, verifying its ability to explicitly penalize loss sharpness by perturbing weights in the direction of maximum loss locally before computing the final gradient update.

## Component Testing: Differentiable Architecture Search (DARTS)

**Script:** `train_darts_component.py`
**Description:** Evaluates a DARTS component to verify its ability to search for optimal architectural operations by continuously relaxing the discrete search space and using a bi-level gradient descent optimization scheme.

## Component Testing: Prototypical Networks (ProtoNet)

**Script:** `train_protonet_component.py`
**Description:** Evaluates a Prototypical Networks (ProtoNet) component, verifying its ability to perform few-shot classification by learning a metric embedding space where query points are classified based on Euclidean distance to class prototypes, utilizing manual backpropagation on the episodic loss.

## Component Testing: Fast Gradient Sign Method (FGSM)

**Script:** `train_fgsm_component.py`
**Description:** Evaluates the Fast Gradient Sign Method (FGSM) for generating adversarial examples and training robust models via Adversarial Training. Demonstrates mathematically that alternating optimization on clean and adversarial inputs increases network robustness against gradient-based perturbations.

## Component Testing: Sparsemax

**Script:** `train_sparsemax_component.py`
**Description:** Evaluates a Sparsemax component, verifying its ability to project logits onto the probability simplex, yielding exactly sparse probability distributions via Euclidean projection, and testing its manual backpropagation mechanism routing gradients only through the non-zero support set.

## Component Testing: Contrastive Predictive Coding (CPC)

**Script:** `train_cpc_component.py`
**Description:** Evaluates Contrastive Predictive Coding (CPC), verifying its ability to learn representations by predicting future latent states autoregressively using the InfoNCE loss.

## Component Testing: Continuous Hopfield Network

**Script:** `train_continuous_hopfield_component.py`
**Description:** Evaluates a Continuous Hopfield Network component mathematically in pure NumPy, testing its ability to retrieve continuous target patterns from noisy initializations by iteratively minimizing a log-sum-exp energy function, demonstrating exponential memory capacity and the connection to self-attention.

## Component Testing: Neural Arithmetic Logic Unit (NALU)

**Script:** `train_nalu_component.py`
**Description:** Evaluates a Neural Arithmetic Logic Unit (NALU) component mathematically in pure NumPy, testing its ability to interpolate between an additive and a multiplicative path via a learned gate to represent numerical relationships.

## Component Testing: Adaptive Computation Time (ACT)

**Script:** `train_act_component.py`
**Description:** Evaluates an Adaptive Computation Time (ACT) component mathematically in pure NumPy, testing its ability to dynamically allocate computation steps per input by iteratively updating a hidden state and computing a halting probability, minimizing a ponder cost alongside the task loss.

## Component Testing: Relational Network (RN)

**Script:** `train_relational_network_component.py`
**Description:** Evaluates a Relational Network (RN) component mathematically in pure NumPy, testing its explicit reasoning capabilities across all pairs of objects within an input set and applying permutation-invariant sum-pooling.

## Component Testing: Pointer Network Component

**Script:** `train_pointer_network_component.py`
**Description:** Evaluates a Pointer Network component mathematically in pure NumPy, testing its ability to solve combinatorial tasks by learning to point to elements of the input sequence directly using an attention mechanism over the encoder states.

## Component Testing: Twin Delayed DDPG (TD3)

**Script:** `train_td3_component.py`
**Description:** Evaluates a Twin Delayed DDPG (TD3) component mathematically in pure NumPy, testing its ability to mitigate overestimation bias in continuous control through clipped double Q-learning and delayed updates.

## Component Testing: Neural Radiance Field (NeRF)

**Script:** `train_nerf_component.py`
**Description:** Evaluates a Neural Radiance Field (NeRF) component mathematically in pure NumPy, testing its ability to represent continuous 3D scenes by mapping spatial coordinates with positional encoding to density and color, and rendering them using volume rendering equations.

## Component Testing: Soft Actor-Critic (SAC)

**Script:** `train_sac_component.py`
**Description:** Evaluates a Soft Actor-Critic (SAC) component mathematically in pure NumPy, testing its ability to optimize a stochastic policy in an off-policy manner using maximum entropy reinforcement learning to encourage exploration and stability.

## Component Testing: Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

**Script:** `train_cmaes_component.py`
**Description:** Evaluates a CMA-ES component mathematically in pure NumPy, testing its ability to optimize parameters without gradients by dynamically adapting a multivariate normal distribution's mean and covariance matrix based on sample fitness.

## Component Testing: Decision Transformer

**Script:** `train_decision_transformer_component.py`
**Description:** Evaluates a Decision Transformer component mathematically in pure NumPy, testing its ability to solve offline reinforcement learning by modeling sequences of states, actions, and returns-to-go using causal self-attention, successfully learning an expert policy via sequence modeling and manual backpropagation.

## Component Testing: K-Means Clustering

**Script:** `train_kmeans_component.py`
**Description:** Evaluates a K-Means clustering component mathematically in pure NumPy, testing its ability to partition data points into clusters by iteratively updating centroids to minimize the within-cluster sum of squared distances.


## Component Testing: Revnet

**Script:** `train_revnet_component.py`
**Description:** Evaluates a Revnet component, verifying its logic mathematically in pure NumPy.

## Component Testing: Evaluation Metrics

**Script:** `train_evaluation_metrics_component.py`
**Description:** Evaluates a Evaluation Metrics component, verifying its logic mathematically in pure NumPy.

## Component Testing: Selective SSM

**Script:** `train_selective_ssm_component.py`
**Description:** Evaluates a Selective SSM component, verifying its logic mathematically in pure NumPy.

## Component Testing: MLP-Mixer

**Script:** `train_mlpmixer_component.py`
**Description:** Evaluates a MLP-Mixer component, verifying its logic mathematically in pure NumPy.

## Component Testing: Alibi

**Script:** `train_alibi_component.py`
**Description:** Evaluates a Alibi component, verifying its logic mathematically in pure NumPy.

## Component Testing: DDPM

**Script:** `train_ddpm_component.py`
**Description:** Evaluates a DDPM component, verifying its logic mathematically in pure NumPy.

## Component Testing: Retention

**Script:** `train_retention_component.py`
**Description:** Evaluates a Retention component, verifying its logic mathematically in pure NumPy.

## Component Testing: VAE

**Script:** `train_vae_component.py`
**Description:** Evaluates a VAE component, verifying its logic mathematically in pure NumPy.

## Component Testing: State Space Model (SSM)

**Script:** `train_ssm_component.py`
**Description:** Evaluates a State Space Model (SSM) component, verifying its logic mathematically in pure NumPy.

## Component Testing: Grokking

**Script:** `train_grokking_component.py`
**Description:** Evaluates a Grokking component, verifying its logic mathematically in pure NumPy.

## Component Testing: Feed Forward

**Script:** `train_ff_component.py`
**Description:** Evaluates a Feed Forward component, verifying its logic mathematically in pure NumPy.

## Component Testing: LSTM

**Script:** `train_lstm_component.py`
**Description:** Evaluates a LSTM component, verifying its logic mathematically in pure NumPy.

## Component Testing: RNN

**Script:** `train_rnn_component.py`
**Description:** Evaluates a RNN component, verifying its logic mathematically in pure NumPy.

## Component Testing: Scaled Exponential Linear Unit (SELU)

**Script:** `train_selu_component.py`
**Description:** Evaluates a Scaled Exponential Linear Unit (SELU) component mathematically in pure NumPy, testing its ability to induce self-normalization in deep networks, maintaining a mean of 0 and variance of 1 across layers.

## Component Testing: Gaussian Mixture Models (GMM)

**Script:** `train_gmm_component.py`
**Description:** Evaluates a Gaussian Mixture Model (GMM) component mathematically in pure NumPy, testing its ability to perform soft clustering and density estimation via the Expectation-Maximization (EM) algorithm, iteratively updating component responsibilities and distribution parameters.

## Component Testing: DBSCAN

**Script:** `train_dbscan_component.py`
**Description:** Evaluates a DBSCAN component mathematically in pure NumPy, testing its ability to identify non-convex clusters and isolate noise based on local density metrics.

## Component Testing: Principal Component Analysis (PCA)

**Script:** `train_pca_component.py`
**Description:** Evaluates a Principal Component Analysis (PCA) component mathematically in pure NumPy, testing its ability to perform dimensionality reduction by finding orthogonal directions of maximum variance in the data via eigendecomposition of the covariance matrix.

## Component Testing: Non-negative Matrix Factorization (NMF)

**Script:** `train_nmf_component.py`
**Description:** Evaluates a Non-negative Matrix Factorization (NMF) component mathematically in pure NumPy, testing its ability to factorize non-negative data into lower-rank non-negative matrices using multiplicative update rules.

## Component Testing: Particle Swarm Optimization (PSO)

**Script:** `train_pso_component.py`
**Description:** Evaluates a Particle Swarm Optimization (PSO) component mathematically in pure NumPy, testing its ability to find the global minimum of the non-convex Rastrigin function using a population of particles that adjust their trajectories based on cognitive and social intelligence.

## Component Testing: Cross-Entropy Method (CEM)

**Script:** `train_cem_component.py`
**Description:** Evaluates a Cross-Entropy Method (CEM) component mathematically in pure NumPy, testing its ability to solve a continuous control problem by maintaining a parameterized probability distribution over action sequences and iteratively updating the distribution's mean and covariance matrix based on sample fitness.

## Component Testing: Tabular Q-Learning

**Script:** `train_q_learning_component.py`
**Description:** Evaluates a Tabular Q-Learning component mathematically in pure NumPy, testing its ability to find an optimal policy in a GridWorld environment using an off-policy temporal difference control algorithm to iteratively approximate the Bellman equation.

## Component Testing: SARSA

**Script:** `train_sarsa_component.py`
**Description:** Evaluates a SARSA component mathematically in pure NumPy, testing its ability to find an optimal policy in a GridWorld environment using an on-policy temporal difference control algorithm that updates state-action values based on the specific actions taken by the epsilon-greedy policy.

## Component Testing: Support Vector Machine (SVM)

**Script:** `train_svm_component.py`
**Description:** Evaluates a linear Support Vector Machine component mathematically in pure NumPy, testing its ability to learn a maximum margin decision boundary for a linearly separable dataset by optimizing the hinge loss with subgradient descent.

## Component Testing: K-Nearest Neighbors (KNN)

**Script:** `train_knn_component.py`
**Description:** Evaluates a K-Nearest Neighbors component mathematically in pure NumPy, testing its ability to classify samples based on the majority vote of their $k$-nearest neighbors in the feature space using Euclidean distance, effectively handling non-linear decision boundaries without explicit distributional assumptions.

## Component Testing: Simulated Annealing
**Script:** `train_simulated_annealing_component.py`
**Description:** Evaluates a Simulated Annealing component mathematically in pure NumPy for function minimization, testing it on the Rosenbrock function.

## Component Testing: Linear Discriminant Analysis (LDA)
**Script:** `train_lda_component.py`
**Description:** Evaluates a Linear Discriminant Analysis (LDA) component mathematically in pure NumPy, finding axes that maximize the separation between multiple classes.

## Component Testing: Gaussian Naive Bayes
**Script:** `train_naive_bayes_component.py`
**Description:** Evaluates a Gaussian Naive Bayes component mathematically in pure NumPy, using Bayes' theorem with a conditional independence assumption.

## Component Testing: Logistic Regression
**Script:** `train_logistic_regression_component.py`
**Description:** Evaluates a Logistic Regression component mathematically in pure NumPy for binary classification.

## Component Testing: Decision Tree
**Script:** `train_decision_tree_component.py`
**Description:** Evaluates a Decision Tree classifier mathematically in pure NumPy by recursively splitting the feature space based on Information Gain.

## Component Testing: Random Forest
**Script:** `train_random_forest_component.py`
**Description:** Evaluates a Random Forest classifier mathematically in pure NumPy using an ensemble of decision trees trained on bootstrap samples.

## Component Testing: AdaBoost
**Script:** `train_adaboost_component.py`
**Description:** Evaluates an AdaBoost classifier mathematically in pure NumPy using decision stumps and iterative weighting.

## Component Testing: t-Distributed Stochastic Neighbor Embedding (t-SNE)

**Script:** `train_tsne_component.py`
**Description:** Evaluates a t-SNE component mathematically in pure NumPy, testing its ability to perform non-linear dimensionality reduction by converting high-dimensional Euclidean distances into conditional probabilities and minimizing the Kullback-Leibler divergence with a Student-t distribution in the low-dimensional space.

## Component Testing: Spectral Clustering

**Script:** `train_spectral_clustering_component.py`
**Description:** Evaluates a Spectral Clustering component mathematically in pure NumPy, testing its ability to cluster non-convex shapes like the Two Moons dataset using an RBF affinity matrix and normalized graph Laplacian.

## Component Testing: Gaussian Process Regression (GPR)

**Script:** `train_gaussian_process_component.py`
**Description:** Evaluates a Gaussian Process Regression (GPR) component mathematically in pure NumPy, testing its ability to perform non-parametric probabilistic regression using an RBF kernel and Cholesky decomposition to output predictive means and standard deviations (uncertainties).

## Component Testing: Hidden Markov Model (HMM)

**Script:** `train_hmm_component.py`
**Description:** Evaluates a Hidden Markov Model (HMM) component mathematically in pure NumPy, testing its ability to infer hidden states from a sequence of observations using the Expectation-Maximization (EM) algorithm (Baum-Welch algorithm).

## Component Testing: Bayesian Optimization

**Script:** `train_bayesian_optimization_component.py`
**Description:** Evaluates a Bayesian Optimization component mathematically in pure NumPy, testing its ability to find the global maximum of a non-convex 1D function efficiently by fitting a Gaussian Process and maximizing the Expected Improvement acquisition function.

## Component Testing: Dictionary Learning

**Script:** `train_dictionary_learning_component.py`
**Description:** Evaluates a Dictionary Learning component mathematically in pure NumPy, testing its ability to learn a dictionary that provides sparse representations of synthetic data using an alternating optimization scheme with FISTA.

## Component Testing: Ridge Regression

**Script:** `train_ridge_regression_component.py`
**Description:** Evaluates a Ridge Regression component mathematically in pure NumPy, testing its ability to perform regularized linear regression using the closed-form solution with an L2 penalty to prevent overfitting.

## Component Testing: Lasso Regression

**Script:** `train_lasso_regression_component.py`
**Description:** Evaluates a Lasso Regression component mathematically in pure NumPy, testing its ability to perform regularized linear regression using subgradient descent with an L1 penalty to encourage sparse weights and feature selection.
## Component Testing: Elastic Net Regression

**Script:** `train_elastic_net_component.py`
**Description:** Evaluates an Elastic Net Regression component mathematically in pure NumPy, testing its ability to perform regularized linear regression using subgradient descent with a combination of L1 and L2 penalties to encourage sparse weights and prevent overfitting simultaneously.

## Component Testing: Decision Tree Regression

**Script:** `train_decision_tree_regression_component.py`
**Description:** Evaluates a Decision Tree Regressor mathematically in pure NumPy, recursively splitting the feature space to maximize variance reduction for non-linear regression.

## Component Testing: Random Forest Regression

**Script:** `train_random_forest_regression_component.py`
**Description:** Evaluates a Random Forest Regressor mathematically in pure NumPy, testing its ability to reduce variance and overfitting compared to a single decision tree on non-linear regression tasks using bootstrap aggregation and random feature subsets.

## Component Testing: XGBoost Regression

**Script:** `train_xgboost_component.py`
**Description:** Evaluates an XGBoost Regressor mathematically in pure NumPy, testing its ability to reduce variance and bias by iteratively training regression trees on the gradients and hessians of the objective function, with L2 regularization to prevent overfitting.

## Component Testing: Multimodal Training

**Script:** `train_multimodal_component.py`
**Description:** Evaluates a Multimodal training component mathematically in pure NumPy, testing its ability to align different modalities (e.g., vision and language) into a shared representation space using a contrastive loss similar to CLIP, pushing matching pairs closer together and pulling non-matching pairs apart.

## Component Testing: Canonical Correlation Analysis (CCA)

**Script:** `train_cca_component.py`
**Description:** Evaluates a Canonical Correlation Analysis (CCA) component mathematically in pure NumPy, testing its ability to find linear projections that maximize the cross-covariance between two multimodal variable sets, learning a shared representation space.

## Full Model Training Process
To start the overarching AGI/ASI training process mathematically, run the following script:
    python3 train_agi_asi_component.py
This script evaluates the meta-learning component mathematically using NumPy to approximate the final training phase.

## Component Testing: Chinchilla Optimality Scaling

**Script:** `train_chinchilla_optimality_component.py`
**Description:** Evaluates a Chinchilla Optimality Scaling component mathematically in pure NumPy, testing its ability to determine the optimal allocation of a given compute budget between parameter count and dataset size (D ~ 20N) to minimize the loss according to empirical scaling laws.

## Component Testing: Agglomerative Clustering

**Script:** `train_agglomerative_clustering_component.py`
**Description:** Evaluates an Agglomerative Clustering component mathematically in pure NumPy, testing its ability to perform bottom-up hierarchical clustering by iteratively merging the closest clusters based on a single linkage criterion using pairwise Euclidean distances.
## Component Testing: Mean Shift Clustering

**Script:** `train_mean_shift_component.py`
**Description:** Evaluates a Mean Shift Clustering component mathematically in pure NumPy, testing its ability to non-parametrically find clusters and determine the number of clusters by shifting points towards regions of higher density based on an RBF kernel.

## Component Testing: K-Medoids Clustering

**Script:** `train_kmedoids_component.py`
**Description:** Evaluates a K-Medoids Clustering component mathematically in pure NumPy, testing its ability to partition data into clusters by minimizing the distance between points and their assigned medoids (representative points from the dataset).

## Component Testing: Label Propagation

**Script:** `train_label_propagation_component.py`
**Description:** Evaluates a Label Propagation component mathematically in pure NumPy, testing its ability to propagate labels from a small set of labeled data to unlabeled data using an affinity graph and an iterative transition matrix approach.

## Component Testing: Isolation Forest

**Script:** `train_isolation_forest_component.py`
**Description:** Evaluates an Isolation Forest component mathematically in pure NumPy, testing its ability to identify anomalies by isolating them in random trees, utilizing the principle that anomalies require shorter path lengths to be isolated.

## Component Testing: Partial Least Squares (PLS) Regression

**Script:** `train_pls_component.py`
**Description:** Evaluates a Partial Least Squares (PLS) Regression component mathematically in pure NumPy, testing its ability to model relationships between multiple input features and target variables by finding latent components that maximize their covariance using the NIPALS algorithm.

## Component Testing: Quadratic Discriminant Analysis (QDA)

**Script:** `train_qda_component.py`
**Description:** Evaluates a Quadratic Discriminant Analysis (QDA) component mathematically in pure NumPy, testing its ability to classify instances by calculating class-specific priors, means, and covariances.
**Description:** Evaluates a Kernel Density Estimation (KDE) component mathematically in pure NumPy, testing its ability to estimate the probability density function of continuous data distributions using Gaussian kernels.
- **Slow Feature Analysis (SFA)**: Extracts slowly varying temporal features from rapidly varying signals.

## Component Testing: Deep Belief Network (DBN)

**Script:** `train_dbn_component.py`
**Description:** Evaluates a Deep Belief Network (DBN) component mathematically in pure NumPy, testing its ability to learn hierarchical features by greedily training stacked Restricted Boltzmann Machines (RBMs) layer-by-layer using Contrastive Divergence (CD-1).

## Component Testing: Upper Confidence Bound (UCB)
**Script:** `train_upper_confidence_bound_component.py`
**Description:** Evaluates an Upper Confidence Bound (UCB) component mathematically in pure NumPy, testing its ability to solve the multi-armed bandit problem by balancing exploration and exploitation using an optimistic value estimate based on action selection frequency.

## Component Testing: Graph Isomorphism Network (GIN)
**Script:** `train_graph_isomorphism_network_component.py`
**Description:** Evaluates a Graph Isomorphism Network (GIN) component mathematically in pure NumPy, testing its ability to achieve maximum expressive power for graph representation by using an injective aggregation function (summation with epsilon) and MLPs.

## Component Testing: Fuzzy C-Means (FCM)
**Script:** `train_fuzzy_c_means_component.py`
**Description:** Evaluates a Fuzzy C-Means clustering component mathematically in pure NumPy, testing its ability to assign soft membership probabilities to data points across multiple clusters, accommodating overlapping cluster boundaries.

## Component Testing: Kernel Ridge Regression

**Script:** `train_kernel_ridge_regression_component.py`
**Description:** Evaluates a Kernel Ridge Regression component mathematically in pure NumPy, testing its ability to perform non-linear regression by projecting data into a high-dimensional feature space using the RBF kernel trick and solving the regularized optimization problem in dual form.

## Component Testing: Local Outlier Factor (LOF)

**Script:** `train_local_outlier_factor_component.py`
**Description:** Evaluates a Local Outlier Factor (LOF) component mathematically in pure NumPy, testing its ability to identify anomalies by computing the local reachability density of data points and comparing them to their k-nearest neighbors.

## Component Testing: Contractive Autoencoder (CAE)

**Script:** `train_cae_component.py`
**Description:** Evaluates a Contractive Autoencoder (CAE) component mathematically in pure NumPy, testing its ability to learn robust representations by penalizing the Frobenius norm of the Jacobian of hidden representations with respect to inputs, minimizing sensitivity to small perturbations.

## Component Testing: Conditional Random Field (CRF)

**Script:** `train_crf_component.py`
**Description:** Evaluates a Conditional Random Field (CRF) component mathematically in pure NumPy, testing its ability to model sequence tagging tasks using the forward-backward algorithm to compute exact marginals and gradients for transition and emission potentials.
## GraphSAGE added for GNN experiments
- Particle Filter component added to the system.

## Component Testing: Value Iteration

**Script:** `train_value_iter_component.py`
**Description:** Evaluates a Value Iteration component mathematically in pure NumPy, testing its ability to find the optimal value function and extract the optimal policy for a Markov Decision Process by repeatedly applying the Bellman optimality update.

## Component Testing: Policy Iteration

**Script:** `train_policy_iteration_component.py`
**Description:** Evaluates a Policy Iteration component mathematically in pure NumPy, testing its ability to find the optimal policy for a Markov Decision Process by alternating between iterative policy evaluation and greedy policy improvement.
- **Sparse PCA**: Explored sparse principal component analysis using L1-regularized power iteration.
- **CP Decomposition**: Explored CANDECOMP/PARAFAC tensor decomposition via simulated ALS.
- **Tucker Decomposition**: Explored Tucker tensor decomposition (HOSVD) via simulated HOOI.

## Component Testing: Markov Chain Monte Carlo (MCMC)

**Script:** `train_mcmc_component.py`
**Description:** Evaluates a Markov Chain Monte Carlo (MCMC) component mathematically in pure NumPy, testing its ability to sample from a complex target distribution using the Metropolis-Hastings algorithm and accurately estimate theoretical distribution statistics.
- **Advantage Actor-Critic (A2C):** Implemented an on-policy RL algorithm that combines a parameterized policy (actor) and a value function (critic) using an advantage estimate to reduce variance.
## Component Testing: Genetic Algorithm

**Script:** `train_genetic_algorithm_component.py`
**Description:** Evaluates a Genetic Algorithm component mathematically in pure NumPy, testing its ability to find the global minimum of a non-convex function (Rastrigin) using population-based search mechanisms like selection, crossover, and mutation without requiring gradients.
## Component Testing: Differential Evolution (DE)
**Script:** `train_differential_evolution_component.py`
**Description:** Evaluates a Differential Evolution component, verifying gradient-free black-box optimization over continuous functions using scaled vector differences, successfully finding the global minimum of a multi-modal test function without backpropagation.
- **Gaussian Process Regression (GPR):** Implemented for non-parametric Bayesian modeling and uncertainty estimation over functions.
## Component Testing: Ant Colony Optimization

**Script:** `train_aco_component.py`
**Description:** Evaluates an Ant Colony Optimization component mathematically in pure NumPy, testing its ability to solve the Traveling Salesperson Problem (TSP) using swarm intelligence principles, pheromone updates, and heuristic information.
- Disentangled Continuous Representations (Beta-VAE)
## Component Testing: GloVe

**Script:** `train_glove_component.py`
**Description:** Evaluates a GloVe component mathematically in pure NumPy, testing its ability to learn word embeddings by explicitly factorizing a word co-occurrence matrix using a weighted least squares objective.

## Component Testing: Latent Semantic Analysis (LSA)

**Script:** `train_lsa_component.py`
**Description:** Evaluates a Latent Semantic Analysis (LSA) component mathematically in pure NumPy, testing its ability to extract latent topics and relationships between words and documents by applying Singular Value Decomposition (SVD) to a term-frequency inverse-document-frequency (TF-IDF) matrix.
## Component Testing: Laplacian Eigenmaps

**Script:** `train_laplacian_eigenmaps_component.py`
**Description:** Evaluates a Laplacian Eigenmaps component mathematically in pure NumPy, testing its ability to map data into a lower-dimensional space while optimally preserving local neighbor relations by solving a generalized eigenvalue problem on the graph Laplacian.

## Component Testing: Squeeze-and-Excitation Block

**Script:** `train_se_block_component.py`
**Description:** Evaluates a Squeeze-and-Excitation (SE) block component mathematically in pure NumPy, testing its ability to recalibrate channel-wise feature responses by explicitly modeling interdependencies between channels.
- **Thompson Sampling**: Explored Bayesian multi-armed bandits, balancing exploration and exploitation using Beta distribution priors mathematically.
- Train DNC component.
- Train NEAT component.
- Train CMA-ES component.

## Component Testing: Kernel PCA
**Script:** `train_kpca_component.py`
**Description:** Evaluates a Kernel PCA component mathematically in pure NumPy, testing its ability to perform non-linear dimensionality reduction by projecting data into a high-dimensional feature space using the RBF kernel trick and finding the principal components in that space.

## Component Testing: LinUCB
**Script:** `train_linucb_component.py`
**Description:** Evaluates a contextual multi-armed bandit component mathematically using LinUCB, testing its ability to estimate expected rewards from contexts and minimizing regret over time.

## Component Testing: Matrix Factorization
**Script:** train_matrix_factorization_component.py
**Description:** Implemented Matrix Factorization mathematically for collaborative filtering using Stochastic Gradient Descent to learn latent factor representations for recommendation systems.

## Component Testing: Siamese Network

**Script:** `train_siamese_network_component.py`
**Description:** Evaluates a Siamese Network mathematically in pure NumPy, testing its ability to learn a distance metric embedding space using Triplet Loss, mapping similar inputs closer together and dissimilar inputs further apart.

## Component Testing: Tsetlin Machine
**Script:** `train_tsetlin_machine_component.py`
**Description:** Evaluates a Tsetlin Machine component mathematically in pure NumPy, testing its ability to solve the non-linear XOR problem by learning propositional logic clauses using Tsetlin Automata and Type I/II feedback.

## Component Testing: Affinity Propagation
**Script:** train_affinity_propagation_component.py
**Description:** Implemented and verified Affinity Propagation clustering algorithm by modeling data points as a network that exchanges messages until high-quality exemplars and clusters emerge.

## Component Testing: LDA Topic Model
**Script:** `train_lda_topic_model_component.py`
**Description:** Evaluates a Latent Dirichlet Allocation (LDA) Topic Model mathematically in pure NumPy, testing its ability to discover latent thematic structures in documents using Gibbs Sampling for iterative word-topic assignment based on Dirichlet priors.

## Component Testing: SimCLR
**Script:** train_simclr_component.py
**Description:** Implemented SimCLR for self-supervised contrastive learning using the NT-Xent loss.

## Component Testing: Echo State Network (ESN)
**Script:** `train_echo_state_network_component.py`
**Description:** Evaluates an Echo State Network (ESN) component mathematically in pure NumPy, testing its ability to forecast time-series data using a fixed, randomly initialized recurrent reservoir and a trainable linear readout layer optimized via Ridge Regression.

## Component Testing: Hopfield Network
**Script:** `train_hopfield_network_component.py`
**Description:** Evaluates a discrete Hopfield Network component mathematically in pure NumPy, testing its ability to function as an associative memory system by storing patterns using Hebbian learning and recalling them from corrupted inputs via asynchronous updates.

## Component Testing: Factorization Machine
**Script:** `train_factorization_machine_component.py`
**Description:** Evaluates a Factorization Machine component mathematically in pure NumPy, testing its ability to model pairwise interactions in sparse data by factoring interaction weights into latent vectors, optimized using Stochastic Gradient Descent.

## Component Testing: MADE
**Script:** train_made_component.py
**Description:** Implements and verifies MADE (Masked Autoencoder for Distribution Estimation) for auto-regressive density estimation.

## Component Testing: NCF
**Script:** train_ncf_component.py
**Description:** Implemented and tested a Neural Collaborative Filtering model using PyTorch, capturing complex non-linear patterns of user-item interactions via GMF and MLP layers.

## Component Testing: PageRank
**Script:** train_pagerank_component.py
**Description:** As part of our exploration into node centrality and link analysis, we have implemented PageRank mathematically to evaluate the stationary distribution over network nodes.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the PageRank component test:**
    ```bash
    python3 train_pagerank_component.py
    ```

## Component Testing: HITS
**Script:** train_hits_component.py
**Description:** As part of our exploration into link analysis and node ranking, we have implemented the HITS algorithm mathematically to estimate hub and authority scores in a directed network.

1.  **Ensure you have NumPy installed:**
    ```bash
    pip install numpy
    ```
2.  **Run the HITS component test:**
    ```bash
    python3 train_hits_component.py
    ```

## Component Testing: Sparse Coding
**Script:** train_sparse_coding_component.py
**Description:** Implements Sparse Coding to learn an overcomplete dictionary and sparse representations using ISTA-like inference and gradient descent updates, verifying sparse feature extraction properties.

## Component Testing: Robust PCA
**Script:** train_robust_pca_component.py
**Description:** Implemented Robust Principal Component Analysis (Robust PCA) mathematically to decompose a matrix into a low-rank and sparse component using the Alternating Direction Method of Multipliers (ADMM).

## Component Testing: Locality Sensitive Hashing (LSH)

**Script:** `train_lsh_component.py`
**Description:** Evaluates a Locality Sensitive Hashing component mathematically in pure NumPy, testing its ability to group similar items into buckets using random projections, accelerating approximate nearest neighbor search based on cosine similarity.

## Component Testing: Bayesian Ridge Regression
**Script:** train_bayesian_ridge_component.py
**Description:** Trains a Bayesian Ridge Regression model, which estimates not only the model weights but also their uncertainties via a probabilistic formulation, returning both predictions and standard deviations.

## Component Testing: Reptile Meta-Learning
**Script:** train_reptile_component.py
**Description:** Evaluates a Reptile meta-learning component mathematically in pure NumPy, testing its ability to find optimal initialization weights across a family of sine-wave regression tasks by accumulating inner-loop parameter differences instead of full second-order derivatives.

## Component Testing: Conditional Generative Adversarial Network
**Script:** train_cgan_component.py
**Description:** Implemented and verified Conditional Generative Adversarial Network (CGAN) training for 1D conditional Gaussian generation.

## Component Testing: Fourier Neural Operator (FNO)
**Script:** train_fourier_neural_operator_component.py
**Description:** Implements a 1D Fourier Neural Operator to learn mappings between continuous functions, typically used as neural PDE solvers.

## Component Testing: Spline Regression
**Script:** train_spline_regression_component.py
**Description:** Explores non-linear regression mathematically by implementing cubic spline regression with explicit basis functions and knots using ordinary least squares in pure NumPy.

## Component Testing: Singular Value Decomposition
**Script:** train_svd_component.py
**Description:** Train a singular value decomposition (SVD) component on synthetic data.

## Component Testing: Orthogonal Matching Pursuit (OMP)
**Script:** `train_omp_component.py`
**Description:** Evaluates an Orthogonal Matching Pursuit (OMP) component mathematically in pure NumPy, testing its ability to recover sparse representations by greedily selecting dictionary atoms that most correlate with the current residual, updating coefficients via orthogonal projection.

## Component Testing: Locality Preserving Projections
**Script:** train_lpp_component.py
**Description:** Train a Locality Preserving Projections (LPP) component mathematically to perform linear dimensionality reduction while optimally preserving the local neighborhood structure of the original data points using a k-nearest neighbor graph.

## Component Testing: Relevance Vector Machine
**Script:** train_rvm_component.py
**Description:** Train a Relevance Vector Machine (RVM) component mathematically to perform sparse Bayesian learning, identifying a highly sparse set of relevance vectors that form the decision boundary without the rigid margin-based constraints of SVMs.

## Component Testing: Learning Vector Quantization (LVQ)

**Script:** `train_lvq_component.py`
**Description:** Evaluates a Learning Vector Quantization (LVQ) component mathematically in pure NumPy, testing its ability to learn class-specific prototypes and classify based on the nearest prototype.

## Component Testing: Nadaraya-Watson Kernel Regression
**Script:** train_kernel_regression_component.py
**Description:** Train a Nadaraya-Watson Kernel Regression component mathematically to perform non-parametric non-linear regression, estimating continuous targets as a distance-weighted average of training samples using a Gaussian RBF kernel.

## Component Testing: Bayesian Network
**Script:** train_bayesian_network_component.py
**Description:** Implements a simple Bayesian Network (Naive Bayes structure) to model probabilistic dependencies between variables, compute likelihoods, and perform conditional inference on synthetic data.

## Component Testing: Graph Autoencoder (GAE)

**Script:** `train_gae_component.py`
**Description:** Evaluates a Graph Autoencoder (GAE) component mathematically in pure NumPy, testing its ability to encode graph node features and topology into a lower-dimensional space and then decode them to reconstruct the original adjacency matrix via binary cross-entropy loss.

## Component Testing: Simple Diffusion Model
**Script:** `train_diffusion_model_component.py`
**Description:** Train a Simple Diffusion Model mathematically to perform score matching on a forward continuous-time noising process, learning to map from a prior normal distribution to the simple 2D target data distribution (circle).

## Component Testing: Wasserstein Autoencoder (WAE)
**Script:** `train_wae_component.py`
**Description:** Implemented and verified Wasserstein Autoencoder - WAE with MMD.

## Component Testing: train_fno_component
**Script:** train_fno_component.py
**Description:** Implements a 1D Fourier Neural Operator (FNO) for learning operators, testing mapping between functional spaces rather than finite-dimensional vectors, verifying capability of operator learning methodologies.

## Component Testing: Extreme Learning Machine (ELM)
**Script:** `train_elm_component.py`
**Description:** Implemented and evaluated an Extreme Learning Machine (ELM) component using pure NumPy. This component verifies the mathematical hypothesis that randomly initializing hidden layer weights and analytically solving for the output weights using the Moore-Penrose pseudoinverse can provide rapid, one-shot learning of non-linear boundaries without iterative backpropagation.

## Component Testing: Independent Component Analysis (ICA)
**Script:** `train_ica_component.py`
**Description:** Implement and verify Independent Component Analysis (ICA) in pure NumPy using the FastICA algorithm. This explores unsupervised representation learning for separating linearly mixed, non-Gaussian source signals (blind source separation), modeling the cocktail party problem.

## Component Testing: Hidden Markov Model (HMM)
**Script:** `train_hmm_component.py`
**Description:** Evaluates a Hidden Markov Model (HMM) component mathematically in pure NumPy, testing its ability to estimate hidden state transition and observation emission probabilities given only a sequence of observations using the Baum-Welch (Expectation-Maximization) algorithm.

## Component Testing: AdaBoost
**Script:** `train_adaboost_component.py`
**Description:** Evaluates an AdaBoost component mathematically in pure NumPy, testing its ability to combine weak learners (Decision Stumps) into a strong classifier by sequentially updating sample weights based on classification errors.

## Component Testing: Bayesian Neural Network (BNN)
**Script:** `train_bnn_component.py`
**Description:** Implement and train a Bayesian Neural Network (BNN) component mathematically in pure NumPy using the Bayes by Backprop algorithm to learn a non-linear dataset (XOR) while estimating uncertainty.

## Component Testing: Gaussian Mixture Model (GMM)
**Script:** `train_gmm_component.py`
**Description:** Evaluates a Gaussian Mixture Model (GMM) component mathematically in pure NumPy, testing its ability to estimate the parameters of a mixture of multiple Gaussian distributions using the Expectation-Maximization (EM) algorithm.
## Component Testing: Linear Regression
**Script:** train_linear_regression_component.py
**Description:** Train a Linear Regression component mathematically to find the best fit line using the Normal Equation.

## Component Testing: Perceptron
**Script:** `train_perceptron_component.py`
**Description:** Evaluates a single-layer perceptron component mathematically in pure NumPy, testing its ability to learn a linearly separable decision boundary via manual backpropagation.
