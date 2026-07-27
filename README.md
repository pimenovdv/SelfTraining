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
    python train_bpe.py
    ```
    This script will read the sample text data from `data/sample_text.txt`, perform BPE merges, and save the resulting vocabulary and merges to `models/tokenizer/`.
3.  **Adjusting hyperparameters:**
    You can customize the tokenizer training using command-line arguments:
    ```bash
    python train_bpe.py --data_path "data/sample_text.txt" --num_merges 200 --output_dir "models/my_tokenizer"
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
