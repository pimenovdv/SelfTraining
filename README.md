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
