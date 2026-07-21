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
