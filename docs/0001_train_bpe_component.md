# Experiment 0001: Train BPE Tokenizer

## Objective
To implement and train a simple Byte-Pair Encoding (BPE) tokenizer from scratch. This serves as a foundational exercise to understand tokenization, a core component of modern language models, before attempting any large-scale integration.

## Setup
*   **Script:** `train_bpe_component.py`
*   **Data:** A small sample text corpus about Artificial General Intelligence located in `data/sample_text.txt`.
*   **Hyperparameters:** `num_merges` set to 100.

## Execution
The training script was executed successfully:
```bash
python train_bpe_component.py
```

## Results
*   **Status:** Success.
*   **Vocabulary Size:** The tokenizer learned a vocabulary of 144 tokens based on the 100 merges and base characters.
*   **Output Files:** The vocabulary (`vocab.json`) and merge rules (`merges.json`) were successfully saved to `models/tokenizer/`.

## Observations & Next Steps
*   The implementation correctly extracted base characters and performed iterative merging based on frequency.
*   For future iterations, we may want to test this implementation on a larger, more diverse dataset to observe scaling behavior.
*   We can also benchmark its performance against established tokenizer libraries (like Hugging Face's `tokenizers`) to validate correctness and efficiency.
