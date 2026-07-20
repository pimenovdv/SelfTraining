import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


def get_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """Given a vocabulary (dictionary mapping words to frequency counts), returns a dictionary of pairs and their frequencies."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair: Tuple[str, str], v_in: Dict[str, int]) -> Dict[str, int]:
    """Merges a pair of symbols in a vocabulary."""
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word in v_in:
        w_out = p.sub("".join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def train_bpe(text: str, num_merges: int) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """Trains a BPE tokenizer on a given text."""
    # Pre-tokenize: split text into words and add </w> to end of each word
    words = text.strip().split()
    vocab_base = defaultdict(int)
    for word in words:
        vocab_base[" ".join(list(word)) + " </w>"] += 1

    vocab = vocab_base
    merges = []

    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
        print(f"Merge {i + 1}/{num_merges}: {best}")

    # Extract final token vocabulary
    final_vocab = set()
    for word in vocab:
        final_vocab.update(word.split())

    # Assign integer IDs to tokens
    token_to_id = {token: i for i, token in enumerate(sorted(list(final_vocab)))}

    return token_to_id, merges


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from scratch.")
    parser.add_argument("--data_path", type=str, default="data/sample_text.txt", help="Path to training data.")
    parser.add_argument("--num_merges", type=int, default=100, help="Number of BPE merges to perform.")
    parser.add_argument("--output_dir", type=str, default="models/tokenizer", help="Directory to save tokenizer.")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return

    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Training BPE tokenizer with {args.num_merges} merges...")
    vocab, merges = train_bpe(text, args.num_merges)

    print(f"Training complete. Vocabulary size: {len(vocab)}")

    os.makedirs(args.output_dir, exist_ok=True)

    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    merges_path = os.path.join(args.output_dir, "merges.json")
    with open(merges_path, "w", encoding="utf-8") as f:
        # Convert tuples to strings for JSON serialization
        json.dump([f"{m[0]} {m[1]}" for m in merges], f, ensure_ascii=False, indent=2)

    print(f"Tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()
