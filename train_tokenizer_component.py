import collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word in v_in:
        w_out = word.replace(bigram, replacement)
        v_out[w_out] = v_in[word]
    return v_out

class BPETokenizer:
    def __init__(self, num_merges=10):
        self.num_merges = num_merges
        self.merges = []

    def fit(self, text):
        words = text.strip().split()
        # Initialize vocabulary with character-level tokens and </w>
        vocab = collections.defaultdict(int)
        for word in words:
            vocab[" ".join(list(word)) + " </w>"] += 1

        for i in range(self.num_merges):
            pairs = get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = merge_vocab(best, vocab)
            self.merges.append(best)

        return vocab

if __name__ == "__main__":
    text = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"
    print("Training Tokenizer (Byte Pair Encoding proxy)...")
    tokenizer = BPETokenizer(num_merges=10)
    final_vocab = tokenizer.fit(text)
    print(f"Completed {len(tokenizer.merges)} merges.")
    print("Learned merges:", tokenizer.merges)
    print("Final vocabulary state:", dict(final_vocab))
    print("Success")
