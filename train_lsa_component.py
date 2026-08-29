"""
Latent Semantic Analysis (LSA) Component
"""
import numpy as np

def compute_tf(document):
    tf_dict = {}
    total_words = len(document)
    if total_words == 0:
        return tf_dict
    for word in document:
        tf_dict[word] = tf_dict.get(word, 0) + 1
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / total_words
    return tf_dict

def compute_idf(documents):
    import math
    N = len(documents)
    idf_dict = {}
    all_words = set(word for doc in documents for word in doc)
    for word in all_words:
        count = sum(1 for doc in documents if word in doc)
        idf_dict[word] = math.log((1 + N) / (1 + count)) + 1 # smoothed IDF
    return idf_dict

def compute_tfidf(documents):
    idf = compute_idf(documents)
    tfidf_matrix = []
    all_words = sorted(list(idf.keys()))

    for doc in documents:
        tf = compute_tf(doc)
        tfidf_vector = []
        for word in all_words:
            tfidf_vector.append(tf.get(word, 0) * idf[word])
        # L2 normalization
        norm = np.linalg.norm(tfidf_vector)
        if norm > 0:
            tfidf_vector = (np.array(tfidf_vector) / norm).tolist()
        tfidf_matrix.append(tfidf_vector)

    return np.array(tfidf_matrix), all_words

def lsa(tfidf_matrix, k):
    # Perform SVD
    U, S, Vt = np.linalg.svd(tfidf_matrix, full_matrices=False)
    # Truncate to k dimensions
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    # Document embeddings
    doc_embeddings = np.dot(U_k, S_k)
    return doc_embeddings, U, S, Vt

if __name__ == "__main__":
    print("Testing Latent Semantic Analysis (LSA) Component...")
    docs = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "barked"],
        ["the", "cat", "meowed"],
        ["a", "car", "drove", "by"]
    ]
    tfidf, vocab = compute_tfidf(docs)
    print(f"Vocabulary size: {len(vocab)}")

    k = 2
    doc_embeddings, U, S, Vt = lsa(tfidf, k=k)

    print(f"Original TF-IDF shape: {tfidf.shape}")
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    assert doc_embeddings.shape == (4, 2)
    print("LSA component executed successfully.")
