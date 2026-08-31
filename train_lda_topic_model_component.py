import numpy as np

def test_lda_topic_model():
    print("Initializing Latent Dirichlet Allocation (LDA) Topic Model test...")
    print("Testing mathematical principles of LDA for topic modeling.")

    # Simple simulated document-term matrix (documents x words)
    # 4 documents, 5 unique words
    # Doc 0 and 1 are about "topic 0" (words 0, 1)
    # Doc 2 and 3 are about "topic 1" (words 2, 3)
    # Word 4 is a common word
    docs = [
        [0, 1, 0, 4],
        [1, 0, 1, 4],
        [2, 3, 2, 4],
        [3, 2, 3, 4]
    ]

    D = len(docs)
    W = 5 # Vocab size
    K = 2 # Number of topics

    # Hyperparameters
    alpha = 0.1 # Document-topic Dirichlet prior
    beta = 0.1 # Topic-word Dirichlet prior

    # Initialization
    # Assign random topics to each word in each document
    np.random.seed(42)
    doc_topic_counts = np.zeros((D, K))
    topic_word_counts = np.zeros((K, W))
    topic_counts = np.zeros(K)

    word_topics = []
    for d, doc in enumerate(docs):
        topics_for_doc = []
        for w in doc:
            z = np.random.randint(K)
            topics_for_doc.append(z)
            doc_topic_counts[d, z] += 1
            topic_word_counts[z, w] += 1
            topic_counts[z] += 1
        word_topics.append(topics_for_doc)

    print(f"Initial doc_topic_counts:\n{doc_topic_counts}")
    print(f"Initial topic_word_counts:\n{topic_word_counts}")

    # Gibbs Sampling
    n_iters = 100
    for iter in range(n_iters):
        for d, doc in enumerate(docs):
            for i, w in enumerate(doc):
                # Current topic
                z = word_topics[d][i]

                # Decrement counts
                doc_topic_counts[d, z] -= 1
                topic_word_counts[z, w] -= 1
                topic_counts[z] -= 1

                # Calculate probabilities for new topic
                # p(z | ...) \propto (n_{d,k} + alpha) * (n_{k,w} + beta) / (n_k + W * beta)
                p_z = np.zeros(K)
                for k in range(K):
                    p_doc_topic = (doc_topic_counts[d, k] + alpha) / (len(doc) - 1 + K * alpha)
                    p_topic_word = (topic_word_counts[k, w] + beta) / (topic_counts[k] + W * beta)
                    p_z[k] = p_doc_topic * p_topic_word

                # Normalize probabilities
                p_z /= np.sum(p_z)

                # Sample new topic
                new_z = np.random.choice(K, p=p_z)

                # Increment counts
                word_topics[d][i] = new_z
                doc_topic_counts[d, new_z] += 1
                topic_word_counts[new_z, w] += 1
                topic_counts[new_z] += 1

    print(f"\nFinal doc_topic_counts (after {n_iters} iterations):\n{doc_topic_counts}")
    print(f"Final topic_word_counts:\n{topic_word_counts}")

    # Calculate topic distributions
    theta = np.zeros((D, K)) # Document-topic distribution
    phi = np.zeros((K, W)) # Topic-word distribution

    for d in range(D):
        for k in range(K):
            theta[d, k] = (doc_topic_counts[d, k] + alpha) / (len(docs[d]) + K * alpha)

    for k in range(K):
        for w in range(W):
            phi[k, w] = (topic_word_counts[k, w] + beta) / (topic_counts[k] + W * beta)

    print(f"\nDocument-Topic distribution (Theta):\n{theta}")
    print(f"Topic-Word distribution (Phi):\n{phi}")

    # Assertions to check if it learned something reasonable
    # Docs 0,1 should have similar topic distributions, Docs 2,3 should have similar topic distributions
    # Since topic assignments are random, we just check if it separated them

    topic_doc_0 = np.argmax(theta[0])
    topic_doc_1 = np.argmax(theta[1])
    topic_doc_2 = np.argmax(theta[2])
    topic_doc_3 = np.argmax(theta[3])

    assert topic_doc_0 == topic_doc_1, "Doc 0 and 1 should be assigned to the same dominant topic"
    assert topic_doc_2 == topic_doc_3, "Doc 2 and 3 should be assigned to the same dominant topic"
    assert topic_doc_0 != topic_doc_2, "Docs 0,1 and Docs 2,3 should have different dominant topics"

    print("LDA Topic Model test passed successfully.")

if __name__ == "__main__":
    test_lda_topic_model()
