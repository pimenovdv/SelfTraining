import numpy as np

def cosine_similarity(y_true, y_pred):
    """
    Computes the cosine similarity between two sets of vectors.
    Returns a value between -1 and 1.
    """
    y_true_norm = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True) + 1e-7)
    y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-7)
    return np.sum(y_true_norm * y_pred_norm, axis=1)

def cosine_similarity_loss(y_true, y_pred):
    """
    Computes the cosine similarity loss.
    Loss is minimized when similarity is 1. Loss = 1 - similarity.
    """
    similarity = cosine_similarity(y_true, y_pred)
    return np.mean(1 - similarity)

def cosine_similarity_loss_gradient(y_true, y_pred):
    """
    Computes the gradient of the cosine similarity loss with respect to y_pred.
    """
    norm_y_true = np.linalg.norm(y_true, axis=1, keepdims=True) + 1e-7
    norm_y_pred = np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-7

    y_true_norm = y_true / norm_y_true
    y_pred_norm = y_pred / norm_y_pred

    similarity = np.sum(y_true_norm * y_pred_norm, axis=1, keepdims=True)

    # Gradient of (1 - similarity) wrt y_pred
    # The gradient of similarity wrt y_pred is:
    # 1/norm_y_pred * (y_true_norm - similarity * y_pred_norm)
    # So gradient of loss is:
    N = y_true.shape[0]
    grad = -(y_true_norm - similarity * y_pred_norm) / norm_y_pred
    return grad / N

def test_cosine_similarity_loss():
    print("Testing Cosine Similarity Loss Component...")

    np.random.seed(42)
    N = 100
    D = 10

    y_true = np.random.randn(N, D)
    y_pred = np.random.randn(N, D)

    initial_loss = cosine_similarity_loss(y_true, y_pred)
    print(f"Initial Cosine Similarity Loss: {initial_loss:.4f}")

    learning_rate = 500.0  # Increased learning rate for faster convergence
    epochs = 200

    for epoch in range(epochs):
        grad = cosine_similarity_loss_gradient(y_true, y_pred)
        y_pred -= learning_rate * grad

        if epoch % 50 == 0:
            loss = cosine_similarity_loss(y_true, y_pred)
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    final_loss = cosine_similarity_loss(y_true, y_pred)
    print(f"Final Cosine Similarity Loss: {final_loss:.4f}")

    if final_loss < initial_loss and final_loss < 0.1:
        print("Optimization successful! Predicted vectors aligned with target vectors.")
    else:
        print("Optimization failed.")

if __name__ == "__main__":
    test_cosine_similarity_loss()
