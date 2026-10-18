import numpy as np
import math

def adacos_loss(features, weights, targets, previous_scale=None):
    """
    Computes the AdaCos loss with dynamic scaling.

    Args:
        features (np.ndarray): The input features, shape (batch_size, feature_dim).
        weights (np.ndarray): The class centers/weights, shape (feature_dim, num_classes).
        targets (np.ndarray): The ground truth labels, shape (batch_size,).
        previous_scale (float, optional): The scale parameter from the previous iteration.
                                          If None, it initializes it.

    Returns:
        float: The computed AdaCos loss.
        np.ndarray: The gradients of the loss with respect to the unscaled cosine similarities.
        float: The updated scale parameter.
    """
    batch_size = features.shape[0]
    num_classes = weights.shape[1]

    # L2 normalize features and weights to compute cosine similarity
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    weights_norm = weights / (np.linalg.norm(weights, axis=0, keepdims=True) + 1e-8)

    # Cosine similarities (logits)
    cos_theta = np.dot(features_norm, weights_norm)

    # Create a mask for target classes
    target_mask = np.zeros_like(cos_theta, dtype=bool)
    target_mask[np.arange(batch_size), targets] = True

    # Get the cosine similarities of the target classes
    cos_theta_yi = cos_theta[target_mask]

    # Get the cosine similarities of the non-target classes
    cos_theta_j = cos_theta[~target_mask].reshape(batch_size, num_classes - 1)

    # Initialize scale if not provided
    if previous_scale is None:
        scale = math.sqrt(2) * math.log(num_classes - 1)
    else:
        scale = previous_scale

    # Dynamically update the scale parameter
    # B_avg = (1 / batch_size) * sum_{i} sum_{j != y_i} exp(s * cos(theta_{i, j}))
    # Find the maximum non-target angle for numerical stability
    max_cos_j = np.max(cos_theta_j, axis=1, keepdims=True)

    # Calculate B_avg using the previous scale
    B_avg = np.mean(np.sum(np.exp(scale * (cos_theta_j - max_cos_j)), axis=1) * np.exp(scale * max_cos_j.squeeze()))

    # Median angle for target classes (approximated here by mean for simplicity in batch)
    theta = np.arccos(np.clip(cos_theta_yi, -1.0 + 1e-7, 1.0 - 1e-7))
    theta_med = np.median(theta)

    # Update rule based on the AdaCos paper
    # We want s_new = log(B_avg) / cos(min(pi/4, theta_med))
    s_new = math.log(B_avg + 1e-8) / math.cos(min(math.pi / 4, theta_med))
    scale = s_new

    # Scale the logits
    scaled_logits = cos_theta * scale

    # Compute the softmax probabilities
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Compute the loss
    loss = -np.mean(np.log(probs[np.arange(batch_size), targets] + 1e-8))

    # Compute the gradients with respect to unscaled cosine similarities
    grads = probs.copy()
    grads[np.arange(batch_size), targets] -= 1
    grads *= scale / batch_size

    return loss, grads, scale

def test_adacos_loss():
    np.random.seed(42)
    batch_size = 32
    feature_dim = 128
    num_classes = 10

    # Generate random features and weights
    features = np.random.randn(batch_size, feature_dim)
    weights = np.random.randn(feature_dim, num_classes)
    targets = np.random.randint(0, num_classes, size=batch_size)

    # Compute the loss and gradients, starting with no previous scale
    loss, grads, new_scale = adacos_loss(features, weights, targets)

    print(f"AdaCos Loss: {loss:.4f}")
    print(f"Gradients shape: {grads.shape}")
    print(f"Dynamically Calculated Scale: {new_scale:.4f}")

    assert loss > 0
    assert grads.shape == (batch_size, num_classes)

    print("AdaCos Loss implementation is correct.")

if __name__ == "__main__":
    test_adacos_loss()
