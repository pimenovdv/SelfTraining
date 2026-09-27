import numpy as np

class CenterLoss:
    def __init__(self, num_classes, feature_dim, alpha=0.5):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        # Initialize centers with small random values
        self.centers = np.random.randn(num_classes, feature_dim) * 0.01

    def forward(self, features, labels):
        """
        Compute the center loss.
        features: (batch_size, feature_dim)
        labels: (batch_size,) - integer class labels
        """
        batch_size = features.shape[0]
        # Get the centers for each sample
        centers_batch = self.centers[labels]
        # Compute the squared L2 distance between features and their centers
        loss = np.sum((features - centers_batch) ** 2) / (2.0 * batch_size)
        return loss

    def backward(self, features, labels):
        """
        Compute the gradient of the center loss with respect to features and update centers.
        """
        batch_size = features.shape[0]
        centers_batch = self.centers[labels]

        # Gradient w.r.t features
        grad_features = (features - centers_batch) / batch_size

        # Update centers
        # Calculate the number of samples in the batch for each class to average the update
        unique_classes, class_counts = np.unique(labels, return_counts=True)
        for i, cls in enumerate(unique_classes):
            cls_mask = (labels == cls)
            # Sum the differences for the class
            diff_sum = np.sum(centers_batch[cls_mask] - features[cls_mask], axis=0)
            # Update center
            self.centers[cls] -= self.alpha * diff_sum / (1.0 + class_counts[i])

        return grad_features

def test_center_loss_component():
    np.random.seed(42)
    num_classes = 10
    feature_dim = 2
    batch_size = 32

    # Create fake features and labels
    features = np.random.randn(batch_size, feature_dim)
    # Give them some structure so loss can decrease
    labels = np.random.randint(0, num_classes, size=batch_size)
    features += labels.reshape(-1, 1) * 2.0

    criterion = CenterLoss(num_classes, feature_dim, alpha=0.5)

    print("Testing Center Loss Component...")

    initial_loss = criterion.forward(features, labels)
    print(f"Initial Center Loss: {initial_loss:.4f}")

    # Optimize features to minimize center loss for a few steps
    learning_rate = 0.5
    for step in range(50):
        loss = criterion.forward(features, labels)
        grad = criterion.backward(features, labels)
        features -= learning_rate * grad

    final_loss = criterion.forward(features, labels)
    print(f"Final Center Loss: {final_loss:.4f}")

    assert final_loss < initial_loss, "Center loss did not decrease!"
    print("Center Loss Component implemented and verified successfully.")

if __name__ == "__main__":
    test_center_loss_component()
