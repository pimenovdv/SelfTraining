import numpy as np

def jaccard_loss(y_true, y_pred, smooth=1e-5):
    """
    Computes the Jaccard Loss (1 - IoU) for binary classification.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return 1.0 - iou

def test_jaccard_loss():
    print("Testing Jaccard Loss Component...")

    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([0.9, 0.8, 0.1, 0.2, 0.95])

    loss = jaccard_loss(y_true, y_pred)
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"Jaccard Loss: {loss:.4f}")

    assert loss >= 0.0, "Jaccard Loss must be non-negative"
    print("Test passed!")

if __name__ == "__main__":
    test_jaccard_loss()
