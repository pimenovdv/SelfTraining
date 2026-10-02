import numpy as np
import json
import os

def cutmix_augmentation(x1, x2, y1, y2, alpha=1.0):
    """
    CutMix augmentation mathematically.
    x1, x2: (B, C, H, W)
    y1, y2: (B, num_classes) (one-hot or probabilities)
    """
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x1.shape

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x_out = np.copy(x1)
    x_out[:, :, bby1:bby2, bbx1:bbx2] = x2[:, :, bby1:bby2, bbx1:bbx2]

    lam_adjusted = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_out = lam_adjusted * y1 + (1 - lam_adjusted) * y2

    return x_out, y_out

def test_cutmix_component():
    print("Testing CutMix Component...")
    B, C, H, W = 2, 3, 32, 32
    num_classes = 10

    np.random.seed(42)
    x1 = np.random.randn(B, C, H, W)
    x2 = np.random.randn(B, C, H, W)

    y1 = np.zeros((B, num_classes))
    y1[:, 1] = 1.0 # Class 1

    y2 = np.zeros((B, num_classes))
    y2[:, 2] = 1.0 # Class 2

    x_out, y_out = cutmix_augmentation(x1, x2, y1, y2, alpha=1.0)

    print(f"Original x1 shape: {x1.shape}")
    print(f"Output x_out shape: {x_out.shape}")
    print(f"Output y_out: \n{y_out}")

    assert x_out.shape == x1.shape, "Output shape mismatch"
    assert y_out.shape == y1.shape, "Output label shape mismatch"

    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        "x_out_shape": list(x_out.shape),
        "y_out": y_out.tolist()
    }
    with open("results/cutmix_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to results/cutmix_results.json")

if __name__ == "__main__":
    test_cutmix_component()
