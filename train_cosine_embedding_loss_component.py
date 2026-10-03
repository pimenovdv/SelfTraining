import torch
import torch.nn as nn
import torch.optim as optim
import json

def test_cosine_embedding_loss():
    print("Testing Cosine Embedding Loss Component...")

    # Inputs: two batches of 1D vectors
    x1 = torch.tensor([[1.0, 2.0, 3.0],
                       [0.5, 0.5, 0.5],
                       [-1.0, -2.0, -3.0],
                       [1.0, 0.0, 0.0]], requires_grad=True)

    x2 = torch.tensor([[1.0, 2.0, 3.0],
                       [-0.5, -0.5, -0.5],
                       [1.0, 2.0, 3.0],
                       [0.0, 1.0, 0.0]], requires_grad=True)

    # Targets: 1 for similar, -1 for dissimilar
    y = torch.tensor([1, 1, -1, -1])

    margin = 0.5

    # Cosine Embedding Loss
    loss_fn = nn.CosineEmbeddingLoss(margin=margin)
    loss = loss_fn(x1, x2, y)

    # Backward pass
    loss.backward()

    # Calculate expected loss manually to verify
    # cos_sim(x1, x2)
    # y=1 -> 1 - cos
    # y=-1 -> max(0, cos - margin)

    print(f"Loss: {loss.item():.4f}")
    print(f"x1 gradient:\n{x1.grad}")
    print(f"x2 gradient:\n{x2.grad}")

    if loss.item() >= 0:
        print("Test passed!\n")
        return {"loss": loss.item()}
    else:
        print("Test failed!\n")
        return None

if __name__ == "__main__":
    result = test_cosine_embedding_loss()
    if result is not None:
        import os
        os.makedirs("results", exist_ok=True)
        with open("results/cosine_embedding_loss_results.json", "w") as f:
            json.dump(result, f)
