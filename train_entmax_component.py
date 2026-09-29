import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import time

def entmax15(x, dim=-1):
    """
    Entmax 1.5 in pure PyTorch (mathematical implementation).
    Finds a sparse probability distribution p minimizing the alpha-divergence (alpha=1.5).
    """
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x = x - x_max

    sorted_x, _ = torch.sort(x, dim=dim, descending=True)
    d = x.size(dim)

    rho = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    shape = [1] * x.dim()
    shape[dim] = d
    rho = rho.view(shape)

    cumsum_x = torch.cumsum(sorted_x, dim=dim)
    cumsum_x_sq = torch.cumsum(sorted_x ** 2, dim=dim)

    delta = rho - rho * cumsum_x_sq + cumsum_x ** 2
    delta = torch.clamp(delta, min=0)

    tau = (cumsum_x - torch.sqrt(delta)) / rho
    support = sorted_x > tau

    rho_idx = support.sum(dim=dim, keepdim=True) - 1
    rho_idx = torch.clamp(rho_idx, min=0)

    tau_star = torch.gather(tau, dim, rho_idx)
    p = torch.clamp(x - tau_star, min=0) ** 2

    return p


class Entmax15Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=-1):
        ctx.dim = dim
        p = entmax15(x, dim=dim)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, grad_output):
        p, = ctx.saved_tensors
        dim = ctx.dim
        sqrt_p = torch.sqrt(p)
        sum_sqrt_p = torch.sum(sqrt_p, dim=dim, keepdim=True)
        sum_grad_sqrt = torch.sum(grad_output * sqrt_p, dim=dim, keepdim=True)
        grad_input = sqrt_p * (grad_output - sum_grad_sqrt / sum_sqrt_p)
        return grad_input, None

class Entmax15Layer(nn.Module):
    def __init__(self, dim=-1):
        super(Entmax15Layer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return Entmax15Function.apply(x, self.dim)

class EntmaxClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EntmaxClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.entmax = Entmax15Layer(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.entmax(x)
        return x

def main():
    print("Testing Entmax15 Component...")
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=4, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = EntmaxClassifier(input_dim=20, hidden_dim=64, output_dim=4)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 100
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        probs = model(X_train_t)
        log_probs = torch.log(probs + 1e-8)
        loss = criterion(log_probs, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    train_time = time.time() - start_time
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test_t)
        predictions = torch.argmax(test_probs, dim=1)
        accuracy = accuracy_score(y_test, predictions.numpy())
        sparsity = (test_probs == 0).float().mean().item()

    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Sparsity (fraction of exactly zero probabilities): {sparsity * 100:.2f}%")

    if accuracy > 0.7:
        print("Entmax15 component successfully learned!")
    else:
        print("Warning: Accuracy is lower than expected.")

if __name__ == "__main__":
    main()
