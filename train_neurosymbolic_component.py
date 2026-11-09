import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LogicalRule:
    """Represents a simple differentiable logical rule."""
    def __init__(self, weight=1.0):
        self.weight = nn.Parameter(torch.tensor(weight))

    def forward(self, predicates):
        # Example rule: if A and B, then C
        # Represented softly using product T-norm for AND
        # predicates: (batch_size, num_predicates)
        # return satisfaction degree
        pass # To be implemented later, for now we will do a simpler neurosymbolic model

class NeurosymbolicModel(nn.Module):
    def __init__(self, input_dim, num_symbols, hidden_dim=32):
        super().__init__()
        # Neural perception module
        self.perception = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_symbols),
            nn.Sigmoid() # Output probabilities of symbols
        )

        # Symbolic reasoning module (soft logic)
        # We define a simple rule: Symbol 0 AND Symbol 1 -> Target
        # In a real system, this would be a complex set of rules.

    def forward(self, x):
        # 1. Perception
        symbol_probs = self.perception(x)

        # 2. Reasoning (differentiable logic)
        # Using product T-norm for AND: P(A and B) = P(A) * P(B)
        # Assuming we want to predict a target based on the first two symbols being true.
        rule_satisfaction = symbol_probs[:, 0] * symbol_probs[:, 1]

        return symbol_probs, rule_satisfaction

def generate_neurosymbolic_data(num_samples=1000):
    """
    Generates data where the true label depends on a logical combination
    of underlying hidden features that need to be perceived.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    # Inputs are random points in 2D
    X = np.random.randn(num_samples, 2)

    # Hidden concepts (symbols):
    # Symbol 0: X[0] > 0
    # Symbol 1: X[1] > 0
    symbols = np.zeros((num_samples, 2))
    symbols[:, 0] = (X[:, 0] > 0).astype(float)
    symbols[:, 1] = (X[:, 1] > 0).astype(float)

    # Final label based on rule: Symbol 0 AND Symbol 1
    y = (symbols[:, 0] * symbols[:, 1]).astype(float)

    # Add some noise to inputs to make perception non-trivial
    X += np.random.randn(num_samples, 2) * 0.1

    return torch.FloatTensor(X), torch.FloatTensor(symbols), torch.FloatTensor(y)

def test_neurosymbolic():
    print("Testing Neurosymbolic AI component...")

    X, true_symbols, y = generate_neurosymbolic_data(500)

    model = NeurosymbolicModel(input_dim=2, num_symbols=2)

    # Loss incorporates both task loss and symbolic constraints
    # Unsupervised reasoning could also be applied here using constraints
    bce = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()

        pred_symbols, pred_y = model(X)

        # Semi-supervised setup: we supervise the final output
        # and optionally the intermediate symbols.
        # Let's assume we have full supervision for demonstration.
        loss_task = bce(pred_y, y)
        loss_perception = bce(pred_symbols, true_symbols)

        loss = loss_task + 0.5 * loss_perception

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Task Loss: {loss_task.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        _, pred_y = model(X)
        predictions = (pred_y > 0.5).float()
        accuracy = (predictions == y).float().mean()
        print(f"Final Accuracy: {accuracy.item() * 100:.2f}%")

    assert accuracy > 0.8, "Model failed to learn the neurosymbolic mapping."
    print("Neurosymbolic AI component test passed!")

if __name__ == "__main__":
    test_neurosymbolic()
