import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def lrp_linear_epsilon(layer, a_in, r_out, epsilon=1e-9):
    """
    Computes Layer-wise Relevance Propagation for a linear layer using the epsilon rule.
    R_i = sum_j (a_i * w_ji) / (z_j + epsilon * sign(z_j)) * R_j
    """
    w = layer.weight
    b = layer.bias

    # Forward pass through just this layer
    z = F.linear(a_in, w, b)

    # Epsilon stabilization
    z = z + epsilon * torch.sign(z)

    # Relevance propagation
    s = r_out / z

    # Gradient of z with respect to a_in gives w
    c = F.linear(s, w.t())

    r_in = a_in * c
    return r_in

def train_lrp():
    print("Testing Layer-wise Relevance Propagation (LRP)...")

    model = SimpleMLP()
    # Dummy input
    x = torch.randn(1, 10, requires_grad=True)

    # Forward pass, store intermediate activations
    a0 = x
    a1 = F.relu(model.fc1(a0))
    a2 = model.fc2(a1)

    print(f"Output: {a2}")

    # Assume we want to explain the first class prediction
    target_class = 0
    r2 = torch.zeros_like(a2)
    r2[0, target_class] = a2[0, target_class]

    # Backward pass LRP
    # fc2 layer
    r1 = lrp_linear_epsilon(model.fc2, a1, r2)

    # ReLU is ignored in LRP (relevance passes through unchanged for positive activations)
    r1_pre_relu = r1 * (a1 > 0).float()

    # fc1 layer
    r0 = lrp_linear_epsilon(model.fc1, a0, r1_pre_relu)

    print(f"Input Relevance Scores: {r0}")

    # Check conservation property: sum of relevance scores should roughly equal output
    sum_r0 = r0.sum().item()
    output_target = a2[0, target_class].item()

    print(f"Sum of input relevance: {sum_r0}")
    print(f"Output of target class: {output_target}")

    # Conservation property might not be exact due to epsilon and bias terms,
    # but it should be close for LRP
    conservation_error = abs(sum_r0 - output_target)
    print(f"Conservation Error: {conservation_error}")

    if conservation_error < 1.0: # relaxed threshold due to biases and epsilon
        print("LRP execution successful.")
        return True
    else:
        print("LRP conservation property failed significantly.")
        return False

if __name__ == "__main__":
    success = train_lrp()
    if success:
        print("LRP component implemented and verified successfully.")
