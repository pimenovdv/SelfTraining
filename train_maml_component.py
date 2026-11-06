import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def generate_task():
    a = np.random.uniform(0.1, 5.0)
    b = np.random.uniform(0.0, 2.0 * np.pi)

    def target_function(x):
        return a * torch.sin(x + b)

    return target_function

def sample_points(n):
    return torch.rand(n, 1) * 10.0 - 5.0

def train_maml():
    torch.manual_seed(42)
    np.random.seed(42)

    model = SimpleModel()
    meta_optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    inner_lr = 0.1
    num_meta_iterations = 100
    meta_batch_size = 10
    num_inner_steps = 1

    print("Starting MAML training...")

    for meta_iter in range(num_meta_iterations):
        meta_loss = 0.0

        # Manually accumulate gradients
        meta_grads = [torch.zeros_like(p) for p in model.parameters()]

        for task_idx in range(meta_batch_size):
            task = generate_task()

            # Support set
            x_support = sample_points(10)
            y_support = task(x_support)

            # Query set
            x_query = sample_points(10)
            y_query = task(x_query)

            # Clone model for inner loop (using higher order gradients manually)
            # For simplicity in this base implementation, we'll do one step inner update
            # and compute grads w.r.t initial params

            fast_weights = {}
            for name, param in model.named_parameters():
                fast_weights[name] = param.clone()

            # Inner loop (support set)
            predictions_support = model.fc2(torch.relu(model.fc1(x_support)))
            # actually we need to use fast_weights
            h = torch.relu(torch.nn.functional.linear(x_support, fast_weights['fc1.weight'], fast_weights['fc1.bias']))
            preds = torch.nn.functional.linear(h, fast_weights['fc2.weight'], fast_weights['fc2.bias'])

            loss_support = loss_fn(preds, y_support)

            grads = torch.autograd.grad(loss_support, fast_weights.values(), create_graph=True)

            fast_weights_new = {}
            for (name, param), grad in zip(fast_weights.items(), grads):
                fast_weights_new[name] = param - inner_lr * grad

            # Outer loop (query set)
            h_q = torch.relu(torch.nn.functional.linear(x_query, fast_weights_new['fc1.weight'], fast_weights_new['fc1.bias']))
            preds_q = torch.nn.functional.linear(h_q, fast_weights_new['fc2.weight'], fast_weights_new['fc2.bias'])

            loss_query = loss_fn(preds_q, y_query)
            meta_loss += loss_query

            # Compute gradients of meta-loss w.r.t initial parameters
            task_grads = torch.autograd.grad(loss_query, model.parameters(), retain_graph=True)
            for i, grad in enumerate(task_grads):
                meta_grads[i] += grad

        # Meta-update
        meta_optimizer.zero_grad()
        for p, g in zip(model.parameters(), meta_grads):
            p.grad = g / meta_batch_size
        meta_optimizer.step()

        if (meta_iter + 1) % 20 == 0:
            print(f"Meta-iteration {meta_iter + 1}/{num_meta_iterations}, Meta-Loss: {meta_loss.item() / meta_batch_size:.4f}")

    print("MAML training completed.")

if __name__ == "__main__":
    train_maml()
