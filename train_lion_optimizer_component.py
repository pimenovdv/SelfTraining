import torch
import torch.nn as nn

class Lion(torch.optim.Optimizer):
    """
    Lion optimizer from Google Research (EvoLved Sign Momentum).
    Algorithm:
    c = beta1 * m + (1 - beta1) * g
    update = sign(c)
    w = w - lr * (update + weight_decay * w)
    m = beta2 * m + (1 - beta2) * g
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Update step: c = beta1 * m + (1 - beta1) * g
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])

                # Update momentum: m = beta2 * m + (1 - beta2) * g
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

def test_lion_optimizer():
    torch.manual_seed(42)
    # Simple linear regression task
    X = torch.randn(100, 10)
    true_w = torch.randn(10, 1)
    y = X @ true_w + 0.1 * torch.randn(100, 1)

    model = nn.Linear(10, 1, bias=False)
    criterion = nn.MSELoss()
    optimizer = Lion(model.parameters(), lr=0.1, weight_decay=1e-3)

    print("Initial Loss:", criterion(model(X), y).item())

    for epoch in range(1000):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(X), y).item()
    print("Final Loss:", final_loss)

    if final_loss < 0.1:
        print("Success: Lion optimizer converged well.")
    else:
        print("Warning: Lion optimizer didn't converge as expected.")

if __name__ == "__main__":
    test_lion_optimizer()
