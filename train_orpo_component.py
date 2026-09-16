import numpy as np
import argparse

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_orpo_component(X_chosen, X_rejected, d_model, epochs, learning_rate, lambda_param=0.1):
    num_samples = X_chosen.shape[0]

    np.random.seed(42)
    W_policy = np.random.randn(d_model, 1) * 0.1

    for epoch in range(epochs):
        log_prob_chosen = np.sum(np.dot(X_chosen, W_policy), axis=1)
        log_prob_rejected = np.sum(np.dot(X_rejected, W_policy), axis=1)

        log_odds_ratio = log_prob_chosen - log_prob_rejected

        loss_or = np.mean(-np.log(sigmoid(log_odds_ratio) + 1e-10))
        loss_sft = np.mean(-log_prob_chosen)

        loss = loss_sft + lambda_param * loss_or

        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Total Loss = {loss:.4f}, L_SFT = {loss_sft:.4f}, L_OR = {loss_or:.4f}")

        d_log_odds_ratio = (sigmoid(log_odds_ratio) - 1.0)

        d_or_chosen = d_log_odds_ratio * lambda_param / num_samples
        d_or_rejected = -d_log_odds_ratio * lambda_param / num_samples

        d_log_prob_chosen = (-1.0 / num_samples) + d_or_chosen
        d_log_prob_rejected = d_or_rejected

        X_chosen_sum = np.sum(X_chosen, axis=1)
        X_rejected_sum = np.sum(X_rejected, axis=1)

        dW_policy = np.dot(X_chosen_sum.T, d_log_prob_chosen) + np.dot(X_rejected_sum.T, d_log_prob_rejected)

        W_policy -= learning_rate * dW_policy

    return W_policy

def main():
    parser = argparse.ArgumentParser(description="Train an ORPO component on synthetic data.")
    parser.add_argument("--d_model", type=int, default=4, help="Dimension of the model.")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--lambda_param", type=float, default=0.1, help="ORPO lambda parameter.")
    args = parser.parse_args()

    X_chosen = np.array([
        [[1.0, 0.5, 0.0, 0.0], [0.8, 0.6, 0.0, 0.0], [0.9, 0.7, 0.0, 0.0]],
        [[0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.8, 0.6], [0.0, 0.0, 0.9, 0.7]]
    ])
    X_rejected = np.array([
        [[-1.0, -0.5, 0.0, 0.0], [-0.8, -0.6, 0.0, 0.0], [-0.9, -0.7, 0.0, 0.0]],
        [[0.0, 0.0, -1.0, -0.5], [0.0, 0.0, -0.8, -0.6], [0.0, 0.0, -0.9, -0.7]]
    ])

    print(f"Training ORPO Component with d_model={args.d_model}, epochs={args.epochs}, lr={args.lr}, lambda={args.lambda_param}")

    W_policy = train_orpo_component(X_chosen, X_rejected, args.d_model, args.epochs, args.lr, args.lambda_param)

    print("\nTraining Complete.")
    print("Final Policy Weights:")
    print(W_policy)

if __name__ == "__main__":
    main()
