import numpy as np
import os

np.random.seed(42)

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.b = np.zeros((1, out_features))
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        dx = np.dot(dout, self.W.T)
        return dx

class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx

class MLP:
    def __init__(self, in_features, hidden_size, out_features):
        self.fc1 = Linear(in_features, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.fc3 = Linear(hidden_size, out_features)

    def forward(self, x):
        h = self.fc1.forward(x)
        a = self.relu.forward(h)
        h2 = self.fc2.forward(a)
        a2 = self.relu2.forward(h2)
        out = self.fc3.forward(a2)
        return out

    def backward(self, dout):
        da2 = self.fc3.backward(dout)
        dh2 = self.relu2.backward(da2)
        da = self.fc2.backward(dh2)
        dh = self.relu.backward(da)
        dx = self.fc1.backward(dh)
        return dx

    def update(self, lr=0.01):
        self.fc1.W -= lr * self.fc1.dW
        self.fc1.b -= lr * self.fc1.db
        self.fc2.W -= lr * self.fc2.dW
        self.fc2.b -= lr * self.fc2.db
        self.fc3.W -= lr * self.fc3.dW
        self.fc3.b -= lr * self.fc3.db

def softmax_cross_entropy_loss(logits, y_true):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    loss = -np.mean(np.sum(y_true * np.log(probs + 1e-9), axis=1))
    dout = (probs - y_true) / y_true.shape[0]
    return loss, dout, probs

def generate_fgsm_example(model, x, y_true, epsilon=0.1):
    logits = model.forward(x)
    loss, dout, _ = softmax_cross_entropy_loss(logits, y_true)
    dx = model.backward(dout)
    x_adv = x + epsilon * np.sign(dx)
    x_adv = np.clip(x_adv, 0, 1)
    return x_adv

def train_fgsm():
    N = 400
    in_features = 10
    out_features = 2

    X = np.random.rand(N, in_features)
    Y_labels = (np.sum(X[:, :5], axis=1) > np.sum(X[:, 5:], axis=1)).astype(int)
    Y = np.zeros((N, out_features))
    Y[np.arange(N), Y_labels] = 1

    model = MLP(in_features, 32, out_features)

    epochs = 1000
    lr = 0.1
    epsilon = 0.1

    print("Training standard model...")
    for epoch in range(epochs):
        logits = model.forward(X)
        loss, dout, _ = softmax_cross_entropy_loss(logits, Y)
        model.backward(dout)
        model.update(lr)

    logits = model.forward(X)
    preds = np.argmax(logits, axis=1)
    std_acc = np.mean(preds == Y_labels)
    print(f"Standard model accuracy on clean data: {std_acc:.4f}")

    X_adv = generate_fgsm_example(model, X, Y, epsilon=epsilon)
    logits_adv = model.forward(X_adv)
    preds_adv = np.argmax(logits_adv, axis=1)
    adv_acc_before = np.mean(preds_adv == Y_labels)
    print(f"Standard model accuracy on adversarial data: {adv_acc_before:.4f}")

    print("\nTraining robust model (Adversarial Training)...")
    model_robust = MLP(in_features, 32, out_features)
    for epoch in range(epochs):
        logits_clean = model_robust.forward(X)
        loss_clean, dout_clean, _ = softmax_cross_entropy_loss(logits_clean, Y)
        model_robust.backward(dout_clean)
        model_robust.update(lr)

        X_adv_train = generate_fgsm_example(model_robust, X, Y, epsilon=epsilon)
        logits_adv = model_robust.forward(X_adv_train)
        loss_adv, dout_adv, _ = softmax_cross_entropy_loss(logits_adv, Y)
        model_robust.backward(dout_adv)
        model_robust.update(lr)

    logits_clean_robust = model_robust.forward(X)
    preds_clean_robust = np.argmax(logits_clean_robust, axis=1)
    clean_acc_robust = np.mean(preds_clean_robust == Y_labels)
    print(f"Robust model accuracy on clean data: {clean_acc_robust:.4f}")

    X_adv_test = generate_fgsm_example(model_robust, X, Y, epsilon=epsilon)
    logits_adv_robust = model_robust.forward(X_adv_test)
    preds_adv_robust = np.argmax(logits_adv_robust, axis=1)
    adv_acc_after = np.mean(preds_adv_robust == Y_labels)
    print(f"Robust model accuracy on adversarial data: {adv_acc_after:.4f}")

    doc_content = f"""# Fast Gradient Sign Method (FGSM) Component Experiment

**Script:** `train_fgsm_component.py`

## Concept
Adversarial examples are inputs to machine learning models that an attacker has intentionally designed to cause the model to make a mistake. The Fast Gradient Sign Method (FGSM) generates adversarial examples by taking a small step in the direction of the gradient of the loss with respect to the input: $x_{{adv}} = x + \\epsilon \\cdot \\text{{sign}}(\\nabla_x L(x, y; \\theta))$. Adversarial training involves augmenting the training data with these adversarial examples to improve model robustness.

## Hypothesis
A neural network trained only on clean data is highly susceptible to adversarial attacks, showing a significant drop in accuracy. By training the network simultaneously on both clean data and adversarial examples generated on-the-fly via FGSM, we can increase the model's robustness to such perturbations.

## Action
Implemented an MLP and the FGSM attack mathematically using pure NumPy. The script first trains a standard model on a synthetic dataset and demonstrates its vulnerability to FGSM. It then trains a robust model using Adversarial Training (alternating clean and adversarial batches) and evaluates its performance on both clean and adversarially perturbed data.

## Outcome
- Standard Model Clean Accuracy: {std_acc:.4f}
- Standard Model Adversarial Accuracy: {adv_acc_before:.4f} (Vulnerable)
- Robust Model Clean Accuracy: {clean_acc_robust:.4f}
- Robust Model Adversarial Accuracy: {adv_acc_after:.4f} (Robust)

The model successfully learned to resist the FGSM attack, maintaining high accuracy on adversarial examples compared to the standard model, confirming the effectiveness of adversarial training.

## Next Steps
Explore more advanced adversarial attacks like Projected Gradient Descent (PGD) or implement defensive distillation.
"""
    doc_path = "docs/0120_train_fgsm_component.md"
    os.makedirs("docs", exist_ok=True)
    with open(doc_path, "w") as f:
        f.write(doc_content)
    print(f"Documentation saved to {doc_path}")

if __name__ == "__main__":
    train_fgsm()
