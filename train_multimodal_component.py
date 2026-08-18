import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class MultimodalModel:
    def __init__(self, input_dim_a, input_dim_b, hidden_dim, shared_dim):
        self.W1_a = np.random.randn(input_dim_a, hidden_dim) * np.sqrt(2. / input_dim_a)
        self.b1_a = np.zeros(hidden_dim)
        self.W2_a = np.random.randn(hidden_dim, shared_dim) * np.sqrt(2. / hidden_dim)
        self.b2_a = np.zeros(shared_dim)

        self.W1_b = np.random.randn(input_dim_b, hidden_dim) * np.sqrt(2. / input_dim_b)
        self.b1_b = np.zeros(hidden_dim)
        self.W2_b = np.random.randn(hidden_dim, shared_dim) * np.sqrt(2. / hidden_dim)
        self.b2_b = np.zeros(shared_dim)

    def forward(self, x_a, x_b):
        self.x_a = x_a
        self.z1_a = np.dot(x_a, self.W1_a) + self.b1_a
        self.a1_a = relu(self.z1_a)
        self.out_a = np.dot(self.a1_a, self.W2_a) + self.b2_a

        self.x_b = x_b
        self.z1_b = np.dot(x_b, self.W1_b) + self.b1_b
        self.a1_b = relu(self.z1_b)
        self.out_b = np.dot(self.a1_b, self.W2_b) + self.b2_b

        self.norm_a = np.linalg.norm(self.out_a, axis=1, keepdims=True) + 1e-8
        self.norm_b = np.linalg.norm(self.out_b, axis=1, keepdims=True) + 1e-8

        self.proj_a = self.out_a / self.norm_a
        self.proj_b = self.out_b / self.norm_b

        # Scaling parameter for contrastive learning (often called temperature)
        self.tau = 0.1
        self.logits = np.dot(self.proj_a, self.proj_b.T) / self.tau
        return self.logits

    def backward(self, d_logits, lr=0.01):
        batch_size = d_logits.shape[0]
        d_logits = d_logits / self.tau

        d_proj_a = np.dot(d_logits, self.proj_b)
        d_proj_b = np.dot(d_logits.T, self.proj_a)

        d_out_a = (d_proj_a - self.proj_a * np.sum(d_proj_a * self.proj_a, axis=1, keepdims=True)) / self.norm_a
        d_out_b = (d_proj_b - self.proj_b * np.sum(d_proj_b * self.proj_b, axis=1, keepdims=True)) / self.norm_b

        dW2_a = np.dot(self.a1_a.T, d_out_a) / batch_size
        db2_a = np.sum(d_out_a, axis=0) / batch_size
        d_a1_a = np.dot(d_out_a, self.W2_a.T)
        d_z1_a = d_a1_a * relu_deriv(self.z1_a)
        dW1_a = np.dot(self.x_a.T, d_z1_a) / batch_size
        db1_a = np.sum(d_z1_a, axis=0) / batch_size

        dW2_b = np.dot(self.a1_b.T, d_out_b) / batch_size
        db2_b = np.sum(d_out_b, axis=0) / batch_size
        d_a1_b = np.dot(d_out_b, self.W2_b.T)
        d_z1_b = d_a1_b * relu_deriv(self.z1_b)
        dW1_b = np.dot(self.x_b.T, d_z1_b) / batch_size
        db1_b = np.sum(d_z1_b, axis=0) / batch_size

        self.W1_a -= lr * dW1_a
        self.b1_a -= lr * db1_a
        self.W2_a -= lr * dW2_a
        self.b2_a -= lr * db2_a

        self.W1_b -= lr * dW1_b
        self.b1_b -= lr * db1_b
        self.W2_b -= lr * dW2_b
        self.b2_b -= lr * db2_b

def train_multimodal_model():
    np.random.seed(42)
    batch_size = 32
    input_dim_a = 64
    input_dim_b = 128
    hidden_dim = 32
    shared_dim = 16
    epochs = 2000
    lr = 0.1

    model = MultimodalModel(input_dim_a, input_dim_b, hidden_dim, shared_dim)

    latent_concepts = np.random.randn(batch_size, 8)
    A_proj = np.random.randn(8, input_dim_a)
    X_a = np.dot(latent_concepts, A_proj) + np.random.randn(batch_size, input_dim_a) * 0.1
    B_proj = np.random.randn(8, input_dim_b)
    X_b = np.dot(latent_concepts, B_proj) + np.random.randn(batch_size, input_dim_b) * 0.1

    labels = np.arange(batch_size)

    for epoch in range(epochs):
        logits = model.forward(X_a, X_b)

        probs = softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(batch_size), labels] + 1e-8))

        d_logits = probs.copy()
        d_logits[np.arange(batch_size), labels] -= 1

        # Compute gradient (the division by batch_size happens in backward)
        model.backward(d_logits, lr=lr)

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    logits = model.forward(X_a, X_b)
    accuracy = np.mean(np.argmax(logits, axis=1) == labels)
    print(f"Final Alignment Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    train_multimodal_model()
