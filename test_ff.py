import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

class FFLayer:
    def __init__(self, in_features, out_features, lr=0.01, threshold=2.0):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.b = np.zeros(out_features)
        self.lr = lr
        self.threshold = threshold

    def normalize(self, x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norms + 1e-8)

    def forward(self, x):
        x_norm = self.normalize(x)
        return relu(np.dot(x_norm, self.W) + self.b)

    def train_step(self, x_pos, x_neg):
        x_pos_norm = self.normalize(x_pos)
        x_neg_norm = self.normalize(x_neg)

        a_pos = relu(np.dot(x_pos_norm, self.W) + self.b)
        a_neg = relu(np.dot(x_neg_norm, self.W) + self.b)

        g_pos = np.sum(a_pos ** 2, axis=1)
        g_neg = np.sum(a_neg ** 2, axis=1)

        p_pos = sigmoid(g_pos - self.threshold)
        p_neg = sigmoid(g_neg - self.threshold)

        grad_g_pos = p_pos - 1.0
        grad_g_neg = p_neg

        grad_z_pos = 2 * a_pos * grad_g_pos[:, np.newaxis]
        grad_z_neg = 2 * a_neg * grad_g_neg[:, np.newaxis]

        dW = np.dot(x_pos_norm.T, grad_z_pos) + np.dot(x_neg_norm.T, grad_z_neg)
        db = np.sum(grad_z_pos, axis=0) + np.sum(grad_z_neg, axis=0)

        self.W -= self.lr * dW / x_pos.shape[0]
        self.b -= self.lr * db / x_pos.shape[0]

        loss = np.mean(-np.log(p_pos + 1e-8) - np.log(1 - p_neg + 1e-8))
        return loss

# Generate data
np.random.seed(42)
N = 1000
X = np.random.randn(N, 2)
y = (X[:, 0] * X[:, 1] > 0).astype(int)
Y_onehot = np.zeros((N, 2))
Y_onehot[np.arange(N), y] = 1

Y_wrong = np.zeros((N, 2))
Y_wrong[np.arange(N), 1 - y] = 1

X_pos = np.hstack((X, Y_onehot))
X_neg = np.hstack((X, Y_wrong))

layer1 = FFLayer(4, 32, lr=0.1, threshold=2.0)
layer2 = FFLayer(32, 16, lr=0.1, threshold=2.0)

for epoch in range(100):
    loss1 = layer1.train_step(X_pos, X_neg)

    h_pos = layer1.forward(X_pos)
    h_neg = layer1.forward(X_neg)

    loss2 = layer2.train_step(h_pos, h_neg)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: L1={loss1:.4f}, L2={loss2:.4f}")

# Predict
def predict(x):
    goodness = []
    for label in [0, 1]:
        y_oh = np.zeros((x.shape[0], 2))
        y_oh[:, label] = 1
        x_in = np.hstack((x, y_oh))

        h1 = layer1.forward(x_in)
        h2 = layer2.forward(h1)
        g = np.sum(h1**2, axis=1) + np.sum(h2**2, axis=1)
        goodness.append(g)

    return np.argmax(np.array(goodness), axis=0)

preds = predict(X)
acc = np.mean(preds == y)
print("Accuracy:", acc)
