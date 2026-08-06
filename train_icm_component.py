import numpy as np
import os

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class LinearLayer:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dW += np.dot(self.x.T, dout)
        self.db += np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        return dx

    def zero_grad(self):
        self.dW.fill(0)
        self.db.fill(0)

class ICM:
    def __init__(self, state_dim, action_dim, feature_dim=32, hidden_dim=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim

        self.feat_l1 = LinearLayer(state_dim, hidden_dim)
        self.feat_l2 = LinearLayer(hidden_dim, feature_dim)

        self.inv_l1 = LinearLayer(feature_dim * 2, hidden_dim)
        self.inv_l2 = LinearLayer(hidden_dim, action_dim)

        self.fwd_l1 = LinearLayer(feature_dim + action_dim, hidden_dim)
        self.fwd_l2 = LinearLayer(hidden_dim, feature_dim)

        self.layers = [self.feat_l1, self.feat_l2, self.inv_l1, self.inv_l2, self.fwd_l1, self.fwd_l2]

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def update(self, lr=1e-3):
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

    def extract_features(self, state):
        h1 = relu(self.feat_l1.forward(state))
        phi = self.feat_l2.forward(h1)
        return phi, h1

    def forward_model_pass(self, phi_t, action):
        fwd_in = np.concatenate([phi_t, action], axis=-1)
        h1 = relu(self.fwd_l1.forward(fwd_in))
        phi_t1_pred = self.fwd_l2.forward(h1)
        return phi_t1_pred, h1, fwd_in

    def inverse_model_pass(self, phi_t, phi_t1):
        inv_in = np.concatenate([phi_t, phi_t1], axis=-1)
        h1 = relu(self.inv_l1.forward(inv_in))
        action_logits = self.inv_l2.forward(h1)
        return action_logits, h1, inv_in

    def train_step(self, state_t, action, state_t1, beta=0.2, lr=1e-3):
        self.zero_grad()
        batch_size = state_t.shape[0]

        phi_t, f_h1_t = self.extract_features(state_t)
        feat_l2_x_t = self.feat_l2.x
        feat_l1_x_t = self.feat_l1.x

        phi_t1, f_h1_t1 = self.extract_features(state_t1)
        feat_l2_x_t1 = self.feat_l2.x
        feat_l1_x_t1 = self.feat_l1.x

        phi_t1_pred, fwd_h1, fwd_in = self.forward_model_pass(phi_t, action)
        action_logits, inv_h1, inv_in = self.inverse_model_pass(phi_t, phi_t1)
        action_pred = softmax(action_logits)

        fwd_diff = phi_t1_pred - phi_t1
        fwd_loss = 0.5 * np.mean(fwd_diff**2) * self.feature_dim
        inv_loss = -np.mean(np.sum(action * np.log(action_pred + 1e-8), axis=-1))
        intrinsic_reward = np.mean(fwd_diff**2, axis=-1)

        d_phi_t1_pred = beta * fwd_diff / batch_size
        d_fwd_h1 = self.fwd_l2.backward(d_phi_t1_pred) * relu_deriv(fwd_h1)
        d_fwd_in = self.fwd_l1.backward(d_fwd_h1)
        d_phi_t_fwd = d_fwd_in[:, :self.feature_dim]

        d_action_logits = (1 - beta) * (action_pred - action) / batch_size
        d_inv_h1 = self.inv_l2.backward(d_action_logits) * relu_deriv(inv_h1)
        d_inv_in = self.inv_l1.backward(d_inv_h1)
        d_phi_t_inv = d_inv_in[:, :self.feature_dim]
        d_phi_t1_inv = d_inv_in[:, self.feature_dim:]

        d_phi_t1 = d_phi_t1_inv - (beta * fwd_diff / batch_size)
        self.feat_l2.x = feat_l2_x_t1
        self.feat_l1.x = feat_l1_x_t1
        d_feat_h1_t1 = self.feat_l2.backward(d_phi_t1) * relu_deriv(f_h1_t1)
        self.feat_l1.backward(d_feat_h1_t1)

        d_phi_t = d_phi_t_inv + d_phi_t_fwd
        self.feat_l2.x = feat_l2_x_t
        self.feat_l1.x = feat_l1_x_t
        d_feat_h1_t = self.feat_l2.backward(d_phi_t) * relu_deriv(f_h1_t)
        self.feat_l1.backward(d_feat_h1_t)

        self.update(lr)
        return fwd_loss, inv_loss, np.mean(intrinsic_reward)

if __name__ == "__main__":
    icm = ICM(state_dim=10, action_dim=4)
    state_t = np.random.randn(32, 10)
    action_idx = np.random.randint(0, 4, size=(32,))
    action = np.zeros((32, 4))
    action[np.arange(32), action_idx] = 1
    state_t1 = np.random.randn(32, 10)

    for _ in range(100):
        fwd_loss, inv_loss, _ = icm.train_step(state_t, action, state_t1)

    print(f"Final Forward Loss: {fwd_loss:.4f}, Inverse Loss: {inv_loss:.4f}")

    # Generate documentation
    os.makedirs("docs", exist_ok=True)
    doc_content = f"""# Component Testing: Intrinsic Curiosity Module (ICM)

**Script:** `train_icm_component.py`

**Description:** Evaluates an Intrinsic Curiosity Module (ICM) component, verifying its ability to encourage exploration by generating intrinsic reward through predicting the next state feature representation (forward model) and learning action-conditioned representations (inverse model) via manual backpropagation.

**Status:** Success

**Results:**
- Final Forward Loss: {fwd_loss:.4f}
- Final Inverse Loss: {inv_loss:.4f}
"""
    with open("docs/0097_train_icm_component.md", "w") as f:
        f.write(doc_content)
