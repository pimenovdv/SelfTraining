import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

class CVAE:
    def __init__(self, input_dim, cond_dim, hidden_dim, latent_dim):
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        enc_in_dim = input_dim + cond_dim
        self.W_e1 = np.random.randn(enc_in_dim, hidden_dim) * np.sqrt(2. / enc_in_dim)
        self.b_e1 = np.zeros(hidden_dim)

        self.W_mu = np.random.randn(hidden_dim, latent_dim) * np.sqrt(1. / hidden_dim)
        self.b_mu = np.zeros(latent_dim)

        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * np.sqrt(1. / hidden_dim)
        self.b_logvar = np.zeros(latent_dim)

        dec_in_dim = latent_dim + cond_dim
        self.W_d1 = np.random.randn(dec_in_dim, hidden_dim) * np.sqrt(2. / dec_in_dim)
        self.b_d1 = np.zeros(hidden_dim)

        self.W_out = np.random.randn(hidden_dim, input_dim) * np.sqrt(1. / hidden_dim)
        self.b_out = np.zeros(input_dim)

    def forward(self, X, Y):
        self.X = X
        self.Y = Y
        self.enc_in = np.concatenate([X, Y], axis=1)

        self.H_e_z = self.enc_in @ self.W_e1 + self.b_e1
        self.H_e = relu(self.H_e_z)

        self.mu = self.H_e @ self.W_mu + self.b_mu
        self.logvar = self.H_e @ self.W_logvar + self.b_logvar

        self.std = np.exp(0.5 * self.logvar)
        self.eps = np.random.randn(*self.mu.shape)
        self.Z = self.mu + self.std * self.eps

        self.dec_in = np.concatenate([self.Z, Y], axis=1)

        self.H_d_z = self.dec_in @ self.W_d1 + self.b_d1
        self.H_d = relu(self.H_d_z)

        self.Out_z = self.H_d @ self.W_out + self.b_out
        self.Out = sigmoid(self.Out_z)

        return self.Out

    def backward(self, X, Y):
        batch_size = X.shape[0]

        dOut_z = (self.Out - X) / batch_size

        dW_out = self.H_d.T @ dOut_z
        db_out = np.sum(dOut_z, axis=0)
        dH_d = dOut_z @ self.W_out.T

        dH_d_z = dH_d * relu_deriv(self.H_d_z)

        dW_d1 = self.dec_in.T @ dH_d_z
        db_d1 = np.sum(dH_d_z, axis=0)
        ddec_in = dH_d_z @ self.W_d1.T

        dZ = ddec_in[:, :self.latent_dim]

        dmu_recon = dZ
        dstd = dZ * self.eps
        dlogvar_recon = dstd * 0.5 * self.std

        dmu_kl = self.mu / batch_size
        dlogvar_kl = 0.5 * (np.exp(self.logvar) - 1.0) / batch_size

        dmu = dmu_recon + dmu_kl
        dlogvar = dlogvar_recon + dlogvar_kl

        dW_mu = self.H_e.T @ dmu
        db_mu = np.sum(dmu, axis=0)
        dH_e_mu = dmu @ self.W_mu.T

        dW_logvar = self.H_e.T @ dlogvar
        db_logvar = np.sum(dlogvar, axis=0)
        dH_e_logvar = dlogvar @ self.W_logvar.T

        dH_e = dH_e_mu + dH_e_logvar
        dH_e_z = dH_e * relu_deriv(self.H_e_z)

        dW_e1 = self.enc_in.T @ dH_e_z
        db_e1 = np.sum(dH_e_z, axis=0)

        return {
            'W_e1': dW_e1, 'b_e1': db_e1,
            'W_mu': dW_mu, 'b_mu': db_mu,
            'W_logvar': dW_logvar, 'b_logvar': db_logvar,
            'W_d1': dW_d1, 'b_d1': db_d1,
            'W_out': dW_out, 'b_out': db_out
        }

class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)

            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

if __name__ == "__main__":
    np.random.seed(42)

    X = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 1]
    ], dtype=float)

    Y = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ], dtype=float)

    cvae = CVAE(input_dim=4, cond_dim=4, hidden_dim=16, latent_dim=2)

    params = {
        'W_e1': cvae.W_e1, 'b_e1': cvae.b_e1,
        'W_mu': cvae.W_mu, 'b_mu': cvae.b_mu,
        'W_logvar': cvae.W_logvar, 'b_logvar': cvae.b_logvar,
        'W_d1': cvae.W_d1, 'b_d1': cvae.b_d1,
        'W_out': cvae.W_out, 'b_out': cvae.b_out
    }
    optimizer = AdamOptimizer(params, lr=0.01)

    epochs = 3000
    for epoch in range(epochs):
        Out = cvae.forward(X, Y)
        recon_loss = -np.sum(X * np.log(Out + 1e-8) + (1 - X) * np.log(1 - Out + 1e-8)) / X.shape[0]
        kl_loss = -0.5 * np.sum(1 + cvae.logvar - cvae.mu**2 - np.exp(cvae.logvar)) / X.shape[0]
        total_loss = recon_loss + kl_loss

        grads = cvae.backward(X, Y)
        optimizer.step(params, grads)

    print(f"Final Loss: {total_loss:.4f} (Recon: {recon_loss:.4f}, KL: {kl_loss:.4f})")
    print("Sample reconstructions (rounded):")
    print(np.round(Out[:4], 2))
