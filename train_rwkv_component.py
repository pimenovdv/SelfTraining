import numpy as np

class RWKVTimeMixing:
    def __init__(self, channels):
        self.channels = channels
        self.time_decay = np.random.randn(channels)
        self.time_first = np.random.randn(channels)
        self.time_mix_k = np.random.rand(channels)
        self.time_mix_v = np.random.rand(channels)
        self.time_mix_r = np.random.rand(channels)
        self.Wk = np.random.randn(channels, channels) / np.sqrt(channels)
        self.Wv = np.random.randn(channels, channels) / np.sqrt(channels)
        self.Wr = np.random.randn(channels, channels) / np.sqrt(channels)
        self.Wo = np.random.randn(channels, channels) / np.sqrt(channels)

    def forward(self, x):
        seq_len, channels = x.shape
        out = np.zeros_like(x)
        aa = np.zeros(channels)
        bb = np.zeros(channels)
        pp = np.full(channels, -np.inf)
        prev_x = np.zeros(channels)
        for t in range(seq_len):
            xt = x[t]
            xk = x[t] * self.time_mix_k + prev_x * (1 - self.time_mix_k)
            xv = x[t] * self.time_mix_v + prev_x * (1 - self.time_mix_v)
            xr = x[t] * self.time_mix_r + prev_x * (1 - self.time_mix_r)
            prev_x = xt
            k = xk @ self.Wk
            v = xv @ self.Wv
            r = xr @ self.Wr
            r = 1 / (1 + np.exp(-r))
            ww = self.time_first + k
            p = np.maximum(pp, ww)
            e1 = np.exp(pp - p)
            e2 = np.exp(ww - p)
            wkv = (e1 * aa + e2 * v) / (e1 * bb + e2)
            ww = self.time_decay + pp
            p = np.maximum(ww, k)
            e1 = np.exp(ww - p)
            e2 = np.exp(k - p)
            aa = e1 * aa + e2 * v
            bb = e1 * bb + e2
            pp = p
            out[t] = (r * wkv) @ self.Wo
        return out

if __name__ == "__main__":
    np.random.seed(42)
    seq_len = 10
    channels = 16
    x = np.random.randn(seq_len, channels)
    rwkv = RWKVTimeMixing(channels)
    out = rwkv.forward(x)
    print("RWKV Component output shape:", out.shape)
    print(f"RWKV outputs mean: {np.mean(out):.4f}")
    print("Success")
