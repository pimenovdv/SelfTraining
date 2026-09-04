import numpy as np

def conv2d_forward(X, W, b, stride=1, padding=0):
    n, c, h, w = X.shape
    f, _, fh, fw = W.shape

    out_h = (h + 2 * padding - fh) // stride + 1
    out_w = (w + 2 * padding - fw) // stride + 1

    if padding > 0:
        X_padded = np.pad(X, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
    else:
        X_padded = X

    out = np.zeros((n, f, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + fh
            w_start = j * stride
            w_end = w_start + fw

            X_slice = X_padded[:, :, h_start:h_end, w_start:w_end]
            out[:, :, i, j] = np.einsum('ncyx,fcyx->nf', X_slice, W) + b

    return out, X_padded

def conv2d_backward(dout, X_padded, W, stride=1, padding=0):
    n, f, out_h, out_w = dout.shape
    _, c, h_padded, w_padded = X_padded.shape
    _, _, fh, fw = W.shape

    dX_padded = np.zeros_like(X_padded)
    dW = np.zeros_like(W)
    db = np.zeros((f,))

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + fh
            w_start = j * stride
            w_end = w_start + fw

            X_slice = X_padded[:, :, h_start:h_end, w_start:w_end]

            for n_idx in range(n):
                for f_idx in range(f):
                    dW[f_idx] += X_slice[n_idx] * dout[n_idx, f_idx, i, j]
                    dX_padded[n_idx, :, h_start:h_end, w_start:w_end] += W[f_idx] * dout[n_idx, f_idx, i, j]

    db = np.sum(dout, axis=(0, 2, 3))

    if padding > 0:
        dX = dX_padded[:, :, padding:-padding, padding:-padding]
    else:
        dX = dX_padded

    return dX, dW, db

def maxpool2d_forward(X, size=2, stride=2):
    n, c, h, w = X.shape
    out_h = (h - size) // stride + 1
    out_w = (w - size) // stride + 1

    out = np.zeros((n, c, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + size
            w_start = j * stride
            w_end = w_start + size

            X_slice = X[:, :, h_start:h_end, w_start:w_end]
            out[:, :, i, j] = np.max(X_slice, axis=(2, 3))

    return out, X

def maxpool2d_backward(dout, X, size=2, stride=2):
    n, c, out_h, out_w = dout.shape
    dX = np.zeros_like(X)

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + size
            w_start = j * stride
            w_end = w_start + size

            X_slice = X[:, :, h_start:h_end, w_start:w_end]

            for n_idx in range(n):
                for c_idx in range(c):
                    x_slice_nc = X_slice[n_idx, c_idx]
                    mask = (x_slice_nc == np.max(x_slice_nc))
                    dX[n_idx, c_idx, h_start:h_end, w_start:w_end] += mask * dout[n_idx, c_idx, i, j]

    return dX

def relu(x):
    return np.maximum(0, x)

def relu_backward(dout, x):
    dX = dout.copy()
    dX[x <= 0] = 0
    return dX

def cross_entropy_loss(logits, targets):
    m = targets.shape[0]
    p = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    p /= np.sum(p, axis=1, keepdims=True)

    log_likelihood = -np.log(p[range(m), targets] + 1e-9)
    loss = np.sum(log_likelihood) / m

    dp = p.copy()
    dp[range(m), targets] -= 1
    dp /= m
    return loss, dp

if __name__ == "__main__":
    np.random.seed(42)

    N = 100
    X = np.random.randn(N, 1, 8, 8)
    y = np.random.randint(0, 2, size=(N,))
    X[y == 1, 0, 3:5, 3:5] += 2.0

    f_num = 4
    W_conv = np.random.randn(f_num, 1, 3, 3) * 0.1
    b_conv = np.zeros(f_num)

    W_fc = np.random.randn(36, 2) * 0.1
    b_fc = np.zeros(2)

    lr = 0.1
    epochs = 100

    for epoch in range(epochs):
        out_conv, X_padded = conv2d_forward(X, W_conv, b_conv, stride=1, padding=0)
        out_relu = relu(out_conv)
        out_pool, _ = maxpool2d_forward(out_relu, size=2, stride=2)

        out_flat = out_pool.reshape(N, -1)

        logits = out_flat.dot(W_fc) + b_fc

        loss, dlogits = cross_entropy_loss(logits, y)

        dW_fc = out_flat.T.dot(dlogits)
        db_fc = np.sum(dlogits, axis=0)
        dout_flat = dlogits.dot(W_fc.T)

        dout_pool = dout_flat.reshape(out_pool.shape)
        dout_relu = maxpool2d_backward(dout_pool, out_relu, size=2, stride=2)
        dout_conv = relu_backward(dout_relu, out_conv)
        dX, dW_conv, db_conv = conv2d_backward(dout_conv, X_padded, W_conv, stride=1, padding=0)

        W_fc -= lr * dW_fc
        b_fc -= lr * db_fc
        W_conv -= lr * dW_conv
        b_conv -= lr * db_conv

        if epoch % 10 == 0:
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y)
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    print("Training complete.")
