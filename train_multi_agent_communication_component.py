import numpy as np

class MultiAgentCommunication:
    def __init__(self, vocab_size=10, message_dim=16, learning_rate=0.1):
        self.vocab_size = vocab_size
        self.message_dim = message_dim
        self.lr = learning_rate

        # Sender: maps target index to a continuous message
        self.W_sender = np.random.randn(vocab_size, message_dim) * 0.1

        # Receiver: maps message back to a probability distribution over targets
        self.W_receiver = np.random.randn(message_dim, vocab_size) * 0.1

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def train_step(self, target_idx):
        # Sender generates message
        x = np.zeros(self.vocab_size)
        x[target_idx] = 1.0

        message = np.dot(x, self.W_sender)

        # Receiver processes message
        logits = np.dot(message, self.W_receiver)
        probs = self.softmax(logits)

        # Compute loss (Cross Entropy)
        loss = -np.log(probs[target_idx] + 1e-8)

        # Backpropagation
        dlogits = probs.copy()
        dlogits[target_idx] -= 1.0

        # Gradients
        dW_receiver = np.outer(message, dlogits)
        dmessage = np.dot(self.W_receiver, dlogits)
        dW_sender = np.outer(x, dmessage)

        # Update weights
        self.W_receiver -= self.lr * dW_receiver
        self.W_sender -= self.lr * dW_sender

        return loss, probs

if __name__ == "__main__":
    np.random.seed(42)
    mac = MultiAgentCommunication(learning_rate=0.05)

    epochs = 2000
    loss_history = []

    for epoch in range(epochs):
        target = np.random.randint(0, mac.vocab_size)
        loss, _ = mac.train_step(target)
        loss_history.append(loss)

    initial_loss = np.mean(loss_history[:50])
    final_loss = np.mean(loss_history[-50:])

    print("Multi-Agent Communication Component Testing...")
    print(f"Initial Average Loss: {initial_loss:.4f}")
    print(f"Final Average Loss: {final_loss:.4f}")

    assert final_loss < initial_loss, "Loss did not decrease."
    assert final_loss < 0.5, "Model failed to converge."
    print("Success: Agents learned a continuous communication protocol.")
