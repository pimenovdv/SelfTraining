import numpy as np
import argparse
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        self.W = np.random.normal(0, 0.01, (num_visible, num_hidden))
        self.h_bias = np.zeros(num_hidden)
        self.v_bias = np.zeros(num_visible)

    def sample_hidden(self, v):
        activation = np.dot(v, self.W) + self.h_bias
        prob = sigmoid(activation)
        h_sample = np.random.binomial(1, prob)
        return prob, h_sample

    def sample_visible(self, h):
        activation = np.dot(h, self.W.T) + self.v_bias
        prob = sigmoid(activation)
        v_sample = np.random.binomial(1, prob)
        return prob, v_sample

    def train_step(self, v0, lr):
        batch_size = v0.shape[0]

        ph0_prob, ph0_sample = self.sample_hidden(v0)

        pv1_prob, pv1_sample = self.sample_visible(ph0_sample)
        ph1_prob, ph1_sample = self.sample_hidden(pv1_sample)

        pos_associations = np.dot(v0.T, ph0_prob)
        neg_associations = np.dot(pv1_sample.T, ph1_prob)

        self.W += lr * (pos_associations - neg_associations) / batch_size
        self.v_bias += lr * np.mean(v0 - pv1_sample, axis=0)
        self.h_bias += lr * np.mean(ph0_prob - ph1_prob, axis=0)

        error = np.mean(np.sum((v0 - pv1_prob)**2, axis=1))
        return error

class DBN:
    def __init__(self, layer_sizes):
        self.rbms = []
        for i in range(len(layer_sizes) - 1):
            self.rbms.append(RBM(layer_sizes[i], layer_sizes[i+1]))

    def train(self, data, epochs, lr, batch_size):
        input_data = data
        errors = []
        for i, rbm in enumerate(self.rbms):
            num_batches = input_data.shape[0] // batch_size
            final_error = 0
            for epoch in range(epochs):
                epoch_error = 0
                indices = np.random.permutation(input_data.shape[0])
                shuffled_data = input_data[indices]
                for b in range(num_batches):
                    batch = shuffled_data[b*batch_size : (b+1)*batch_size]
                    error = rbm.train_step(batch, lr)
                    epoch_error += error
                epoch_error /= num_batches
                final_error = epoch_error
            errors.append(final_error)

            next_input = np.zeros((input_data.shape[0], rbm.num_hidden))
            for b in range(num_batches):
                batch = input_data[b*batch_size : (b+1)*batch_size]
                next_input[b*batch_size : (b+1)*batch_size], _ = rbm.sample_hidden(batch)
            input_data = next_input

        return errors

def generate_synthetic_data(num_samples, num_visible):
    data = np.zeros((num_samples, num_visible))
    for i in range(num_samples):
        if np.random.rand() > 0.5:
            data[i, :num_visible//2] = 1
        else:
            data[i, num_visible//2:] = 1
    return data

def main():
    parser = argparse.ArgumentParser(description="Train a Deep Belief Network (DBN)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs per layer")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of synthetic samples")
    args = parser.parse_args()

    np.random.seed(42)

    data = generate_synthetic_data(args.num_samples, 8)
    layer_sizes = [8, 6, 4]

    dbn = DBN(layer_sizes)

    errors = dbn.train(data, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    success = all(e < 0.2 for e in errors)
    status = "Success" if success else "Failure"

    print(f"Final Layer Errors: {errors}")
    print(f"Status: {status}")

    if not success:
        exit(1)

if __name__ == "__main__":
    main()
