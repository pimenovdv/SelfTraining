import torch
import torch.nn as nn
import torch.optim as optim

class NoiseContrastiveEstimationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target_idx, context_idx, noise_idx):
        # target: (batch_size,)
        # context: (batch_size,)
        # noise: (batch_size, num_noise)

        target_emb = self.target_embeddings(target_idx) # (batch_size, embedding_dim)
        context_emb = self.context_embeddings(context_idx) # (batch_size, embedding_dim)
        noise_emb = self.context_embeddings(noise_idx) # (batch_size, num_noise, embedding_dim)

        # Positive score
        pos_score = torch.sum(target_emb * context_emb, dim=1) # (batch_size,)

        # Negative scores
        neg_scores = torch.bmm(target_emb.unsqueeze(1), noise_emb.transpose(1, 2)).squeeze(1) # (batch_size, num_noise)

        return pos_score, neg_scores

def nce_loss(pos_score, neg_scores):
    # NCE minimizes negative log likelihood of binary classification

    # Positive log probability
    pos_loss = -torch.nn.functional.logsigmoid(pos_score).mean()

    # Negative log probability
    neg_loss = -torch.nn.functional.logsigmoid(-neg_scores).sum(dim=1).mean()

    return pos_loss + neg_loss

def test_nce():
    print("Testing NCE component...")
    torch.manual_seed(42)

    vocab_size = 500
    embedding_dim = 16
    batch_size = 64
    num_noise = 5

    model = NoiseContrastiveEstimationModel(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # Dummy data
    target_idx = torch.randint(0, vocab_size, (batch_size,))
    context_idx = target_idx # Easiest case, context is same as target

    # Train for a few epochs
    for epoch in range(100):
        optimizer.zero_grad()
        noise_idx = torch.randint(0, vocab_size, (batch_size, num_noise))

        pos_score, neg_scores = model(target_idx, context_idx, noise_idx)
        loss = nce_loss(pos_score, neg_scores)

        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    assert loss.item() < 3.0, "NCE failed to converge."
    print("NCE component test passed.")

if __name__ == "__main__":
    test_nce()
