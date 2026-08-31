import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=16, hidden_layers=[64, 32, 16]):
        super(NCF, self).__init__()

        # GMF embeddings
        self.embedding_user_mf = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.embedding_item_mf = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        # MLP embeddings
        self.embedding_user_mlp = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        # MLP layers
        mlp_modules = []
        input_size = embedding_dim * 2
        for hidden_size in hidden_layers:
            mlp_modules.append(nn.Linear(input_size, hidden_size))
            mlp_modules.append(nn.ReLU())
            input_size = hidden_size
        self.mlp_layers = nn.Sequential(*mlp_modules)

        # Prediction layer
        self.prediction_layer = nn.Linear(embedding_dim + hidden_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        # GMF part
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        # MLP part
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mlp_vector = self.mlp_layers(mlp_vector)

        # Concatenate GMF and MLP parts
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)

        # Final prediction
        prediction = self.prediction_layer(predict_vector)
        return self.sigmoid(prediction).squeeze()

def train_and_evaluate():
    torch.manual_seed(42)
    np.random.seed(42)

    num_users = 20
    num_items = 20
    num_samples = 2000

    # Generate synthetic dataset (users, items, ratings 0 or 1)
    users = torch.randint(0, num_users, (num_samples,))
    items = torch.randint(0, num_items, (num_samples,))

    # Simple underlying pattern: user i likes item j if i % 2 == j % 2
    ratings = ((users % 2) == (items % 2)).float()

    model = NCF(num_users=num_users, num_items=num_items, embedding_dim=16, hidden_layers=[64, 32, 16])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 200
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(users, items)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 40 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_users = torch.randint(0, num_users, (500,))
        test_items = torch.randint(0, num_items, (500,))
        test_ratings = ((test_users % 2) == (test_items % 2)).float()

        preds = model(test_users, test_items)
        predicted_classes = (preds > 0.5).float()
        accuracy = (predicted_classes == test_ratings).float().mean().item()

        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        if accuracy > 0.8:
            print("NCF model successfully learned the interaction pattern.")
        else:
            print("NCF model failed to learn the interaction pattern.")

if __name__ == "__main__":
    train_and_evaluate()
