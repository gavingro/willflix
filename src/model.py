import torch
import torch.nn as nn


class MatrixFactorizationExplicitFeedback(nn.Module):
    def __init__(self, num_viewers, num_movies, k_embed_dims):
        super().__init__()
        self.viewer_emb = nn.Embedding(num_viewers, k_embed_dims)
        self.movie_emb = nn.Embedding(num_movies, k_embed_dims)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.viewer_emb.weight, std=0.1)
        nn.init.normal_(self.movie_emb.weight, std=0.1)

    def forward(self, viewer_ids, movie_ids):
        u = self.embed_viewers(viewer_ids)
        v = self.movie_emb(movie_ids)  # (batch_size, k_embed_dims)
        return (u * v).sum(dim=1)  # dot product

    def embed_viewers(self, viewer_ids):
        return self.viewer_emb(viewer_ids)  # (k_embed_dims)

    def embed_movies(self, movie_ids):
        return self.movie_emb(movie_ids)  # (batch_size, k_embed_dims)


def fit_mf(
    model: MatrixFactorizationExplicitFeedback,
    user_ids,
    item_ids,
    ratings,
    epochs: int = 1000,
    lr=1e-2,
    weight_decay=1e-4,
    early_stopping_tolerance=0.05,
    epoch_check_every=250,
) -> list:
    """Fit the Matrix Factorization model, return a series of loss over time to plot."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay  # ← L2 regularization
    )
    loss_fn = nn.MSELoss()
    loss_series = []
    prev_losses = [float("inf")] * 3
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(user_ids, item_ids)
        loss = loss_fn(preds, ratings)
        loss.backward()
        optimizer.step()
        if epoch % epoch_check_every == 0:
            # print(f"Epoch {epoch}: loss={loss.item():.6f}")
            loss_series.append((epoch, loss.item()))
            # Early Stopping
            if all(
                [
                    (prev_loss - loss.item()) < early_stopping_tolerance
                    for prev_loss in prev_losses
                ]
            ):
                print(f"Stopping at {epoch} as loss has halted at {loss.item():.6f}.")
                break
            else:
                prev_losses.pop(0)
                prev_losses.append(loss)
    return loss_series
