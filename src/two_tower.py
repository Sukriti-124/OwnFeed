"""Two-tower finder: a user tower and an item tower on top of frozen MiniLM article vectors."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.data import explode_impressions
from src.finder import ARTIFACTS_DIR

MODEL_PATH = ARTIFACTS_DIR / "two_tower.pt"
HISTORY_LEN = 50  # most recent past clicks used to describe a user


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TwoTower(nn.Module):
    def __init__(self, dim_in: int = 384, hidden: int = 256, dim_out: int = 128):
        super().__init__()
        self.user_tower = nn.Sequential(nn.Linear(dim_in, hidden), nn.ReLU(), nn.Linear(hidden, dim_out))
        self.item_tower = nn.Sequential(nn.Linear(dim_in, hidden), nn.ReLU(), nn.Linear(hidden, dim_out))

    def users(self, history_vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average the user's past-click vectors, then pass through the user tower. Shapes: (B, L, 384), (B, L)."""
        summed = (history_vectors * mask.unsqueeze(-1)).sum(1)
        mean = summed / mask.sum(1, keepdim=True).clamp(min=1)
        return F.normalize(self.user_tower(mean), dim=-1)

    def items(self, vectors: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_tower(vectors), dim=-1)


def history_matrix(behaviors: pd.DataFrame, index_of: dict) -> tuple[dict, np.ndarray]:
    """Row per user with the article indices of their last HISTORY_LEN clicks (padded with -1)."""
    users = behaviors.drop_duplicates("user_id")[["user_id", "history"]]
    row_of = {user: i for i, user in enumerate(users.user_id)}
    matrix = np.full((len(users), HISTORY_LEN), -1, dtype=np.int64)
    for i, history in enumerate(users.history):
        ids = [index_of[n] for n in history.split() if n in index_of][-HISTORY_LEN:]
        matrix[i, : len(ids)] = ids
    return row_of, matrix


def training_examples(behaviors: pd.DataFrame, index_of: dict, row_of: dict, n_negatives: int = 4, seed: int = 0):
    """One example per click: (user row, clicked article, n_negatives skipped articles from the same page)."""
    rng = np.random.default_rng(seed)
    shown = explode_impressions(behaviors)
    shown["item"] = shown.news_id.map(index_of)
    skipped = shown[shown.clicked == 0].groupby("impression_id").item.agg(list)
    clicks = shown[shown.clicked == 1]

    users, positives, negatives = [], [], []
    for impression_id, user_id, item in zip(clicks.impression_id, clicks.user_id, clicks.item):
        pool = skipped.get(impression_id)
        if not pool:  # no skipped articles on this page: fall back to random ones
            pool = rng.integers(0, len(index_of), n_negatives).tolist()
        users.append(row_of[user_id])
        positives.append(item)
        negatives.append(rng.choice(pool, n_negatives, replace=len(pool) < n_negatives))
    return np.array(users), np.array(positives), np.array(negatives), clicks.time.dt.floor("h").values


def popularity_table(hourly: pd.DataFrame, hours, index_of: dict, window_hours: int = 24) -> tuple[dict, np.ndarray]:
    """log(1 + clicks in the previous window_hours) for every article at each hour. Rows = hours, columns = article index."""
    hours = sorted(set(pd.to_datetime(hours)))
    table = np.zeros((len(hours), len(index_of)), dtype=np.float32)
    cols = np.array([index_of.get(n, -1) for n in hourly.columns])
    known = cols >= 0
    for row, hour in enumerate(hours):
        window = hourly[(hourly.index >= hour - pd.Timedelta(hours=window_hours)) & (hourly.index < hour)].sum().values
        table[row, cols[known]] = np.log1p(window[known])
    return {h: i for i, h in enumerate(hours)}, table


def user_vectors(model: TwoTower, history: torch.Tensor, article_vectors: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    """User-tower output for every row of the history matrix."""
    out = []
    for start in range(0, len(history), batch_size):
        h = history[start : start + batch_size]
        out.append(model.users(article_vectors[h.clamp(min=0)], (h >= 0).float()))
    return torch.cat(out)


def train(
    model: TwoTower, article_vectors: torch.Tensor, history: torch.Tensor, examples: tuple,
    epochs: int = 3, batch_size: int = 512, lr: float = 1e-3, temperature: float = 0.05, on_epoch_end=None,
    popularity: torch.Tensor | None = None, example_hours: torch.Tensor | None = None, beta: float = 0.0,
) -> list[float]:
    """Pull each clicked article toward its user, push away skipped articles and other users' clicks in the batch.

    With `popularity` (hours x articles) and beta > 0, the trending boost is added to every score during training,
    so the towers only have to learn what trending doesn't already explain.
    """
    dev = article_vectors.device
    users, positives, negatives = (torch.as_tensor(x, device=dev) for x in examples[:3])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(users), device=dev)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            h = history[users[batch]]
            u = model.users(article_vectors[h.clamp(min=0)], (h >= 0).float())
            pos = model.items(article_vectors[positives[batch]])
            neg = model.items(article_vectors[negatives[batch]])
            scores = torch.cat([u @ pos.T, torch.einsum("bd,bnd->bn", u, neg)], dim=1)
            if beta > 0:
                hot = popularity[example_hours[batch]]  # (B, all articles) trending at each example's hour
                scores = scores + beta * torch.cat([hot[:, positives[batch]], hot.gather(1, negatives[batch])], dim=1)
            logits = scores / temperature
            loss = F.cross_entropy(logits, torch.arange(len(batch), device=dev))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(order))
        if on_epoch_end:
            model.eval()
            with torch.no_grad():
                on_epoch_end(epoch, losses[-1])
    return losses


def top_k_with_quotas(scores: torch.Tensor, item_buckets: np.ndarray, buckets: list, per_bucket: int = 100):
    """For each user (row), the top per_bucket items of every bucket, merged and sorted by score. Returns item columns."""
    picked_idx, picked_scores = [], []
    for bucket in buckets:
        cols = torch.as_tensor(np.flatnonzero(item_buckets == bucket), device=scores.device)
        s, i = scores[:, cols].topk(min(per_bucket, len(cols)), dim=1)
        picked_idx.append(cols[i])
        picked_scores.append(s)
    idx, s = torch.cat(picked_idx, 1), torch.cat(picked_scores, 1)
    order = s.argsort(dim=1, descending=True)
    return idx.gather(1, order)
