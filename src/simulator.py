"""Simulator: fake users who click like real MIND users, using a click model fitted on real clicks."""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch

from src.arranger import PAGE_SIZE, arrange, mix_to_slots, normalize_scores
from src.two_tower import TwoTower, top_k_with_quotas, user_vectors


def bucket_affinity(history: torch.Tensor, article_buckets: torch.Tensor, n_buckets: int, smoothing: float = 1.0) -> torch.Tensor:
    """Each user's share of past clicks per bucket, smoothed so no bucket is exactly 0. Shapes: (N, L) -> (N, buckets).

    history holds article indices (-1 = padding); article_buckets maps article index -> bucket index (-1 = no bucket).
    """
    buckets = article_buckets[history.clamp(min=0)].masked_fill(history < 0, -1)
    counts = torch.stack([(buckets == b).sum(1) for b in range(n_buckets)], dim=1).float()
    return (counts + smoothing) / (counts.sum(1, keepdim=True) + smoothing * n_buckets)


def click_features(scores: torch.Tensor, buckets: torch.Tensor, affinity: torch.Tensor) -> torch.Tensor:
    """Inputs to the click model for each (user, article): finder score, how much the user likes its bucket,
    and a one-hot bucket indicator (so each bucket gets its own baseline click rate).

    Bucket liking is log(buckets x affinity): 0 for an average bucket, positive for favorites, negative for avoided ones.
    """
    n_buckets = affinity.shape[1]
    liking = torch.log(n_buckets * affinity.gather(1, buckets))
    one_hot = torch.nn.functional.one_hot(buckets, n_buckets).float()
    return torch.cat([scores.unsqueeze(-1), liking.unsqueeze(-1), one_hot], dim=-1)


class ClickModel:
    """Logistic regression: P(click) = sigmoid(w_score x score + w_liking x bucket liking + baseline of the bucket)."""

    def __init__(self):
        self.weights = torch.zeros(0)
        self.bias = torch.zeros(1)

    def fit(self, features: torch.Tensor, clicked: torch.Tensor, steps: int = 500, lr: float = 0.1) -> "ClickModel":
        w = torch.zeros(features.shape[-1], requires_grad=True, device=features.device)
        b = torch.zeros(1, requires_grad=True, device=features.device)
        optimizer = torch.optim.LBFGS([w, b], lr=lr, max_iter=steps)

        def loss_fn():
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(features @ w + b, clicked.float())
            loss.backward()
            return loss

        optimizer.step(loss_fn)
        self.weights, self.bias = w.detach(), b.detach()
        return self

    def probability(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(features @ self.weights.to(features.device) + self.bias.to(features.device))


def auc(probabilities: np.ndarray, clicked: np.ndarray) -> float:
    """Chance that a random clicked article gets a higher probability than a random skipped one (0.5 = guessing)."""
    ranks = pd.Series(probabilities).rank().values
    n_pos = clicked.sum()
    return (ranks[clicked == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(clicked) - n_pos))


@dataclass
class World:
    """Everything a scrolling session needs that doesn't change from page to page."""
    model: TwoTower
    article_vectors: torch.Tensor  # MiniLM vectors of every article
    pool_idx: torch.Tensor         # article index of each live article
    pool_buckets: np.ndarray       # bucket name of each live article
    pool_bucket_idx: torch.Tensor  # bucket index of each live article
    trending: torch.Tensor         # (users, pool) log(1 + clicks in last 24h) at each user's session time
    click_model: ClickModel
    buckets: list
    beta: float = 0.05             # trending weight in the finder score


def simulate(
    world: World, history: torch.Tensor, affinity: torch.Tensor, targets: np.ndarray,
    policy: Callable | None, pages: int = 20, strictness: float = 0.5, seed: int = 0,
) -> dict:
    """Scroll `pages` pages for many users at once.

    Each page: the finder scores the live articles from the user's current history, the arranger builds 20 articles
    for the mix the policy asks for, the fake user clicks each article with the click model's probability,
    and clicked articles join the user's history (so the feed learns from clicks, like a real one).

    policy(page, clicks_so_far, targets) -> page mix (users, buckets), or None for a plain feed that ignores sliders.
    Returns per-page bucket counts of what was shown, clicked, and expected to be clicked (sum of click
    probabilities, a noise-free version of clicks), each (users, pages, buckets).
    """
    generator = torch.Generator(device=history.device).manual_seed(seed)
    n_users, n_buckets = len(history), len(world.buckets)
    history = history.clone()
    rows = torch.arange(n_users, device=history.device)
    already_shown = torch.zeros(n_users, len(world.pool_idx), dtype=torch.bool, device=history.device)
    clicked_counts = torch.zeros(n_users, n_buckets, device=history.device)
    shown_log, clicked_log, expected_log = [], [], []

    with torch.no_grad():
        items = world.model.items(world.article_vectors[world.pool_idx])
        # The user's real taste is fixed at the start; only the feed's picture of them changes as they click
        true_scores = user_vectors(world.model, history, world.article_vectors) @ items.T + world.beta * world.trending
        for page in range(pages):
            # Finder: score every live article for the user's current history, never repeat an article
            scores = user_vectors(world.model, history, world.article_vectors) @ items.T + world.beta * world.trending
            scores = scores.masked_fill(already_shown, float("-inf"))
            cols, cand_scores = top_k_with_quotas(scores, world.pool_buckets, world.buckets, return_scores=True)
            cand_buckets = world.pool_bucket_idx[cols]

            # Arranger: plain top 20, or the mix the policy asks for
            mix = policy(page, clicked_counts.cpu().numpy(), targets) if policy else None
            if mix is None:
                chosen = cand_scores.argsort(1, descending=True)[:, :PAGE_SIZE]
            else:
                slots = torch.as_tensor(mix_to_slots(mix), device=history.device)
                chosen = arrange(normalize_scores(cand_scores), cand_buckets, slots, strictness)
            page_cols, page_buckets = cols.gather(1, chosen), cand_buckets.gather(1, chosen)

            # Fake user clicks, based on their real (fixed) taste
            p = world.click_model.probability(click_features(true_scores.gather(1, page_cols), page_buckets, affinity))
            clicks = torch.bernoulli(p, generator=generator).bool()

            # Bookkeeping: log, mark shown, add clicks to history (most recent last, keep the last L)
            one_hot = torch.nn.functional.one_hot(page_buckets, n_buckets).float()
            shown_log.append(one_hot.sum(1))
            clicked_log.append((one_hot * clicks.unsqueeze(-1)).sum(1))
            expected_log.append((one_hot * p.unsqueeze(-1)).sum(1))
            clicked_counts += clicked_log[-1]
            already_shown[rows.unsqueeze(1), page_cols] = True
            for user in torch.nonzero(clicks.any(1)).flatten().tolist():
                new = world.pool_idx[page_cols[user, clicks[user]]]
                merged = torch.cat([history[user][history[user] >= 0], new])[-history.shape[1]:]
                history[user] = -1
                history[user, : len(merged)] = merged

    return {
        "shown": torch.stack(shown_log, 1).cpu().numpy(),
        "clicked": torch.stack(clicked_log, 1).cpu().numpy(),
        "expected": torch.stack(expected_log, 1).cpu().numpy(),
    }


def bucket_click_chance(world: World, history: torch.Tensor, affinity: torch.Tensor, click_model: ClickModel) -> np.ndarray:
    """Average click probability per bucket over each user's finder candidates, according to `click_model`.

    With the simulator's own click model this is the truth (an oracle); with a model fitted on other days it's
    the feed's honest estimate. Shape (users, buckets).
    """
    with torch.no_grad():
        items = world.model.items(world.article_vectors[world.pool_idx])
        scores = user_vectors(world.model, history, world.article_vectors) @ items.T + world.beta * world.trending
        cols, cand_scores = top_k_with_quotas(scores, world.pool_buckets, world.buckets, return_scores=True)
        cand_buckets = world.pool_bucket_idx[cols]
        p = click_model.probability(click_features(cand_scores, cand_buckets, affinity))
        chance = [(p * (cand_buckets == b)).sum(1) / (cand_buckets == b).sum(1) for b in range(len(world.buckets))]
    return torch.stack(chance, 1).cpu().numpy()


def shift_taste(affinity: torch.Tensor, mood: float, seed: int = 0) -> torch.Tensor:
    """Today's taste = history taste x random per-bucket factor (log-normal with spread `mood`), renormalized.

    mood = 0 means the user behaves exactly like their history; larger values mean today differs more.
    """
    generator = torch.Generator().manual_seed(seed)
    shifted = affinity * torch.exp(mood * torch.randn(affinity.shape, generator=generator)).to(affinity.device)
    return shifted / shifted.sum(1, keepdim=True)
