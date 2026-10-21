"""Metrics: how well a ranked list covers what the user clicked, and how well a page matches the slider mix."""
import numpy as np
import pandas as pd


def recall_at_k(ranked: list, clicked: list, k: int) -> float:
    """Share of the clicked articles that appear in the top k."""
    top = set(ranked[:k])
    return sum(news_id in top for news_id in clicked) / len(clicked)


def ndcg_at_k(ranked: list, clicked: list, k: int) -> float:
    """1.0 if the clicked articles sit at the very top of the list, lower the further down they are."""
    clicked = set(clicked)
    dcg = sum(1 / np.log2(i + 2) for i, news_id in enumerate(ranked[:k]) if news_id in clicked)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(clicked), k)))
    return dcg / ideal


def evaluate(rankings: list[list], clicked_lists: list[list], ks=(10, 50, 100, 500), ndcg_k: int = 10) -> pd.DataFrame:
    """One row per impression with Recall@k for each k and NDCG@ndcg_k."""
    rows = []
    for ranked, clicked in zip(rankings, clicked_lists):
        row = {f"recall@{k}": recall_at_k(ranked, clicked, k) for k in ks}
        row[f"ndcg@{ndcg_k}"] = ndcg_at_k(ranked, clicked, ndcg_k)
        rows.append(row)
    return pd.DataFrame(rows)


def mix_error(page_buckets: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per page: half the summed gap between the page's bucket shares and the target (0 = perfect, 1 = completely off).

    page_buckets: (pages, page_size) bucket index of each article; targets: (pages, buckets) shares summing to 1.
    """
    n_buckets = targets.shape[1]
    shares = np.stack([(page_buckets == b).mean(1) for b in range(n_buckets)], axis=1)
    return 0.5 * np.abs(shares - targets).sum(1)


def relevance_kept(chosen_relevance: np.ndarray, top_relevance: np.ndarray) -> np.ndarray:
    """Per page: total relevance of the chosen articles as a share of the plain top-k's (1 = nothing lost)."""
    return chosen_relevance.sum(1) / top_relevance.sum(1)
