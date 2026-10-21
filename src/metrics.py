"""Ranking metrics: how well a ranked list of articles covers what the user actually clicked."""
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
