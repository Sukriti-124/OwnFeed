"""Finders: pick candidate articles for a user. The popularity ("trending") baseline and the two-tower model."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import explode_impressions

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "news_embeddings.npz"


def hourly_clicks(behaviors: pd.DataFrame) -> pd.DataFrame:
    """Clicks per article per hour: rows = hour, columns = news_id."""
    shown = explode_impressions(behaviors)
    clicks = shown[shown.clicked == 1]
    return clicks.groupby([clicks.time.dt.floor("h"), "news_id"]).size().unstack(fill_value=0)


def trending_scores(hourly: pd.DataFrame, hour: pd.Timestamp, window_hours: int = 24) -> pd.Series:
    """Clicks per article in the window_hours before `hour`. The current hour is excluded, so nothing peeks ahead."""
    start = hour - pd.Timedelta(hours=window_hours)
    return hourly[(hourly.index >= start) & (hourly.index < hour)].sum()


def rank(scores: pd.Series, pool: list) -> list:
    """Articles in the pool sorted by score, highest first. Articles with no score go last."""
    return scores.reindex(pool, fill_value=0).sort_values(ascending=False, kind="stable").index.tolist()


def with_bucket_quotas(ranked: list, bucket_of: dict, per_bucket: int = 100) -> list:
    """Keep the top `per_bucket` articles from each bucket, preserving overall order, so no bucket runs dry."""
    taken, kept = {}, []
    for news_id in ranked:
        bucket = bucket_of[news_id]
        if taken.get(bucket, 0) < per_bucket:
            taken[bucket] = taken.get(bucket, 0) + 1
            kept.append(news_id)
    return kept


# --- Two-tower: article text -> numbers ---

def article_text(news: pd.DataFrame) -> pd.Series:
    """The text MiniLM reads for each article: category, subcategory, title and abstract."""
    return news["category"] + " / " + news["subcategory"] + ". " + news["title"] + ". " + news["abstract"]


def embed_articles(news: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
    """One 384-number vector per article from frozen MiniLM, scaled to length 1 so dot product = similarity."""
    from sentence_transformers import SentenceTransformer  # imported here so the trending baseline doesn't load torch

    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(
        article_text(news).tolist(), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    )


def save_embeddings(news_ids, vectors: np.ndarray, path: Path = EMBEDDINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, news_ids=np.asarray(news_ids, dtype=str), vectors=vectors)


def load_embeddings(path: Path = EMBEDDINGS_PATH) -> tuple[np.ndarray, np.ndarray]:
    """(news_ids, vectors) in matching order."""
    saved = np.load(path)
    return saved["news_ids"], saved["vectors"]
