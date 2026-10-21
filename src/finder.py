"""Finders: pick candidate articles for a user. Starts with the popularity ("trending") baseline."""
import pandas as pd

from src.data import explode_impressions


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
