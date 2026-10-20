"""Load MIND-small, assign articles to slider buckets, and split impressions by time."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DIR = DATA_DIR / "MINDsmall_train"
TEST_DIR = DATA_DIR / "MINDsmall_test"  # locked test set: only for final results

# Last day of the train folder is held out for validation (split by time to avoid peeking at the future)
VALIDATION_DAY = "2019-11-14"

BUCKETS = ["News", "Learn", "Lifestyle", "Entertain", "Sports"]

CATEGORY_TO_BUCKET = {
    "news": "News", "weather": "News",
    "finance": "Learn", "health": "Learn",
    "lifestyle": "Lifestyle", "foodanddrink": "Lifestyle", "travel": "Lifestyle", "autos": "Lifestyle",
    "entertainment": "Entertain", "movies": "Entertain", "music": "Entertain", "tv": "Entertain", "video": "Entertain",
    "sports": "Sports",
}

# Subcategories that belong in a different bucket than their category
SUBCATEGORY_OVERRIDES = {
    "newsscienceandtechnology": "Learn",
    "lifestyleroyals": "Entertain", "lifestylebuzz": "Entertain", "lifestylecelebstyle": "Entertain", "awardstyle": "Entertain",
}


def assign_bucket(news: pd.DataFrame) -> pd.Series:
    """Bucket for each article: subcategory override first, then category. NaN for dropped categories (kids, etc.)."""
    return news["subcategory"].map(SUBCATEGORY_OVERRIDES).fillna(news["category"].map(CATEGORY_TO_BUCKET))


def load_news(folder: Path = TRAIN_DIR) -> pd.DataFrame:
    """One row per article, with its bucket."""
    news = pd.read_csv(
        folder / "news.tsv", sep="\t", header=None, quoting=3,
        names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"],
        usecols=["news_id", "category", "subcategory", "title", "abstract"],
    )
    news["abstract"] = news["abstract"].fillna("")
    news["bucket"] = assign_bucket(news)
    return news


def load_behaviors(folder: Path = TRAIN_DIR) -> pd.DataFrame:
    """One row per impression (one page view): user, time, past clicks, and the articles shown."""
    behaviors = pd.read_csv(
        folder / "behaviors.tsv", sep="\t", header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"],
    )
    behaviors["time"] = pd.to_datetime(behaviors["time"], format="%m/%d/%Y %I:%M:%S %p")
    behaviors["history"] = behaviors["history"].fillna("")
    return behaviors


def explode_impressions(behaviors: pd.DataFrame, news: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (impression, article shown) with clicked = 1/0. Adds category/subcategory/bucket if news is given."""
    shown = behaviors[["impression_id", "user_id", "time", "impressions"]].copy()
    shown["item"] = shown["impressions"].str.split()
    shown = shown.explode("item").drop(columns="impressions")
    shown[["news_id", "clicked"]] = shown["item"].str.split("-", expand=True)
    shown["clicked"] = shown["clicked"].astype(int)
    shown = shown.drop(columns="item").reset_index(drop=True)
    if news is not None:
        shown = shown.merge(news[["news_id", "category", "subcategory", "bucket"]], on="news_id", how="left")
    return shown


def split_by_time(behaviors: pd.DataFrame, validation_day: str = VALIDATION_DAY) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impressions before validation_day go to train, the rest to validation."""
    is_val = behaviors["time"] >= pd.Timestamp(validation_day)
    return behaviors[~is_val].reset_index(drop=True), behaviors[is_val].reset_index(drop=True)
