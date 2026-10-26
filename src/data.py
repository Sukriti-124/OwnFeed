"""Load MIND-small, assign articles to slider buckets, and split impressions by time.

Reads from the warehouse (warehouse/ownfeed.duckdb, built by pipeline/ingest.py + dbt) when it exists, otherwise
from the raw TSV files. Both give identical tables (pipeline/check_parity.py). Set OWNFEED_DATA=raw to force the
raw files.
"""
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DIR = DATA_DIR / "MINDsmall_train"
TEST_DIR = DATA_DIR / "MINDsmall_test"  # locked test set:
WAREHOUSE_PATH = DATA_DIR.parent / "warehouse" / "ownfeed.duckdb"
SOURCE_OF = {TRAIN_DIR: "train", TEST_DIR: "test"}

# Last day of the train folder is held out for validation
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


def use_warehouse() -> bool:
    """True if the warehouse has been built and OWNFEED_DATA isn't set to 'raw'."""
    return WAREHOUSE_PATH.exists() and os.environ.get("OWNFEED_DATA", "warehouse") != "raw"


def _query(sql: str, params: list) -> pd.DataFrame:
    import duckdb  # only needed with the warehouse

    with duckdb.connect(str(WAREHOUSE_PATH), read_only=True) as con:
        return con.execute(sql, params).df()


def load_news(folder: Path = TRAIN_DIR) -> pd.DataFrame:
    """One row per article, with its bucket (in the raw file's order, which saved embeddings follow)."""
    if use_warehouse():
        news = _query("""
            SELECT a.news_id, a.category, a.subcategory, a.title, a.abstract, a.bucket
            FROM marts.dim_article_sources AS s JOIN marts.dim_articles AS a USING (news_id)
            WHERE s.source = ? ORDER BY s.file_row
        """, [SOURCE_OF[folder]])
        news["bucket"] = news["bucket"].astype("str")  # same dtype as the raw path (missing stays NaN)
        return news
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
    if use_warehouse():
        return _query("""
            SELECT impression_id, user_id, time, history, impressions
            FROM marts.fct_page_views WHERE source = ? ORDER BY impression_id
        """, [SOURCE_OF[folder]])
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
