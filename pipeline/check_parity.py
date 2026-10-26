"""Check that the warehouse (warehouse/ownfeed.duckdb) gives exactly what the ML code computes from the raw files.

    python -m pipeline.check_parity

Compares: page-view and row counts per split, every article's bucket, trending scores at sample hours,
and users' last-50 click histories.
"""
import sys

import duckdb
import numpy as np
import pandas as pd

from src.data import TEST_DIR, TRAIN_DIR, assign_bucket, explode_impressions, load_behaviors, load_news, split_by_time
from src.finder import hourly_clicks, trending_scores
from src.two_tower import HISTORY_LEN

WAREHOUSE = TRAIN_DIR.parent.parent / "warehouse" / "ownfeed.duckdb"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    print(f"{'✓' if ok else '✗'} {name}" + (f"  ({detail})" if detail else ""))


def main() -> None:
    if not WAREHOUSE.exists():
        sys.exit("Warehouse not built yet. Run: cd warehouse && dbt build --profiles-dir .")
    con = duckdb.connect(str(WAREHOUSE), read_only=True)

    train_b, test_b = load_behaviors(TRAIN_DIR), load_behaviors(TEST_DIR)
    train_part, val_part = split_by_time(train_b)

    # 1. Page views and rows per split
    views = dict(con.execute("SELECT split, count(DISTINCT impression_key) FROM marts.fct_impressions GROUP BY 1").fetchall())
    expected = {"train": len(train_part), "validation": len(val_part), "test": len(test_b)}
    check("page views per split", views == expected, f"{views}")
    rows = dict(con.execute("SELECT split, count(*) FROM marts.fct_impressions GROUP BY 1").fetchall())
    expected_rows = {s: len(explode_impressions(b)) for s, b in [("train", train_part), ("validation", val_part), ("test", test_b)]}
    check("articles shown per split", rows == expected_rows, f"{rows}")

    # 2. Clicks per bucket per split
    news = pd.concat([load_news(TRAIN_DIR), load_news(TEST_DIR)]).drop_duplicates("news_id")
    bucket_of = news.set_index("news_id")["bucket"]
    py = []
    for split, b in [("train", train_part), ("validation", val_part), ("test", test_b)]:
        s = explode_impressions(b)
        s = s[s.clicked == 1]
        py.append(s.news_id.map(bucket_of).fillna("(none)").value_counts().rename(split))
    py = pd.concat(py, axis=1).fillna(0).astype(int).sort_index()
    wh = con.execute("SELECT coalesce(bucket, '(none)') AS bucket, split, sum(clicked) AS clicks FROM marts.fct_impressions GROUP BY ALL").df()
    wh = wh.pivot(index="bucket", columns="split", values="clicks").fillna(0).astype(int)[py.columns].sort_index()
    check("clicks per bucket per split", py.equals(wh))

    # 3. Every article's bucket
    articles = con.execute("SELECT news_id, category, subcategory, coalesce(bucket, '(none)') AS bucket FROM marts.dim_articles").df()
    expected_bucket = assign_bucket(articles).fillna("(none)")
    check("bucket of every article", (articles.bucket == expected_bucket).all(), f"{len(articles):,} articles")

    # 4. Trending at sample hours (train day, validation day, test day)
    hourly = hourly_clicks(pd.concat([train_b, test_b]))
    trending_ok = True
    for hour in ["2019-11-12 08:00", "2019-11-14 09:00", "2019-11-14 18:00", "2019-11-15 12:00", "2019-11-15 23:00"]:
        py_t = trending_scores(hourly, pd.Timestamp(hour))
        py_t = py_t[py_t > 0].sort_index()
        wh_t = con.execute("SELECT news_id, clicks_last_24h FROM marts.fct_hourly_trending WHERE hour = ? AND clicks_last_24h > 0",
                           [pd.Timestamp(hour)]).df().set_index("news_id")["clicks_last_24h"].sort_index()
        trending_ok &= py_t.index.equals(wh_t.index) and np.array_equal(py_t.values, wh_t.values)
    check("trending (24 h clicks) at 5 sample hours", trending_ok)

    # 5. Users' last-50 histories
    sample = train_b.drop_duplicates("user_id").sample(2000, random_state=0)
    wh_h = con.execute("SELECT user_id, recent_clicks FROM marts.dim_user_history WHERE source = 'train'").df().set_index("user_id")["recent_clicks"]
    same = all(list(wh_h[u]) == h.split()[-HISTORY_LEN:] for u, h in zip(sample.user_id, sample.history))
    check("last-50 click history (2,000 users)", same)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
