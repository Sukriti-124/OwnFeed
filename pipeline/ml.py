"""ML tasks for the orchestrated pipeline: embeddings, daily evaluation, training challengers, and a model registry.

Data comes from the warehouse marts (fct_impressions, fct_hourly_trending, dim_user_history). Models live in a
small registry (artifacts/models/registry.json). The model behind the reported results is pinned as "v1-paper"
(artifacts/two_tower.pt) and is never overwritten.

Rule: the locked test day can be *monitored* (evaluate_day) but never used to train or choose a model.

    python -m pipeline.ml embed                      # embed articles that have no vector yet
    python -m pipeline.ml evaluate 2019-11-14        # score the champion (and trending) on one day
    python -m pipeline.ml retrain                    # train a challenger, compare with the champion, promote if better
"""
import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from src.arranger import PAGE_SIZE  # noqa: F401  (kept for readers: pages are 20 articles)
from src.data import BUCKETS, TRAIN_DIR, VALIDATION_DAY, WAREHOUSE_PATH, load_behaviors
from src.feed import ALL_EMBEDDINGS, BETA
from src.finder import ARTIFACTS_DIR, embed_articles, load_embeddings, save_embeddings
from src.metrics import evaluate
from src.two_tower import (
    HISTORY_LEN, MODEL_PATH, TwoTower, device, find_candidates, history_matrix, train, training_examples,
)

MODELS_DIR = ARTIFACTS_DIR / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"
MONITORING_PATH = ARTIFACTS_DIR.parent / "results" / "model_monitoring.csv"
TEST_DAY = date(2019, 11, 15)  # locked test set (MINDsmall_dev)
PAPER_VERSION = "v1-paper"
PROMOTE_MARGIN = 0.002  # a challenger must beat the champion's Recall@500 by this much (and not lose NDCG@10 by more)


# ---------------- Registry ----------------

def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {
        "champion": PAPER_VERSION,
        "versions": {PAPER_VERSION: {
            "path": str(MODEL_PATH.relative_to(ARTIFACTS_DIR.parent)), "trained_on": "2019-11-09..2019-11-13",
            "pinned": True, "note": "the model behind the README / report results; never overwritten",
        }},
        "decisions": [],
    }


def save_registry(registry: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def load_model(version: str | None = None, dev: torch.device | None = None) -> tuple[str, TwoTower]:
    registry = load_registry()
    version = version or registry["champion"]
    model = TwoTower().to(dev or device())
    model.load_state_dict(torch.load(ARTIFACTS_DIR.parent / registry["versions"][version]["path"], map_location=dev or device()))
    model.eval()
    return version, model


def _warehouse() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(WAREHOUSE_PATH), read_only=True)


# ---------------- Embeddings ----------------

def embed_new_articles() -> int:
    """Give every article in the warehouse a MiniLM vector (appends to news_embeddings_all.npz). Returns how many."""
    with _warehouse() as con:
        articles = con.execute("SELECT news_id, category, subcategory, title, abstract FROM marts.dim_articles").df()
    if ALL_EMBEDDINGS.exists():
        ids, vectors = load_embeddings(ALL_EMBEDDINGS)
    else:
        ids, vectors = np.array([], dtype=str), np.zeros((0, 384), dtype=np.float32)
    new = articles[~articles.news_id.isin(set(ids))]
    if len(new):
        save_embeddings(np.concatenate([ids, new.news_id.values]), np.concatenate([vectors, embed_articles(new)]), ALL_EMBEDDINGS)
    return len(new)


# ---------------- Evaluation ----------------

def score_day(model: TwoTower, day: date, dev: torch.device) -> pd.DataFrame:
    """Recall@K / NDCG@10 on one day for the model (taste + BETA x trending) and for trending alone."""
    news_ids, vectors = load_embeddings(ALL_EMBEDDINGS)
    index_of = {n: i for i, n in enumerate(news_ids)}
    with _warehouse() as con:
        shown = con.execute("""
            SELECT impression_key, user_id, hour, news_id, clicked, bucket, source
            FROM marts.fct_impressions WHERE date = ? AND bucket IS NOT NULL
        """, [day]).df()
        if shown.empty:
            raise ValueError(f"No impressions in the warehouse for {day}")
        source = shown.source.iloc[0]
        histories = con.execute("SELECT user_id, recent_clicks FROM marts.dim_user_history WHERE source = ?", [source]).df()
        hours = sorted(shown.hour.unique())
        trending = con.execute("SELECT hour, news_id, trending_score FROM marts.fct_hourly_trending WHERE hour IN (SELECT unnest(?))",
                               [hours]).df()

    pool = np.array(sorted(shown.news_id.unique()))
    pool_buckets = shown.drop_duplicates("news_id").set_index("news_id").loc[pool, "bucket"].to_numpy(dtype=object)
    pool_idx = torch.as_tensor([index_of[n] for n in pool], device=dev)
    clicks = shown[shown.clicked == 1].groupby(["impression_key", "user_id", "hour"]).news_id.agg(list).reset_index()

    # Trending table: hours x pool, log(1 + clicks in the previous 24 h)
    hour_row = {h: i for i, h in enumerate(hours)}
    col_of = {n: i for i, n in enumerate(pool)}
    table = np.zeros((len(hours), len(pool)), dtype=np.float32)
    t = trending[trending.news_id.isin(col_of)]
    table[t.hour.map(hour_row).values, t.news_id.map(col_of).values] = t.trending_score.values
    popularity = torch.as_tensor(table, device=dev)

    # History matrix for the day's users (last HISTORY_LEN clicks, padded with -1)
    row_of = {u: i for i, u in enumerate(histories.user_id)}
    hist = np.full((len(histories), HISTORY_LEN), -1, dtype=np.int64)
    for i, recent in enumerate(histories.recent_clicks):
        idx = [index_of[n] for n in recent if n in index_of][-HISTORY_LEN:]
        hist[i, : len(idx)] = idx

    user_rows = torch.as_tensor([row_of[u] for u in clicks.user_id], device=dev)
    hour_rows = torch.as_tensor([hour_row[h] for h in clicks.hour], device=dev)
    article_vectors = torch.as_tensor(vectors, device=dev)
    full_pop = torch.zeros((len(hours), len(news_ids)), device=dev)
    full_pop[:, pool_idx] = popularity

    cols, _ = find_candidates(model, user_rows, hour_rows, torch.as_tensor(hist, device=dev), article_vectors, full_pop,
                              pool_idx, pool_buckets, BUCKETS, beta=BETA)
    model_rank = [list(pool[row]) for row in cols.cpu().numpy()]

    # Trending alone: top 100 per bucket by trending score, same for everyone in an hour
    trending_rank = {}
    for h in hours:
        order = np.argsort(-table[hour_row[h]], kind="stable")
        picked, taken = [], {}
        for c in order:
            b = pool_buckets[c]
            if taken.get(b, 0) < 100:
                taken[b] = taken.get(b, 0) + 1
                picked.append(pool[c])
        trending_rank[h] = picked

    clicked = clicks.news_id.tolist()
    return pd.DataFrame({
        "model": evaluate(model_rank, clicked).mean(),
        "trending": evaluate([trending_rank[h] for h in clicks.hour], clicked).mean(),
    }).T


def evaluate_day(day: str, version: str | None = None) -> dict:
    """Score a model on one day and record it in results/model_monitoring.csv (one row per day and version)."""
    dev = device()
    version, model = load_model(version, dev)
    d = date.fromisoformat(day)
    scores = score_day(model, d, dev)
    row = {
        "day": day, "version": version, "split": "test" if d >= TEST_DAY else ("validation" if d == date.fromisoformat(VALIDATION_DAY) else "train"),
        "recall@500": round(float(scores.loc["model", "recall@500"]), 4), "ndcg@10": round(float(scores.loc["model", "ndcg@10"]), 4),
        "trending_recall@500": round(float(scores.loc["trending", "recall@500"]), 4), "trending_ndcg@10": round(float(scores.loc["trending", "ndcg@10"]), 4),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    log = pd.read_csv(MONITORING_PATH) if MONITORING_PATH.exists() else pd.DataFrame(columns=list(row))
    log = log[~((log.day == day) & (log.version == version))]
    pd.concat([log, pd.DataFrame([row])]).sort_values(["day", "version"]).to_csv(MONITORING_PATH, index=False)
    return row


# ---------------- Training and promotion ----------------

def train_challenger(train_until: str = VALIDATION_DAY, epochs: int = 3, seed: int = 0) -> str:
    """Train a new candidate on all train-folder days before `train_until`. Never touches the locked test day."""
    if date.fromisoformat(train_until) > TEST_DAY:
        raise ValueError("Training may not include the locked test day")
    dev = device()
    news_ids, vectors = load_embeddings(ALL_EMBEDDINGS)
    index_of = {n: i for i, n in enumerate(news_ids)}
    behaviors = load_behaviors(TRAIN_DIR)  # train folder only (Nov 9–14)
    train_part = behaviors[behaviors.time < pd.Timestamp(train_until)]
    row_of, hist = history_matrix(behaviors, index_of)
    examples = training_examples(train_part, index_of, row_of)
    torch.manual_seed(seed)
    model = TwoTower().to(dev)
    losses = train(model, torch.as_tensor(vectors, device=dev), torch.as_tensor(hist, device=dev), examples, epochs=epochs)

    version = f"v{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    path = MODELS_DIR / version / "two_tower.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    registry = load_registry()
    registry["versions"][version] = {
        "path": str(path.relative_to(ARTIFACTS_DIR.parent)), "trained_on": f"2019-11-09..{train_until} (exclusive)",
        "epochs": epochs, "seed": seed, "final_loss": round(float(losses[-1]), 4), "pinned": False,
    }
    save_registry(registry)
    return version


def compare_and_promote(challenger: str, eval_day: str = VALIDATION_DAY) -> dict:
    """Evaluate challenger and champion on eval_day (never the test day) and promote the challenger only if it wins."""
    if date.fromisoformat(eval_day) >= TEST_DAY:
        raise ValueError("Model selection may not use the locked test day")
    registry = load_registry()
    champion = registry["champion"]
    c = evaluate_day(eval_day, challenger)
    k = evaluate_day(eval_day, champion)
    wins = c["recall@500"] >= k["recall@500"] + PROMOTE_MARGIN and c["ndcg@10"] >= k["ndcg@10"] - PROMOTE_MARGIN
    decision = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "challenger": challenger, "champion": champion,
        "eval_day": eval_day, "challenger_recall@500": c["recall@500"], "champion_recall@500": k["recall@500"],
        "challenger_ndcg@10": c["ndcg@10"], "champion_ndcg@10": k["ndcg@10"], "promoted": bool(wins),
    }
    registry = load_registry()
    registry["decisions"].append(decision)
    if wins:
        registry["champion"] = challenger
    save_registry(registry)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("embed")
    ev = sub.add_parser("evaluate")
    ev.add_argument("day")
    ev.add_argument("--version")
    rt = sub.add_parser("retrain")
    rt.add_argument("--train-until", default=VALIDATION_DAY)
    rt.add_argument("--eval-day", default=VALIDATION_DAY)
    args = parser.parse_args()

    if args.command == "embed":
        print(f"embedded {embed_new_articles()} new articles")
    elif args.command == "evaluate":
        print(evaluate_day(args.day, args.version))
    else:
        challenger = train_challenger(args.train_until)
        print(json.dumps(compare_and_promote(challenger, args.eval_day), indent=2))


if __name__ == "__main__":
    sys.exit(main())
