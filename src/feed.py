"""The OwnFeed engine for one reader at a time: finder -> arranger -> balancer, page by page.

Used by the Streamlit demo (and later a web API). `build_demo_assets` precomputes everything that doesn't change
during a demo (the day's live articles, trending, example readers, click models) so the app starts in seconds.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.arranger import PAGE_SIZE, arrange, mix_to_slots, normalize_scores
from src.balancer import PIDBalancer, feedforward_mix
from src.data import BUCKETS, TEST_DIR, TRAIN_DIR, explode_impressions, load_behaviors, load_news, split_by_time
from src.finder import ARTIFACTS_DIR, hourly_clicks, load_embeddings
from src.simulator import ClickModel, bucket_affinity, click_features
from src.two_tower import MODEL_PATH, TwoTower, history_matrix, popularity_table, top_k_with_quotas, user_vectors

ALL_EMBEDDINGS = ARTIFACTS_DIR / "news_embeddings_all.npz"  # train + test articles (made in notebook 08)
DEMO_ASSETS = ARTIFACTS_DIR / "demo_assets.pt"
DEMO_HOUR = pd.Timestamp("2019-11-15 12:00")  # the moment the demo feed is served (test day)
STRICTNESS = 0.5
BETA = 0.05
PERSONAS = {"News lover": "News", "Sports fan": "Sports", "Celebrity fan": "Entertain", "Lifestyle reader": "Lifestyle", "Learner": "Learn"}


def _click_rows(behaviors_part, history, row_of, index_of, article_buckets, hourly, model, article_vectors, dev):
    """Click-model features and labels for every bucketed article shown in `behaviors_part`."""
    rows = explode_impressions(behaviors_part)
    rows["item"] = rows.news_id.map(index_of)
    rows = rows[article_buckets.cpu().numpy()[rows.item.values] >= 0].reset_index(drop=True)
    hours = rows.time.dt.floor("h")
    hour_row, table = popularity_table(hourly, hours.unique(), index_of)
    with torch.no_grad():
        users = user_vectors(model, history, article_vectors)
        items = model.items(article_vectors)
        affinity = bucket_affinity(history, article_buckets, len(BUCKETS))
        r = torch.as_tensor(rows.user_id.map(row_of).values, device=dev)
        it = torch.as_tensor(rows.item.values, device=dev)
        hr = torch.as_tensor(hours.map(hour_row).values, device=dev)
        score = (users[r] * items[it]).sum(1) + BETA * torch.as_tensor(table, device=dev)[hr, it]
        features = click_features(score[:, None], article_buckets[it][:, None], affinity[r])[:, 0, :]
    return features, torch.as_tensor(rows.clicked.values, device=dev)


def build_demo_assets(path: Path = DEMO_ASSETS) -> None:
    """Precompute the demo's fixed inputs once (~1 minute) and save them to artifacts/."""
    dev = torch.device("cpu")
    news = pd.concat([load_news(TRAIN_DIR), load_news(TEST_DIR)]).drop_duplicates("news_id")
    news_ids, vectors = load_embeddings(ALL_EMBEDDINGS)
    news = news.set_index("news_id").loc[news_ids].reset_index()
    index_of = {n: i for i, n in enumerate(news_ids)}
    article_buckets = torch.as_tensor(news.bucket.map({b: i for i, b in enumerate(BUCKETS)}).fillna(-1).astype(int).values)
    article_vectors = torch.as_tensor(vectors)
    model = TwoTower()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    model.eval()

    train_b, test_b = load_behaviors(TRAIN_DIR), load_behaviors(TEST_DIR)
    hourly = hourly_clicks(pd.concat([train_b, test_b]))
    row_of, hist = history_matrix(test_b, index_of)
    history = torch.as_tensor(hist)

    # Live articles: everything with a bucket shown on the test day; trending as of DEMO_HOUR
    shown = explode_impressions(test_b)
    pool = np.array(sorted(set(shown.news_id) & set(news.dropna(subset=["bucket"]).news_id)))
    _, table = popularity_table(hourly, [DEMO_HOUR], index_of)
    trending = table[0, [index_of[n] for n in pool]]

    # Example readers: for each persona, the test user with 30+ past clicks who leans most toward that bucket
    affinity = bucket_affinity(history, article_buckets, len(BUCKETS)).numpy()
    lengths = (hist >= 0).sum(1)
    users = np.array(list(row_of))
    personas = {}
    for name, bucket in PERSONAS.items():
        share = np.where(lengths >= 30, affinity[:, BUCKETS.index(bucket)], -1)
        personas[name] = users[share.argmax()]

    # Click models: the reader's "twin" (fitted on test-day clicks) and the feed's head start (fitted on Nov 9–13)
    truth = ClickModel().fit(*_click_rows(test_b, history, row_of, index_of, article_buckets, hourly, model, article_vectors, dev))
    train_row_of, train_hist = history_matrix(train_b, index_of)
    train_part, _ = split_by_time(train_b)
    feed_model = ClickModel().fit(*_click_rows(train_part.sample(frac=0.3, random_state=0), torch.as_tensor(train_hist), train_row_of,
                                               index_of, article_buckets, hourly, model, article_vectors, dev))

    torch.save({
        "pool": pool.tolist(),
        "trending": torch.as_tensor(trending),
        "personas": personas,
        "persona_history": {name: history[row_of[user]] for name, user in personas.items()},
        "truth": truth.state(),
        "feed_model": feed_model.state(),
    }, path)


def spread_topics(buckets: list) -> list:
    """Reorder a best-first page so topics are spread out: each next story is the best remaining one whose topic
    differs from the story just before it (falls back to the best remaining if no other topic is left).
    Returns positions into the original page."""
    remaining, order = list(range(len(buckets))), []
    while remaining:
        pick = next((i for i in remaining if not order or buckets[i] != buckets[order[-1]]), remaining[0])
        order.append(pick)
        remaining.remove(pick)
    return order


class FeedEngine:
    """Everything shared by all readers: articles, model, today's live pool and trending. Load once."""

    def __init__(self, path: Path = DEMO_ASSETS):
        assets = torch.load(path, weights_only=False)
        self.dev = torch.device("cpu")  # a single reader is faster on CPU than on the GPU
        news_ids, vectors = load_embeddings(ALL_EMBEDDINGS)
        news = pd.concat([load_news(TRAIN_DIR), load_news(TEST_DIR)]).drop_duplicates("news_id").set_index("news_id")
        index_of = {n: i for i, n in enumerate(news_ids)}

        self.pool = np.array(assets["pool"])
        self.articles = news.loc[self.pool, ["bucket", "subcategory", "title", "abstract"]].reset_index()
        self.article_vectors = torch.as_tensor(vectors)
        self.pool_idx = torch.as_tensor([index_of[n] for n in self.pool])
        self.pool_buckets = self.articles.bucket.to_numpy(dtype=object)
        self.pool_bucket_idx = torch.as_tensor([BUCKETS.index(b) for b in self.pool_buckets])
        article_buckets = torch.as_tensor(news.loc[news_ids].bucket.map({b: i for i, b in enumerate(BUCKETS)}).fillna(-1).astype(int).values)
        self.article_buckets = article_buckets
        self.trending = assets["trending"].float()

        self.model = TwoTower()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.dev))
        self.model.eval()
        with torch.no_grad():
            self.items = self.model.items(self.article_vectors[self.pool_idx])

        self.personas = assets["personas"]
        self.persona_history = assets["persona_history"]
        self.truth = ClickModel.from_state(assets["truth"])
        self.feed_model = ClickModel.from_state(assets["feed_model"])

    def scores(self, history: torch.Tensor) -> torch.Tensor:
        """Finder score of every live article for one reader: taste + BETA x trending. Shape (1, pool)."""
        with torch.no_grad():
            return user_vectors(self.model, history, self.article_vectors) @ self.items.T + BETA * self.trending


class FeedSession:
    """One reader scrolling: remembers their history, what they've seen and read, and builds each next page."""

    def __init__(self, engine: FeedEngine, persona: str, targets, mode: str = "OwnFeed", seed: int = 0):
        self.engine = engine
        self.history = engine.persona_history[persona].clone().unsqueeze(0)
        self.start_history = self.history.clone()
        self.affinity = bucket_affinity(self.history, engine.article_buckets, len(BUCKETS))
        self.already_shown = torch.zeros(1, len(engine.pool), dtype=torch.bool)
        self.shown_counts = np.zeros(len(BUCKETS))
        self.read_counts = np.zeros(len(BUCKETS))
        self.page_number = 0
        self.latencies_ms = []
        self.page = pd.DataFrame()
        self.generator = torch.Generator().manual_seed(seed)
        self.set_controls(targets, mode)

    def set_controls(self, targets, mode: str) -> None:
        """New slider mix or feed type: restart the balancer from a fresh head start."""
        self.targets = np.asarray(targets, dtype=float).reshape(1, -1)
        self.mode = mode
        with torch.no_grad():
            scores = self.engine.scores(self.start_history)
            cols, cand = top_k_with_quotas(scores, self.engine.pool_buckets, BUCKETS, return_scores=True)
            buckets = self.engine.pool_bucket_idx[cols]
            p = self.engine.feed_model.probability(click_features(cand, buckets, self.affinity))
            chance = torch.stack([(p * (buckets == b)).sum(1) / (buckets == b).sum(1) for b in range(len(BUCKETS))], 1).numpy()
        self.balancer = PIDBalancer(kp=0.5, ki=0.05, start_mix=feedforward_mix(self.targets, chance))
        self.balancer_page = 0

    def next_page(self) -> pd.DataFrame:
        """Build and return the next 20 articles."""
        start = time.perf_counter()
        scores = self.engine.scores(self.history).masked_fill(self.already_shown, float("-inf"))
        cols, cand = top_k_with_quotas(scores, self.engine.pool_buckets, BUCKETS, return_scores=True)
        cand_buckets = self.engine.pool_bucket_idx[cols]
        if self.mode == "Normal feed":
            chosen = cand.argsort(1, descending=True)[:, :PAGE_SIZE]
        else:
            if self.mode == "Sliders only":  # the page matches the sliders exactly, no balancer
                mix = self.targets
            else:  # "OwnFeed": the balancer adjusts the page so what you read lands on the sliders
                mix = self.balancer(self.balancer_page, self.read_counts.reshape(1, -1), self.targets)
                self.balancer_page += 1
            slots = torch.as_tensor(mix_to_slots(mix))
            chosen = arrange(normalize_scores(cand), cand_buckets, slots, STRICTNESS)
        page_cols = cols.gather(1, chosen)[0]
        self.latencies_ms.append(1000 * (time.perf_counter() - start))

        if self.mode != "Normal feed":
            page_cols = page_cols[spread_topics(self.engine.pool_bucket_idx[page_cols].tolist())]
        self.already_shown[0, page_cols] = True
        self.page = self.engine.articles.iloc[page_cols.numpy()].reset_index(drop=True)
        self.page["pool_col"] = page_cols.numpy()
        self.page["read"] = False
        self.shown_counts += np.bincount(self.engine.pool_bucket_idx[page_cols].numpy(), minlength=len(BUCKETS))
        self.page_number += 1
        return self.page

    def read(self, position: int) -> None:
        """The reader clicks article `position` on the current page."""
        if self.page.loc[position, "read"]:
            return
        self.page.loc[position, "read"] = True
        col = int(self.page.loc[position, "pool_col"])
        self.read_counts[int(self.engine.pool_bucket_idx[col])] += 1
        kept = self.history[0][self.history[0] >= 0]
        merged = torch.cat([kept, self.engine.pool_idx[col : col + 1]])[-self.history.shape[1]:]
        self.history[0] = -1
        self.history[0, : len(merged)] = merged

    def twin_reads(self) -> None:
        """The reader's simulated twin decides what to read on the current page (click model fitted on real clicks)."""
        cols = torch.as_tensor(self.page.pool_col.values).unsqueeze(0)
        true_scores = self.engine.scores(self.start_history).gather(1, cols)
        buckets = self.engine.pool_bucket_idx[cols]
        p = self.engine.truth.probability(click_features(true_scores, buckets, self.affinity))[0]
        for position in torch.nonzero(torch.bernoulli(p, generator=self.generator)).flatten().tolist():
            self.read(position)

    def page_mix(self) -> np.ndarray:
        """Share of each bucket on the current page."""
        counts = self.page.bucket.value_counts().reindex(BUCKETS, fill_value=0).values
        return counts / counts.sum()

    def mix(self) -> pd.DataFrame:
        """Target, shown-so-far and read-so-far share per bucket."""
        def share(counts):
            return counts / counts.sum() if counts.sum() else np.zeros_like(counts)
        return pd.DataFrame({"Target": self.targets[0], "Shown": share(self.shown_counts), "Read": share(self.read_counts)}, index=BUCKETS)

    def gap(self) -> float | None:
        """Distance between what was read so far and the target (0 = perfect), or None before any reads."""
        if not self.read_counts.sum():
            return None
        return 0.5 * np.abs(self.read_counts / self.read_counts.sum() - self.targets[0]).sum()
