"""Build everything the demo needs from the raw MIND data, in one command.

    python -m src.prepare            # skips steps whose output already exists
    python -m src.prepare --force    # rebuilds everything

Steps (same recipes as the notebooks):
  1. MiniLM vectors for train-folder articles        -> artifacts/news_embeddings.npz      (notebook 04)
  2. MiniLM vectors for train + test articles        -> artifacts/news_embeddings_all.npz  (notebook 08)
  3. Two-tower model trained on Nov 9–13             -> artifacts/two_tower.pt             (notebook 05)
  4. Demo assets (live articles, readers, models)    -> artifacts/demo_assets.pt
"""
import argparse
import sys
import time

import numpy as np
import torch

from src.data import TEST_DIR, TRAIN_DIR, load_behaviors, load_news, split_by_time
from src.feed import ALL_EMBEDDINGS, DEMO_ASSETS, build_demo_assets
from src.finder import EMBEDDINGS_PATH, embed_articles, load_embeddings, save_embeddings
from src.two_tower import MODEL_PATH, TwoTower, device, history_matrix, train, training_examples

REQUIRED_FILES = [d / f for d in (TRAIN_DIR, TEST_DIR) for f in ("news.tsv", "behaviors.tsv")]


def missing_data() -> list:
    return [str(p) for p in REQUIRED_FILES if not p.exists()]


def step(name: str, output, force: bool, build) -> None:
    if output.exists() and not force:
        print(f"✓ {name}: already built ({output.name})")
        return
    start = time.time()
    print(f"… {name}")
    build()
    print(f"✓ {name}: done in {time.time() - start:.0f}s")


def embed_train() -> None:
    news = load_news(TRAIN_DIR)
    save_embeddings(news.news_id, embed_articles(news))


def embed_all() -> None:
    train_ids, train_vectors = load_embeddings()
    test_news = load_news(TEST_DIR)
    new = test_news[~test_news.news_id.isin(set(train_ids))]
    save_embeddings(np.concatenate([train_ids, new.news_id.values]),
                    np.concatenate([train_vectors, embed_articles(new)]), ALL_EMBEDDINGS)


def train_model() -> None:
    dev = device()
    news_ids, vectors = load_embeddings()
    index_of = {n: i for i, n in enumerate(news_ids)}
    behaviors = load_behaviors(TRAIN_DIR)
    train_b, _ = split_by_time(behaviors)
    row_of, hist = history_matrix(behaviors, index_of)
    examples = training_examples(train_b, index_of, row_of)
    torch.manual_seed(0)
    model = TwoTower().to(dev)
    train(model, torch.as_tensor(vectors, device=dev), torch.as_tensor(hist, device=dev), examples, epochs=3)
    torch.save(model.state_dict(), MODEL_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild every step even if its output exists")
    force = parser.parse_args().force

    if missing := missing_data():
        sys.exit("MIND-small data not found. Expected:\n  " + "\n  ".join(missing) +
                 "\nSee README → Run locally → step 2.")

    step("Embed train-folder articles", EMBEDDINGS_PATH, force, embed_train)
    step("Embed test-folder articles", ALL_EMBEDDINGS, force, embed_all)
    step("Train the two-tower model", MODEL_PATH, force, train_model)
    step("Build demo assets", DEMO_ASSETS, force, build_demo_assets)
    print("\nAll set. Start the demo with:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
