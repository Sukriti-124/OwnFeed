# OwnFeed

**Your feed, your rules.** A news recommender where you set the topic mix with sliders, and the feed sticks to it, instead of drifting toward whatever you click most.

![OwnFeed demo: Same reader's normal feed vs. OwnFeed, side by side](results/figures/demo.gif)

*Same reader, same articles. The normal feed gives a news lover ~80% News no matter what. OwnFeed follows the sliders. When the mix changes, OwnFeed follows; the normal feed doesn't.*

**End to end:** raw click logs → tested data pipeline (DuckDB + dbt) → recommender (two-tower + re-ranker + PID controller) → model monitoring and gated retraining (Airflow) → interactive demo (Streamlit). Everything runs locally; Airflow runs in Docker.

---

## The problem

Engagement-optimized feeds show you more of what you already click. That works for clicks, but it means **you don't get to decide what your feed is made of**. Someone who reads a lot of news gets an ever-newsier feed, even if they'd *like* more science, food or sport.

OwnFeed asks: **Can a recommender follow a mix the user chooses, and what does that cost?**

## How it works

Three stages, each with one job. Think of a bookshop assistant (finder), a shelf arranger (arranger) and cruise control (balancer):

```mermaid
flowchart LR
    A["~6,000 articles<br/>live today"] --> B["<b>Finder</b><br/>two-tower model<br/>+ what's trending"]
    B -->|"500 candidates<br/>(100 per topic)"| C["<b>Arranger</b><br/>pick 20 that match<br/>your sliders"]
    C -->|"page of 20"| U(("Reader"))
    U -->|"clicks"| D["<b>Balancer</b><br/>PID controller<br/>+ head start"]
    D -->|"adjusted mix<br/>for next page"| C
    S["Your sliders<br/>News · Learn · Lifestyle<br/>Entertain · Sports"] --> C
    S --> D
```

| Stage | What it does | How |
|---|---|---|
| **Finder** | Narrows ~6,000 live articles to 500 the reader would likely enjoy, **100 from each topic** so no topic runs dry | Two-tower model on frozen [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) text embeddings (title, abstract, category). Score = taste similarity + 0.05 × log(1 + clicks in the last 24 h) |
| **Arranger** | Picks the 20 articles you see so the page matches your sliders | Greedy re-ranker: each pick maximizes `relevance + strictness × open slots in that topic`. Strictness 0 = normal feed, 0.5 = matches the sliders on ~95% of pages |
| **Balancer** | Keeps what you actually **read** close to your sliders as you scroll | **Head start:** from your past clicks, predict which topics you'll under-read and show a bit more of them. **PID controller:** after each page, compare clicks with the target and nudge the next page |

The five topic sliders group MIND's 17 categories: **News** (news, weather), **Learn** (finance, health, science & tech), **Lifestyle** (food, travel, autos, home), **Entertain** (TV, movies, music, celebrities, video) and **Sports**.

## Results

All numbers are on the **locked test day (Nov 15)**, which was run **once** with every setting frozen. Every choice was made on the validation day (Nov 14). Validation numbers are in [`results/test_vs_validation.csv`](results/test_vs_validation.csv).

### 1. The finder beats "show what's trending"

| Finder (100 per topic) | Recall@500 | NDCG@10 |
|---|---|---|
| Random | 0.093 | – |
| Trending (most clicks, last 24 h) | 0.838 | **0.152** |
| **OwnFeed: two-tower + trending** | **0.859** | 0.150 |

The 500 candidates catch 86% of what readers clicked. At the very top of the list it **ties** with trending; see the findings below for why.

### 2. Following your sliders keeps ~90% of relevance

![Chart 1: relevance kept vs. distance from the slider mix](results/figures/chart1_relevance_vs_mix_test.png)

A normal feed is **36–55 points** away from the requested mix. At the chosen strictness (0.5), **96% of pages match the sliders exactly** while keeping **91% of relevance**. Strictness 0.5 is the smallest value where at least 90% of pages match exactly; the choice is documented in [notebook 06](notebooks/06_arranger.ipynb).

### 3. OwnFeed keeps what people *read* closest to their mix

Showing the right mix isn't enough: people still click their favorite topic more. We tested this with **simulated readers**, built from a click model fitted on real MIND clicks (AUC 0.70 on held-out users), each scrolling 30 pages.

![Chart 2: gap between reading and the slider mix over a session](results/figures/chart2_reading_gap_test.png)

| After 30 pages (target: 20% per topic) | Gap from sliders | Favorite topic's share of reading | Click rate |
|---|---|---|---|
| Normal feed | 30% | 41% | 3.7% |
| Fixed quotas (arranger only) | 22% | 34% | 3.4% |
| PID only | 15% | 26% | 2.9% |
| **OwnFeed (head start + PID)** | **12%** | **25%** | 2.9% |

2,000 simulated readers × 5 runs; bands show ±2 standard errors.

### Speed

Building one page (finder + arranger + balancer) takes **~1 ms** on a laptop CPU.

## Key findings

1. **Control is cheap on relevance.** Matching the slider mix exactly keeps about 90% of the relevance of an unconstrained feed.
2. **Control costs clicks.** Click rate drops from 3.7% to 2.9% (about −20%). People click less when they read what they *asked for* rather than what they *naturally* click. This is the stated-vs-revealed preference trade-off, measured.
3. **The head start does most of the balancing; the PID is the safety net.** In a validation stress test, when readers behave exactly like their history, the head start alone is as good as the full balancer (gap 9.5% vs 9.8%). When today's taste differs strongly from history, the PID clearly helps (13% vs 16%).
4. **Personalization helps less on big-news days.** On the test day the 10 most-clicked articles took **32% of all clicks** (vs. 21% on the validation day). On a day like that, "what's hot" ties with personal taste at the top of the feed. Taste still helps further down the list.
5. **A normal feed doesn't trap you within one session.** The expected "filter bubble" drift didn't show up within a 30-page session: the normal feed runs out of fresh favorite-topic articles. Bubbles build over weeks, not one scroll. The real gap is that a normal feed **ignores your stated preferences from the start**.

## Data platform and MLOps

The recommender reads its data from a **tested warehouse**, and the whole loop (load data → build tables → check them → score the model → retrain) runs as **Airflow pipelines**.

```mermaid
flowchart LR
    R["Raw MIND files<br/>data/ (TSV)"] --> I["<b>Ingest</b><br/>typed Parquet,<br/>one folder per day<br/>lake/bronze/"]
    I --> W["<b>dbt + DuckDB</b><br/>staging → marts<br/>27 data tests<br/>warehouse/ownfeed.duckdb"]
    W --> P["<b>Parity check</b><br/>warehouse = original<br/>Python logic"]
    P --> M["<b>ML</b><br/>embed new articles<br/>score the model daily"]
    M --> A["<b>Demo assets</b><br/>Streamlit app"]
    W -.-> T["<b>Retrain (on demand)</b><br/>train challenger →<br/>promote only if better"]
```

### Layers

| Layer | Where | What's in it |
|---|---|---|
| **Raw** | `data/` | MIND files exactly as downloaded (never modified, never committed) |
| **Bronze** | `lake/bronze/` | A faithful, typed copy as Parquet, one folder per day; 200 MB of TSV → 50 MB. Each row records its source file and load time; every load is logged |
| **Staging** | `warehouse/ownfeed.duckdb` | Cleaned tables: empty abstracts filled, a safe page-view key, parsed dates and hours |
| **Marts** | `warehouse/ownfeed.duckdb` | What the ML code reads: `dim_articles` (with topic), `fct_impressions` (8.6 M rows: article shown, clicked, topic, train/validation/test), `fct_hourly_trending`, `dim_user_history`, `fct_page_views` |

The **topic mapping is a version-controlled dbt seed** (`warehouse/seeds/bucket_mapping.csv`), with a reason column for every row.

### Data tests (27, run on every build)

- **Keys and references:** IDs unique and present; every article shown exists in the catalog.
- **Valid values:** topic is one of the five; clicked is 0/1; split is train/validation/test.
- **No leakage:** no row in the wrong time split; user histories never change inside the log window; "trending" never counts the current hour (re-computed independently on 2,000 random rows).
- **Sanity:** each day's click rate is 2–6%; every topic has articles every day.

### Parity check

`python -m pipeline.check_parity` proves the warehouse gives **exactly** what the original Python code computes from the raw files: page views and rows per split, clicks per topic, the topic of all 65,238 articles, trending at sample hours, user histories, and the data loaders (values, order and types). **7/7 checks pass**, so moving the ML code onto the warehouse changed no result.

> A real bug the platform caught: **page-view IDs restart at 1 in each MIND folder.** Joining the folders on that ID would silently mix up different days. The warehouse uses *folder + ID* as the key, and a data test enforces it.

### Airflow pipelines

![Airflow: the daily pipeline](results/figures/airflow_daily_dag.png)

| Pipeline | When | Steps |
|---|---|---|
| **`ownfeed_daily`** | Once per day (Nov 9 → 15 replayed with catch-up) | ingest the day → **dbt: each model and its tests as separate tasks** ([expanded view](results/figures/airflow_daily_dag_expanded.png)) → parity check → embed new articles → score the model → refresh the demo. A failed test stops the run before anything downstream |
| **`ownfeed_retrain`** | On demand | embed → **train a challenger** → compare with the champion on validation → **promote only if it wins** (Recall@500 +0.002, NDCG not worse) |

- **Model registry:** every model is a version with its metrics. The model behind the results above is pinned as **`v1-paper`** and never overwritten. In the last retrain, the challenger had a better top 10 (NDCG 0.078 vs 0.074) but no better Recall@500, so it was **not promoted**.
- **The test day is protected in code:** it can be *monitored*, but `pipeline/ml.py` refuses to use it for training or choosing a model.
- **Daily model monitoring** ([`results/model_monitoring.csv`](results/model_monitoring.csv)): OwnFeed's Recall@500 beats trending on all 7 days (0.81–0.89 vs 0.78–0.86). Nov 9–13 are training days, so those scores are optimistic.

### Engineering notes

- **Memory:** inside Docker (7.75 GB), two steps ran out of memory because they unpacked all 8.6 M "article shown" rows to find ~160 K clicks. Both were rewritten to read only what they need, with output **proven identical**: trending counts are 30× faster, and training-data preparation dropped from ~6 GB to 0.9 GB.
- **DuckDB has a single writer**, so Airflow runs one task at a time.
- **Reproducibility:** the Airflow image pins the same library versions as the project, and scores computed in Docker (CPU) match the Mac (Apple GPU) exactly.

## Design decisions

- **Time-based evaluation.** Train on Nov 9–13, validate on Nov 14, test on Nov 15 (`MINDsmall_dev`). Never a random split, because news is about *today*. 53% of validation articles never appear in training, which is why the item tower reads text instead of memorizing article IDs.
- **Locked test set.** Every setting (trending weight 0.05, strictness 0.5, PID gains 0.5 / 0.05) was chosen on validation. The test day was run once, and the results above are reported as they came out.
- **Per-topic candidate quotas.** Without them, a news lover's 500 candidates would be nearly all News and the arranger couldn't fill a Sports slot. Quotas cost almost nothing (Recall@500 0.786 → 0.777 on validation).
- **Honest simulation.** Simulated readers have a *fixed* true taste. Only the feed's picture of them changes as they click, so the simulation doesn't build in drift. The feed's own head-start model is fitted on different days from the simulator's "truth", so it never sees the answers.
- **Bronze stays faithful; cleaning happens in dbt.** If a cleaning rule is wrong, fix it and rebuild from bronze in seconds without touching the raw files.
- **Retraining is gated, not automatic.** A new model must beat the champion on held-out data before it's used, and the reported model is pinned.
- **What didn't work.** Training the two towers with the trending signal inside the loss was mixed (better top 10, worse Recall@500), so we kept the simpler setup: train on taste, add trending at scoring time. The towers also learn little beyond raw MiniLM similarity. Both experiments are documented in notebooks 05 and 07.

For every decision, the alternatives we considered and a reference for every function, see the **[technical report](docs/REPORT.md)**.

## Limitations

- **Simulated readers, not real people.** The balancer results come from a click model fitted on real clicks, not a live A/B test.
- **One day of articles per evaluation.** MIND-small has no publish times, so "live" means "shown to anyone that day".
- **Balanced targets only** in the balancer study; other slider settings were tested for the arranger.
- **The balancer can feel counter-intuitive.** To make a news lover *read* 20% News, it may *show* only ~5–10% News. A product would cap that (see future work).
- **The pipeline replays a fixed dataset.** The daily schedule simulates a live feed with MIND's seven days; the warehouse is rebuilt in full each run (it takes ~2 s).
- **MIND is news, not social posts.** It stands in for a social feed.

## Beyond news: where this applies

The idea isn't specific to news: **let people choose the *mix* of what they get, build each page to match it, and correct for what they actually consume.** Any product that ranks a stream of items of different kinds can reuse the same three stages.

| Product | What the sliders could be | What changes from OwnFeed |
|---|---|---|
| **News apps** (Google News, Apple News) | Topics; local vs. world; politics vs. science | Almost nothing. Add a "breaking news" exception so major stories get through a low News slider |
| **Instagram, TikTok, YouTube** | Friends vs. recommended; topics; calm vs. high-energy | Image/video embeddings instead of text; **watch time** instead of clicks |
| **Bluesky** | Same as OwnFeed | Custom feeds are open to anyone: the most realistic way to put OwnFeed in front of real users |
| **Spotify, Apple Music** | Genres; familiar vs. new; mood (focus / chill) | The "page" is a queue; **skips and listen-through** replace clicks |
| **Podcasts, audiobooks** | Learning vs. entertainment; episode length | The balancer works over **listening hours per week**, not per page |
| **Netflix, long video** | Genres; "finish what I started" vs. something new | The arranger fills rows on a home screen |
| **Online learning** | Subjects; review vs. new material | The balancer tracks learning goals ("you've skipped science all week") |
| **E-commerce** | Categories; budget vs. premium; sustainable share | The cost of following the sliders shows up as revenue, not clicks |
| **Parental controls** | Learning / creative / entertainment | Keeps what a child actually *watches* near the chosen mix |

**What carries over as is:** the finder → arranger → balancer design, per-category candidate quotas, the strictness rule, head start + PID, the offline evaluation (time split, locked test set, simulator fitted on real logs), and the data platform. **What changes per product:** item embeddings, the "consumed" signal, session length and ordering rules.

**Why it matters now:** the EU's Digital Services Act already requires very large platforms to explain their recommender systems and offer an option not based on profiling. OwnFeed puts a number on what user control costs (~10% relevance and ~20% clicks in a session), which a company can weigh against long-term trust. More detail is in the [technical report](docs/REPORT.md#8-beyond-news-where-this-applies).

## Run locally

The demo isn't hosted: the [MIND license](https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf) allows public demonstrations but not redistributing the data, so you need your own copy.

**1. Set up the environment** (Python 3.10+, developed on 3.14)

```bash
git clone https://github.com/Sukriti-124/OwnFeed.git
cd OwnFeed
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Get MIND-small** from [msnews.github.io](https://msnews.github.io) (requires accepting the Microsoft Research License) and unzip it so you have:

```
data/MINDsmall_train/   news.tsv  behaviors.tsv  …   (from MINDsmall_train.zip)
data/MINDsmall_test/    news.tsv  behaviors.tsv  …   (from MINDsmall_dev.zip, used as the locked test set)
```

### Option A: just the demo (~3 minutes)

```bash
python -m src.prepare              # embeddings, model, demo files
streamlit run app/streamlit_app.py
```

### Option B: the full data platform

```bash
python -m pipeline.ingest --all                          # raw → bronze Parquet (~2 s)
(cd warehouse && dbt build --profiles-dir .)             # staging + marts + 27 tests (~2 s)
python -m pipeline.check_parity                          # warehouse = original logic (7 checks)
python -m src.prepare                                    # model files (if not built yet)
```

Once the warehouse exists, all code reads from it automatically (set `OWNFEED_DATA=raw` to use the raw files instead).

### Option C: everything in Airflow (Docker)

```bash
docker compose up                  # builds the image on first run, then http://localhost:8080
```

Turn on `ownfeed_daily` to replay all seven days, or trigger `ownfeed_retrain`. Your data folders are mounted into the container, never copied into the image. Stop with `docker compose down`.

To reproduce the analysis, run the notebooks in order (`notebooks/01` → `08`).

## Repository

```
OwnFeed/
├── app/streamlit_app.py      # the demo
├── src/                      # the recommender
│   ├── data.py               # loading (warehouse or raw files), topic buckets, time split
│   ├── finder.py             # trending baseline, MiniLM embeddings
│   ├── two_tower.py          # two-tower model, training, candidate retrieval
│   ├── arranger.py           # slider mix → slots, greedy re-ranker
│   ├── balancer.py           # PID controller + head start
│   ├── simulator.py          # click model fitted on real clicks, scrolling sessions
│   ├── metrics.py            # Recall@K, NDCG, mix error, relevance kept
│   ├── feed.py               # one-reader engine used by the demo
│   └── prepare.py            # one-command setup
├── pipeline/                 # data platform + MLOps tasks
│   ├── ingest.py             # raw TSV → bronze Parquet (per day, checked, logged)
│   ├── check_parity.py       # warehouse vs. original Python logic
│   └── ml.py                 # embed, daily scoring, train challenger, promote, model registry
├── warehouse/                # dbt project (seed, staging, marts, 27 tests)
├── airflow/                  # Dockerfile + DAGs (ownfeed_daily, ownfeed_retrain)
├── docker-compose.yml        # Airflow in Docker
├── notebooks/                # the analysis, in order (below)
├── results/                  # metrics CSVs and figures
└── docs/REPORT.md            # every decision and every function
```

| Notebook | What it covers |
|---|---|
| [01 · EDA](notebooks/01_eda.ipynb) | Categories, subcategories, click shares, choosing the 5 topic buckets |
| [02 · Split](notebooks/02_split.ipynb) | Time-based train/validation split and leakage checks |
| [03 · Baseline](notebooks/03_baseline.ipynb) | Trending finder and evaluation setup |
| [04 · Embeddings](notebooks/04_two_tower.ipynb) | MiniLM article vectors and sanity checks |
| [05 · Two-tower](notebooks/05_two_tower_train.ipynb) | Training, taste + trending blend, what didn't work |
| [06 · Arranger](notebooks/06_arranger.ipynb) | Strictness sweep, Chart 1, choosing strictness 0.5 |
| [07 · Balancer](notebooks/07_balancer.ipynb) | Simulator checks, head start + PID, Chart 2, mood-shift stress test |
| [08 · Test](notebooks/08_test_results.ipynb) | The single run on the locked test day |

## Future work

- **Bounded balancer:** keep each topic's share of the page within ½–2× its slider, so the page always *looks* like what you asked for.
- **"Breaking news" exception:** let a truly major story through even when your News slider is low.
- **Feed concierge (agentic):** type "more science, less politics this week" and an LLM agent sets the sliders through tools, and checks in when you keep skipping a topic you asked for.
- **Streaming:** demo clicks as events (Kafka/Redpanda) feeding live "trending in the last hour", plus incremental dbt models instead of full rebuilds.
- **A/B test plan** for a real launch: engagement *and* "did the feed match what I asked for?" survey metrics.
- **Stronger models:** fine-tuned text encoder, MIND entity embeddings, MIND-large.

## Data and acknowledgements

Built on **MIND: Microsoft News Recommendation Dataset** (Wu et al., ACL 2020), used under the [Microsoft Research License Terms](https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf) for non-commercial research and demonstration. No MIND data is included in this repository; `data/`, `lake/`, `warehouse/ownfeed.duckdb` and `artifacts/` are all gitignored. Text embeddings: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Orchestration: Apache Airflow with Astronomer Cosmos for dbt.
