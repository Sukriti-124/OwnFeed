# OwnFeed: Technical Report

A detailed record of how OwnFeed was built: **Every Decision, The Alternatives Considered, Why We Chose What We Did, and The Evidence**, plus a reference for every module and function. 

All "validation" numbers are from **Nov 14** (`MINDsmall_train`, last day). All "test" numbers are from **Nov 15** (`MINDsmall_dev`, run once at the end with everything frozen).

---

## Contents

1. [Goal and system overview](#1-goal-and-system-overview)
2. [Decision log](#2-decision-log)
   - [2.1 Data and topic buckets](#21-data-and-topic-buckets)
   - [2.2 Train / validation / test split](#22-train--validation--test-split)
   - [2.3 Finder](#23-finder)
   - [2.4 Arranger](#24-arranger)
   - [2.5 Simulator](#25-simulator)
   - [2.6 Balancer](#26-balancer)
   - [2.7 Final test](#27-final-test)
   - [2.8 Demo](#28-demo)
   - [2.9 Hosting and the data license](#29-hosting-and-the-data-license)
   - [2.10 Deliberately deferred](#210-deliberately-deferred)
   - [2.11 Data platform and MLOps](#211-data-platform-and-mlops)
3. [Code reference](#3-code-reference)
4. [Notebooks](#4-notebooks)
5. [Metrics glossary](#5-metrics-glossary)
6. [All key numbers](#6-all-key-numbers)
7. [Limitations and open questions](#7-limitations-and-open-questions)
8. [Beyond news: where this applies](#8-beyond-news-where-this-applies)

---

## 1. Goal and system overview

**Goal:** A feed where the user sets a topic mix with sliders (e.g. 20% each of News, Learn, Lifestyle, Entertain, Sports), and the feed keeps to it instead of drifting toward what they click most. **Question:** Can a recommender do this, and what does it cost in relevance and engagement?

**Pipeline:**

```
~6,000 live articles ──▶ FINDER ──500 candidates (100/topic)──▶ ARRANGER ──20 articles──▶ reader
                          ▲                                       ▲                         │
                          │ reader's click history                │ page mix                │ clicks
                          │                                       └──── BALANCER ◀──────────┘
                          └──────────────── sliders ─────────────────────▲
```

| Stage | Job | One-line method |
|---|---|---|
| Finder | Don't miss anything the reader would like | Two-tower on frozen MiniLM vectors + 0.05 × log(1 + 24 h clicks), top 100 per topic |
| Arranger | Make the page match the sliders | Greedy pick of 20: `relevance + strictness × open slots`, strictness 0.5 |
| Balancer | Make what's *read* match the sliders | Head start from history (feedforward) + gentle PID on clicks (feedback) |
| Simulator | Test the balancer without real users | Logistic click model fitted on real MIND clicks; fixed true taste per reader |

---

## 2. Decision log

Each entry gives the **decision**, the **alternatives** considered, **why** we chose it, and the **evidence**.

### 2.1 Data and topic buckets

**Dataset: MIND-small.** 50,000 users, 51,282 train-folder articles, 156,965 impressions (Nov 9–14), plus a separate Nov 15 folder.
- *Alternatives:* MIND-large (more compute for the same story); other news datasets (fewer real impression logs).
- *Why:* real impressions with clicked/skipped labels, article text and categories, and a laptop-sized dataset.

**Five buckets, not the four originally planned.**
- *History:* the first draft had 4 buckets with "Learn" = news + finance + health + weather. That bucket held **~45% of clicks**, mostly plain news. The user proposed splitting it, giving **News** (current affairs) and **Learn** (money, health, science).
- *Final mapping:* by category, with 5 subcategory exceptions:

| Bucket | Categories | Subcategory overrides | Share of clicks (train) | CTR |
|---|---|---|---|---|
| News | news, weather | | 30.7% | 4.4% |
| Learn | finance, health | + `newsscienceandtechnology` | 13.9% | 3.6% |
| Lifestyle | lifestyle, foodanddrink, travel, autos | | 15.4% | 2.9% |
| Entertain | entertainment, movies, music, tv, video | + `lifestyleroyals`, `lifestylebuzz`, `lifestylecelebstyle`, `awardstyle` | 28.1% | 4.6% |
| Sports | sports | | 11.9% | 4.8% |

- *Why the overrides:* exploring subcategories showed **~60% of "lifestyle" clicks were celebrity gossip** (royals, buzz, celeb style). Moving those into Entertain means **all celebrity content lives in one bucket**, so turning Entertain down really removes it, and Lifestyle means food, travel, cars and home. Science and tech is the one part of `news` that's about learning rather than current events.
- *Why finance stays whole in Learn:* moving business and markets news into News shrank Learn to **~7% of clicks**, too thin for a slider.
- *Why category level, not hand-mapping all 264 subcategories:* simple, explainable, and 5 overrides fixed the only real problems.
- *Dropped:* `kids`, `middleeast`, `northamerica` (20 articles, almost no clicks).
- *Evidence the buckets are useful:* every bucket has 6,000+ articles; **41% of active users get more than half their clicks from one bucket** (people genuinely differ, so sliders matter); Learn and Lifestyle are the "wish" buckets, with the lowest CTR and a quarter of active users never clicking them.

### 2.2 Train / validation / test split

**Decision: Split by time.** Train = Nov 9–13 (126,695 impressions, 81%), validation = Nov 14 (30,270, 19%), test = Nov 15 (`MINDsmall_dev`, 73,152 impressions), **locked until the end**.
- *Alternative:* a random split, rejected because it lets the model see the future. News is about *today*.
- *Leakage check:* every user's `history` is **identical across all their impressions**, so it was collected before the log window and is safe to use as input.
- *What the split revealed (and how it shaped the model):*
  - **53% of validation articles never appear in training, and they get 36% of clicks.** So the item tower must read text, not memorize article IDs.
  - **20% of validation users are new**, but only 94 of them have no history. So the user tower builds a user from their history, not from a user ID.
  - The daily bucket mix wobbles (Sports 8–19% of clicks by day), so small validation differences can be noise.

### 2.3 Finder

**Candidate pool:** Every bucketed article shown to anyone that day (6,141 on Nov 14, 5,365 on Nov 15). MIND has no publish times, so "shown that day" is the best proxy for "live".

**Baseline: trending.** For each page view, clicks per article in the **24 hours before**, excluding the current hour (no peeking). Computed per hour for speed.
- *Why a strong baseline:* "beats popular" is the bar every recommender must clear, and a weak baseline makes a win meaningless.
- *Result (validation):* Recall@500 **0.786** (random 0.081), NDCG@10 only **0.048**, because everyone gets the same top 10.

**Per-bucket quotas (100 per topic).**
- *Why:* Without them a news lover's 500 could be nearly all News, and the arranger couldn't fill a Sports slot ("candidate starvation").
- *Cost:* Recall@500 0.786 → **0.777**. It actually *raised* recall for Sports (0.60 → 0.69) and Learn (0.77 → 0.80).

**Text representation: frozen MiniLM** (`all-MiniLM-L6-v2`, 384 numbers per article; text = category / subcategory / title / abstract).

| Option considered | Verdict |
|---|---|
| IDs + category only | Rejected: fails on the 53% of new articles |
| TF-IDF | Rejected: matches words, not meaning |
| MIND entity vectors | Kept as a possible add-on |
| **Frozen small pre-trained model (MiniLM)** | **Chosen:** understands meaning, handles new articles, ~80 s for 51K articles on a laptop |
| Bigger frozen model (mpnet) | 3–5× slower for small gains on headlines |
| Fine-tuning the language model | Best accuracy, but hours of training; future work |
| Paid embedding API | Cost, keys, and harder to reproduce |

- *Sanity check:* nearest neighbors make sense (a Patriots–Ravens preview finds other Patriots–Ravens stories, similarity 0.81–0.85). **82% of 5 nearest neighbors share a bucket vs 24% by chance.**

**Two-tower model.** User = mean of the last 50 clicked articles' vectors → MLP (384→256→128). Item = article vector → MLP (384→256→128). Both are L2-normalized; score = dot product.
- *Training:* each Nov 9–13 click is one example. Negatives are 4 skipped articles from the same page plus the other clicks in the batch (in-batch negatives). Cross-entropy with temperature 0.05, Adam with learning rate 1e-3, batch 512, **3 epochs**, seed 0. 189,519 training clicks.

**Finding: taste alone loses badly to trending, and training adds little.**
- Two-tower alone: Recall@500 **0.225** / NDCG@10 0.014. Raw MiniLM (no training): **0.231** / 0.013.
- *Why:* taste says "football", but there are hundreds of football stories; people click the one everyone is talking about.
- *Bug check before concluding:* only 2% of users have empty histories, and the data lines up.

**Decision: blend taste with trending.** Score = similarity + **β** × log(1 + clicks in last 24 h).

| β (validation) | Recall@500 | NDCG@10 |
|---|---|---|
| 0 | 0.225 | 0.014 |
| 0.03 | 0.784 | 0.075 |
| **0.05** | **0.811** | **0.074** |
| 0.10 | 0.804 | 0.063 |

β = 0.05 beats trending on every metric (NDCG@10 is about 50% higher). *Caveat:* β was chosen on validation, so the honest number comes from test.

**Tried and rejected: trending-aware training (option B).** Adding β × trending inside the training scores, so the towers learn only what trending doesn't explain.
- β = 0.05 or 0.1 during training was clearly worse (Recall@500 0.44–0.79).
- Skipping Nov 9 (when trending is zero for every article) with β = 0.02: NDCG@10 0.081, but Recall@500 0.799 < 0.811. Mixed.
- *Decision:* keep the simple version (train on taste, add trending at scoring time), because recall is the finder's main job.

### 2.4 Arranger

**Decision: greedy re-ranker with a strictness knob.**
1. Slider percentages → 20 integer slots (largest-remainder rounding, so they always sum to 20).
2. Pick 20 articles one at a time. Each pick maximizes `relevance + strictness × (open slots left in the article's bucket)`.
3. Relevance = the finder score rescaled per page to 0..1 (best candidate = 1).

- *Why greedy + knob, not hard quotas:* one knob gives the whole trade-off curve (Chart 1). 0 = normal feed; large values = exact slots.
- *Evaluation:* 4 slider settings (Balanced 20/20/20/20/20, Less news 10/30/20/20/20, Learner 10/50/20/10/10, and a random mix per page view), over every validation page view.
- *Result:* a normal feed is **37–54 points** from the requested mix. Exact matching keeps **88–93%** of relevance; the further the mix is from the reader's taste, the more it costs.

**Choosing strictness = 0.5.**
- *History:* the first sweep used 6 hand-picked values (0, 0.01, 0.03, 0.1, 0.3, 1), and 0.3 was proposed. The user asked for a finer comparison. A 15-value sweep showed that at 0.3, **two in three pages were still about one article off** (only 34% exact).
- *Rule adopted:* **the smallest strictness where at least 90% of pages match the sliders exactly** (averaged over the 4 settings). People judge sliders by counting articles, and the *smallest* such value avoids paying for strictness we don't need.

| Strictness | Pages exact | Articles off per page | Relevance kept |
|---|---|---|---|
| 0.3 | 34% | 0.68 | 91.8% |
| 0.4 | 72% | 0.28 | 90.9% |
| **0.5** | **94%** | **0.06** | **90.3%** |
| 0.7–1.0 | ~100% | 0 | 90.1% |

- *Why not finer (e.g. 0.47, where the rule is first met):* 0.47 vs 0.5 differs by 0.1% in relevance, which is below day-to-day noise. Reporting "0.47" would be false precision from overfitting to one day.
- *Note:* random mixes are hardest (84% exact at 0.5, 90% at 0.6), and they can't reach 0 mix error because each article is 5% of a 20-article page.

**Recall@20 caveat:** real clicks landing in the 20 drop from 0.22 to ~0.15 at strictness 0.5. This is biased against OwnFeed, because those clicks were made on MSN's normal feed.

### 2.5 Simulator

**Why a simulator:** the balancer reacts to clicks over a session, and MIND has no "what if the page had been different" data.

**Click model: logistic regression on 7 features**, fitted on real clicks:
1. finder score (similarity + 0.05 × trending)
2. bucket liking = log(5 × the user's smoothed share of past clicks in that bucket)
3–7. a one-hot bucket indicator (each bucket's own baseline click rate)

| Version (fit on half of Nov 14 users, tested on the other half) | AUC |
|---|---|
| score only | 0.654 |
| bucket liking only | 0.602 |
| score + liking | 0.667 |
| **+ bucket baselines** | **0.672** |

- *Why add bucket baselines:* without them, predicted CTR was off per bucket (Lifestyle 3.6% vs real 2.7%, Sports 2.8% vs 3.7%). With them, **every bucket is within 0.2 points**. That matters because the balancer reacts to per-bucket click rates.
- *Credibility:* AUC 0.67–0.70 is in line with published MIND news models (~0.65–0.68).

**Session design decisions:**
- **Fixed true taste.** Click chances come from the reader's *starting* history. Only the feed's picture of them updates with clicks. *Why:* letting clicks change true taste would build drift into the simulation.
- **No repeats.** Already-shown articles are excluded.
- **Measure with expected clicks.** The "reading mix" uses the sum of click probabilities, not sampled clicks. *Why:* with ~1 click per page, a sampled reading mix after page 1 is 100% one bucket, which is pure noise.
- **Per-user drift, not averages.** Averaging over users hides drift (one user drifting to News and another to Sports cancel out), so drift is measured as each user's favorite bucket's share.
- **Readers:** Nov 14 users with 10+ past clicks; session time = their first Nov 14 page view; Balanced target.

**Finding: no filter bubble within one session.** The normal feed's favorite-topic share of the page *fell* (45% → 31% over 30 pages), because it runs out of fresh favorite-topic articles, and one session adds only ~17 clicks to a 50-click history. The original plan's claim ("the normal feed drifts") was replaced with the honest version: **a normal feed ignores your sliders from the start, and fixed quotas show the mix but people still read their favorite.**

**Mood shift (stress test).** Today's taste = history taste × a random log-normal factor per bucket (spread 0 / 0.5 / 1.0). *Why:* real people don't behave exactly like their history, and that's the case feedback exists for.

### 2.6 Balancer

**PID controller (feedback).** After each page: error = target − clicked share. The clicked share is smoothed with 5 "prior clicks" on target, to calm early noise. Next page mix = target + clip(kp·e + ki·Σe + kd·Δe, ±max_push), clipped at 0 and renormalized.

**PID alone was noisy and slow:**

| Feed (validation, 30 pages) | Gap at page 30 | Favorite share of reading |
|---|---|---|
| Normal feed | 0.28 | 41% |
| Fixed quotas | 0.19 | 34% |
| PID only (best gains) | 0.14 | 25% |

**Diagnostic: an oracle that knows each reader's true per-bucket click chance** and shows mix ∝ target / chance reached **0.07–0.10**. So the floor wasn't built in: the PID was limited by learning from ~1 click per page, starting from zero.

**Decision: add a head start (feedforward).** Before the session, estimate each bucket's click chance for this reader with **a separate click model fitted on the training days (Nov 9–13)**, never on the simulator's Nov 14 truth. Then start from mix ∝ target / estimated chance ("a bucket you click half as often gets shown twice as much"). The PID corrects around that.
- The estimate correlates **0.991** with the truth, so the head start alone ≈ the oracle.
- Adding the PID on top made things slightly *worse* when readers behaved exactly like their history, because it was reacting to noise. **The mood-shift test showed why it's still needed:**

| Gap after 30 pages (validation, seed 0) | No shift | Some (0.5) | Big (1.0) |
|---|---|---|---|
| Fixed quotas | 0.188 | 0.199 | 0.226 |
| PID only | 0.140 | 0.147 | 0.160 |
| Head start only | **0.095** | 0.116 | 0.162 |
| **Head start + PID (kp 0.5, ki 0.05)** | 0.098 | **0.109** | **0.133** |

- *Gains:* gentle values (kp 0.5, ki 0.05, kd 0, max_push 0.2) beat stronger ones (kp 1–2, ki 0.1–0.3), which overreact to a handful of clicks.
- *Final:* **head start + gentle PID.** The head start does most of the work; the PID is the safety net when today differs from history.
- *Cost measured:* click rate 3.8% → 3.0% (validation). People click less when they read what they asked for: the **stated-vs-revealed preference trade-off**.

**Open design issue:** to make a news lover *read* 20% News, the head start may *show* only 3–9% News, which feels like a broken slider. The proposed fix (keep each topic within ½–2× its slider) was not adopted, because it would change the balancer after the test run. It's listed as future work.

### 2.7 Final test

**Run once on Nov 15 with everything frozen.** The only thing fitted on Nov 15 is the simulator's truth click model, which describes the test world rather than an OwnFeed setting. 14K new test articles were embedded with the same frozen MiniLM.

| Metric | Validation | Test | Held? |
|---|---|---|---|
| Recall@500, trending → OwnFeed | 0.777 → 0.811 | 0.838 → **0.859** | ✅ |
| NDCG@10, trending → OwnFeed | 0.048 → 0.074 | 0.152 → **0.150** | ❌ tie |
| Arranger pages exact (λ 0.5) | 94.5% | 95.8% | ✅ |
| Arranger relevance kept | 90.3% | 90.7% | ✅ |
| Reading gap: normal / quotas / OwnFeed | 29 / 20 / 11% | 30 / 22 / **12%** | ✅ |
| Click rate: normal → OwnFeed | 3.8 → 3.0% | 3.7 → 2.9% | ✅ same cost |
| Simulator AUC (held-out users) | 0.672 | 0.702 | ✅ |

**Why NDCG@10 tied:** Nov 15 was a big-news day. **The 10 most-clicked articles got 31.6% of clicks vs 21.0% on Nov 14**, and on such a day "what's hot" is hard to beat at the very top.
- *Decision:* don't tweak. The core claims held, and tuning after seeing test would compromise it. Reported as is.

### 2.8 Demo

**Iteration 1 (rejected): phone feed + control panel.** A reader picker, sliders, a Normal/OwnFeed toggle, ♥ Read buttons, next page and auto-scroll, and a Target/Shown/Read chart. The user found it **confusing**: too many concepts at once, a 3-bar chart, nothing happening without clicks, and the balancer showing very little of a topic the user had set to 20%.

**Iteration 2 (final): one top-to-bottom page.**
1. **Pick a reader**, a real Nov 15 user with the strongest lean toward one topic. Shown with a plain sentence: "96% of their past clicks were News" and "Borrow a real reader's click history so the feed knows their taste." *Why readers at all:* without a history there's no "natural pull" to fight, and both feeds would look the same.
2. **Set the mix:** five sliders, scaled to 100%.
3. **Normal feed vs OwnFeed side by side** (same reader, same articles). OwnFeed here is **"sliders only"** (the arranger, no balancer), so the page visibly matches the sliders. Each column has a 2-bar chart (your slider vs this page) and 20 stories.
4. **"Simulate 10 pages of reading":** 20 twin sessions plotting the reading gap for Normal / OwnFeed (sliders only) / OwnFeed + balancer. This is where the balancer and its counter-intuitive "show less of your favorite" behavior appear, with an explanation.

**Other demo decisions:**
- **Topic spreading:** OwnFeed pages are reordered so each next story is the best remaining one from a different topic than the one before. *Why:* both feeds sort by relevance, so the first 5 headlines were identical and the difference was hidden below. Spreading changes only the order, not which 20 are picked, so no measured result changes. The top 2–3 stories still match the normal feed, which is realistic (the day's biggest stories).
- **Read button:** considered and left out. With one page per column it would have no visible effect, and the in-session effect on the normal feed is small.
- **Style:** "Warm editorial" (cream paper `#FAF6EF`, ink text, Fraunces serif headlines, earthy topic colors), chosen over "dark social app" and "bright & playful".
- **Visual QA:** screenshots via a headless browser (Playwright) caught a clipped title, squashed charts with missing labels, and invisible button text after a click. All were fixed.
- **Speed:** ~1 ms per page for one reader on CPU (finder + arranger + balancer).
- **GIF:** recorded headlessly with Playwright driving the live app (49 s) and converted with ffmpeg at 820 px / 8 fps (6.9 MB).

### 2.9 Hosting and the data license

**The MIND license (Microsoft Research License Terms)** allows use for "non-commercial… research purposes", including **"public demonstrations"** (1a), but says you will not **"distribute the Dataset"** (2a). Publications may not include "a material portion" (1c).

- A Streamlit Cloud deploy failed because `data/` isn't in the repo (correctly gitignored).
- *Option "download from Microsoft at startup":* **not possible.** The official Azure storage now returns *"Public access is not permitted"*, and the `recommenders` mirror no longer resolves.
- *Option "private bundle on Hugging Face + public Space":* a grey area under "do not distribute", so the user judged it too risky.
- **Decision: no hosting.** The README shows the demo GIF plus "run locally" instructions, and `python -m src.prepare` builds everything from the user's own MIND copy in ~2.5 minutes. **No MIND data or derived article text is committed.**

### 2.10 Deliberately deferred

| Item | Status / reason |
|---|---|
| ~~Data engineering layer~~ | **Built** (§2.11): ingestion → Parquet → DuckDB + dbt → Airflow in Docker |
| Streaming (Kafka/Redpanda) | Future work: live "trending in the last hour" from click events; incremental dbt models |
| Kubernetes | Optional; Airflow already runs in Docker |
| Feed concierge (agentic) | Future work: an LLM agent turns plain-language requests into slider settings via tools, and checks in when a requested topic keeps being skipped. Only the request and topic names would go to the LLM, never MIND text |
| Bounded balancer (½–2× slider) | Future work: would change results after the test run |
| "Breaking news" exception | Future work: let major stories through a low News slider |
| A/B test | A written plan only; simulated A/B results would be circular |
| Fine-tuned encoder, entity vectors, MIND-large | Future model upgrades |

### 2.11 Data platform and MLOps

**Why add it at all.** For the ML results alone it adds nothing: the numbers are identical. Its value is in how the data is handled. Before it, the project had real problems a data layer prevents:
- the bucket mapping existed in two places (notebook 01 kept its own copy);
- every notebook re-read and re-unpacked the raw TSVs;
- "trending" was computed in four places;
- click-data preparation was copy-pasted into notebook 07, notebook 08 and `feed.py`;
- leakage was checked once, by hand.

It also shows the full "raw logs → tested pipeline → model → product" story that data-engineering and ML-platform roles look for.

**Phase 1: ingestion (`pipeline/ingest.py`).** DuckDB reads the raw TSVs and writes ZSTD Parquet to `lake/bronze/`, news by source folder and behaviors **one folder per day** (Nov 9–15).
- *Bronze stays faithful:* the impressions list stays one text column, because unpacking it is a transformation for dbt. If a cleaning rule is wrong, rebuild from bronze without touching the raw files.
- *Idempotent:* each partition is written to a `.tmp` folder and swapped in, so reloading a day replaces it.
- *Checked at load time:* rows > 0, parsable times, keys present, every impressions entry matches `^N\d+-[01]( N\d+-[01])*$`. A failure writes nothing and is logged.
- *Traceable:* every row has `_source_file` and `_ingested_at`; `lake/_load_log.csv` records rows, file MD5, duration and status.
- *Verified:* content identical to the pandas loaders; reloads add no duplicates; a planted malformed line is rejected; 200 MB of TSV → 50 MB of Parquet in ~2 s.
- *Finding:* **`impression_id` restarts at 1 in each source folder** (train 1–156,965, test 1–73,152). The warehouse key is therefore `source + impression_id`, enforced by a uniqueness test.

**Phase 2: warehouse (DuckDB + dbt, `warehouse/`).**
- *Seed:* `bucket_mapping.csv`, 19 rows with a `reason` column; overrides join on subcategory only, exactly like `src/data.py`.
- *Staging:* `stg_news` (one row per source and article, keeping `file_row` because saved embeddings follow the raw file's order) and `stg_behaviors`. These are stored as **tables, not views**: views pointed at `../lake` by a relative path and broke when queried from anywhere but `warehouse/`.
- *Intermediate:* `int_impressions` unpacks each impression list.
- *Marts:* `dim_articles`, `dim_article_sources`, `fct_impressions` (8,584,442 rows with bucket and split), `fct_page_views`, `fct_hourly_trending` (dense hourly grid per article; window = 24 preceding to 1 preceding hour) and `dim_user_history` (last 50 clicks per source and user).
- *27 data tests:* 20 schema tests (unique, not null, accepted values, relationships) and 7 singular ones: no split leakage, history fixed per user, one history per user, one row per source and article, daily CTR 2–6%, every bucket shown every day, and trending excludes the current hour (recomputed independently on 2,000 sampled rows).
- *Paths:* the profile and sources read `OWNFEED_WAREHOUSE` / `OWNFEED_LAKE` with relative defaults. Airflow runs dbt from a temporary copy of the project, so relative paths alone would break. dbt doesn't evaluate `env_var` inside `vars:`, so it's called where the path is used.

**Parity check (`pipeline/check_parity.py`).** Proves the warehouse gives exactly what the original Python code computes from the raw files: page views and rows per split, clicks per bucket, every article's bucket, trending at 5 sample hours, last-50 histories, and the `src/data.py` loaders (values, order and dtypes). It forces `OWNFEED_DATA=raw` for its Python side so it never compares the warehouse with itself. **7/7 pass.**
- *Why keep it after the switch:* it guards future warehouse edits, and it keeps the warehouse and raw-file paths identical. Without the raw-file path, "run locally" wouldn't work for people who skip the data layer. If the raw path were ever removed, the plan is to replace it with cheaper "golden number" tests inside dbt.

**Switching the ML code to the warehouse.** `src/data.py` reads `marts.dim_article_sources` + `dim_articles` (ordered by `file_row`) and `marts.fct_page_views` (ordered by `impression_id`) when `warehouse/ownfeed.duckdb` exists, and falls back to the raw files otherwise. Function names and outputs are unchanged, so no notebook or app code changed. Notebooks 02/03/04/06 were re-run through the warehouse and rewrote their result files identically.
- *Lesson:* a first verification "passed" only because the test scripts had been deleted and the notebooks never ran. The final check confirms output files were actually rewritten (timestamp) before comparing.

**Phase 3: orchestration (Airflow in Docker).**
- *Airflow over Dagster:* Airflow is far more common in job postings, its daily runs and catch-up fit replaying MIND's seven days, and running it in Docker avoids Python 3.14 compatibility issues.
- *Setup:* `docker-compose.yml` + `airflow/Dockerfile` (Airflow 3.3.2, Python 3.12; dbt in its own virtualenv as Cosmos recommends; the same torch / pandas / DuckDB versions as the project). The repo is mounted, so data is never baked into the image.
- *`ownfeed_daily`:* runs @daily from 2019-11-09 to 2019-11-15 with catch-up: ingest the day → Cosmos `DbtTaskGroup` (each model's run and test as separate tasks) → parity → embed new articles → score the champion → refresh the demo. `max_active_tasks=1` because DuckDB has a single writer. All 7 days succeeded.
- *`ownfeed_retrain`:* manual, with `train_until` / `eval_day` parameters: embed → train challenger → evaluate and promote.
- *MLOps rules (`pipeline/ml.py`):*
  - The model registry (`artifacts/models/registry.json`) pins `v1-paper`, the model behind every reported number, and never overwrites it.
  - A challenger is promoted only if Recall@500 improves by at least 0.002 and NDCG@10 doesn't drop by more than 0.002.
  - **The test day may be monitored but never used to train or choose a model**; the code raises an error.
  - `evaluate_day` rebuilt only from warehouse marts reproduces 0.811 / 0.074 (validation) and 0.859 / 0.150 (test) exactly.
  - In Airflow, a challenger trained on CPU scored 0.8105 / 0.0783 vs the champion's 0.8106 / 0.0743, and was **not promoted**.
- *Memory (Docker VM 7.75 GB):* two steps were killed for running out of memory, because both unpacked all 8.6 M shown rows to reach ~160 K clicks. Both were rewritten in `src/` (with the user's agreement) and proven identical:
  - `finder.hourly_clicks` extracts clicked IDs only: an identical table, 11.9 s → 0.4 s.
  - `two_tower.training_examples` walks impressions one by one: all 189,519 examples identical to the committed version (same random negatives), peak ~6 GB → 0.9 GB, 4× faster.
- *Reproducibility:* scores computed in Docker (CPU) match the Mac (Apple GPU) exactly.

---

## 3. Code reference

All code lives in `src/` (a package). Notebooks import it with `sys.path.insert(0, "..")`.

### `src/data.py`: loading MIND, buckets, time split

| Name | What it does |
|---|---|
| `DATA_DIR`, `TRAIN_DIR`, `TEST_DIR` | Paths to `data/`, `data/MINDsmall_train`, `data/MINDsmall_test` (the locked test set) |
| `VALIDATION_DAY = "2019-11-14"` | Impressions on or after this day are validation |
| `BUCKETS` | `["News", "Learn", "Lifestyle", "Entertain", "Sports"]`, the fixed order used everywhere |
| `CATEGORY_TO_BUCKET` | Category → bucket map (§2.1) |
| `SUBCATEGORY_OVERRIDES` | 5 subcategories that move bucket (science & tech → Learn; 4 celebrity subcategories → Entertain) |
| `assign_bucket(news)` | Bucket per article: the subcategory override wins, otherwise the category's bucket; NaN for dropped categories |
| `WAREHOUSE_PATH`, `use_warehouse()` | `warehouse/ownfeed.duckdb`; the warehouse is used when it exists unless `OWNFEED_DATA=raw` |
| `load_news(folder)` | Articles of one source folder (id, category, subcategory, title, abstract, bucket) in the raw file's order: from the warehouse marts, or from `news.tsv` (empty abstracts filled, `bucket` assigned) |
| `load_behaviors(folder)` | Page views (impression id, user, time, history, impressions): from `marts.fct_page_views`, or from `behaviors.tsv` (time parsed, empty histories as `""`). Both paths are identical (parity check) |
| `explode_impressions(behaviors, news=None)` | One row per *(page view, article shown)* with `clicked` 1/0; optionally adds category/subcategory/bucket |
| `split_by_time(behaviors, validation_day)` | Returns `(train, validation)` split at the validation day |

### `src/finder.py`: trending baseline and text embeddings

| Name | What it does |
|---|---|
| `ARTIFACTS_DIR` | `artifacts/` (gitignored) for models and embeddings |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDINGS_PATH` | `artifacts/news_embeddings.npz` (train-folder articles) |
| `hourly_clicks(behaviors)` | Clicks per article per hour: rows = hour, columns = news_id. Extracts only clicked IDs (`N123-1`) instead of unpacking every article shown, so it's ~30× faster with the same table |
| `trending_scores(hourly, hour, window_hours=24)` | Clicks per article in the 24 h before `hour`, **excluding the current hour** (no peeking) |
| `rank(scores, pool)` | Pool articles sorted by score (articles without clicks last, stable order) |
| `with_bucket_quotas(ranked, bucket_of, per_bucket=100)` | Keeps the top 100 of each bucket from a ranked list, preserving overall order |
| `article_text(news)` | The text MiniLM reads: `category / subcategory. title. abstract` |
| `embed_articles(news, batch_size=256)` | MiniLM vector (384, unit length) per article; imports torch lazily so the trending baseline stays light |
| `save_embeddings(news_ids, vectors, path)` / `load_embeddings(path)` | Save/load `(news_ids, vectors)` as `.npz`; IDs are stored as plain strings so they load without pickle |

### `src/two_tower.py`: model, training, candidate retrieval

| Name | What it does |
|---|---|
| `MODEL_PATH` | `artifacts/two_tower.pt` |
| `HISTORY_LEN = 50` | Number of most recent past clicks that describe a user |
| `device()` | Apple GPU (`mps`) if available, else CPU |
| `TwoTower` | `user_tower` and `item_tower`, each Linear(384→256)–ReLU–Linear(256→128) |
| `TwoTower.users(history_vectors, mask)` | Masked mean of history vectors → user tower → unit vector |
| `TwoTower.items(vectors)` | Article vector → item tower → unit vector |
| `history_matrix(behaviors, index_of)` | `(row_of, matrix)`: one row per user with their last 50 clicked article indices, padded with −1 |
| `training_examples(behaviors, index_of, row_of, n_negatives=4, seed=0)` | One example per click: `(user row, clicked article, 4 skipped articles from the same page, hour)`; falls back to random negatives if a page had no skips. Walks page views one at a time (peak memory ~6 GB → 0.9 GB), with output identical to the earlier table-based version |
| `popularity_table(hourly, hours, index_of, window_hours=24)` | `log(1 + clicks in the previous 24 h)` for every article at each requested hour; returns `(hour → row, table)` |
| `user_vectors(model, history, article_vectors)` | User-tower output for every history row, in batches |
| `train(model, article_vectors, history, examples, …)` | Training loop: in-batch + same-page negatives, cross-entropy at temperature 0.05. Optional `popularity` / `example_hours` / `beta` add trending inside training (the rejected option B). Optional `on_epoch_end` callback |
| `top_k_with_quotas(scores, item_buckets, buckets, per_bucket=100, return_scores=False)` | For each row, the top 100 items per bucket, merged and sorted by score; optionally also returns the scores |
| `find_candidates(model, user_rows, hour_rows, …, beta=0.05)` | **The final finder** for many page views at once: similarity + β × trending, quotas, chunked. Returns `(pool columns, scores)` |

### `src/arranger.py`: matching the slider mix

| Name | What it does |
|---|---|
| `PAGE_SIZE = 20` | Articles per page |
| `mix_to_slots(targets, page_size)` | Slider shares → whole article counts that sum to 20 (largest-remainder rounding) |
| `normalize_scores(scores)` | Rescales each row of candidate scores to 0..1 (best = 1, worst = 0) |
| `arrange(relevance, candidate_buckets, slots, strictness)` | Greedy, batched pick of 20: each step takes the max of `relevance + strictness × open slots in its bucket`. Returns chosen positions, best relevance first |

### `src/metrics.py`

| Name | What it does |
|---|---|
| `recall_at_k(ranked, clicked, k)` | Share of clicked articles in the top k |
| `ndcg_at_k(ranked, clicked, k)` | 1.0 if the clicked articles are at the very top; discounted by log of position |
| `evaluate(rankings, clicked_lists, ks, ndcg_k)` | One row per page view with Recall@k for each k and NDCG@10 |
| `mix_error(page_buckets, targets)` | Per page: ½ × Σ\|page share − target\| (0 = perfect, 1 = completely off) |
| `relevance_kept(chosen_relevance, top_relevance)` | Per page: chosen relevance ÷ plain top-20 relevance |

### `src/simulator.py`: fake readers fitted on real clicks

| Name | What it does |
|---|---|
| `bucket_affinity(history, article_buckets, n_buckets, smoothing=1)` | Each user's share of past clicks per bucket, add-one smoothed |
| `click_features(scores, buckets, affinity)` | The 7 click-model inputs: score, bucket liking `log(5 × affinity)`, one-hot bucket |
| `ClickModel` | Logistic regression; `fit` (L-BFGS), `probability`, `state` / `from_state` (save/load the 7 weights + bias) |
| `auc(probabilities, clicked)` | Rank-based AUC (0.5 = guessing) |
| `World` | Everything fixed during a session: model, article vectors, pool, pool buckets, per-user trending, truth click model, β |
| `simulate(world, history, affinity, targets, policy, pages=20, strictness=0.5, seed=0)` | Batched scrolling. Per page: finder from the *current* history (no repeats) → arranger with the policy's mix (or plain top 20 if the policy is `None`) → clicks sampled from the *fixed true* taste → clicks join the history. Returns per-page bucket counts of shown, clicked and expected clicks |
| `bucket_click_chance(world, history, affinity, click_model)` | Average click probability per bucket over each user's candidates: the oracle with the truth model, or the head-start estimate with the feed's model |
| `shift_taste(affinity, mood, seed)` | Today's taste = history × log-normal noise (spread `mood`), renormalized |

### `src/balancer.py`

| Name | What it does |
|---|---|
| `PIDBalancer(kp, ki, kd, max_push, prior_clicks, start_mix)` | Callable policy `(page, clicked_counts, targets) → page mix`. Resets on page 0; error = target − smoothed clicked share; correction clipped to ±`max_push`; corrects around `start_mix` (the head start) if given |
| `feedforward_mix(targets, click_chance)` | Head start: mix ∝ target / expected click chance |

### `src/feed.py`: the one-reader engine behind the demo

| Name | What it does |
|---|---|
| `ALL_EMBEDDINGS`, `DEMO_ASSETS` | `artifacts/news_embeddings_all.npz` (train + test articles), `artifacts/demo_assets.pt` |
| `DEMO_HOUR`, `STRICTNESS`, `BETA` | Nov 15 12:00, 0.5, 0.05: the frozen settings |
| `PERSONAS` | Demo reader name → the bucket they lean toward |
| `_click_rows(...)` | Click-model features and labels for every bucketed article shown in a set of impressions |
| `build_demo_assets(path)` | Precomputes the Nov 15 live pool, trending at 12:00, the 5 personas (users with 30+ clicks and the strongest lean per bucket), and both click models (truth on Nov 15, feed model on Nov 9–13) |
| `spread_topics(buckets)` | Reorders a page so consecutive stories have different topics (display only) |
| `FeedEngine` | Loads everything once (CPU): articles, vectors, model, item vectors, pool, trending, personas, click models. `scores(history)` = finder score for one reader |
| `FeedSession(engine, persona, targets, mode, seed)` | One reader scrolling. Modes: `"Normal feed"` (plain top 20), `"Sliders only"` (arranger), `"OwnFeed"` (arranger + balancer) |
| `FeedSession.set_controls(targets, mode)` | New sliders or mode: recompute the head start and reset the balancer |
| `FeedSession.next_page()` | Builds the next 20 articles (never repeats), times it, spreads topics, updates the shown counts |
| `FeedSession.read(position)` | The reader clicks an article: counts it and adds it to the history |
| `FeedSession.twin_reads()` | The reader's simulated twin clicks articles on the current page (truth click model) |
| `page_mix()`, `mix()`, `gap()` | Current page's bucket shares; target/shown/read table; reading gap (None before any reads) |

### `src/prepare.py`: one-command setup

| Name | What it does |
|---|---|
| `REQUIRED_FILES`, `missing_data()` | The four MIND files the project needs, and which are missing |
| `step(name, output, force, build)` | Runs a build step unless its output exists (or `--force`) |
| `embed_train()` / `embed_all()` / `train_model()` | Rebuild train embeddings, train + test embeddings, and the two-tower (same recipe as notebook 05, seed 0) |
| `main()` | `python -m src.prepare [--force]`: all four steps, ~2.5 min from scratch |

### `app/streamlit_app.py`

Streamlit page (see §2.8). Shows a friendly error if the MIND data or built artifacts are missing. Caches the `FeedEngine` with `st.cache_resource` and the step-4 simulation with `st.cache_data`. Theme in `.streamlit/config.toml`.

### `pipeline/ingest.py`: raw files → bronze Parquet

| Name | What it does |
|---|---|
| `LANDING`, `LAKE`, `BRONZE`, `LOAD_LOG` | `data/` (raw), `lake/`, `lake/bronze/`, `lake/_load_log.csv` |
| `read_tsv(path, columns)` | DuckDB expression that reads a headerless MIND TSV with explicit types and no quoting |
| `write_partition(con, query, target_dir)` | Writes a query to Parquet in a `.tmp` folder, then swaps it in (idempotent) |
| `ingest_news(con, source)` | One news snapshot per source folder, with `_file_row` (line number) and metadata; checks for missing or duplicate IDs |
| `behaviors_view(con, source)` | Typed view over a behaviors file with parsed time and `date` |
| `ingest_behaviors_day(con, source, day, …)` | One day → `bronze/behaviors/date=YYYY-MM-DD/`, after checks (rows, times, keys, impressions format) |
| `ingest(days=None)` / `main()` | All days and both news snapshots, or the given days: `python -m pipeline.ingest --all \| --day YYYY-MM-DD` |
| `log_load(...)`, `md5(path)` | Load log rows: rows, source checksum, duration, status, message |

### `warehouse/`: dbt project (DuckDB)

| Node | What it is |
|---|---|
| `seeds/bucket_mapping.csv` | The topic mapping with a reason per row |
| `stg_news`, `stg_behaviors` | Cleaned staging tables (composite key `source-impression_id`, `file_row`, date, hour) |
| `int_impressions` | View: one row per (page view, article shown, clicked) |
| `dim_articles`, `dim_article_sources` | Each article once with its bucket; which articles each folder lists, in file order |
| `fct_impressions`, `fct_page_views` | Articles shown with clicked/bucket/split; page views in raw form for `src/data.py` |
| `fct_hourly_trending` | Clicks in the 24 h before each hour (current hour excluded) and `ln(1 + x)` |
| `dim_user_history` | Per source and user: history length and the last 50 clicks |
| `tests/*.sql` | 7 singular tests (leakage, history, uniqueness, CTR range, bucket coverage, trending window) |
| `macros/generate_schema_name.sql` | Uses folder schemas (`staging`, `marts`) as-is |

Run: `cd warehouse && dbt build --profiles-dir .` (1 seed + 9 models + 27 tests, ~2 s).

### `pipeline/check_parity.py`

`python -m pipeline.check_parity`: 7 checks that the warehouse matches the raw-file Python code exactly (§2.11). It exits non-zero on any mismatch, which makes the Airflow task fail.

### `pipeline/ml.py`: MLOps tasks

| Name | What it does |
|---|---|
| `load_registry()` / `save_registry()` | `artifacts/models/registry.json`: champion, versions (path, training days, metrics), promotion decisions. Starts with `v1-paper` pinned |
| `load_model(version=None)` | Loads a registered model (default: the champion) |
| `embed_new_articles()` | MiniLM vectors for warehouse articles that have none yet (appends to `news_embeddings_all.npz`) |
| `score_day(model, day, dev)` | Recall@K / NDCG@10 on one day for the model (taste + 0.05 × trending, quotas) and for trending alone, from warehouse marts only |
| `evaluate_day(day, version=None)` | Scores and upserts a row in `results/model_monitoring.csv` (day, version, split, model vs trending) |
| `train_challenger(train_until, epochs=3, seed=0)` | Trains a new version on train-folder days before `train_until` and registers it; refuses the test day |
| `compare_and_promote(challenger, eval_day)` | Scores challenger and champion on `eval_day` (never the test day); promotes only on a +0.002 Recall@500 win without losing more than 0.002 NDCG@10; logs the decision |
| `main()` | `python -m pipeline.ml embed \| evaluate DAY \| retrain` |

### `airflow/` and `docker-compose.yml`

| File | What it is |
|---|---|
| `airflow/Dockerfile` | Airflow 3.3.2 (Python 3.12) + dbt in `/home/airflow/dbt_venv` + CPU PyTorch, pandas, DuckDB, sentence-transformers, Cosmos (versions pinned to the project's) |
| `docker-compose.yml` | `airflow standalone` on :8080, no login (local demo), repo mounted at `/opt/ownfeed`, absolute warehouse/lake paths, MiniLM cache in `artifacts/` |
| `airflow/dags/ownfeed_daily.py` | Daily DAG with catch-up (Nov 9–15): `ingest_day` → `dbt_warehouse` (Cosmos task group, tests after each model) → `parity_check` → `embed_new_articles` → `evaluate_day` → `refresh_demo_assets` |
| `airflow/dags/ownfeed_retrain.py` | Manual DAG with `train_until` / `eval_day` params: `embed_new_articles` → `train_challenger` → `evaluate_and_promote` |

---

## 4. Notebooks

| Notebook | Purpose | Main outputs |
|---|---|---|
| 01 · EDA | Categories, subcategories, click shares, CTR, per-user mixes, choosing the buckets | `bucket_click_share.png`, `user_bucket_mix.png` |
| 02 · Split | Time split; checks for history leakage, new users, new articles, mix drift | |
| 03 · Baseline | Trending finder, quotas, evaluation setup | `finder_baseline_metrics.csv` |
| 04 · Embeddings | MiniLM vectors, nearest-neighbor checks, same-bucket rate | `news_embeddings.npz` |
| 05 · Two-tower | Training, β blend, trending-aware experiment | `two_tower.pt`, `finder_metrics.csv` |
| 06 · Arranger | Strictness sweep, Chart 1, choosing 0.5, example page | `arranger_sweep.csv`, `chart1_relevance_vs_mix.png`, `choosing_strictness.png` |
| 07 · Balancer | Simulator checks, four feeds, Chart 2, mood-shift table | `balancer_results.csv`, `chart2_reading_gap.png` |
| 08 · Test | The single frozen run on Nov 15, validation vs test | `test_*.csv`, `*_test.png`, `test_vs_validation.csv` |

---

## 5. Metrics glossary

| Metric | Plain meaning |
|---|---|
| **CTR** | Click-through rate: clicks ÷ times shown |
| **Click share** | Share of *all* clicks that went to a bucket (the "natural feed mix") |
| **Recall@K** | Of the articles a user clicked, the share found in the top K. Recall@500 is the finder's main metric |
| **NDCG@10** | Are the clicked articles near the very top? 1 = perfect |
| **Mix error** | Distance between a page's topic shares and the slider target: ½ × Σ\|share − target\|, 0 = perfect |
| **Articles off** | The same in articles: how many of the 20 are in the wrong bucket |
| **Pages exact** | Share of pages with 0 articles off |
| **Relevance kept** | Total relevance of the chosen 20 ÷ the plain top 20 (relevance = finder score rescaled 0..1 per page) |
| **Reading gap** | Mix error between what a reader has *read* (expected clicks) and the target |
| **AUC** | Chance a random clicked article gets a higher predicted click probability than a random skipped one (0.5 = guessing) |

---

## 6. All key numbers

| | Validation (Nov 14) | Test (Nov 15) |
|---|---|---|
| Random Recall@500 | 0.081 | 0.093 |
| Trending Recall@500 / NDCG@10 | 0.777 / 0.048 | 0.838 / 0.152 |
| OwnFeed finder Recall@500 / NDCG@10 | 0.811 / 0.074 | 0.859 / 0.150 |
| Normal feed mix error (strictness 0) | 37–54 pts | 36–55 pts |
| Arranger at 0.5: pages exact / relevance kept | 94.5% / 90.3% | 95.8% / 90.7% |
| Reading gap after 30 pages: normal / quotas / PID / OwnFeed | 0.291 / 0.199 / 0.146 / 0.109 | 0.304 / 0.221 / 0.153 / 0.125 |
| Favorite topic's share of reading (target 20%) | 41 / 34 / 26 / 24% | 41 / 34 / 26 / 25% |
| Click rate: normal → OwnFeed | 3.78% → 3.03% | 3.68% → 2.89% |
| Simulator AUC (held-out users) | 0.672 | 0.702 |
| Top-10 articles' share of clicks | 21.0% | 31.6% |
| Warehouse: page views / articles shown / articles (all days) | 230,117 / 8,584,442 / 65,238 | |
| dbt build (1 seed + 9 models + 27 tests) / parity checks | ~2 s, all pass / 7 of 7 | |
| Daily monitoring Nov 9–15, OwnFeed vs trending Recall@500 | 0.81–0.89 vs 0.78–0.86 (OwnFeed higher every day) | |
| Page build time (one reader, CPU) | ~1 ms | ~1 ms |

---

## 7. Limitations and open questions

- **Simulated readers.** The balancer evidence comes from a click model fitted on real clicks, not real people reacting to a new feed.
- **One-day pools.** No publish times; "live" = shown that day.
- **Balancer tested on the Balanced target** only; the arranger was tested on 4 settings.
- **Choices tuned on one validation day.** Small validation differences may be noise (daily mix wobble).
- **Counter-intuitive balancer pages** (showing very little of a favorite topic); the bounded variant is future work.
- **The towers learn little beyond raw MiniLM.** Richer user modeling (attention over history, recent-click weighting) or a fine-tuned encoder may help.
- **No in-session bubble observed.** Long-horizon (multi-day) simulation would be needed to study real filter-bubble dynamics.
- **MIND is news.** It stands in for a social feed.

---

## 8. Beyond news: where this applies

OwnFeed's core idea isn't specific to news: **let people choose the *mix* of what they get, then build each page to match it and correct for what they actually consume.** Any product that ranks a stream of items from several kinds of content can use the same three stages.

### How the pieces map to other products

| OwnFeed part | General idea | Example elsewhere |
|---|---|---|
| Topic buckets | The categories a user can steer | Genres, creators, formats, moods, "friends vs. recommended" |
| Finder (two-tower + trending, quotas per bucket) | Retrieve good candidates **from every bucket** | Song, video or post retrieval with per-genre/per-format quotas |
| Arranger (strictness knob) | Fill a page/queue to match the chosen mix | Building a playlist, a Reels queue, a podcast queue |
| Balancer (head start + PID) | Correct for what the user actually **consumes** | Correct by minutes listened/watched, not just clicks |
| Click model + simulator | Test steering policies offline before real users | Simulated listeners/viewers fitted on real logs |

### Where it could be used

**News apps** (Google News, Apple News, MSN, newspapers' own apps)
- *Sliders:* News, Learn, Lifestyle, Entertainment, Sports, or finer: local vs. world, politics vs. science.
- *Fit:* this is exactly what OwnFeed built. The main product work is the **"breaking news exception"** (a major story should get through even with a low News slider) and a **bounded balancer**, so pages always *look* like the sliders.
- *Why a publisher would want it:* readers who feel in control are more likely to trust and return, even if a single session gets fewer clicks.

**Instagram, TikTok, YouTube, X, Threads** (social and short-video feeds)
- *Sliders:* **friends vs. recommended creators**, and topics (comedy, fitness, cooking, news), or "calm vs. high-energy".
- *What changes:* items are images and videos, so the item tower needs **image/video embeddings** (e.g. CLIP-style models) instead of MiniLM. "Read" becomes **watch time** and **completion rate**, not clicks.
- *Big challenge:* the engagement cost. Our ~20% click drop is exactly the tension these platforms face. OwnFeed's charts make that trade-off measurable instead of hidden.
- *Note:* Instagram's terms forbid scraping, so for a portfolio this stays a **design mockup**, not a working integration.

**Bluesky** (custom feeds)
- Bluesky lets anyone publish a **custom feed generator** through its open API, and users choose which feed to follow.
- *Fit:* the most realistic way to put OwnFeed in front of **real users** legally: classify posts into buckets, rank with the finder, arrange with the user's mix. This is the stretch goal from the original plan.

**Spotify, Apple Music, YouTube Music** (music)
- *Sliders:* genres (pop / indie / hip-hop / classical), **familiar vs. new discoveries**, mood (focus / energy / chill), decades.
- *What changes:* the "page" is a **queue or playlist**. "Read" becomes **listened past 30 seconds / skipped**, which gives a cleaner signal than clicks. Order also matters (transitions between songs).
- *Good use of the balancer:* "I asked for 30% new music but keep skipping it." That's the stated-vs-revealed question again. The strictness knob lets the user decide whether to push or back off.

**Spotify podcasts, Apple Podcasts, Audible** (podcasts and audiobooks)
- *Sliders:* news / learning / comedy / true crime / business, plus **episode length** (short vs. long).
- *What changes:* items are long (30–90 min), so a "session" spans days, not pages. The balancer would work over **listening hours per week** rather than per page. Text embeddings still work well (titles, descriptions, transcripts).
- *Natural fit:* "learning vs. entertainment" balance is something podcast listeners already say they want.

**Netflix, Prime Video, YouTube** (long video)
- *Sliders:* genres, **"finish what I started" vs. "something new"**, documentaries vs. fiction.
- *What changes:* rows of items instead of one list. The arranger would fill each row or the home screen to match the mix.

**Online learning** (Coursera, Duolingo, Khan Academy)
- *Sliders:* subjects or skills (e.g. 50% math, 30% reading, 20% science), or "review vs. new material".
- *Fit:* the balancer maps naturally onto *learning goals*. If a student keeps skipping science, the system can push a bit more or ask whether the goal should change.

**E-commerce** (Amazon, Etsy, grocery apps)
- *Sliders:* categories, **budget vs. premium**, **sustainable / local** share, "new brands vs. ones I know".
- *What changes:* the cost of following the sliders shows up as **revenue**, not clicks, so the trade-off chart becomes a business decision tool.

**Wellbeing and parental controls**
- Parents could set a child's video mix (e.g. 50% learning, 30% creative, 20% entertainment), and the balancer keeps *watched* content near that, not just *shown* content.

### What transfers directly, and what needs work

| Transfers as is | Needs changing per product |
|---|---|
| The three-stage design (finder → arranger → balancer) | Item embeddings: images, audio and video need their own encoders |
| Per-bucket candidate quotas | The "consumed" signal: watch time, listen-through and purchases instead of clicks |
| The strictness knob and the "smallest strictness that matches X% of pages" rule | Session length: a page for news, a queue for music, a week for podcasts |
| Head start + gentle PID | Ordering rules: song transitions, rows on a home screen |
| Offline evaluation: time split, locked test set, simulator fitted on real logs | Real A/B tests to measure long-term trust and retention, which simulation can't capture |

### Why this matters now

- **Regulation is moving this way.** The EU's Digital Services Act requires very large platforms to explain the main parameters of their recommender systems and to offer at least one option *not* based on profiling. User-controlled mixes are one concrete way to give people that kind of control.
- **The trade-off is now measurable.** OwnFeed shows the cost of user control in numbers: ~10% relevance and ~20% clicks in a single session. A company can weigh that against the long-term value of users trusting the feed, which a single session can't measure and an A/B test over weeks could.
