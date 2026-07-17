# Personalized Conversational Advertisement Ranking & Recommendation System

An end-to-end **retrieval → learning-to-rank → evaluation** pipeline for
conversational ads. A user speaks a request — *"I'm looking for running shoes
under ₹5000"* — and the system returns a **ranked list of ads personalized to
that user's history**, using a **LightGBM CTR-prediction / Learning-to-Rank
model**. It ships with a full offline evaluation (NDCG@10, Precision@10, a
simulated A/B test) **and validation on the recognized MovieLens-100K
benchmark**, so the ranking approach is proven on real, public data — not just
synthetic data.

Runs fully offline with graceful fallbacks (no API key or GPU required).

---

## Architecture

```
             Natural-language query ("running shoes under ₹5000")  +  user_id
                                        │
                    ┌───────────────────▼────────────────────┐
             Step 1 │  Intent parsing  → {category, budget}   │  Claude API (regex fallback)
                    └───────────────────┬────────────────────┘
                                        │
                    ┌───────────────────▼────────────────────┐
             Step 2 │  Candidate retrieval  (~150 → ~25 ads)  │  embeddings + FAISS
                    └───────────────────┬────────────────────┘
                                        │
                    ┌───────────────────▼────────────────────┐
             Step 3 │  Feature engineering  (per candidate)   │  similarity · category-match
                    │  semantic_sim, category_match,          │  · price-fit · user history
                    │  price_fit, user_cat_clicks             │
                    └───────────────────┬────────────────────┘
                                        │
                    ┌───────────────────▼────────────────────┐
             Step 4 │  Learning-to-Rank  (CTR prediction)     │  LightGBM
                    └───────────────────┬────────────────────┘
                                        │
                                 Personalized ranked ads
                                        │
                    ┌───────────────────▼────────────────────┐
           Step 5-6 │  Evaluation: NDCG@10 / Precision@10     │  offline metrics
                    │  + simulated A/B test (z-test on CTR)   │  + public benchmark (MovieLens)
                    └─────────────────────────────────────────┘
```

| Step | What happens | Code |
|------|--------------|------|
| 0 | Build a fake world: 150 ads, 80 users, ~4.2k click events with a **baked-in behavioural signal** | [data/generate_data.py](data/generate_data.py) |
| 1 | Parse the sentence → `{category, budget}` (Claude API, regex fallback) | [src/query_parser.py](src/query_parser.py) |
| 2 | Retrieve ~25 candidate ads (embeddings + FAISS), scoped to the requested intent | [src/retrieval.py](src/retrieval.py) |
| 3 | Build per-candidate features: `semantic_sim, category_match, price_fit, user_cat_clicks` | [src/features.py](src/features.py) |
| 4 | Rank by predicted click probability with a trained **LightGBM** model | [src/ranker.py](src/ranker.py) |
| 5 | Offline eval: **NDCG@10 / Precision@10**, model vs baseline | [evaluate.py](evaluate.py) |
| 6 | Simulated **A/B test** with a two-proportion z-test | [evaluate.py](evaluate.py) |
| ★ | **Recognized-benchmark validation on MovieLens-100K** (LambdaRank vs popularity) | [benchmark/benchmark_movielens.py](benchmark/benchmark_movielens.py) |

---

## Results

### 1 · Recognized public benchmark — MovieLens-100K

Standard recsys/LTR benchmark (100k ratings, 943 users, 1,682 movies), evaluated
with the leave-one-out + 99-negative protocol from *Neural Collaborative
Filtering* (He et al., WWW 2017). Full details in [benchmark/RESULTS.md](benchmark/RESULTS.md).

| method | HR@10 | NDCG@10 |
|---|---|---|
| popularity baseline | 0.409 | 0.229 |
| **LightGBM LambdaRank** | **0.449** | **0.248** |

→ **+8.0% NDCG@10** over the popularity baseline; `genre_affinity`
(personalization) is the model's #2 feature after popularity.

This trains the project's own ranker (`src.ranker.train_lambdarank`) — not a
separate re-implementation. It validates the **ranking** claim on data nobody
invented; it does not exercise Steps 1-2, because no public dataset pairs
natural-language shopping queries with priced ads. That gap is exactly why
`data/` is synthetic: it's the only way to demo the conversational half.

### 2 · Conversational ad pipeline — controlled synthetic data

| method | NDCG@10 | Precision@10 |
|---|---|---|
| baseline (similarity only) | 0.828 | 0.747 |
| **LightGBM ranker** | **0.925** | **0.804** |

**Simulated A/B test:** Group B (model) showed a **+19.1% relative CTR lift**
over Group A (baseline) — 0.928 vs 0.779, **z = 4.16, p < 0.001**, 95% CI on the
absolute lift **[+8.0%, +21.7%]** — statistically significant at α = 0.05.

*(The synthetic dataset has a deliberately strong, learnable signal so the CTR
lift is large and the mechanics are easy to inspect; the MovieLens numbers above
are the honest, real-data reference point.)*

---

## Quick start

```bash
pip install -r requirements.txt            # core only — reproduces every number below

python run_all.py                          # data → train → evaluate → live demo → benchmark
python -m benchmark.benchmark_movielens    # recognized-benchmark validation (~2-3 min)
```

Optional extras (not needed to reproduce results): `pip install -r requirements-optional.txt`
adds Claude-based query parsing (`ANTHROPIC_API_KEY`) and MiniLM embeddings
(`EMBED_BACKEND=sentence-transformers`).

Individual pieces:

```bash
python data/generate_data.py                                     # Step 0
python train.py                                                  # Steps 3-4 → model/ranker.txt
python evaluate.py                                               # Steps 5-6
python pipeline.py --user u11 --text "running shoes under 5000"  # Steps 1-4, live
```

A step-by-step walkthrough with output at each stage: [notebook.ipynb](notebook.ipynb).

---

## Résumé bullet

> **Personalized Conversational Advertisement Ranking & Recommendation System** —
> Built an end-to-end recommendation pipeline combining semantic retrieval (FAISS),
> feature engineering, and a **LightGBM Learning-to-Rank / CTR-prediction model** to
> personalize ad recommendations from natural-language queries. Designed offline
> evaluation with **NDCG@10 / Precision@10** and a simulated **A/B testing framework**
> (two-proportion z-test); **validated on the MovieLens-100K benchmark**, improving
> **NDCG@10 by 8%** over a popularity baseline.

---

## Layout

```
data/       generate_data.py + the 3 generated CSVs
src/        query_parser, retrieval, features, ranker, config, splits
train.py        trains + saves the LightGBM model
evaluate.py     Steps 5-6 (offline metrics + simulated A/B test)
pipeline.py     end-to-end text + user_id → ranked ads (CLI)
run_all.py      runs every step in order, end to end
benchmark/      benchmark_movielens.py + RESULTS.md (recognized-dataset validation)
notebook.ipynb  walkthrough of Steps 1-6
```

## Design choices
- **Learning-to-Rank / CTR prediction:** LightGBM ranks candidates by predicted
  click probability; the MovieLens benchmark uses LightGBM **LambdaRank** with
  per-user query groups.
- **Reproducible by default, upgradable by choice:** retrieval embeds with
  TF-IDF and searches a **FAISS `IndexFlatIP`**; the embedding backend is set
  explicitly (`EMBED_BACKEND`), not auto-detected, so the numbers above don't
  silently change based on what happens to be installed. Opt into MiniLM with
  `pip install -r requirements-optional.txt && EMBED_BACKEND=sentence-transformers`.
  Step 1 uses Claude when `ANTHROPIC_API_KEY` is set and a deterministic regex
  parser otherwise; retrieval falls back to NumPy cosine if FAISS is absent.
- **Graded retrieval back-off:** filters try `category + budget` → `category`
  → pure semantic, so an over-tight query (*"headphones under ₹8000"* when few
  qualify) still returns on-intent ads instead of a random mixed bag.
- **No leakage:** history-count features and every model are fit on the training
  split only; all metrics (NDCG/Precision/CTR/HR) are measured on held-out data.
- **Intent-scoped retrieval, personalized ranking:** retrieval narrows to the
  requested category/budget first, then the model personalizes *within* that
  shortlist — mirroring real ad-retrieval systems.

## Limitations (and what I'd do next)

Stated plainly, because these are the first things worth asking about:

- **The synthetic CTR (~0.71) is unrealistically high by design.** The generator
  bakes in a strong, learnable signal so the mechanics are easy to inspect; real
  ad CTRs are low single digits. The MovieLens numbers are the honest reference.
- **The simulated A/B test splits users into disjoint groups**, so part of the
  CTR gap can reflect group composition rather than ranker quality. The offline
  NDCG@10/Precision@10 are the cleaner signal — they rank *the same* candidate
  set per user, isolating the ranker. A paired within-user design would remove
  the confound.
- **It's a replay-style simulation, not a true counterfactual.** Clicks are only
  observed for ads the logging policy already showed, so unshown ads count as
  non-clicks. Proper off-policy evaluation (IPS / doubly-robust) is the next step.
- **Scale:** 150 ads / 80 users, and MovieLens-100K. Nothing here is tested at
  production cardinality — FAISS is exact (`IndexFlatIP`), not an approximate
  index, because at 150 ads that's the correct choice.
- **Next:** MovieLens-1M, a LambdaRank objective for the ads model too (it's
  pointwise today), and IPS-weighted evaluation.
