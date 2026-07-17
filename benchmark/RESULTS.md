# MovieLens-100K benchmark results

Public dataset: [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) (GroupLens) — the
standard academic recsys / learning-to-rank benchmark (100,000 ratings, 943 users, 1,682 movies).

Protocol: **leave-one-out + 99 sampled negatives**, the widely-cited evaluation from
He et al., *Neural Collaborative Filtering* (WWW 2017). For each user the most recent
interaction is held out, mixed with 99 random unseen items, and the model must rank the
held-out item highly. Metrics averaged over all 943 test users.

| method               | HR@10 | NDCG@10 |
|----------------------|-------|---------|
| popularity baseline  | 0.409 | 0.229   |
| **LightGBM LambdaRank** | **0.449** | **0.248** |

**LambdaRank improves NDCG@10 by +8.0% over the popularity baseline.**

Feature importances (gain) — personalization (`genre_affinity`) is the #2 signal,
i.e. the model adds real per-user value on top of raw popularity:

```
item_pop            89444.7
genre_affinity       6219.7
user_activity        3697.0
item_mean_rating     2992.9
user_mean_rating     1548.1
genre_overlap         981.2
```

Reproduce: `python -m benchmark.benchmark_movielens` (auto-downloads the dataset, ~2-3 min).

**Scope.** This validates the *ranking* claim — the project's own ranker
(`src.ranker.train_lambdarank`) beats a strong baseline on data nobody invented.
It does not exercise Steps 1-2 (query parsing, semantic retrieval): no public
dataset pairs natural-language shopping queries with priced ads, which is
precisely why `data/` is synthetic.
