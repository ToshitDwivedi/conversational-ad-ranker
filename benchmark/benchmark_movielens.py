"""
Benchmark the Learning-to-Rank approach on a RECOGNISED public dataset:
MovieLens-100K (GroupLens) -- the standard academic recsys/ranking benchmark.

This trains the project's own ranker (`src.ranker.train_lambdarank`) on real
data and evaluates it with the widely-cited leave-one-out + 99-negative-sampling
protocol from He et al., "Neural Collaborative Filtering" (WWW 2017).
Metrics: HR@10 and NDCG@10.

Baseline : rank candidates by item popularity (a strong classic baseline).
Model    : LightGBM LambdaRank over popularity + rating + genre-affinity
           (personalisation) features.

SCOPE -- what this does and does not prove:
  * It validates the RANKING claim: a LightGBM LTR model over behavioural +
    content features beats a strong baseline on data nobody invented.
  * It does NOT exercise Steps 1-2 (query parsing, semantic retrieval): no
    public dataset pairs natural-language shopping queries with priced ads,
    which is exactly why data/ is synthetic. The features here are MovieLens's
    own (popularity, ratings, genres) rather than the ad features.

Run:  python -m benchmark.benchmark_movielens
      (auto-downloads ml-100k the first time)
"""

import os
import io
import zipfile
import urllib.request

import numpy as np
import pandas as pd

from src.ranker import train_lambdarank

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "ml-100k")
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
RNG = np.random.default_rng(42)

N_TRAIN_NEG = 4       # negatives per positive when training the ranker
N_EVAL_NEG = 99       # negatives per test positive (NCF protocol)
K = 10                # HR@K / NDCG@K
GENRE_COLS = [f"g{i}" for i in range(19)]
FEATURES = ["item_pop", "item_mean_rating", "user_mean_rating",
            "user_activity", "genre_affinity", "genre_overlap"]


# ----------------------------------------------------------------------------- data
def ensure_data():
    if os.path.exists(os.path.join(DATA, "u.data")):
        return
    print("Downloading MovieLens-100K ...", flush=True)
    raw = urllib.request.urlopen(URL, timeout=60).read()
    zipfile.ZipFile(io.BytesIO(raw)).extractall(HERE)


def load():
    ensure_data()
    ratings = pd.read_csv(os.path.join(DATA, "u.data"), sep="\t",
                          names=["user", "item", "rating", "ts"])
    items = pd.read_csv(os.path.join(DATA, "u.item"), sep="|", encoding="latin-1",
                        header=None,
                        names=["item", "title", "date", "x", "url"] + GENRE_COLS,
                        usecols=["item"] + GENRE_COLS)
    return ratings, items


def loo_split(ratings):
    """Each user's most recent interaction -> test; the rest -> train."""
    ratings = ratings.sort_values(["user", "ts"])
    test = ratings.groupby("user").tail(1)
    train = ratings.drop(test.index)
    ok = train.user.value_counts()
    keep = set(ok[ok >= 5].index)                 # users with enough history
    return (train[train.user.isin(keep)].reset_index(drop=True),
            test[test.user.isin(keep)].reset_index(drop=True))


# ------------------------------------------------------------------- feature building
class FeatureStore:
    """All aggregates computed on the TRAIN split only (no leakage).

    Everything is stored in dense arrays indexed by id so features for
    millions of (user, item) pairs can be built with pure NumPy.
    """

    def __init__(self, train, items):
        self.max_item = int(items.item.max())
        self.max_user = int(train.user.max())
        gmean = train.rating.mean()

        # item genre matrix  [item_id x 19]
        self.gmat = np.zeros((self.max_item + 1, 19))
        self.gmat[items.item.to_numpy()] = items[GENRE_COLS].to_numpy(float)

        # item aggregates
        self.item_pop = np.zeros(self.max_item + 1)
        pop = train.groupby("item").size()
        self.item_pop[pop.index.to_numpy()] = pop.to_numpy()

        self.item_mean = np.full(self.max_item + 1, gmean)
        im = train.groupby("item").rating.mean()
        self.item_mean[im.index.to_numpy()] = im.to_numpy()

        # user aggregates
        self.user_mean = np.full(self.max_user + 1, gmean)
        um = train.groupby("user").rating.mean()
        self.user_mean[um.index.to_numpy()] = um.to_numpy()

        self.user_act = np.zeros(self.max_user + 1)
        ua = train.groupby("user").size()
        self.user_act[ua.index.to_numpy()] = ua.to_numpy()

        # user genre-preference matrix [user_id x 19] from liked (>=4) items
        self.user_gpref = np.zeros((self.max_user + 1, 19))
        liked = train[train.rating >= 4]
        for uid, grp in liked.groupby("user"):
            v = self.gmat[grp.item.to_numpy()].sum(axis=0)
            n = np.linalg.norm(v)
            self.user_gpref[uid] = v / n if n else v

        self.all_items = items.item.to_numpy()
        self.user_seen = train.groupby("user").item.apply(set).to_dict()

    def batch(self, users, items):
        """Vectorised feature matrix for aligned arrays of users & items."""
        users = np.asarray(users)
        items = np.asarray(items)
        gpref = self.user_gpref[users]          # [N x 19]
        gvec = self.gmat[items]                 # [N x 19]
        df = pd.DataFrame({
            "item_pop": np.log1p(self.item_pop[items]),
            "item_mean_rating": self.item_mean[items],
            "user_mean_rating": self.user_mean[users],
            "user_activity": np.log1p(self.user_act[users]),
            "genre_affinity": np.einsum("ij,ij->i", gpref, gvec),
            "genre_overlap": ((gpref > 0) & (gvec > 0)).sum(axis=1).astype(float),
        }, columns=FEATURES)
        return df

    def sample_negatives(self, user, n, seen_extra=frozenset()):
        seen = self.user_seen.get(user, set()) | seen_extra
        # never ask for more negatives than the catalogue can supply
        n = min(n, len(self.all_items) - len(seen))
        out = set()
        while len(out) < n:
            picks = RNG.choice(self.all_items, size=(n - len(out)) * 2 + 8)
            for c in picks:
                c = int(c)
                if c not in seen and c not in out:
                    out.add(c)
                    if len(out) == n:
                        break
        return list(out)


# --------------------------------------------------------------------- training LTR
def build_train_matrix(train, fs):
    """Per user: rated items (graded relevance) + sampled negatives (relevance 0)."""
    u_all, i_all, y_all, groups = [], [], [], []
    for uid, grp in train.groupby("user"):
        items = grp.item.tolist()
        rels = grp.rating.tolist()
        negs = fs.sample_negatives(uid, len(items) * N_TRAIN_NEG)
        block_items = items + negs
        block_rel = rels + [0] * len(negs)
        u_all.extend([uid] * len(block_items))
        i_all.extend(block_items)
        y_all.extend(block_rel)
        groups.append(len(block_items))
    X = fs.batch(u_all, i_all)
    return X, np.array(y_all), np.array(groups)


# --------------------------------------------------------------------- evaluation
def evaluate(test, fs, ranker):
    """Leave-one-out + 99 sampled negatives, scored in one vectorised batch."""
    users, items, is_pos = [], [], []
    for r in test.itertuples():
        negs = fs.sample_negatives(r.user, N_EVAL_NEG, seen_extra={r.item})
        cands = [r.item] + negs
        users.extend([r.user] * len(cands))
        items.extend(cands)
        is_pos.extend([1] + [0] * len(negs))

    feats = fs.batch(users, items)
    model_scores = ranker.predict(feats)
    pop_scores = feats["item_pop"].to_numpy()
    is_pos = np.array(is_pos)

    block = 1 + N_EVAL_NEG
    n = len(test)
    disc = 1.0 / np.log2(np.arange(2, K + 2))

    def metrics(scores):
        s = scores.reshape(n, block)
        p = is_pos.reshape(n, block)
        order = np.argsort(-s, axis=1)
        ranked_pos = np.take_along_axis(p, order, axis=1)[:, :K]  # [n x K]
        hr = ranked_pos.max(axis=1).mean()                       # hit if pos in top-K
        ndcg = (ranked_pos * disc).sum(axis=1).mean()            # ideal DCG = 1
        return hr, ndcg

    hr_b, ndcg_b = metrics(pop_scores)
    hr_m, ndcg_m = metrics(model_scores)
    return pd.DataFrame([
        {"method": "popularity baseline", f"HR@{K}": hr_b, f"NDCG@{K}": ndcg_b},
        {"method": "LightGBM LambdaRank", f"HR@{K}": hr_m, f"NDCG@{K}": ndcg_m},
    ]), n


def main():
    ratings, items = load()
    print(f"MovieLens-100K: {len(ratings)} ratings, "
          f"{ratings.user.nunique()} users, {ratings.item.nunique()} items", flush=True)

    train, test = loo_split(ratings)
    fs = FeatureStore(train, items)
    print(f"Train interactions: {len(train)} | "
          f"Test users (leave-one-out): {len(test)}", flush=True)

    X, y, groups = build_train_matrix(train, fs)
    print(f"Training LightGBM LambdaRank (src.ranker.train_lambdarank) on "
          f"{len(X)} rows / {len(groups)} query groups ...", flush=True)
    ranker = train_lambdarank(X, y, groups)

    print("Evaluating (leave-one-out + 99 sampled negatives, NCF protocol) ...",
          flush=True)
    table, n = evaluate(test, fs, ranker)
    print("\n" + "=" * 60)
    print(f"MovieLens-100K benchmark  (n = {n} test users)")
    print("=" * 60)
    print(table.to_string(index=False))

    b, m = table.iloc[0], table.iloc[1]
    lift = (m[f"NDCG@{K}"] - b[f"NDCG@{K}"]) / b[f"NDCG@{K}"] * 100
    print(f"\nLambdaRank improves NDCG@{K} by {lift:+.1f}% over the popularity baseline.")
    print("Feature importances (gain):")
    for f, g in sorted(zip(FEATURES, ranker.booster_.feature_importance('gain')),
                       key=lambda t: -t[1]):
        print(f"  {f:18} {g:10.1f}")
    return table


if __name__ == "__main__":
    main()
