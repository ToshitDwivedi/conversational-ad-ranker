"""
Step 4 - The trained LightGBM ranking model.

We frame ranking as pointwise click prediction: LightGBM learns
P(click | features), and we sort candidates by that score. This is the
core deliverable -- given a feature table it returns the ads best-to-worst.
"""

import os
import lightgbm as lgb

from .config import FEATURE_COLS, RANKER_PATH


def train_ranker(X, y, params=None):
    """Fit a LightGBM classifier on features -> clicked. Returns a Booster."""
    params = params or dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        verbose=-1,
    )
    dtrain = lgb.Dataset(X[FEATURE_COLS], label=y)
    booster = lgb.train(params, dtrain, num_boost_round=200)
    return booster


def train_lambdarank(X, y, groups, max_relevance=5, params=None):
    """Fit a LightGBM LambdaRank ranker over per-query groups.

    X       : feature frame (columns are used as-is)
    y       : graded relevance per row
    groups  : number of rows belonging to each query, in order

    This is the pairwise/listwise counterpart to `train_ranker` above, and is
    what the MovieLens benchmark trains -- so the benchmark validates this
    repo's ranker rather than a parallel re-implementation of it.
    """
    defaults = dict(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        label_gain=list(range(max_relevance + 1)),
        random_state=42,
        verbose=-1,
    )
    defaults.update(params or {})
    ranker = lgb.LGBMRanker(**defaults)
    ranker.fit(X, y, group=groups)
    return ranker


class Ranker:
    """Loads a trained booster and scores feature tables."""

    def __init__(self, path=RANKER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No model at {path}. Run `python train.py` first.")
        self.booster = lgb.Booster(model_file=path)

    def score(self, feats):
        """Return predicted click probability for each row of `feats`."""
        return self.booster.predict(feats[FEATURE_COLS])

    def rank(self, feats):
        """Return `feats` sorted best-to-worst with a 'score' column added."""
        out = feats.copy()
        out["score"] = self.score(out)
        return out.sort_values("score", ascending=False).reset_index(drop=True)


def save_booster(booster, path=RANKER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    booster.save_model(path)
