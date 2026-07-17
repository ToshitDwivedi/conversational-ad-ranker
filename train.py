"""
Train the LightGBM ranker (Steps 3 + 4, offline).

  1. Load ads/users/interactions and split off a held-out test set.
  2. Build the semantic-similarity feature via the retriever.
  3. Build the full feature table on the TRAIN split (counts from train only).
  4. Fit LightGBM on features -> clicked and save model/ranker.txt.

Run:  python train.py
"""

from sklearn.metrics import roc_auc_score

from src.splits import load_all, split_interactions
from src.retrieval import AdRetriever, build_user_sim_lookup
from src.features import FeatureBuilder
from src.ranker import train_ranker, save_booster
from src.config import FEATURE_COLS, RANKER_PATH


def main():
    ads, users, interactions = load_all()
    train_df, test_df = split_interactions(interactions)
    print(f"Train interactions: {len(train_df)} | Test (held out): {len(test_df)}")

    print("Building retriever + per-user similarity feature ...")
    retriever = AdRetriever(ads)
    print(f"  embedding backend = {retriever.backend}, faiss = {retriever.uses_faiss}")
    sim_lookup = build_user_sim_lookup(retriever, users)

    # Counts come from the TRAIN split only -> no leakage into the feature.
    fb = FeatureBuilder(users=users, ads=ads, interactions=train_df)

    train_feats = fb.for_pairs(train_df, sim_lookup=sim_lookup)
    test_feats = fb.for_pairs(test_df, sim_lookup=sim_lookup)

    print("Training LightGBM ...")
    booster = train_ranker(train_feats, train_feats["clicked"])

    auc = roc_auc_score(test_feats["clicked"], booster.predict(test_feats[FEATURE_COLS]))
    print(f"Held-out AUC: {auc:.3f}")

    save_booster(booster, RANKER_PATH)
    print(f"Saved model -> {RANKER_PATH}")
    print("Feature gain importances:")
    for f, g in sorted(zip(FEATURE_COLS, booster.feature_importance('gain')),
                       key=lambda t: -t[1]):
        print(f"  {f:18} {g:10.1f}")


if __name__ == "__main__":
    main()
