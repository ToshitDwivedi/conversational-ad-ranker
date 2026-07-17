"""
Steps 5 + 6 - Prove the model beats a naive baseline.

Baseline : rank the retrieved candidates by semantic similarity ALONE
           (no model, no personalisation).
Model    : rank by the trained LightGBM click probability.

Step 5 (offline metrics): on held-out interactions, for each user we rank
        that user's candidate ads with both methods and score them against
        the real click labels -> NDCG@10 and Precision@10.

Step 6 (simulated A/B test): split users into group A (served baseline) and
        group B (served model). "Show" each user's top-K, count realised
        clicks, and run a two-proportion z-test on the two CTRs.

Run:  python evaluate.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from scipy import stats

from src.splits import load_all, split_interactions
from src.retrieval import AdRetriever, build_user_sim_lookup
from src.features import FeatureBuilder
from src.ranker import Ranker

K_METRIC = 10     # NDCG@ / Precision@
K_SERVE = 5       # how many ads we "show" in the A/B simulation
MIN_CANDIDATES = 6  # a user needs enough ads for ranking order to matter


def precision_at_k(labels, k):
    top = labels[:k]
    return float(np.sum(top)) / min(k, len(labels)) if len(labels) else 0.0


def ndcg_at_k(labels, scores, k):
    if len(labels) < 2 or np.sum(labels) == 0:
        return None  # undefined / uninformative for this user
    return float(ndcg_score([labels], [scores], k=k))


def build_per_user_frames(test_df, fb, sim_lookup, ranker):
    """For each qualifying user return candidate rows with baseline+model scores."""
    feats = fb.for_pairs(test_df, sim_lookup=sim_lookup)
    feats["model_score"] = ranker.score(feats)
    feats["baseline_score"] = feats["semantic_sim"]

    frames = {}
    for uid, grp in feats.groupby("user_id"):
        if len(grp) >= MIN_CANDIDATES:
            frames[uid] = grp.reset_index(drop=True)
    return frames


def offline_metrics(frames):
    rows = []
    for method, col in [("baseline", "baseline_score"), ("model", "model_score")]:
        prec, ndcg = [], []
        for grp in frames.values():
            ordered = grp.sort_values(col, ascending=False)
            labels = ordered["clicked"].to_numpy()
            prec.append(precision_at_k(labels, K_METRIC))
            n = ndcg_at_k(grp["clicked"].to_numpy(), grp[col].to_numpy(), K_METRIC)
            if n is not None:
                ndcg.append(n)
        rows.append({
            "method": method,
            f"NDCG@{K_METRIC}": np.mean(ndcg),
            f"Precision@{K_METRIC}": np.mean(prec),
            "users_scored": len(frames),
        })
    return pd.DataFrame(rows)


def two_prop_ztest(clicks_a, n_a, clicks_b, n_b):
    p_a, p_b = clicks_a / n_a, clicks_b / n_b
    p_pool = (clicks_a + clicks_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    # 95% CI on the difference (unpooled SE)
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    lo, hi = (p_b - p_a) - 1.96 * se_diff, (p_b - p_a) + 1.96 * se_diff
    return p_a, p_b, z, p_value, lo, hi


def simulate_ab(frames, seed=42):
    uids = list(frames.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(uids)
    half = len(uids) // 2
    group_a, group_b = uids[:half], uids[half:]  # A=baseline, B=model

    def served_clicks(uids_, col):
        clicks = impressions = 0
        for uid in uids_:
            grp = frames[uid].sort_values(col, ascending=False)
            shown = grp["clicked"].to_numpy()[:K_SERVE]
            clicks += int(shown.sum())
            impressions += len(shown)
        return clicks, impressions

    ca, na = served_clicks(group_a, "baseline_score")
    cb, nb = served_clicks(group_b, "model_score")
    p_a, p_b, z, p, lo, hi = two_prop_ztest(ca, na, cb, nb)
    lift = (p_b - p_a) / p_a * 100 if p_a else float("nan")
    return dict(group_a_users=len(group_a), group_b_users=len(group_b),
                ctr_a=p_a, ctr_b=p_b, abs_lift=p_b - p_a, rel_lift_pct=lift,
                z=z, p_value=p, ci=(lo, hi))


def main():
    ads, users, interactions = load_all()
    train_df, test_df = split_interactions(interactions)

    retriever = AdRetriever(ads)
    sim_lookup = build_user_sim_lookup(retriever, users)
    fb = FeatureBuilder(users=users, ads=ads, interactions=train_df)  # train counts
    ranker = Ranker()

    frames = build_per_user_frames(test_df, fb, sim_lookup, ranker)

    print("=" * 60)
    print("STEP 5 - Offline ranking metrics (held-out)")
    print("=" * 60)
    metrics = offline_metrics(frames)
    print(metrics.to_string(index=False))

    print("\n" + "=" * 60)
    print("STEP 6 - Simulated A/B test (two-proportion z-test)")
    print("=" * 60)
    ab = simulate_ab(frames)
    print(f"Group A (baseline): {ab['group_a_users']} users | "
          f"CTR = {ab['ctr_a']:.3f}")
    print(f"Group B (model)   : {ab['group_b_users']} users | "
          f"CTR = {ab['ctr_b']:.3f}")
    print(f"\nGroup B showed a {ab['rel_lift_pct']:+.1f}% relative CTR lift "
          f"over Group A")
    print(f"  absolute lift = {ab['abs_lift']*100:+.1f} pts | "
          f"z = {ab['z']:.2f} | p = {ab['p_value']:.3f}")
    print(f"  95% CI on absolute lift = "
          f"[{ab['ci'][0]*100:+.1f}%, {ab['ci'][1]*100:+.1f}%]")
    verdict = "STATISTICALLY SIGNIFICANT" if ab["p_value"] < 0.05 else "not significant"
    print(f"  -> {verdict} at alpha=0.05")

    return metrics, ab


if __name__ == "__main__":
    main()
