"""
Step 3 - Build a feature table for a shortlist of candidate ads + a user.

For each candidate ad we compute the columns in config.FEATURE_COLS:
  semantic_sim    : cosine similarity from Step 2 (passed in)
  category_match  : is the ad's category one the user bought before?  (1/0)
  price_fit       : is the ad's price <= the user's avg budget?       (1/0)
  user_cat_clicks : how many times has this user clicked this category?

The click counts come from an interactions frame that the caller supplies.
During training/eval we pass ONLY the training split so counts never leak
labels from the held-out data.
"""

import pandas as pd

from .config import USERS_CSV, INTERACTIONS_CSV, ADS_CSV, FEATURE_COLS


class FeatureBuilder:
    def __init__(self, users=None, ads=None, interactions=None):
        self.users = users if users is not None else pd.read_csv(USERS_CSV)
        self.ads = ads if ads is not None else pd.read_csv(ADS_CSV)
        interactions = (interactions if interactions is not None
                        else pd.read_csv(INTERACTIONS_CSV))

        self._user_prefs = {
            r.user_id: set(str(r.past_categories_bought).split(","))
            for r in self.users.itertuples()
        }
        self._user_budget = dict(zip(self.users.user_id, self.users.avg_budget))
        self._ad_cat = dict(zip(self.ads.ad_id, self.ads.category))
        self._ad_price = dict(zip(self.ads.ad_id, self.ads.price))

        # (user_id, category) -> number of clicks, from supplied interactions only
        clicked = interactions[interactions.clicked == 1].copy()
        clicked["category"] = clicked.ad_id.map(self._ad_cat)
        self._cat_clicks = (clicked.groupby(["user_id", "category"]).size()
                            .to_dict())

    def _row(self, user_id, ad_id, semantic_sim):
        cat = self._ad_cat.get(ad_id)
        prefs = self._user_prefs.get(user_id, set())
        budget = self._user_budget.get(user_id, 0)
        price = self._ad_price.get(ad_id, 0)
        return {
            "semantic_sim": float(semantic_sim),
            "category_match": int(cat in prefs),
            "price_fit": int(price <= budget),
            "user_cat_clicks": int(self._cat_clicks.get((user_id, cat), 0)),
        }

    def for_candidates(self, user_id: str, candidates: pd.DataFrame) -> pd.DataFrame:
        """`candidates` must have columns ad_id + semantic_sim (from retrieval)."""
        rows = [self._row(user_id, int(c.ad_id), c.semantic_sim)
                for c in candidates.itertuples()]
        feats = pd.DataFrame(rows, columns=FEATURE_COLS)
        # drop any columns the candidates frame already carries (e.g. semantic_sim)
        base = candidates.reset_index(drop=True).drop(
            columns=[c for c in FEATURE_COLS if c in candidates.columns])
        return pd.concat([base, feats], axis=1)

    def for_pairs(self, interactions: pd.DataFrame,
                  sim_lookup=None) -> pd.DataFrame:
        """Vectorised feature build for a whole interactions frame (training).

        sim_lookup(user_id, ad_id) -> semantic_sim, or None to fill 0.0.
        """
        recs = []
        for r in interactions.itertuples():
            sim = 0.0 if sim_lookup is None else sim_lookup(r.user_id, r.ad_id)
            row = self._row(r.user_id, int(r.ad_id), sim)
            row["user_id"] = r.user_id
            row["ad_id"] = int(r.ad_id)
            if hasattr(r, "clicked"):
                row["clicked"] = int(r.clicked)
            recs.append(row)
        return pd.DataFrame(recs)


if __name__ == "__main__":
    fb = FeatureBuilder()
    demo = pd.DataFrame({"ad_id": [1, 2, 3], "semantic_sim": [0.9, 0.8, 0.7]})
    print(fb.for_candidates("u1", demo))
