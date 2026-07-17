"""
Step 0 - Build the fake "world".

Generates three CSVs under data/:
  - ads.csv          (~150 ads across 10 categories)
  - users.csv        (80 fake users with past behaviour + budget)
  - interactions.csv (~4500 user x ad click events)

The interactions are NOT random. We bake in a deliberate signal:
a user clicks far more often on ads that (a) match a category they have
bought before and (b) are within their average budget. That baked-in
signal is exactly what the LightGBM ranker later learns to detect -- and
what the naive similarity-only baseline cannot see.

Run:  python data/generate_data.py
"""

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
HERE = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Catalogue definition: category -> (brands, price range in INR, descriptors)
# -----------------------------------------------------------------------------
CATALOGUE = {
    "running_shoes": {
        "brands": ["Nike", "Adidas", "Puma", "Asics", "Reebok"],
        "price": (2500, 9000),
        "words": ["lightweight running shoes", "breathable trainers",
                  "cushioned road-running shoes", "marathon racing shoes"],
    },
    "headphones": {
        "brands": ["Sony", "Boat", "JBL", "Sennheiser", "Bose"],
        "price": (1200, 25000),
        "words": ["wireless noise-cancelling headphones", "over-ear headphones",
                  "bluetooth earbuds", "studio headphones"],
    },
    "coffee": {
        "brands": ["Blue Tokai", "Davidoff", "Nescafe", "Lavazza", "Sleepy Owl"],
        "price": (300, 2500),
        "words": ["single-origin arabica coffee beans", "instant coffee jar",
                  "cold brew coffee pack", "medium roast ground coffee"],
    },
    "laptops": {
        "brands": ["Dell", "HP", "Lenovo", "Asus", "Apple"],
        "price": (35000, 180000),
        "words": ["thin and light laptop", "gaming laptop with RTX graphics",
                  "business ultrabook", "student laptop for everyday use"],
    },
    "backpacks": {
        "brands": ["Wildcraft", "American Tourister", "Skybags", "Nike", "Fjallraven"],
        "price": (900, 6000),
        "words": ["waterproof laptop backpack", "trekking rucksack",
                  "casual college backpack", "anti-theft travel backpack"],
    },
    "smartwatches": {
        "brands": ["Apple", "Samsung", "Noise", "Fire-Boltt", "Garmin"],
        "price": (1500, 45000),
        "words": ["fitness tracking smartwatch", "AMOLED display smartwatch",
                  "GPS running watch", "bluetooth calling smartwatch"],
    },
    "yoga_mats": {
        "brands": ["Boldfit", "Amazon Basics", "Nivia", "Reebok", "Manduka"],
        "price": (500, 4500),
        "words": ["anti-slip yoga mat", "extra thick exercise mat",
                  "eco-friendly TPE yoga mat", "cushioned fitness mat"],
    },
    "sunglasses": {
        "brands": ["Ray-Ban", "Fastrack", "Oakley", "Vincent Chase", "Carrera"],
        "price": (700, 12000),
        "words": ["polarized aviator sunglasses", "UV-protection wayfarer sunglasses",
                  "sports sunglasses", "retro round sunglasses"],
    },
    "cookware": {
        "brands": ["Prestige", "Hawkins", "Pigeon", "Wonderchef", "Bergner"],
        "price": (600, 8000),
        "words": ["non-stick frying pan", "stainless steel cookware set",
                  "induction base pressure cooker", "cast iron skillet"],
    },
    "books": {
        "brands": ["Penguin", "HarperCollins", "Rupa", "Bloomsbury", "OReilly"],
        "price": (150, 1500),
        "words": ["bestselling fiction novel", "self-help paperback",
                  "hardcover science book", "programming reference book"],
    },
}
CATEGORIES = list(CATALOGUE.keys())


# -----------------------------------------------------------------------------
# ads.csv
# -----------------------------------------------------------------------------
def build_ads(n_ads=150):
    rows = []
    for ad_id in range(1, n_ads + 1):
        cat = RNG.choice(CATEGORIES)
        spec = CATALOGUE[cat]
        brand = RNG.choice(spec["brands"])
        lo, hi = spec["price"]
        price = int(RNG.integers(lo, hi))
        phrase = RNG.choice(spec["words"])
        desc = f"{brand} {phrase}, priced at Rs {price}."
        rows.append((ad_id, brand, cat, price, desc))
    return pd.DataFrame(rows, columns=["ad_id", "brand", "category", "price", "description"])


# -----------------------------------------------------------------------------
# users.csv
# -----------------------------------------------------------------------------
def build_users(n_users=80):
    rows = []
    for i in range(1, n_users + 1):
        k = RNG.integers(1, 4)  # 1-3 past categories
        cats = list(RNG.choice(CATEGORIES, size=k, replace=False))
        # budget loosely anchored to the mid-price of their favourite categories
        mids = [np.mean(CATALOGUE[c]["price"]) for c in cats]
        budget = int(np.clip(RNG.normal(np.mean(mids), np.mean(mids) * 0.3),
                             300, 180000))
        rows.append((f"u{i}", ",".join(cats), budget))
    return pd.DataFrame(rows, columns=["user_id", "past_categories_bought", "avg_budget"])


# -----------------------------------------------------------------------------
# interactions.csv  -- the important one, with a baked-in click pattern
# -----------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_interactions(ads, users, n_per_user=(40, 65)):
    ads_by_cat = {c: ads[ads.category == c].ad_id.tolist() for c in CATEGORIES}
    # per (user, category) latent affinity -> makes "past click count" meaningful
    rows = []
    for _, u in users.iterrows():
        prefs = u.past_categories_bought.split(",")
        n = int(RNG.integers(*n_per_user))
        for _ in range(n):
            # exposure bias: 60% of impressions come from a preferred category
            if RNG.random() < 0.60:
                cat = RNG.choice(prefs)
            else:
                cat = RNG.choice(CATEGORIES)
            ad_id = int(RNG.choice(ads_by_cat[cat]))
            ad = ads.loc[ads.ad_id == ad_id].iloc[0]

            cat_match = 1.0 if cat in prefs else 0.0
            price_fit = 1.0 if ad.price <= u.avg_budget else 0.0

            z = (-1.1
                 + 2.2 * cat_match
                 + 1.2 * price_fit
                 + 0.6 * cat_match * price_fit          # interaction effect
                 + RNG.normal(0, 0.5))                  # noise
            clicked = int(RNG.random() < _sigmoid(z))
            rows.append((u.user_id, ad_id, clicked))
    df = pd.DataFrame(rows, columns=["user_id", "ad_id", "clicked"])
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def main():
    ads = build_ads()
    users = build_users()
    interactions = build_interactions(ads, users)

    ads.to_csv(os.path.join(HERE, "ads.csv"), index=False)
    users.to_csv(os.path.join(HERE, "users.csv"), index=False)
    interactions.to_csv(os.path.join(HERE, "interactions.csv"), index=False)

    print(f"ads.csv          : {len(ads)} rows, {ads.category.nunique()} categories")
    print(f"users.csv        : {len(users)} rows")
    print(f"interactions.csv : {len(interactions)} rows, "
          f"overall CTR = {interactions.clicked.mean():.3f}")


if __name__ == "__main__":
    main()
