"""Shared paths and constants."""
import os

# Embedding backend for Step 2 retrieval.
# "tfidf" is the default on purpose: it needs no model download and makes the
# reported numbers reproducible on a clean `pip install -r requirements.txt`.
# Set EMBED_BACKEND=sentence-transformers (and install requirements-optional.txt)
# to embed with MiniLM instead -- richer vectors, but it changes the scores.
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "tfidf")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "model")

ADS_CSV = os.path.join(DATA_DIR, "ads.csv")
USERS_CSV = os.path.join(DATA_DIR, "users.csv")
INTERACTIONS_CSV = os.path.join(DATA_DIR, "interactions.csv")

RANKER_PATH = os.path.join(MODEL_DIR, "ranker.txt")

# The 10 categories in our fake world (kept here so the parser can snap to them).
CATEGORIES = [
    "running_shoes", "headphones", "coffee", "laptops", "backpacks",
    "smartwatches", "yoga_mats", "sunglasses", "cookware", "books",
]

# Feature columns, in the fixed order the model expects.
FEATURE_COLS = [
    "semantic_sim",       # Step 2 similarity score
    "category_match",     # ad category in user's past categories (1/0)
    "price_fit",          # ad price <= user's avg budget (1/0)
    "user_cat_clicks",    # times this user clicked this category before
]

TOP_K_RETRIEVE = 25       # Step 2 shortlist size
