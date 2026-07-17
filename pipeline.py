"""
The whole thing, end to end (Steps 1 -> 4).

    text query + user_id  ->  ranked list of best-matching ads

    python pipeline.py --user u1 --text "I'm looking for running shoes under Rs 5000"

Step 1  parse the sentence            (Claude API, or regex fallback)
Step 2  retrieve ~25 candidate ads    (embeddings + FAISS)
Step 3  build the feature table       (personalised to the user)
Step 4  rank with the LightGBM model  (sort by predicted click prob)
"""

import argparse
import pandas as pd

from src.splits import load_all
from src.query_parser import parse_query
from src.retrieval import AdRetriever, build_query_text
from src.features import FeatureBuilder
from src.ranker import Ranker
from src.config import TOP_K_RETRIEVE

pd.set_option("display.width", 120)


class AdPipeline:
    """Reusable object so the notebook / eval can call it repeatedly."""

    def __init__(self):
        ads, users, interactions = load_all()
        self.retriever = AdRetriever(ads)
        self.fb = FeatureBuilder(users=users, ads=ads, interactions=interactions)
        self.ranker = Ranker()

    def run(self, text, user_id, k_retrieve=TOP_K_RETRIEVE, use_llm=None, verbose=True):
        parsed = parse_query(text, use_llm=use_llm)
        qtext = build_query_text(parsed, raw_text=text)
        candidates = self.retriever.query(
            qtext, k=k_retrieve, restrict_category=parsed.get("category"),
            max_price=parsed.get("budget"))
        feats = self.fb.for_candidates(user_id, candidates)
        ranked = self.ranker.rank(feats)

        if verbose:
            print(f"\nStep 1  query      : {text!r}")
            print(f"        parsed     : {parsed}")
            print(f"        embed text : {qtext!r}")
            print(f"Step 2  retrieved  : {len(candidates)} candidate ads")
            print(f"Step 3-4 ranked for user {user_id}:\n")
            cols = ["ad_id", "brand", "category", "price", "semantic_sim",
                    "category_match", "price_fit", "user_cat_clicks", "score"]
            print(ranked[cols].head(10).to_string(index=False))
        return ranked


def main():
    ap = argparse.ArgumentParser(description="Alexa-style ad ranker")
    ap.add_argument("--user", default="u1", help="user_id from users.csv")
    ap.add_argument("--text", default="I'm looking for running shoes under Rs 5000")
    ap.add_argument("--k", type=int, default=TOP_K_RETRIEVE)
    ap.add_argument("--no-llm", action="store_true",
                    help="force the regex parser even if ANTHROPIC_API_KEY is set")
    args = ap.parse_args()

    pipe = AdPipeline()
    pipe.run(args.text, args.user, k_retrieve=args.k,
             use_llm=(False if args.no_llm else None))


if __name__ == "__main__":
    main()
