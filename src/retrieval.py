"""
Step 2 - Retrieve candidate ads (narrow ~150 ads down to ~25).

Embed every ad description once, embed the user's query, and pull the
top-K most similar ads by cosine similarity.

Embeddings : chosen explicitly via config.EMBED_BACKEND -- "tfidf" (default:
             no download, reproducible) or "sentence-transformers" (MiniLM).
Index      : FAISS IndexFlatIP if faiss is installed, else a NumPy brute-force
             cosine search. Either way the public API is identical.
"""

import numpy as np
import pandas as pd

from .config import ADS_CSV, TOP_K_RETRIEVE, EMBED_BACKEND

MIN_SHORTLIST = 5   # if a filter tier leaves fewer than this, relax the filters


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class _Embedder:
    """Embeds text with the backend named in config.EMBED_BACKEND.

    The backend is chosen explicitly rather than auto-detected: results would
    otherwise change silently depending on whether sentence-transformers
    happens to be installed on the machine.
    """

    def __init__(self, backend: str | None = None):
        requested = backend or EMBED_BACKEND
        self._st_model = None
        self._tfidf = None

        if requested == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.backend = "sentence-transformers"
                return
            except Exception as e:
                print(f"[retrieval] sentence-transformers unavailable ({e}); "
                      f"falling back to TF-IDF.")

        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.backend = "tfidf"

    def fit_transform(self, texts):
        if self.backend == "sentence-transformers":
            vecs = self._st_model.encode(list(texts), show_progress_bar=False)
        else:
            vecs = self._tfidf.fit_transform(list(texts)).toarray()
        return _l2_normalize(np.asarray(vecs, dtype="float32"))

    def transform(self, texts):
        if self.backend == "sentence-transformers":
            vecs = self._st_model.encode(list(texts), show_progress_bar=False)
        else:
            vecs = self._tfidf.transform(list(texts)).toarray()
        return _l2_normalize(np.asarray(vecs, dtype="float32"))


class AdRetriever:
    """Build an index over ad descriptions and query it."""

    def __init__(self, ads: pd.DataFrame | None = None, backend: str | None = None):
        self.ads = ads if ads is not None else pd.read_csv(ADS_CSV)
        self.embedder = _Embedder(backend)
        self.ad_vecs = self.embedder.fit_transform(self.ads.description.tolist())
        self.ad_ids = self.ads.ad_id.to_numpy()
        self._index = self._build_index(self.ad_vecs)

    @property
    def backend(self) -> str:
        return self.embedder.backend

    @property
    def uses_faiss(self) -> bool:
        return self._faiss

    def _build_index(self, vecs):
        try:
            import faiss
            index = faiss.IndexFlatIP(vecs.shape[1])  # inner product on normed = cosine
            index.add(vecs)
            self._faiss = True
            return index
        except Exception:
            self._faiss = False
            return None  # no index -> NumPy brute force in _search

    def _search(self, qv, n):
        """Top-n ad indices + similarities, via FAISS when available."""
        n = min(n, len(self.ads))
        if self._faiss:
            sims, idx = self._index.search(qv.reshape(1, -1).astype("float32"), n)
            return idx[0], sims[0]
        sims_all = self.ad_vecs @ qv          # brute-force cosine fallback
        idx = np.argsort(-sims_all)[:n]
        return idx, sims_all[idx]

    def query(self, text: str, k: int = TOP_K_RETRIEVE,
              restrict_category=None, max_price=None):
        """Return DataFrame of top-k ads with a 'semantic_sim' column.

        Retrieval scopes to the requested intent before personalisation, with a
        graded back-off so an over-tight query still returns something sensible:
            1. category + budget   2. category only   3. pure semantic
        The first tier yielding >= MIN_SHORTLIST ads wins. The trained ranker
        then personalises the order *within* this shortlist.
        """
        qv = self.embedder.transform([text])[0]
        # Over-fetch when filtering so the filters still have candidates to cut.
        n_fetch = len(self.ads) if (restrict_category or max_price) else k
        idx_all, sims_all = self._search(qv, n_fetch)

        cats = self.ads.category.to_numpy()
        prices = self.ads.price.to_numpy()

        tiers = [(restrict_category, max_price), (restrict_category, None), (None, None)]
        for cat, price_cap in tiers:
            keep = [j for j, i in enumerate(idx_all)
                    if (cat is None or cats[i] == cat)
                    and (price_cap is None or prices[i] <= price_cap)]
            if len(keep) >= min(MIN_SHORTLIST, k) or (cat is None and price_cap is None):
                break

        keep = keep[:k]
        out = self.ads.iloc[idx_all[keep]].copy()
        out["semantic_sim"] = sims_all[keep]
        return out.reset_index(drop=True)


def build_user_sim_lookup(retriever: "AdRetriever", users: pd.DataFrame):
    """Precompute semantic_sim(user, ad) for every ad, per user.

    Each user gets a synthesised query from their first past category +
    budget (the same shape build_query_text produces at inference), so the
    training-time similarity feature matches serving-time behaviour.
    """
    ad_index = {int(a): i for i, a in enumerate(retriever.ad_ids)}
    sims_by_user = {}
    for u in users.itertuples():
        cat = str(u.past_categories_bought).split(",")[0].replace("_", " ")
        qtext = f"{cat} under Rs {u.avg_budget}"
        qv = retriever.embedder.transform([qtext])[0]
        sims_by_user[u.user_id] = retriever.ad_vecs @ qv  # cosine to every ad

    def lookup(user_id, ad_id):
        row = sims_by_user.get(user_id)
        if row is None:
            return 0.0
        return float(row[ad_index.get(int(ad_id), 0)])

    return lookup


def build_query_text(parsed: dict, raw_text: str = "") -> str:
    """Compose the string we actually embed for retrieval."""
    parts = []
    if parsed.get("category"):
        parts.append(parsed["category"].replace("_", " "))
    if parsed.get("budget"):
        parts.append(f"under Rs {parsed['budget']}")
    composed = " ".join(parts).strip()
    return composed or raw_text


if __name__ == "__main__":
    r = AdRetriever()
    print(f"Embedding backend: {r.backend} | FAISS index: {r.uses_faiss}")
    hits = r.query("running shoes under Rs 5000", k=5)
    print(hits[["ad_id", "brand", "category", "price", "semantic_sim"]])
