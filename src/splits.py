"""Shared data loading + a deterministic train/test split of interactions."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ADS_CSV, USERS_CSV, INTERACTIONS_CSV


def load_all():
    ads = pd.read_csv(ADS_CSV)
    users = pd.read_csv(USERS_CSV)
    interactions = pd.read_csv(INTERACTIONS_CSV)
    return ads, users, interactions


def split_interactions(interactions, test_size=0.25, seed=42):
    """Same split everywhere so train.py and evaluate.py agree on held-out data."""
    train, test = train_test_split(
        interactions, test_size=test_size, random_state=seed,
        stratify=interactions.clicked)
    return train.reset_index(drop=True), test.reset_index(drop=True)
