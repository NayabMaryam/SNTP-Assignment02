# src/recommenders/base.py
"""Common interface for all recommenders."""
import numpy as np


class BaseRecommender:
    name = "base"

    def fit(self, R_train, **kwargs):
        """R_train: scipy CSR (n_users, n_items), 0 = missing."""
        raise NotImplementedError

    def predict(self, user_id, item_ids):
        """Return predicted scores for given (user, item) pairs."""
        raise NotImplementedError

    def recommend_topk(self, user_id, k=10, exclude_seen=True, R_train=None):
        """Return top-k item ids for a user."""
        raise NotImplementedError