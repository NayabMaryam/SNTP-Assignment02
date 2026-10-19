# src/recommenders/social_cf.py
"""
Social extension of User-based CF (Req 3).

score(u, i) = (1 - alpha) * CF_score(u, i) + alpha * Social_score(u, i)

Two variants for the Social_score:
  - "friends":    neighbours restricted to social graph neighbours
  - "similar":    neighbours restricted to similar-but-unconnected users

The alpha parameter lets us isolate the contribution of social information.
"""
import numpy as np
import scipy.sparse as sp
from .base import BaseRecommender
from .cf_user_items import UserCF


class SocialUserCF(BaseRecommender):
    name = "SocialUserCF"

    def __init__(self, k_neighbors=30, alpha=0.3, variant="friends",
             min_common=2, social_k=30, use_trust_weights=False):
        self.k = k_neighbors
        self.alpha = alpha
        self.variant = variant
        self.min_common = min_common
        self.social_k = social_k
        self.use_trust_weights = use_trust_weights
        self.A_weighted = None
    # ------------------------------------------------------------------
    def fit(self, R_train, A_social=None, A_weighted=None, **kwargs):
        ...
        self.A_weighted = A_weighted.tocsr() if A_weighted is not None else None
        ...
        """
        R_train : scipy CSR (n_users, n_items) training ratings
        A_social: scipy CSR (n_users, n_users) social adjacency (unweighted)
        """
        assert A_social is not None, "SocialUserCF needs A_social"
        self.R = R_train.tocsr().astype(np.float32)
        self.A = A_social.tocsr().astype(np.float32)
        self.n_users, self.n_items = self.R.shape

        # ---- base CF (same as UserCF) ----
        sums = np.asarray(self.R.sum(axis=1)).ravel()
        counts = np.asarray((self.R > 0).sum(axis=1)).ravel()
        self.user_mean = np.divide(sums, counts, out=np.zeros_like(sums),
                                   where=counts > 0)

        self.R_centred = self.R.copy().astype(np.float32)
        self.R_centred.data -= np.repeat(self.user_mean, np.diff(self.R_centred.indptr))
        self.R_centred = self.R_centred.tocsr()
        self.R_mask = (self.R > 0).astype(np.float32).tocsr()

        # CF similarity (Pearson-like on centred ratings)
        self.sim_cf = self._compute_cf_similarity()

        # Social mask: which users count as "social neighbours" for this variant
        self.social_neighbours = self._compute_social_neighbour_mask()

        return self

    # ------------------------------------------------------------------
    def _compute_cf_similarity(self):
        M = self.R_centred
        dot = (M @ M.T).toarray()
        norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel()) + 1e-9
        sim = dot / np.outer(norms, norms)
        co = (self.R_mask @ self.R_mask.T).toarray()
        sim[co < self.min_common] = 0.0
        np.fill_diagonal(sim, 0.0)
        return sim.astype(np.float32)

    def _compute_social_neighbour_mask(self):
        """
        Build a boolean mask (n_users, n_users) whose True entries are
        users we are allowed to draw social signal from.

        variant == "friends" : direct social neighbours
        variant == "similar" : similar users who are NOT socially connected
        """
        adj = self.A.copy()
        adj.setdiag(0)
        adj.eliminate_zeros()
        adj_bool = (adj > 0).toarray()              # (n_users, n_users) bool

        if self.variant == "friends":
            return adj_bool

        if self.variant == "similar":
            # start from CF similarity (positive, strong)
            sim = self.sim_cf.copy()
            sim[sim < np.percentile(sim[sim > 0], 90)] = 0.0   # keep top ~10%
            similar_mask = sim > 0
            # remove friends (and self)
            mask = similar_mask & ~adj_bool
            np.fill_diagonal(mask, False)
            return mask

        raise ValueError(f"Unknown variant: {self.variant}")

    # ------------------------------------------------------------------
    def _cf_scores_for_user(self, user_id, item_ids):
        """Vectorised UserCF prediction (same logic as UserCF.recommend_topk)."""
        sims = self.sim_cf[user_id]
        if self.k < len(sims):
            idx = np.argpartition(-sims, self.k)[:self.k]
        else:
            idx = np.arange(len(sims))
        idx = idx[sims[idx] > 0]
        scores = np.full(len(item_ids), self.user_mean[user_id], dtype=np.float32)
        if idx.size == 0:
            return scores

        # Build a dense K x |items| matrix of neighbour ratings
        R_nb = self.R[idx][:, item_ids].toarray().astype(np.float32)
        mask = R_nb > 0
        dev = R_nb - self.user_mean[idx][:, None]
        sims_nb = sims[idx].astype(np.float32)
        num = (dev * mask) .T @ sims_nb           # (|items|,)
        den = mask.T @ np.abs(sims_nb) + 1e-9
        pred = self.user_mean[user_id] + num / den
        # where nobody rated, keep user_mean
        no_signal = den.ravel() < 1e-6
        pred[no_signal] = self.user_mean[user_id]
        return np.clip(pred, 1.0, 5.0).astype(np.float32)

    def _social_scores_for_user(self, user_id, item_ids, use_trust_weights=False):
        """
        Prediction from social neighbours only.

        Neighbours = users connected by trust edges (mask from self.social_neighbours).
        Weighting:
        - uniform (default) OR
        - trust edge weight (if use_trust_weights=True and A_weighted supplied)

        Returns (scores, has_signal).
        """
        allowed = self.social_neighbours[user_id]
        neighbours = np.where(allowed)[0]
        scores = np.full(len(item_ids), self.user_mean[user_id], dtype=np.float32)
        has_signal = np.zeros(len(item_ids), dtype=bool)
        if neighbours.size == 0:
            return scores, has_signal

        # ---- weights for neighbours ----
        if use_trust_weights and self.A_weighted is not None:
            w = np.asarray(self.A_weighted[user_id, neighbours]).ravel().astype(np.float32)
        else:
            w = np.ones(len(neighbours), dtype=np.float32)

        if w.sum() < 1e-9:
            return scores, has_signal

        # ---- neighbour ratings on requested items ----
        R_nb = self.R[neighbours][:, item_ids].toarray().astype(np.float32)  # (M, I)
        mask = R_nb > 0
        dev = R_nb - self.user_mean[neighbours][:, None]

        num = (dev * mask).T @ w                  # (I,)
        den = mask.T @ w + 1e-9                   # (I,)
        pred = self.user_mean[user_id] + num / den
        has_signal = den.ravel() >= 1e-6
        pred[~has_signal] = self.user_mean[user_id]
        return np.clip(pred, 1.0, 5.0).astype(np.float32), has_signal
    # ------------------------------------------------------------------
    def predict(self, user_id, item_ids):
        item_ids = np.atleast_1d(item_ids).astype(np.int64)
        cf = self._cf_scores_for_user(user_id, item_ids)
        if self.alpha == 0.0:
            return cf
        social, has_signal = self._social_scores_for_user(
            user_id, item_ids, use_trust_weights=self.use_trust_weights
        )
        out = cf.copy()
        out[has_signal] = (1 - self.alpha) * cf[has_signal] + self.alpha * social[has_signal]
        return np.clip(out, 1.0, 5.0)
    # ------------------------------------------------------------------
    def recommend_topk(self, user_id, k=10, exclude_seen=True):
        item_ids = np.arange(self.n_items, dtype=np.int64)
        scores = self.predict(user_id, item_ids)
        if exclude_seen:
            seen = self.R[user_id].indices
            scores[seen] = -np.inf
        return np.argsort(-scores)[:k], scores