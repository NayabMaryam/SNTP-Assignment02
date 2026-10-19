# src/recommenders/cf_user_item.py
"""
User-based Collaborative Filtering (memory-based, my own implementation).

Prediction for (u, i):
    r_hat(u,i) = mean_u + sum_{v in N_k(u)} sim(u,v) * (r(v,i) - mean_v)
                          ------------------------------------------------
                                 sum_{v in N_k(u)} |sim(u,v)|
only over neighbours v who actually rated i.

Similarity: cosine on mean-centred rating vectors (default), or Pearson.
Neighbourhood: top-K similar users.
"""
import numpy as np
import scipy.sparse as sp
from .base import BaseRecommender


class UserCF(BaseRecommender):
    name = "UserCF"

    def __init__(self, k_neighbors=30, sim_metric="cosine", min_common=2):
        self.k = k_neighbors
        self.sim_metric = sim_metric
        self.min_common = min_common

    # ---------- fit ----------
    def fit(self, R_train, **kwargs):
        self.R = R_train.tocsr().astype(np.float32)
        self.n_users, self.n_items = self.R.shape

        # per-user mean over observed ratings
        sums = np.asarray(self.R.sum(axis=1)).ravel()
        counts = np.asarray((self.R > 0).sum(axis=1)).ravel()
        self.user_mean = np.divide(sums, counts, out=np.zeros_like(sums),
                                   where=counts > 0)

        # centred matrix (0 stays 0 -> masked in similarity)
        R_dense_like = self.R.copy()
        R_centred = self.R.copy().astype(np.float32)
        R_centred.data -= np.repeat(self.user_mean, np.diff(R_centred.indptr))
        # keep only observed entries non-zero for the mask
        self.R_centred = R_centred.tocsr()
        self.R_mask = (self.R > 0).astype(np.float32).tocsr()

        self.sim = self._compute_similarity()
        return self

    def _compute_similarity(self):
        """User-user similarity on centred ratings (cosine or Pearson)."""
        M = self.R_centred
        # cosine similarity over observed entries only
        # sim = (M @ M.T) / (||M_u|| * ||M_v||)
        dot = (M @ M.T).toarray()
        norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel()) + 1e-9
        sim = dot / np.outer(norms, norms)

        # mask: require at least min_common co-rated items
        co = (self.R_mask @ self.R_mask.T).toarray()
        sim[co < self.min_common] = 0.0
        np.fill_diagonal(sim, 0.0)
        return sim.astype(np.float32)

    # ---------- predict ----------
    def predict(self, user_id, item_ids):
        item_ids = np.atleast_1d(item_ids)
        sims = self.sim[user_id]                     # (n_users,)
        # top-k neighbours (all users who are similar)
        if self.k < len(sims):
            idx = np.argpartition(-sims, self.k)[:self.k]
        else:
            idx = np.arange(len(sims))
        idx = idx[sims[idx] > 0]

        preds = np.zeros(len(item_ids), dtype=np.float32)
        for j, i in enumerate(item_ids):
            if not idx.size:
                preds[j] = self.user_mean[user_id] if self.user_mean[user_id] else 3.0
                continue
            r_vi = self.R[idx, i].toarray().ravel()
            rated = r_vi > 0
            if not rated.any():
                preds[j] = self.user_mean[user_id] if self.user_mean[user_id] else 3.0
                continue
            sims_r = sims[idx][rated]
            dev = r_vi[rated] - self.user_mean[idx][rated]
            preds[j] = self.user_mean[user_id] + np.dot(sims_r, dev) / (np.abs(sims_r).sum() + 1e-9)
        return np.clip(preds, 1.0, 5.0)

    # ---------- top-k ----------
    def recommend_topk(self, user_id, k=10, exclude_seen=True):
        """Vectorized top-k: ek hi pass mein saare items ka score compute karo."""
        sims = self.sim[user_id]                      # (n_users,)
        if self.k < len(sims):
            idx = np.argpartition(-sims, self.k)[:self.k]
        else:
            idx = np.arange(len(sims))
        idx = idx[sims[idx] > 0]
        scores = np.full(self.n_items, self.user_mean[user_id], dtype=np.float32)

        if idx.size == 0:
            # no neighbours -> fallback to user mean
            pass
        else:
            # neighbour ratings on all items: (K, n_items) — sparse subset
            R_nb = self.R[idx]                            # CSR (K, n_items)
            # only items rated by at least one neighbour matter
            rated_items = np.unique(R_nb.indices)         # which items to score
            if rated_items.size:
                # dense neighbour ratings on those items (K x |rated_items|)
                R_nb_dense = R_nb[:, rated_items].toarray().astype(np.float32)
                sims_nb = sims[idx].astype(np.float32)    # (K,)
                dev = R_nb_dense - self.user_mean[idx][:, None]
                mask = (R_nb_dense > 0)
                num = sims_nb @ (dev * mask)              # (|rated_items|,)
                den = np.abs(sims_nb) @ mask.astype(np.float32) + 1e-9
                scores[rated_items] = self.user_mean[user_id] + num / den

        scores = np.clip(scores, 1.0, 5.0)
        if exclude_seen:
            seen = self.R[user_id].indices
            scores[seen] = -np.inf
        return np.argsort(-scores)[:k], scores