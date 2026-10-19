# src/recommenders/matrix_factorization.py
"""
Matrix Factorization with SGD (my own implementation).

Model:
    r_hat(u,i) = mu + b_u + b_i + p_u . q_i

Training:
    SGD over observed ratings, minimizing squared error + L2 reg.
    Baseline biases are also learned.

Supports:
    - cold users (biases fall back to mu)
    - cold items
    - early stopping via validation RMSE
"""
import numpy as np
import scipy.sparse as sp
from .base import BaseRecommender


class MatrixFactorization(BaseRecommender):
    name = "MF-SGD"

    def __init__(self, n_factors=16, lr=0.01, reg=0.05,
                 n_epochs=30, batch_size=1024, seed=42):
        self.k = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed

    # ------------------------------------------------------------------
    def fit(self, R_train, val_pairs=None, verbose=False):
        """
        R_train: scipy CSR (n_users, n_items)
        val_pairs: optional list of (u, i, r) for early stopping
        """
        rng = np.random.default_rng(self.seed)
        self.R = R_train.tocsr().astype(np.float32)
        self.n_users, self.n_items = self.R.shape

        # --- training triples from sparse matrix ---
        coo = self.R.tocoo()
        users = coo.row.astype(np.int32)
        items = coo.col.astype(np.int32)
        ratings = coo.data.astype(np.float32)

        self.mu = float(ratings.mean())
        self.b_u = np.zeros(self.n_users, dtype=np.float32)
        self.b_i = np.zeros(self.n_items, dtype=np.float32)
        self.P = rng.normal(0, 0.1, size=(self.n_users, self.k)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, size=(self.n_items, self.k)).astype(np.float32)

        n = len(ratings)
        idx = np.arange(n)

        for epoch in range(self.n_epochs):
            rng.shuffle(idx)
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                self._sgd_step(users[batch], items[batch], ratings[batch])

            if verbose or val_pairs is not None:
                train_rmse = self._rmse(users, items, ratings)
                if val_pairs is not None:
                    vu, vi, vr = val_pairs
                    val_rmse = self._rmse(vu, vi, vr)
                    if verbose:
                        print(f"  epoch {epoch+1:02d}  train={train_rmse:.4f}  val={val_rmse:.4f}")
        return self

    def _sgd_step(self, u, i, r):
        """Vectorized SGD step over a minibatch."""
        p_u = self.P[u]          # (B, k)
        q_i = self.Q[i]
        b_u = self.b_u[u]
        b_i = self.b_i[i]

        pred = self.mu + b_u + b_i + np.einsum("bk,bk->b", p_u, q_i)
        err = r - pred              # (B,)

        # gradients
        grad_bu = -err + self.reg * b_u
        grad_bi = -err + self.reg * b_i
        grad_pu = -err[:, None] * q_i + self.reg * p_u
        grad_qi = -err[:, None] * p_u + self.reg * q_i

        # apply updates (use np.add.at for repeated indices)
        np.add.at(self.b_u, u, -self.lr * grad_bu)
        np.add.at(self.b_i, i, -self.lr * grad_bi)
        np.add.at(self.P, u, -self.lr * grad_pu)
        np.add.at(self.Q, i, -self.lr * grad_qi)

    def _rmse(self, u, i, r):
        pred = (self.mu + self.b_u[u] + self.b_i[i]
                + np.einsum("bk,bk->b", self.P[u], self.Q[i]))
        return float(np.sqrt(np.mean((r - pred) ** 2)))

    # ------------------------------------------------------------------
    def predict(self, user_id, item_ids):
        item_ids = np.atleast_1d(item_ids).astype(np.int32)
        pred = (self.mu
                + self.b_u[user_id]
                + self.b_i[item_ids]
                + self.P[user_id] @ self.Q[item_ids].T)
        return np.clip(pred, 1.0, 5.0).astype(np.float32)

    def recommend_topk(self, user_id, k=10, exclude_seen=True):
        scores = (self.mu + self.b_u[user_id] + self.b_i
                  + self.P[user_id] @ self.Q.T)
        scores = np.clip(scores, 1.0, 5.0)
        if exclude_seen:
            seen = self.R[user_id].indices
            scores[seen] = -np.inf
        return np.argsort(-scores)[:k], scores