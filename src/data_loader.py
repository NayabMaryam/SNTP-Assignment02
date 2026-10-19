# src/data_loader.py
"""
Data loading, preprocessing, and leakage-free train/test splitting.

Observed data  : ratings.csv, social_edges.csv
Created/inferred: user_latent.npy, item_latent.npy,
                  susceptibility.npy, edge_influence.json
                  (ground truth -> ONLY for evaluation, never for training)
"""
import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"


# ----------------------------------------------------------------------
# 1. Load raw data
# ----------------------------------------------------------------------
def load_raw():
    ratings = pd.read_csv(os.path.join(RAW_DIR, "ratings.csv"))
    edges = pd.read_csv(os.path.join(RAW_DIR, "social_edges.csv"))
    with open(os.path.join(RAW_DIR, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(RAW_DIR, "edge_influence.json")) as f:
        edge_influence = json.load(f)  # {"a-b": weight}
    susceptibility = np.load(os.path.join(RAW_DIR, "susceptibility.npy"))
    user_latent = np.load(os.path.join(RAW_DIR, "user_latent.npy"))
    item_latent = np.load(os.path.join(RAW_DIR, "item_latent.npy"))
    return ratings, edges, meta, edge_influence, susceptibility, user_latent, item_latent


# ----------------------------------------------------------------------
# 2. Leakage-free per-user train/test split
# ----------------------------------------------------------------------
def train_test_split_per_user(ratings, test_frac=0.2, min_train=1, seed=42):
    """
    Har user ke ratings mein se test_frac test mein daalo.
    Guarantee: har user ke paas kam se kam `min_train` ratings train mein rahen.
    Isse 'limited history' users ka behaviour bhi test ho sakta hai.
    """
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []

    for uid, grp in ratings.groupby("user_id"):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        # ensure min_train stays in train
        n_test = min(n_test, max(0, len(idx) - min_train))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        train_rows.append(grp.loc[train_idx])
        test_rows.append(grp.loc[test_idx])

    train = pd.concat(train_rows).reset_index(drop=True)
    test = pd.concat(test_rows).reset_index(drop=True)

    # safety: no overlap
    assert len(set(map(tuple, train[["user_id", "item_id"]].values)) &
               set(map(tuple, test[["user_id", "item_id"]].values))) == 0
    return train, test


# ----------------------------------------------------------------------
# 3. Sparse matrix builders
# ----------------------------------------------------------------------
def build_rating_matrix(df, n_users, n_items, value_col="rating"):
    """Returns scipy CSR (n_users, n_items); missing = 0 (never a real rating)."""
    rows = df["user_id"].to_numpy()
    cols = df["item_id"].to_numpy()
    vals = df[value_col].to_numpy(dtype=np.float32)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))


def build_social_adjacency(edges_df, edge_influence, n_users, weighted=True):
    """
    Undirected symmetric adjacency. If weighted=True, edge weight is the
    inferred influence strength (created/inferred ground truth).
    NOTE: for the *recommender* (Req 3) we should use an *unweighted* or
    similarity-inferred weight — using ground-truth influence would be leakage.
    """
    rows, cols, vals = [], [], []
    for _, r in edges_df.iterrows():
        a, b = int(r["user_a"]), int(r["user_b"])
        w = edge_influence.get(f"{min(a,b)}-{max(a,b)}", 1.0) if weighted else 1.0
        rows += [a, b]
        cols += [b, a]
        vals += [w, w]
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_users))


# ----------------------------------------------------------------------
# 4. Full preprocessing + save
# ----------------------------------------------------------------------
def preprocess(test_frac=0.2, seed=42):
    os.makedirs(PROC_DIR, exist_ok=True)
    (ratings, edges, meta, edge_influence,
     susceptibility, user_latent, item_latent) = load_raw()

    n_users, n_items = meta["n_users"], meta["n_items"]

    # --- split ---
    train, test = train_test_split_per_user(ratings, test_frac=test_frac, seed=seed)

    # --- matrices ---
    R_train = build_rating_matrix(train, n_users, n_items)
    R_all   = build_rating_matrix(ratings, n_users, n_items)  # for reporting sparsity

    # social graph (unweighted for recommender; weighted only for diffusion later)
    A_unweighted = build_social_adjacency(edges, edge_influence, n_users, weighted=False)
    A_weighted   = build_social_adjacency(edges, edge_influence, n_users, weighted=True)

    # --- save ---
    train.to_csv(os.path.join(PROC_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(PROC_DIR, "test.csv"), index=False)
    sp.save_npz(os.path.join(PROC_DIR, "R_train.npz"), R_train)
    sp.save_npz(os.path.join(PROC_DIR, "R_all.npz"), R_all)
    sp.save_npz(os.path.join(PROC_DIR, "A_unweighted.npz"), A_unweighted)
    sp.save_npz(os.path.join(PROC_DIR, "A_weighted.npz"), A_weighted)
    edges.to_csv(os.path.join(PROC_DIR, "edges.csv"), index=False)

    # ground truth copy (evaluation only) -- keep SEPARATE from processed/train
    np.save(os.path.join(PROC_DIR, "gt_user_latent.npy"), user_latent)
    np.save(os.path.join(PROC_DIR, "gt_item_latent.npy"), item_latent)
    np.save(os.path.join(PROC_DIR, "gt_susceptibility.npy"), susceptibility)
    with open(os.path.join(PROC_DIR, "gt_edge_influence.json"), "w") as f:
        json.dump(edge_influence, f)

    split_info = {
        "n_users": n_users, "n_items": n_items,
        "n_ratings_total": int(len(ratings)),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "sparsity_all": float(1 - len(ratings) / (n_users * n_items)),
        "n_edges": int(len(edges)),
        "test_frac": test_frac, "seed": seed,
    }
    with open(os.path.join(PROC_DIR, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    print("✅ Preprocessing done")
    print(json.dumps(split_info, indent=2))
    return split_info


if __name__ == "__main__":
    preprocess()