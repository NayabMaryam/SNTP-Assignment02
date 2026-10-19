# data/preprocess_epinions.py
"""
Epinions (Trustlet) preprocessing for SNTP Assignment 2.

Input  : data/raw/epinions/ratings_data.txt  (user_id item_id rating)
         data/raw/epinions/trust_data.txt    (trustor trustree 1)

Output : data/processed/  (same format run_req2.py expects)

Pipeline:
  1. Load raw ratings + trust (memory-efficient for large files).
  2. Filter items with >= MIN_ITEM_RATINGS.
  3. Filter users with >= MIN_USER_RATINGS.
  4. Keep trust edges only between surviving users.
  5. Infer influence strength for each trust edge from
     rating similarity on co-rated items (Pearson). Fallback = 0.5.
  6. Per-user 80/20 train/test split (leakage-free).
  7. Save: train.csv, test.csv, R_train.npz, R_all.npz,
           A_unweighted.npz, A_weighted.npz, edges.csv, split_info.json

No ground-truth latent files (real data -> no synthetic ground truth).
"""
import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

RAW = "data/raw/epinions"
PROC = "data/processed"

# --- filtering thresholds (tune based on dataset size) ---
MIN_ITEM_RATINGS = 10     # items with < 10 ratings dropped
MIN_USER_RATINGS = 20     # users with < 20 ratings dropped
MIN_CO_RATED_FOR_SIM = 5  # need >= 5 common items to compute edge similarity
TEST_FRAC = 0.2
SEED = 42


def load_ratings(path):
    """Trustlet ratings: 'user item rating' whitespace-separated."""
    print(f"Loading ratings from {path}...")
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["user_id", "item_id", "rating"],
                     engine="python", dtype={"user_id": np.int64,
                                             "item_id": np.int64,
                                             "rating": np.float32})
    print(f"  raw ratings: {len(df):,}  users: {df.user_id.nunique():,}"
          f"  items: {df.item_id.nunique():,}")
    return df


def load_trust(path):
    """Trustlet trust: 'trustor trustree 1' whitespace-separated."""
    print(f"Loading trust from {path}...")
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["trustor", "trustee", "trust"],
                     engine="python",
                     dtype={"trustor": np.int64,
                            "trustee": np.int64,
                            "trust": np.int8})
    df = df[df.trustor != df.trustee]                # drop self loops
    print(f"  raw trust edges: {len(df):,}")
    return df


def iter_filter_ratings(ratings, min_item, min_user):
    """Iteratively filter until stable (removing items changes user counts & vice versa)."""
    cur = ratings.copy()
    for it in range(10):
        before = len(cur)
        item_counts = cur.item_id.value_counts()
        keep_items = item_counts[item_counts >= min_item].index
        cur = cur[cur.item_id.isin(keep_items)]
        user_counts = cur.user_id.value_counts()
        keep_users = user_counts[user_counts >= min_user].index
        cur = cur[cur.user_id.isin(keep_users)]
        print(f"  iter {it+1}: {len(cur):,} ratings "
              f"({before - len(cur):,} dropped)")
        if len(cur) == before:
            break
    return cur


def remap_ids(df, col):
    """Remap sparse raw IDs -> contiguous 0..N-1."""
    uniq = np.sort(df[col].unique())
    mapping = {old: new for new, old in enumerate(uniq)}
    df[col] = df[col].map(mapping)
    return df, mapping


def infer_edge_weights(edges, R):
    """
    Influence strength for trust edge (a,b) = Pearson similarity of the
    two users' ratings on commonly-rated items.
    Clipped to [0, 1]. If < MIN_CO_RATED, fallback = 0.5.
    NOTE: this is an INFERRED quantity (not observed) -> report it.
    """
    R = R.tocsr()
    rows, cols = [], []
    for _, r in edges.iterrows():
        rows.append(int(r["trustor"]))
        cols.append(int(r["trustee"]))
    rows = np.array(rows); cols = np.array(cols)

    weights = np.zeros(len(rows), dtype=np.float32)
    for k, (a, b) in enumerate(zip(rows, cols)):
        ra = R[a].toarray().ravel()
        rb = R[b].toarray().ravel()
        mask = (ra > 0) & (rb > 0)
        n_common = mask.sum()
        if n_common < MIN_CO_RATED_FOR_SIM:
            weights[k] = 0.5
            continue
        x, y = ra[mask], rb[mask]
        if x.std() < 1e-6 or y.std() < 1e-6:
            weights[k] = 0.5
            continue
        sim = np.corrcoef(x, y)[0, 1]
        if np.isnan(sim):
            sim = 0.0
        # map [-1, 1] -> [0, 1]
        weights[k] = float(np.clip((sim + 1) / 2, 0.0, 1.0))

    return weights


def train_test_split_per_user(ratings, test_frac=0.2, min_train=1, seed=42):
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    for uid, grp in ratings.groupby("user_id", sort=False):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        n_test = min(n_test, max(0, len(idx) - min_train))
        test_rows.append(grp.loc[idx[:n_test]])
        train_rows.append(grp.loc[idx[n_test:]])
    train = pd.concat(train_rows, ignore_index=True)
    test = pd.concat(test_rows, ignore_index=True)

    overlap = set(map(tuple, train[["user_id", "item_id"]].values)) & \
              set(map(tuple, test[["user_id", "item_id"]].values))
    assert not overlap, f"Leakage! {len(overlap)} overlapping pairs"
    return train, test


def build_rating_matrix(df, n_users, n_items):
    return sp.csr_matrix(
        (df.rating.to_numpy(np.float32),
         (df.user_id.to_numpy(), df.item_id.to_numpy())),
        shape=(n_users, n_items))


def main():
    os.makedirs(PROC, exist_ok=True)

    ratings = load_ratings(f"{RAW}/ratings_data.txt")
    trust = load_trust(f"{RAW}/trust_data.txt")

    # --- filtering ---
    print("\nFiltering ratings (iterative):")
    ratings = iter_filter_ratings(ratings, MIN_ITEM_RATINGS, MIN_USER_RATINGS)

    # drop duplicate (user,item) pairs if any (keep mean)
    dup = ratings.duplicated(["user_id", "item_id"]).sum()
    if dup:
        print(f"  merging {dup} duplicate (user,item) pairs")
        ratings = ratings.groupby(["user_id", "item_id"], as_index=False).rating.mean()

    # --- remap ids ---
    ratings, user_map = remap_ids(ratings, "user_id")
    ratings, item_map = remap_ids(ratings, "item_id")
    n_users, n_items = len(user_map), len(item_map)
    print(f"\nAfter filtering: {n_users:,} users  {n_items:,} items"
          f"  {len(ratings):,} ratings")

    # --- trust edges among surviving users only ---
    # translate raw trust ids to new ids
    raw_to_new_user = user_map  # old_id -> new_id (inverse)
    trust = trust[trust.trustor.isin(raw_to_new_user) &
                  trust.trustee.isin(raw_to_new_user)].copy()
    trust["trustor"] = trust["trustor"].map(raw_to_new_user)
    trust["trustee"] = trust["trustee"].map(raw_to_new_user)
    trust = trust.drop_duplicates(["trustor", "trustee"])
    print(f"Trust edges among filtered users: {len(trust):,}")

    # --- infer edge weights (needs full rating matrix) ---
    R_all = build_rating_matrix(ratings, n_users, n_items)
    print("Inferring influence weights (Pearson on co-rated items)...")
    trust["weight"] = infer_edge_weights(trust, R_all)
    print(f"  weight stats: mean={trust.weight.mean():.3f}"
          f"  std={trust.weight.std():.3f}"
          f"  min={trust.weight.min():.3f}  max={trust.weight.max():.3f}")

    # --- split ---
    train, test = train_test_split_per_user(ratings, TEST_FRAC, 1, SEED)
    R_train = build_rating_matrix(train, n_users, n_items)

    # --- adjacency (undirected; both orientations saved) ---
    def build_adj(weighted):
        rows, cols, vals = [], [], []
        for _, r in trust.iterrows():
            a, b = int(r.trustor), int(r.trustee)
            w = float(r.weight) if weighted else 1.0
            rows += [a, b]; cols += [b, a]; vals += [w, w]
        return sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_users))

    A_unweighted = build_adj(weighted=False)
    A_weighted = build_adj(weighted=True)

    # --- save ---
    train.to_csv(f"{PROC}/train.csv", index=False)
    test.to_csv(f"{PROC}/test.csv", index=False)
    trust.to_csv(f"{PROC}/edges.csv", index=False)
    sp.save_npz(f"{PROC}/R_train.npz", R_train)
    sp.save_npz(f"{PROC}/R_all.npz", R_all)
    sp.save_npz(f"{PROC}/A_unweighted.npz", A_unweighted)
    sp.save_npz(f"{PROC}/A_weighted.npz", A_weighted)

    # metadata
    info = {
        "dataset": "Epinions (Trustlet)",
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_ratings_total": int(len(ratings)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_edges_filtered": int(len(trust)),
        "sparsity_all": float(1 - len(ratings) / (n_users * n_items)),
        "rating_min": float(ratings.rating.min()),
        "rating_max": float(ratings.rating.max()),
        "rating_mean": float(ratings.rating.mean()),
        "filtering": {"min_item_ratings": MIN_ITEM_RATINGS,
                      "min_user_ratings": MIN_USER_RATINGS,
                      "min_co_rated_for_sim": MIN_CO_RATED_FOR_SIM},
        "test_frac": TEST_FRAC,
        "seed": SEED,
        "observed": ["train.csv", "test.csv", "edges.csv (trust links)",
                     "ratings in train/test"],
        "inferred": ["edge 'weight' column (Pearson similarity of raters)"],
    }
    with open(f"{PROC}/split_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n✅ Done")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()