# data/generate_data.py
"""
Synthetic social recommendation dataset generator.
Reproducible: fixed seed -> identical dataset.
"""
import numpy as np
import pandas as pd
import json
import os

def generate_dataset(
    n_users=500,
    n_items=300,
    n_factors=5,
    rating_sparsity=0.12,     # fraction of items each user rates
    n_rating_levels=5,        # ratings 1..5
    homophily=0.7,            # 0=random graph, 1=perfect homophily
    avg_degree=8,
    influence_strength_range=(0.1, 0.9),
    susceptibility_range=(0.1, 0.9),
    seed=42
):
    rng = np.random.default_rng(seed)

    # ---- 1. Latent ground truth (CREATED / INFERRED, not observed by system) ----
    user_latent = rng.normal(0, 1, size=(n_users, n_factors))
    item_latent = rng.normal(0, 1, size=(n_items, n_factors))

    # per-user rating bias (heterogeneous rating patterns)
    user_bias = rng.normal(0, 0.5, size=n_users)

    # ---- 2. True affinity -> ratings ----
    affinity = user_latent @ item_latent.T          # (n_users, n_items)
    raw_scores = affinity + user_bias[:, None]
    # normalize to 1..5
    raw_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    ratings_full = 1 + raw_scores * (n_rating_levels - 1)

    # ---- 3. Observed interactions: sparse sampling ----
    observed_mask = rng.random((n_users, n_items)) < rating_sparsity
    # ensure every user has at least 3 ratings & every item at least 3
    for u in range(n_users):
        if observed_mask[u].sum() < 3:
            idx = rng.choice(n_items, 3, replace=False)
            observed_mask[u, idx] = True
    for i in range(n_items):
        if observed_mask[:, i].sum() < 3:
            idx = rng.choice(n_users, 3, replace=False)
            observed_mask[idx, i] = True
    # ---- 3b. Inject low-activity users (for cold-start investigation) ----
    # 20% users ko deliberately sparse banao (3-8 ratings)
    n_low_activity = int(0.2 * n_users)
    low_activity_users = rng.choice(n_users, n_low_activity, replace=False)
    for u in low_activity_users:
        n_keep = rng.integers(3, 9)  # 3-8 ratings
        # pehle sab hata do, phir n_keep random rakho
        observed_mask[u, :] = False
        idx = rng.choice(n_items, n_keep, replace=False)
        observed_mask[u, idx] = True
    # add small rating noise
    ratings_obs = np.where(observed_mask,
                           np.clip(np.round(ratings_full + rng.normal(0, 0.3, ratings_full.shape)),
                                   1, n_rating_levels),
                           np.nan)

    # ---- 4. Social graph with homophily ----
    # normalize user latents to unit vectors for cosine similarity
    norms = np.linalg.norm(user_latent, axis=1, keepdims=True) + 1e-9
    u_unit = user_latent / norms
    sim = u_unit @ u_unit.T                     # pairwise similarity

    edges = set()
    target_edges = int(n_users * avg_degree / 2)
    attempts = 0
    while len(edges) < target_edges and attempts < n_users * avg_degree * 20:
        attempts += 1
        u, v = rng.integers(0, n_users, size=2)
        if u == v:
            continue
        # homophily: accept edge with prob proportional to similarity
        p_accept = homophily * max(0.0, (sim[u, v] + 1) / 2) + (1 - homophily) * 0.5
        if rng.random() < p_accept:
            edges.add(tuple(sorted((int(u), int(v)))))

    social_edges = sorted(edges)

    # ---- 5. Influence strength & susceptibility (CREATED / INFERRED) ----
    edge_influence = {
        e: float(rng.uniform(*influence_strength_range)) for e in social_edges
    }
    susceptibility = rng.uniform(*susceptibility_range, size=n_users)

    # ---- 6. Save ----
    os.makedirs("data/raw", exist_ok=True)

    # ratings long-format
    rows = []
    for u in range(n_users):
        for i in range(n_items):
            if not np.isnan(ratings_obs[u, i]):
                rows.append((u, i, float(ratings_obs[u, i])))
    pd.DataFrame(rows, columns=["user_id", "item_id", "rating"]).to_csv(
        "data/raw/ratings.csv", index=False)

    pd.DataFrame(social_edges, columns=["user_a", "user_b"]).to_csv(
        "data/raw/social_edges.csv", index=False)

    # ground-truth / created metadata (for evaluation only)
    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "n_factors": n_factors,
        "seed": seed,
        "rating_sparsity": rating_sparsity,
        "homophily": homophily,
        "avg_degree": avg_degree,
        "n_observed_ratings": len(rows),
        "n_edges": len(social_edges),
        "observed": ["ratings.csv", "social_edges.csv"],
        "created_or_inferred": ["user_latent.npy", "item_latent.npy",
                                "edge_influence.json", "susceptibility.npy"]
    }
    with open("data/raw/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    np.save("data/raw/user_latent.npy", user_latent)
    np.save("data/raw/item_latent.npy", item_latent)
    np.save("data/raw/susceptibility.npy", susceptibility)
    with open("data/raw/edge_influence.json", "w") as f:
        json.dump({f"{a}-{b}": w for (a, b), w in edge_influence.items()}, f)

    print(f"✅ Generated: {len(rows)} ratings, {len(social_edges)} edges")
    print(f"   Saved to data/raw/")


if __name__ == "__main__":
    generate_dataset()