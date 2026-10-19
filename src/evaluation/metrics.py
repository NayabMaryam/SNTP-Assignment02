# src/evaluation/metrics.py
"""Evaluation metrics — my own implementations (no sklearn recommender)."""
import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def precision_recall_ndcg_at_k(recommended, relevant_set, k):
    rec = recommended[:k]
    hits = [1 if i in relevant_set else 0 for i in rec]
    p = sum(hits) / k if k else 0.0
    r = sum(hits) / len(relevant_set) if relevant_set else 0.0
    # DCG (binary relevance)
    dcg = sum(h / np.log2(pos + 2) for pos, h in enumerate(hits))
    ideal = sum(1 / np.log2(pos + 2) for pos in range(min(k, len(relevant_set))))
    ndcg = dcg / ideal if ideal > 0 else 0.0
    return p, r, ndcg


def catalog_coverage(recommended_items, n_items):
    return len(set(recommended_items)) / n_items


def intra_list_diversity(R_train, items):
    if len(items) < 2:
        return 0.0
    # only columns (items) of interest
    V = R_train[:, items].toarray().T       # small: (10, 4346) per user
    norms = np.linalg.norm(V, axis=1) + 1e-9
    Vn = V / norms[:, None]
    sims = Vn @ Vn.T
    iu = np.triu_indices(len(items), k=1)
    return float(1 - sims[iu].mean())

def evaluate_ranking_with_negatives(model, test, n_items, k=10,
                                    n_negatives=100,
                                    relevance_threshold=4.0,
                                    seed=42):
    """
    Standard ranking evaluation with negative sampling.

    For each test user:
      - positive items = test items with rating >= relevance_threshold
      - negatives = random unseen items (not in train, not in test)
      - model ranks candidates; measure P@k, R@k, NDCG@k, HR@k
    """
    rng = np.random.default_rng(seed)

    # user -> set of items already rated (train+test) so negatives are truly unseen
    seen_by_user = test.groupby("user_id").item_id.apply(set).to_dict()
    # also mark train-seen to avoid picking them as negatives
    if hasattr(model, "R"):
        for u in range(model.R.shape[0]):
            seen_by_user.setdefault(u, set()).update(model.R[u].indices.tolist())

    test_rel = test[test.rating >= relevance_threshold]

    precisions, recalls, ndcgs, hits = [], [], [], []

    for u, grp in test_rel.groupby("user_id"):
        pos_items = grp.item_id.tolist()
        if not pos_items:
            continue

        seen = seen_by_user.get(u, set())
        # candidate pool: all items minus everything the user has seen
        unseen = np.setdiff1d(np.arange(n_items),
                              np.fromiter(seen, dtype=np.int64),
                              assume_unique=False)
        if len(unseen) < 1:
            continue
        n_neg = min(n_negatives, len(unseen))
        negs = rng.choice(unseen, size=n_neg, replace=False).tolist()

        candidates = np.array(pos_items + negs, dtype=np.int64)
        scores = model.predict(u, candidates)                  # (C,)
        ranked = candidates[np.argsort(-scores)][:k]

        hit_set = set(ranked.tolist()) & set(pos_items)
        precisions.append(len(hit_set) / k)
        recalls.append(len(hit_set) / len(pos_items))

        rel_vec = np.isin(ranked, pos_items).astype(np.float64)
        discounts = 1.0 / np.log2(np.arange(2, len(rel_vec) + 2))
        dcg = float(np.sum(rel_vec * discounts))
        ideal = float(np.sum(
            1.0 / np.log2(np.arange(2, min(k, len(pos_items)) + 2))))
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
        hits.append(1.0 if hit_set else 0.0)

    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "hit_rate_at_k": float(np.mean(hits)),
    }
# __init__.py bhi bana lo agar nahi hai