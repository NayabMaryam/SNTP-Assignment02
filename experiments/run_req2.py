# experiments/run_req2.py
"""
Req 2 evaluation:
 - RMSE / MAE on test
 - Precision@k / Recall@k / NDCG@k (relevance = rating >= 4)
 - Coverage, Diversity
 - Cold-start breakdown (limited history / few item interactions)
"""
# experiments/run_req2.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.recommenders.cf_user_items import UserCF
from src.evaluation.metrics import (
    rmse, mae, precision_recall_ndcg_at_k, catalog_coverage, intra_list_diversity
)

from src.recommenders.matrix_factorization import MatrixFactorization

PROC = "data/processed"
OUT = "results/tables"
os.makedirs(OUT, exist_ok=True)


def load():
    R_train = sp.load_npz(f"{PROC}/R_train.npz")
    test = pd.read_csv(f"{PROC}/test.csv")
    with open(f"{PROC}/split_info.json") as f:
        info = json.load(f)
    return R_train, test, info


def evaluate(model, R_train, test, k=10, relevance_threshold=4.0):
    y_true = test["rating"].values
    y_pred = np.array([model.predict(u, i)[0]
                       for u, i in zip(test["user_id"], test["item_id"])])
    print(f"  RMSE = {rmse(y_true, y_pred):.4f}   MAE = {mae(y_true, y_pred):.4f}")

    from src.evaluation.metrics import evaluate_ranking_with_negatives
    rank_metrics = evaluate_ranking_with_negatives(
        model, test, R_train.shape[1], k=k,
        n_negatives=100, relevance_threshold=relevance_threshold, seed=42
    )
    print(f"  [with 100 negatives] P@{k}={rank_metrics['precision_at_k']:.4f}"
          f"  R@{k}={rank_metrics['recall_at_k']:.4f}"
          f"  NDCG@{k}={rank_metrics['ndcg_at_k']:.4f}"
          f"  HR@{k}={rank_metrics['hit_rate_at_k']:.4f}")

    # coverage + diversity — SIRF ek baar per user, cache
    test_rel = test[test.rating >= relevance_threshold]
    users = test_rel.user_id.unique()
    covered, divs = set(), []
    cache = {}
    for u in users:
        top_k, _ = model.recommend_topk(u, k=k, exclude_seen=True)  # ab fast
        covered.update(top_k.tolist())
        divs.append(intra_list_diversity(R_train, top_k))
    coverage = catalog_coverage(covered, R_train.shape[1])
    print(f"  Coverage = {coverage:.4f}  ILD@{k} = {np.mean(divs):.4f}")

    return {
        "rmse": float(rmse(y_true, y_pred)),
        "mae": float(mae(y_true, y_pred)),
        **rank_metrics,
        "coverage": float(coverage),
        "diversity": float(np.mean(divs)),
    }

def cold_start_breakdown(model, R_train, test, k=10):
    """Split test users by #train ratings."""
    counts = np.asarray((R_train > 0).sum(axis=1)).ravel()
    # data ke hisaab se dynamic thresholds
    q1, q2 = np.percentile(counts, [33, 66])
    results = {}
    for label, cond in [
        (f"cold (<= {int(q1)} train ratings)", counts <= q1),
        (f"warm ({int(q1)+1}-{int(q2)})", (counts > q1) & (counts <= q2)),
        (f"hot (> {int(q2)})", counts > q2),
    ]:
        users = np.where(cond)[0]
        sub = test[test.user_id.isin(users)]
        if len(sub) == 0:
            continue
        y_true = sub.rating.values
        y_pred = np.array([model.predict(u, i)[0]
                           for u, i in zip(sub.user_id, sub.item_id)])
        results[label] = {"n": int(len(sub)),
                          "n_users": int(len(users)),
                          "rmse": float(rmse(y_true, y_pred)),
                          "mae": float(mae(y_true, y_pred))}
        print(f"  {label}: users={len(users)}  n={len(sub)}  RMSE={results[label]['rmse']:.4f}")
    return results

def main():
    R_train, test, info = load()
    print(f"Loaded: {info['n_train']} train / {info['n_test']} test\n")

    print("=" * 60)
    print("Model 1: User-based CF (K=30)")
    model = UserCF(k_neighbors=30).fit(R_train)
    m = evaluate(model, R_train, test)
    m["cold_start"] = cold_start_breakdown(model, R_train, test)

    with open(f"{OUT}/req2_usercf.json", "w") as f:
        json.dump(m, f, indent=2)
    print(f"\n✅ saved -> {OUT}/req2_usercf.json")
    print("\n" + "=" * 60)
    print("Model 2: Matrix Factorization (k=16, SGD)")
    # validation pairs for early-stopping monitoring (use 10% of train)
    import numpy as np
    coo = R_train.tocoo()
    n_val = int(0.1 * len(coo.data))
    rng = np.random.default_rng(42)
    val_idx = rng.choice(len(coo.data), n_val, replace=False)
    val_pairs = (coo.row[val_idx].astype(np.int32),
                 coo.col[val_idx].astype(np.int32),
                 coo.data[val_idx].astype(np.float32))

    mf = MatrixFactorization(n_factors=16, lr=0.01, reg=0.05,
                             n_epochs=30, seed=42)
    mf.fit(R_train, val_pairs=val_pairs, verbose=True)
    m2 = evaluate(mf, R_train, test)
    m2["cold_start"] = cold_start_breakdown(mf, R_train, test)

    with open(f"{OUT}/req2_mf.json", "w") as f:
        json.dump(m2, f, indent=2)
    print(f"\n✅ saved -> {OUT}/req2_mf.json")

if __name__ == "__main__":
    main()