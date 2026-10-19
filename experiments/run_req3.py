# experiments/run_req3.py
"""
Req 3: Social extension evaluation.

Experiments:
  1. Alpha sweep: alpha in {0, 0.2, 0.4, 0.6, 0.8, 1.0}
     - variant = "friends"
     - Report RMSE, MAE, P@10, HR@10, Coverage, ILD
  2. Friends vs Similar-unconnected
     - Fix alpha at best from (1), compare variants
  3. Cold-start breakdown for the best configuration
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.recommenders.social_cf import SocialUserCF
from src.evaluation.metrics import (
    rmse, mae, precision_recall_ndcg_at_k, catalog_coverage,
    intra_list_diversity, evaluate_ranking_with_negatives
)

PROC = "data/processed"
OUT = "results/tables"
os.makedirs(OUT, exist_ok=True)


def load():
    R_train = sp.load_npz(f"{PROC}/R_train.npz")
    A_unweighted = sp.load_npz(f"{PROC}/A_unweighted.npz")
    test = pd.read_csv(f"{PROC}/test.csv")
    return R_train, A_unweighted, test


def quick_eval(model, R_train, test, k=10, rel_thresh=4.0, seed=42):
    """Fast eval: RMSE + negative-sampling ranking metrics + coverage."""
    y_true = test["rating"].values
    y_pred = np.array([model.predict(u, i)[0]
                       for u, i in zip(test["user_id"], test["item_id"])])
    r = {"rmse": float(rmse(y_true, y_pred)),
         "mae": float(mae(y_true, y_pred))}

    rank = evaluate_ranking_with_negatives(
        model, test, R_train.shape[1], k=k,
        n_negatives=100, relevance_threshold=rel_thresh, seed=seed)
    r.update(rank)

    test_rel = test[test.rating >= rel_thresh]
    covered = set()
    divs = []
    for u in test_rel.user_id.unique():
        top, _ = model.recommend_topk(u, k=k, exclude_seen=True)
        covered.update(top.tolist())
        divs.append(intra_list_diversity(R_train, top))
    r["coverage"] = float(catalog_coverage(covered, R_train.shape[1]))
    r["diversity"] = float(np.mean(divs))
    return r


def print_row(label, m):
    print(f"  {label:30s}"
          f"  RMSE={m['rmse']:.4f}"
          f"  P@10={m['precision_at_k']:.4f}"
          f"  HR@10={m['hit_rate_at_k']:.4f}"
          f"  NDCG@10={m['ndcg_at_k']:.4f}"
          f"  Cov={m['coverage']:.3f}")


def main():
    R_train, A, test = load()
    print(f"Loaded: {R_train.shape[0]} users, {R_train.shape[1]} items")

    # --------------------------------------------------------------
    # Experiment 1: alpha sweep (variant = friends)
    # --------------------------------------------------------------
    print("\n" + "=" * 90)
    print("Experiment 1: alpha sweep (variant = 'friends')")
    print("=" * 90)

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sweep_friends = {}
    for a in alphas:
        print(f"\n[alpha = {a}]")
        model = SocialUserCF(k_neighbors=30, alpha=a,
                             variant="friends", social_k=30).fit(R_train, A_social=A)
        m = quick_eval(model, R_train, test)
        sweep_friends[str(a)] = m
        print_row(f"alpha={a}", m)

    with open(f"{OUT}/req3_alpha_friends.json", "w") as f:
        json.dump(sweep_friends, f, indent=2)

    # find best alpha by RMSE
    best_alpha = min(alphas, key=lambda a: sweep_friends[str(a)]["rmse"])
    print(f"\n→ best alpha by RMSE (friends): {best_alpha}")

    # --------------------------------------------------------------
    # Experiment 2: Friends vs Similar-unconnected (at best alpha)
    # --------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"Experiment 2: friends vs similar-but-unconnected (alpha={best_alpha})")
    print("=" * 90)

    comparison = {}
    for variant in ["friends", "similar"]:
        print(f"\n[variant = {variant}]")
        model = SocialUserCF(k_neighbors=30, alpha=best_alpha,
                             variant=variant, social_k=30).fit(R_train, A_social=A)
        m = quick_eval(model, R_train, test)
        comparison[variant] = m
        print_row(variant, m)

    with open(f"{OUT}/req3_friends_vs_similar.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # --------------------------------------------------------------
    # Experiment 3: cold-start breakdown (best config)
    # --------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"Experiment 3: cold-start breakdown (variant=friends, alpha={best_alpha})")
    print("=" * 90)

    model = SocialUserCF(k_neighbors=30, alpha=best_alpha,
                         variant="friends", social_k=30).fit(R_train, A_social=A)
    counts = np.asarray((R_train > 0).sum(axis=1)).ravel()
    q1, q2 = np.percentile(counts, [33, 66])
    cold_report = {}
    for label, cond in [
        (f"cold (<= {int(q1)})", counts <= q1),
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
        cold_report[label] = {"n": int(len(sub)),
                              "n_users": int(len(users)),
                              "rmse": float(rmse(y_true, y_pred)),
                              "mae": float(mae(y_true, y_pred))}
        print(f"  {label}: users={len(users)}  n={len(sub)}"
              f"  RMSE={cold_report[label]['rmse']:.4f}")

    with open(f"{OUT}/req3_cold_start.json", "w") as f:
        json.dump(cold_report, f, indent=2)

    print("\n✅ saved req3 results to results/tables/")


if __name__ == "__main__":
    main()