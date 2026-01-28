# ==========================================
# Model Lead
# train_model_long.py
# Qëllimi: Trajnimi i modelit SVD (user_id, item_id, rating),
# split train/test, vlerësim RMSE, dhe ruajtja e modelit + mapping.
# ==========================================
print("started")

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    # ------------------------------------------
    # Script: model_lead/src/train_model_long.py
    # Data:   Data_Preprocessing_Lead_Project/data/processed/ratings.csv
    # ------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ratings_path = os.path.abspath(
        os.path.join(
            script_dir,
            "..",
            "..",
            "Data_Preprocessing_Lead_Project",
            "data",
            "processed",
            "ratings.csv"
        )
    )

    model_folder = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(model_folder, exist_ok=True)

    model_path = os.path.join(model_folder, "svd_model_long.pkl")
    metrics_path = os.path.join(model_folder, "metrics_long.json")

    print("Duke ngarkuar ratings (long format)...")
    print("Rruga:", ratings_path)

    if not os.path.exists(ratings_path):
        print("Gabim: ratings.csv nuk u gjet.")
        sys.exit(1)

    # ------------------------------------------
    # Load long-format data
    # ------------------------------------------
    df = pd.read_csv(ratings_path)

    # ratings.csv has movie_id, not item_id
    required = {"user_id", "movie_id", "rating"}
    if not required.issubset(df.columns):
        print("Gabim: ratings.csv duhet të ketë kolonat:", required)
        print("Kolonat që u gjetën:", list(df.columns))
        sys.exit(1)

    # Rename movie_id -> item_id
    df = df[["user_id", "movie_id", "rating"]].copy()
    df = df.rename(columns={"movie_id": "item_id"})

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    print("Ratings loaded:", len(df), "rreshta")

    # ------------------------------------------
    # Train/Test split
    # ------------------------------------------
    rng = np.random.RandomState(42)
    test_ratio = 0.2

    idx = np.arange(len(df))
    rng.shuffle(idx)

    n_test = int(len(df) * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Split: train={len(train_df)} test={len(test_df)}")

    # ------------------------------------------
    # Build mappings from TRAIN only
    # ------------------------------------------
    users = sorted(train_df["user_id"].unique())
    items = sorted(train_df["item_id"].unique())

    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {it: j for j, it in enumerate(items)}

    n_users = len(users)
    n_items = len(items)

    if n_users < 2 or n_items < 2:
        print("Gabim: shume pak users/items per SVD.")
        sys.exit(1)

    # ------------------------------------------
    # Krijo user-item matrix (trajnim)
    # ------------------------------------------
    R_train = np.zeros((n_users, n_items), dtype=np.float64)

    grouped = train_df.groupby(["user_id", "item_id"], as_index=False)["rating"].mean()

    for row in grouped.itertuples(index=False):
        u = row.user_id
        it = row.item_id
        r = float(row.rating)
        R_train[user_to_idx[u], item_to_idx[it]] = r

    # ------------------------------------------
    # Trajnim SVD
    # ------------------------------------------
    n_components = min(50, min(R_train.shape) - 1)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(R_train)

    print(f"SVD trained (n_components={n_components})")

    # ------------------------------------------
    # Parashikimet
    # ------------------------------------------
    U = svd.transform(R_train)
    Sigma = svd.singular_values_
    Vt = svd.components_

    R_hat = (U * Sigma) @ Vt

    r_min = float(df["rating"].min())
    r_max = float(df["rating"].max())
    R_hat = np.clip(R_hat, r_min, r_max)

    # ------------------------------------------
    # RMSE
    # ------------------------------------------
    y_true, y_pred = [], []
    skipped = 0

    for row in test_df.itertuples(index=False):
        u = row.user_id
        it = row.item_id
        r = float(row.rating)

        if u not in user_to_idx or it not in item_to_idx:
            skipped += 1
            continue

        pred = float(R_hat[user_to_idx[u], item_to_idx[it]])
        y_true.append(r)
        y_pred.append(pred)

    if len(y_true) == 0:
        print("Gabim: asnje test-shembull nuk u vleresua.")
        sys.exit(1)

    score_rmse = rmse(np.array(y_true), np.array(y_pred))

    print(f"RMSE (test) = {score_rmse:.4f}")
    print(f"Evaluated: {len(y_true)} | Skipped (cold start): {skipped}")

    # ------------------------------------------
    # Ruajtja e modelit + mapping
    # ------------------------------------------
    artifact = {
        "svd_model": svd,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": users,
        "idx_to_item": items,
        "n_components": n_components,
        "rating_min": r_min,
        "rating_max": r_max,
    }

    joblib.dump(artifact, model_path)
    print("Model saved to:", model_path)

    metrics = {
        "rmse_test": score_rmse,
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "n_components": n_components,
        "skipped_cold_start": skipped,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()
