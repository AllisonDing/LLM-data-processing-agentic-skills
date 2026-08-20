# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DBSCAN clustering with eps tuning and noise analysis.

Generates a multi-blob dataset with varying densities, tunes the eps
parameter for DBSCAN, evaluates using silhouette score, and analyzes
noise detection across configurations.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def generate_data():
    """Create blobs with different densities to challenge DBSCAN."""
    centers = [
        [0, 0], [6, 6], [-6, 6], [6, -6], [-6, -6],
        [0, 10], [10, 0],
    ]
    cluster_stds = [0.8, 1.2, 0.5, 1.0, 0.7, 1.5, 0.9]
    samples_per_cluster = [2000, 1500, 1800, 1200, 2500, 1000, 2000]

    X, y_true = make_blobs(
        n_samples=samples_per_cluster,
        centers=centers,
        cluster_std=cluster_stds,
        random_state=42,
    )

    # add uniform noise points
    rng = np.random.default_rng(42)
    n_noise = 500
    noise = rng.uniform(-12, 16, size=(n_noise, 2))
    X = np.vstack([X, noise])
    y_true = np.concatenate([y_true, np.full(n_noise, -1)])

    perm = rng.permutation(len(y_true))
    return X[perm].astype(np.float64), y_true[perm]


def k_distance_analysis(X_scaled, k=5):
    """Compute k-distance graph to help estimate eps."""
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=2)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)

    k_distances = np.sort(distances[:, -1])[::-1]

    print(f"k-distance statistics (k={k}):")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(k_distances, 100 - pct)
        print(f"  {pct}th percentile: {val:.4f}")

    return k_distances


def tune_eps(X_scaled, eps_values, min_samples=5):
    """Run DBSCAN with different eps values and evaluate."""
    results = {}

    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean",
                    algorithm="auto", n_jobs=2)
        labels = db.fit_predict(X_scaled)

        n_clusters = len(set(labels) - {-1})
        n_noise = np.sum(labels == -1)
        noise_pct = n_noise / len(labels) * 100

        if n_clusters >= 2:
            mask = labels != -1
            sil = silhouette_score(X_scaled[mask], labels[mask])
        else:
            sil = -1.0

        results[eps] = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_pct": noise_pct,
            "silhouette": sil,
            "labels": labels,
        }

        print(f"  eps={eps:.3f}: clusters={n_clusters}, "
              f"noise={n_noise} ({noise_pct:.1f}%), "
              f"silhouette={sil:.4f}")

    return results


def analyze_best(X_scaled, labels, eps):
    """Detailed analysis of the best clustering result."""
    n_clusters = len(set(labels) - {-1})
    print(f"\nBest clustering (eps={eps:.3f}):")
    print(f"  Total clusters: {n_clusters}")

    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        count = np.sum(mask)
        centroid = X_scaled[mask].mean(axis=0)
        spread = X_scaled[mask].std(axis=0).mean()
        label = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
        print(f"  {label}: {count} points, centroid={np.round(centroid, 3)}, "
              f"avg_spread={spread:.3f}")

    # cluster size distribution
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) > 0:
        print(f"\n  Cluster sizes: min={counts.min()}, max={counts.max()}, "
              f"mean={counts.mean():.0f}, std={counts.std():.0f}")


def main():
    print("=== DBSCAN Clustering with Eps Tuning ===\n")

    X, y_true = generate_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True clusters: {len(set(y_true) - {-1})}, "
          f"true noise: {np.sum(y_true == -1)}\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("--- k-Distance Analysis ---")
    k_distances = k_distance_analysis(X_scaled, k=5)

    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]
    print(f"\n--- Eps Tuning ({len(eps_values)} values) ---")
    results = tune_eps(X_scaled, eps_values, min_samples=5)

    # pick best by silhouette (among those with 2+ clusters)
    valid = {e: r for e, r in results.items() if r["n_clusters"] >= 2}
    if valid:
        best_eps = max(valid.keys(), key=lambda e: valid[e]["silhouette"])
        analyze_best(X_scaled, results[best_eps]["labels"], best_eps)
    else:
        print("\nNo valid clustering found with 2+ clusters.")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
