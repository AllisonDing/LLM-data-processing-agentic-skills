# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""t-SNE visualization of multi-class data.

Generates a multi-class dataset, applies PCA for initial reduction,
then runs t-SNE at multiple perplexity values to produce 2D embeddings.
Reports per-class cluster statistics in the embedding space.
"""

import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def generate_data():
    X, y = make_classification(
        n_samples=6_000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        n_classes=6,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42,
    )
    return X.astype(np.float64), y


def preprocess(X):
    """Scale and PCA pre-reduce before t-SNE."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=30, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA pre-reduction: {X_scaled.shape[1]} -> {X_pca.shape[1]} dims "
          f"(explained: {pca.explained_variance_ratio_.sum():.4f})")
    return X_pca


def run_tsne(X, perplexity=30, n_iter=1000, learning_rate="auto"):
    """Run t-SNE and return 2D embedding."""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate=learning_rate,
        method="barnes_hut",
        metric="euclidean",
        init="pca",
        random_state=42,
        n_jobs=2,
    )

    t0 = time.perf_counter()
    X_2d = tsne.fit_transform(X)
    elapsed = time.perf_counter() - t0

    print(f"  t-SNE (perplexity={perplexity}): {elapsed:.2f}s, "
          f"KL divergence={tsne.kl_divergence_:.4f}")
    return X_2d, tsne.kl_divergence_


def analyze_embedding(X_2d, y, perplexity):
    """Analyze class separation in the 2D embedding."""
    classes = np.unique(y)
    print(f"\n  Embedding analysis (perplexity={perplexity}):")

    centroids = []
    spreads = []
    for c in classes:
        mask = y == c
        points = X_2d[mask]
        centroid = points.mean(axis=0)
        spread = points.std(axis=0).mean()
        centroids.append(centroid)
        spreads.append(spread)
        print(f"    Class {c}: n={mask.sum()}, centroid=({centroid[0]:.2f}, "
              f"{centroid[1]:.2f}), spread={spread:.2f}")

    centroids = np.array(centroids)

    # inter-class distances
    from scipy.spatial.distance import pdist
    inter_dists = pdist(centroids)
    print(f"  Inter-class centroid distances: "
          f"min={inter_dists.min():.2f}, max={inter_dists.max():.2f}, "
          f"mean={inter_dists.mean():.2f}")

    # separation ratio: mean inter-class / mean intra-class
    mean_spread = np.mean(spreads)
    separation = inter_dists.mean() / mean_spread if mean_spread > 0 else 0
    print(f"  Separation ratio (inter/intra): {separation:.2f}")

    return centroids


def perplexity_comparison(X_pca, y):
    """Compare t-SNE at multiple perplexity settings."""
    print("\n--- Perplexity Comparison ---")
    perplexities = [5, 15, 30, 50, 100]
    results = {}

    for p in perplexities:
        X_2d, kl = run_tsne(X_pca, perplexity=p, n_iter=1000)
        centroids = analyze_embedding(X_2d, y, p)
        results[p] = {
            "embedding": X_2d,
            "kl_divergence": kl,
            "centroids": centroids,
        }

    return results


def main():
    print("=== t-SNE Visualization ===\n")

    X, y = generate_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes")
    print(f"Class distribution: {np.bincount(y)}\n")

    print("Preprocessing...")
    X_pca = preprocess(X)

    print("\nDefault t-SNE (perplexity=30):")
    X_2d_default, kl_default = run_tsne(X_pca, perplexity=30, n_iter=1000)
    analyze_embedding(X_2d_default, y, perplexity=30)

    results = perplexity_comparison(X_pca, y)

    # summary
    print("\n--- Summary ---")
    for p, r in sorted(results.items()):
        print(f"  perplexity={p:3d}: KL={r['kl_divergence']:.4f}")

    # output embedding coordinates for external plotting
    print(f"\nFinal embedding shape: {X_2d_default.shape}")
    print(f"  X range: [{X_2d_default[:, 0].min():.2f}, "
          f"{X_2d_default[:, 0].max():.2f}]")
    print(f"  Y range: [{X_2d_default[:, 1].min():.2f}, "
          f"{X_2d_default[:, 1].max():.2f}]")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
