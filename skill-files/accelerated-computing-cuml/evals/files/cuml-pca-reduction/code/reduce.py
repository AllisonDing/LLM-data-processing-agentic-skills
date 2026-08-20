# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PCA dimensionality reduction with variance analysis and reconstruction.

Generates a high-dimensional dataset, applies PCA at multiple component
counts, reports explained variance (scree plot data), cumulative variance
thresholds, and measures reconstruction error at each level.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def generate_data():
    X, y = make_classification(
        n_samples=10_000,
        n_features=80,
        n_informative=40,
        n_redundant=20,
        n_classes=4,
        n_clusters_per_class=2,
        random_state=42,
    )
    return X.astype(np.float64), y


def scree_analysis(X_scaled):
    """Full PCA to get all component variances for scree plot."""
    pca_full = PCA(n_components=None, svd_solver="full", random_state=42)
    pca_full.fit(X_scaled)

    ratios = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(ratios)

    print("Scree plot data (top 15 components):")
    for i in range(min(15, len(ratios))):
        bar = "#" * int(ratios[i] * 200)
        print(f"  PC{i+1:2d}: {ratios[i]:.4f}  cumulative: {cumulative[i]:.4f}  {bar}")

    # find thresholds
    for threshold in [0.80, 0.90, 0.95, 0.99]:
        n = np.searchsorted(cumulative, threshold) + 1
        print(f"  Components for {threshold:.0%} variance: {n}")

    return ratios, cumulative


def reduce_and_reconstruct(X_scaled, n_components_list):
    """Reduce to various component counts and measure reconstruction error."""
    print(f"\nReconstruction error at different component counts:")
    results = {}

    for n_comp in n_components_list:
        pca = PCA(n_components=n_comp, svd_solver="full", random_state=42)
        X_reduced = pca.fit_transform(X_scaled)
        X_reconstructed = pca.inverse_transform(X_reduced)

        mse = np.mean((X_scaled - X_reconstructed) ** 2)
        explained = pca.explained_variance_ratio_.sum()

        print(f"  n_components={n_comp:3d}: "
              f"MSE={mse:.6f}, explained_var={explained:.4f}, "
              f"reduced_shape={X_reduced.shape}")

        results[n_comp] = {
            "mse": mse,
            "explained_variance": explained,
            "reduced_shape": X_reduced.shape,
        }

    return results


def project_and_separate(X_scaled, y, n_components=3):
    """Project to low dims and check class separability."""
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    X_proj = pca.fit_transform(X_scaled)

    print(f"\nProjection to {n_components}D (explained: "
          f"{pca.explained_variance_ratio_.sum():.4f}):")

    classes = np.unique(y)
    for c in classes:
        mask = y == c
        centroid = X_proj[mask].mean(axis=0)
        spread = X_proj[mask].std(axis=0).mean()
        print(f"  Class {c}: centroid={np.round(centroid, 3)}, avg_spread={spread:.3f}")

    return X_proj


def main():
    print("=== PCA Dimensionality Reduction ===\n")

    X, y = generate_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("--- Scree Analysis ---")
    ratios, cumulative = scree_analysis(X_scaled)

    print("\n--- Reconstruction at Various Levels ---")
    n_components_list = [5, 10, 20, 30, 40, 60]
    results = reduce_and_reconstruct(X_scaled, n_components_list)

    print("\n--- Low-Dimensional Projection ---")
    X_proj = project_and_separate(X_scaled, y, n_components=3)

    # whiten transform
    pca_whiten = PCA(n_components=20, whiten=True, random_state=42)
    X_white = pca_whiten.fit_transform(X_scaled)
    print(f"\nWhitened transform: shape={X_white.shape}, "
          f"feature std (should be ~1): {X_white.std(axis=0)[:5].round(3)}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
