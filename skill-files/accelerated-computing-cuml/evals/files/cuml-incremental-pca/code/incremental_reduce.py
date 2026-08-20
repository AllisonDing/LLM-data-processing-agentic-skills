# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IncrementalPCA on chunked data compared with full PCA.

Generates a large dataset, processes it in batches with IncrementalPCA
via partial_fit, then compares the result with a standard full PCA to
measure how close the incremental approach gets.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import StandardScaler


def generate_data():
    X, y = make_classification(
        n_samples=20_000,
        n_features=60,
        n_informative=35,
        n_redundant=15,
        n_classes=4,
        n_clusters_per_class=2,
        random_state=42,
    )
    return X.astype(np.float64), y


def full_pca(X_scaled, n_components):
    """Full PCA as reference."""
    pca = PCA(n_components=n_components, random_state=42)
    X_full = pca.fit_transform(X_scaled)
    X_recon = pca.inverse_transform(X_full)
    mse = np.mean((X_scaled - X_recon) ** 2)

    print(f"Full PCA ({n_components} components):")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  Reconstruction MSE: {mse:.6f}")
    print(f"  Top 5 variance ratios: "
          f"{np.round(pca.explained_variance_ratio_[:5], 4)}")
    return pca, X_full, mse


def incremental_pca(X_scaled, n_components, batch_size):
    """IncrementalPCA with partial_fit on chunks."""
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    n_samples = X_scaled.shape[0]
    n_batches = 0

    # partial_fit in chunks
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        chunk = X_scaled[start:end]
        ipca.partial_fit(chunk)
        n_batches += 1

    print(f"\nIncrementalPCA ({n_components} components, "
          f"batch_size={batch_size}, {n_batches} batches):")
    print(f"  Explained variance: {ipca.explained_variance_ratio_.sum():.4f}")
    print(f"  Top 5 variance ratios: "
          f"{np.round(ipca.explained_variance_ratio_[:5], 4)}")

    # transform full dataset
    X_inc = ipca.transform(X_scaled)
    X_recon = ipca.inverse_transform(X_inc)
    mse = np.mean((X_scaled - X_recon) ** 2)
    print(f"  Reconstruction MSE: {mse:.6f}")

    return ipca, X_inc, mse


def compare_results(pca_full, ipca, X_full, X_inc, X_scaled):
    """Compare full PCA and IncrementalPCA results."""
    print("\n--- Comparison ---")

    # variance ratio difference
    var_diff = np.abs(
        pca_full.explained_variance_ratio_ - ipca.explained_variance_ratio_
    )
    print(f"Variance ratio diff (mean): {var_diff.mean():.6f}")
    print(f"Variance ratio diff (max):  {var_diff.max():.6f}")

    # component direction similarity (absolute cosine similarity)
    n_comp = min(pca_full.components_.shape[0], ipca.components_.shape[0])
    cosines = []
    for i in range(n_comp):
        c1 = pca_full.components_[i]
        c2 = ipca.components_[i]
        cos = np.abs(np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2)))
        cosines.append(cos)
    cosines = np.array(cosines)
    print(f"Component cosine similarity: mean={cosines.mean():.4f}, "
          f"min={cosines.min():.4f}")

    # projection difference
    # signs may flip, so compare via absolute correlation
    for i in range(min(5, n_comp)):
        corr = np.abs(np.corrcoef(X_full[:, i], X_inc[:, i])[0, 1])
        print(f"  PC{i+1} correlation: {corr:.4f}")


def batch_size_experiment(X_scaled, n_components=15):
    """Try different batch sizes and compare."""
    print("\n--- Batch Size Experiment ---")
    batch_sizes = [500, 1000, 2000, 5000, 10000]

    for bs in batch_sizes:
        ipca = IncrementalPCA(n_components=n_components, batch_size=bs)

        for start in range(0, X_scaled.shape[0], bs):
            end = min(start + bs, X_scaled.shape[0])
            ipca.partial_fit(X_scaled[start:end])

        X_trans = ipca.transform(X_scaled)
        X_recon = ipca.inverse_transform(X_trans)
        mse = np.mean((X_scaled - X_recon) ** 2)

        print(f"  batch_size={bs:5d}: var_explained={ipca.explained_variance_ratio_.sum():.4f}, "
              f"MSE={mse:.6f}")


def main():
    print("=== IncrementalPCA vs Full PCA ===\n")

    X, y = generate_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = 15

    print("--- Full PCA (reference) ---")
    pca_ref, X_full, mse_full = full_pca(X_scaled, n_components)

    batch_size = 2000
    ipca, X_inc, mse_inc = incremental_pca(X_scaled, n_components, batch_size)

    compare_results(pca_ref, ipca, X_full, X_inc, X_scaled)

    batch_size_experiment(X_scaled, n_components)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
