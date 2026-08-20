# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unsupervised clustering pipeline: scale, reduce, embed, cluster, evaluate.

Uses StandardScaler + PCA for initial reduction, UMAP for 2-D embedding,
HDBSCAN for density-based clustering, and NearestNeighbors for a KNN
connectivity graph.  Evaluates with silhouette_score.
"""

import numpy as np
from hdbscan import HDBSCAN
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from umap import UMAP


def generate_data(n_samples=30_000, n_features=100, n_classes=8, seed=42):
    X, y_true = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=40,
        n_redundant=20,
        n_classes=n_classes,
        n_clusters_per_class=2,
        random_state=seed,
    )
    return X, y_true


def preprocess(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: 100 -> 50 components, explained variance: {explained:.2%}")
    return X_pca


def embed_umap(X, n_neighbors=15, min_dist=0.1, metric="euclidean"):
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    embedding = reducer.fit_transform(X)
    print(f"UMAP embedding shape: {embedding.shape}")
    return embedding


def cluster_hdbscan(X, min_cluster_size=50, min_samples=10):
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)
    return labels


def build_knn_graph(X, n_neighbors=15):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=2)
    nn.fit(X)
    graph = nn.kneighbors_graph(mode="connectivity")
    return graph


def evaluate_clusters(X, labels):
    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    noise_ratio = n_noise / len(labels)

    print(f"\nClusters found: {n_clusters}")
    print(f"Noise points: {n_noise} ({noise_ratio:.2%})")

    for cid in sorted(set(labels)):
        count = (labels == cid).sum()
        label = f"Cluster {cid}" if cid >= 0 else "Noise (-1)"
        print(f"  {label}: {count} points")

    non_noise_mask = labels >= 0
    if non_noise_mask.sum() > 1 and len(set(labels[non_noise_mask])) > 1:
        score = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
        print(f"\nSilhouette score (excl. noise): {score:.4f}")
    else:
        print("\nSilhouette score: N/A (too few clusters)")


def main():
    X, y_true = generate_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    X_pca = preprocess(X)

    embedding = embed_umap(X_pca)

    labels = cluster_hdbscan(embedding, min_cluster_size=50, min_samples=10)

    knn_graph = build_knn_graph(X_pca, n_neighbors=15)
    print(f"\nKNN graph: {knn_graph.shape}, {knn_graph.nnz} edges")

    evaluate_clusters(embedding, labels)


if __name__ == "__main__":
    main()
