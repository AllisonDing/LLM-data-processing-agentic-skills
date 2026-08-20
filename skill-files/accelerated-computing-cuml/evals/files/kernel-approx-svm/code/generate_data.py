# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a large nonlinear classification dataset by blending geometric shapes."""

import numpy as np
from sklearn.datasets import make_circles, make_moons

SEED = 33
N_SAMPLES = 80_000
N_NOISY_FEATURES = 30


def generate():
    rng = np.random.default_rng(SEED)
    n_third = N_SAMPLES // 3

    # three non-linear sub-problems stitched together
    X_moons, y_moons = make_moons(n_samples=n_third, noise=0.18, random_state=SEED)
    X_circles, y_circles = make_circles(n_samples=n_third, noise=0.12,
                                         factor=0.4, random_state=SEED + 1)
    y_circles += 2  # classes 2, 3

    # Gaussian blobs with overlap for classes 4, 5
    n_rest = N_SAMPLES - 2 * n_third
    centers = np.array([[3.0, 3.0], [4.5, 3.0], [3.7, 4.5]])
    labels = rng.integers(0, 3, size=n_rest)
    X_blobs = centers[labels] + rng.normal(0, 0.6, (n_rest, 2))
    y_blobs = labels + 4  # classes 4, 5, 6

    X = np.vstack([X_moons, X_circles, X_blobs])
    y = np.concatenate([y_moons, y_circles, y_blobs])

    # add noisy features so the problem isn't trivially 2-D
    noise_feats = rng.normal(0, 1, (X.shape[0], N_NOISY_FEATURES))
    # embed some weak signal in a few noise columns
    noise_feats[:, 0] += 0.3 * X[:, 0]
    noise_feats[:, 3] += 0.2 * X[:, 1]
    X = np.hstack([X, noise_feats])

    # shuffle
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes")
    return X, y


if __name__ == "__main__":
    X, y = generate()
