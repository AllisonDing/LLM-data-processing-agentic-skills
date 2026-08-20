# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Large-scale nonlinear classification: exact SVM vs kernel approximations.

Compares four strategies on a big nonlinear dataset:
  1. Exact SVC (RBF kernel) on a subset
  2. Nystroem approximation + LinearSVC
  3. RBFSampler approximation + LogisticRegression
  4. RandomForest baseline
Each approach is timed end-to-end and evaluated with cross-validation.
"""

import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from generate_data import generate

SUBSET_SIZE = 10_000  # exact SVC is O(n^2-3); cap it


def _timed(fn, label):
    """Run fn(), print elapsed wall-clock time, return result."""
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] {elapsed:.2f}s")
    return result, elapsed


def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=20, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    print(f"  PCA {X_train_s.shape[1]} -> {X_train_p.shape[1]} features, "
          f"explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    return X_train_p, X_test_p


def exact_svc(X_train, y_train, X_test, y_test):
    """Exact RBF SVC on a subset (too slow for full dataset)."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), size=min(SUBSET_SIZE, len(X_train)), replace=False)
    X_sub, y_sub = X_train[idx], y_train[idx]

    svc = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)

    def _fit_predict():
        svc.fit(X_sub, y_sub)
        return svc.predict(X_test)

    y_pred, elapsed = _timed(_fit_predict, "Exact SVC fit+predict")

    cv = cross_val_score(svc, X_sub, y_sub, cv=3, scoring="accuracy")
    print(f"  3-fold CV accuracy (subset): {cv.mean():.4f} +/- {cv.std():.4f}")
    return y_pred, elapsed


def nystroem_linear_svc(X_train, y_train, X_test, y_test):
    """Nystroem kernel approximation piped into LinearSVC."""
    pipe = Pipeline([
        ("nystroem", Nystroem(kernel="rbf", gamma="scale",
                               n_components=500, random_state=42)),
        ("svc", LinearSVC(C=1.0, max_iter=5000, random_state=42)),
    ])

    def _fit_predict():
        pipe.fit(X_train, y_train)
        return pipe.predict(X_test)

    y_pred, elapsed = _timed(_fit_predict, "Nystroem+LinearSVC fit+predict")

    cv = cross_val_score(pipe, X_train, y_train, cv=3, scoring="accuracy",
                         n_jobs=2)
    print(f"  3-fold CV accuracy: {cv.mean():.4f} +/- {cv.std():.4f}")
    return y_pred, elapsed


def rbfsampler_logreg(X_train, y_train, X_test, y_test):
    """RBFSampler approximation piped into LogisticRegression."""
    pipe = Pipeline([
        ("rbf", RBFSampler(gamma="scale", n_components=800, random_state=42)),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                   n_jobs=2)),
    ])

    def _fit_predict():
        pipe.fit(X_train, y_train)
        return pipe.predict(X_test)

    y_pred, elapsed = _timed(_fit_predict, "RBFSampler+LR fit+predict")

    cv = cross_val_score(pipe, X_train, y_train, cv=3, scoring="accuracy",
                         n_jobs=2)
    print(f"  3-fold CV accuracy: {cv.mean():.4f} +/- {cv.std():.4f}")
    return y_pred, elapsed


def random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                 random_state=42, n_jobs=2)

    def _fit_predict():
        rf.fit(X_train, y_train)
        return rf.predict(X_test)

    y_pred, elapsed = _timed(_fit_predict, "RandomForest fit+predict")

    cv = cross_val_score(rf, X_train, y_train, cv=3, scoring="accuracy",
                         n_jobs=2)
    print(f"  3-fold CV accuracy: {cv.mean():.4f} +/- {cv.std():.4f}")
    return y_pred, elapsed


def report(y_test, y_pred, label):
    print(f"\n--- {label} ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix diagonal (correct per class): {np.diag(cm)}")


def main():
    print("=== Large-Scale Nonlinear Classification ===\n")
    X, y = generate()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}\n")

    print("Preprocessing...")
    X_train_p, X_test_p = preprocess(X_train, X_test)

    timings = {}

    print("\n[1] Exact SVC (RBF, subset)...")
    pred_svc, timings["exact_svc"] = exact_svc(X_train_p, y_train, X_test_p, y_test)

    print("\n[2] Nystroem + LinearSVC...")
    pred_nys, timings["nystroem_lsvc"] = nystroem_linear_svc(X_train_p, y_train, X_test_p, y_test)

    print("\n[3] RBFSampler + LogisticRegression...")
    pred_rbf, timings["rbfsampler_lr"] = rbfsampler_logreg(X_train_p, y_train, X_test_p, y_test)

    print("\n[4] RandomForest baseline...")
    pred_rf, timings["random_forest"] = random_forest(X_train_p, y_train, X_test_p, y_test)

    for (label, pred) in [("Exact SVC", pred_svc),
                           ("Nystroem+LinearSVC", pred_nys),
                           ("RBFSampler+LR", pred_rbf),
                           ("RandomForest", pred_rf)]:
        report(y_test, pred, label)

    print("\n=== Timing Summary ===")
    for method, t in sorted(timings.items(), key=lambda x: x[1]):
        print(f"  {method:25s} {t:8.2f}s")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
