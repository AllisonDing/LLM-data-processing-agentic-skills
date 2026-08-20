# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SVM classification with RBF kernel and C parameter tuning.

Generates a binary classification dataset with nonlinear decision boundary,
standardizes features, tunes the C parameter via cross-validation, and
produces detailed classification metrics including confusion matrix.
"""

import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def generate_data():
    X, y = make_classification(
        n_samples=8_000,
        n_features=25,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        class_sep=0.8,
        flip_y=0.05,
        random_state=42,
    )
    return X.astype(np.float64), y


def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s


def tune_c_parameter(X_train, y_train, c_values):
    """Cross-validate SVC across different C values."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for C in c_values:
        fold_accs = []
        fold_aucs = []
        t0 = time.perf_counter()

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            svc = SVC(
                C=C,
                kernel="rbf",
                gamma="scale",
                probability=True,
                cache_size=500,
                random_state=42,
            )
            svc.fit(X_tr, y_tr)

            y_pred = svc.predict(X_val)
            y_proba = svc.predict_proba(X_val)[:, 1]

            fold_accs.append(accuracy_score(y_val, y_pred))
            fold_aucs.append(roc_auc_score(y_val, y_proba))

        elapsed = time.perf_counter() - t0
        mean_acc = np.mean(fold_accs)
        mean_auc = np.mean(fold_aucs)

        results[C] = {"accuracy": mean_acc, "auc": mean_auc, "time": elapsed}
        print(f"  C={C:8.4f}: acc={mean_acc:.4f}, AUC={mean_auc:.4f}, "
              f"time={elapsed:.2f}s")

    return results


def final_evaluation(X_train, X_test, y_train, y_test, best_C):
    """Train final model and produce detailed metrics."""
    svc = SVC(
        C=best_C,
        kernel="rbf",
        gamma="scale",
        probability=True,
        cache_size=500,
        random_state=42,
    )

    t0 = time.perf_counter()
    svc.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = svc.predict(X_test)
    y_proba = svc.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\nFinal SVC (C={best_C}):")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Test AUC:      {auc:.4f}")
    print(f"  Support vectors: {svc.n_support_}")
    print(f"  Total SVs: {svc.n_support_.sum()}")

    print(f"\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    return svc


def main():
    print("=== SVM Classification with C Tuning ===\n")

    X, y = generate_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    print("Preprocessing...")
    X_train_s, X_test_s = preprocess(X_train, X_test)

    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    print(f"\nTuning C ({len(c_values)} values, 5-fold CV):")
    results = tune_c_parameter(X_train_s, y_train, c_values)

    best_C = max(results.keys(), key=lambda c: results[c]["accuracy"])
    print(f"\nBest C: {best_C} (acc={results[best_C]['accuracy']:.4f})")

    final_evaluation(X_train_s, X_test_s, y_train, y_test, best_C)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
