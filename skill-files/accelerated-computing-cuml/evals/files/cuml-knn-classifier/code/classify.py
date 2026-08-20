# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KNN classification with grid search over neighbors and metrics.

Generates a multi-class dataset, standardizes features, then searches
over k values and distance metrics for the best KNN classifier.
Reports per-configuration accuracy and a final classification report.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def generate_data():
    X, y = make_classification(
        n_samples=12_000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_classes=5,
        n_clusters_per_class=2,
        random_state=42,
    )
    return X.astype(np.float64), y


def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s


def grid_search_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_jobs=2)

    param_grid = {
        "n_neighbors": [3, 5, 9, 15, 25],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "p": [1, 2],
    }

    gs = GridSearchCV(
        knn,
        param_grid,
        cv=4,
        scoring="accuracy",
        refit=True,
        n_jobs=2,
    )
    gs.fit(X_train, y_train)

    print(f"Best params: {gs.best_params_}")
    print(f"Best CV accuracy: {gs.best_score_:.4f}")

    # show top 5 configurations
    results = gs.cv_results_
    top_idx = np.argsort(results["mean_test_score"])[::-1][:5]
    print("\nTop 5 configurations:")
    for rank, idx in enumerate(top_idx, 1):
        params = results["params"][idx]
        mean = results["mean_test_score"][idx]
        std = results["std_test_score"][idx]
        print(f"  {rank}. accuracy={mean:.4f} +/- {std:.4f}  {params}")

    return gs.best_estimator_


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix diagonal (correct per class):")
    print(f"  {np.diag(cm)}")
    return acc


def main():
    print("=== KNN Classification with Grid Search ===\n")

    X, y = generate_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    print("Preprocessing...")
    X_train_s, X_test_s = preprocess(X_train, X_test)

    print("\nGrid search over KNN parameters...")
    best_knn = grid_search_knn(X_train_s, y_train)

    print("\nEvaluation on test set:")
    evaluate(best_knn, X_test_s, y_test)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
