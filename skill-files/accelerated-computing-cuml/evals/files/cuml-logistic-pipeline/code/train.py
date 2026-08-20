# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logistic regression in a StandardScaler pipeline with cross-validation.

Generates a binary classification dataset, builds a sklearn Pipeline
(StandardScaler -> LogisticRegression), evaluates with stratified
cross-validation, then trains on full training set and reports metrics.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def generate_data():
    X, y = make_classification(
        n_samples=15_000,
        n_features=40,
        n_informative=25,
        n_redundant=8,
        n_classes=2,
        flip_y=0.05,
        random_state=42,
    )
    return X.astype(np.float64), y


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            n_jobs=2,
            random_state=42,
        )),
    ])


def cross_validate(pipeline, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=2)
    print(f"5-fold CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"  Per-fold scores: {np.round(scores, 4)}")
    return scores


def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["class_0", "class_1"]))

    # inspect coefficients
    coef = pipeline.named_steps["classifier"].coef_
    print(f"Coefficient shape: {coef.shape}")
    top_features = np.argsort(np.abs(coef[0]))[::-1][:5]
    print(f"Top 5 feature indices by |coef|: {top_features}")
    return acc


def main():
    print("=== Logistic Regression Pipeline ===\n")

    X, y = generate_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    pipeline = build_pipeline()

    print("Cross-validation:")
    cross_validate(pipeline, X_train, y_train)

    print("\nFinal training and evaluation:")
    train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
