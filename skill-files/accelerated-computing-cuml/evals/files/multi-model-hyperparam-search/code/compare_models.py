# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-model comparison with hyperparameter tuning.

Generates a classification dataset, preprocesses with StandardScaler + PCA,
tunes five models via GridSearchCV, assembles the top three into a
VotingClassifier, and reports comprehensive metrics.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def generate_dataset():
    X, y = make_classification(
        n_samples=8_000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        n_classes=4,
        n_clusters_per_class=2,
        random_state=42,
    )
    return X, y


def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=20, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)

    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    return X_train_p, X_test_p


MODEL_CONFIGS = {
    "knn": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "n_neighbors": [5, 11, 21],
            "weights": ["uniform", "distance"],
        },
    },
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=2),
        "param_grid": {
            "C": [0.01, 0.1, 1.0, 10.0],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=2),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
        },
    },
    "svc": {
        "estimator": SVC(probability=True, random_state=42),
        "param_grid": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
        },
    },
    "ridge_classifier": {
        "estimator": LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=2000, n_jobs=2
        ),
        "param_grid": {
            "C": [0.01, 0.1, 1.0],
        },
    },
}


def tune_models(X_train, y_train):
    best_models = {}
    for name, config in MODEL_CONFIGS.items():
        print(f"\nTuning {name}...")
        gs = GridSearchCV(
            config["estimator"],
            config["param_grid"],
            cv=3,
            scoring="accuracy",
            refit=True,
        )
        gs.fit(X_train, y_train)
        best_models[name] = {
            "model": gs.best_estimator_,
            "score": gs.best_score_,
            "params": gs.best_params_,
        }
        print(f"  Best CV score: {gs.best_score_:.4f}  params: {gs.best_params_}")
    return best_models


def build_ensemble(best_models, X_train, y_train):
    ranked = sorted(best_models.items(), key=lambda x: x[1]["score"], reverse=True)
    top3 = ranked[:3]
    print(f"\nTop 3 models for ensemble: {[name for name, _ in top3]}")

    estimators = [(name, info["model"]) for name, info in top3]
    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=2)
    ensemble.fit(X_train, y_train)
    return ensemble


def evaluate(model, X_test, y_test, label="Model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{label} — Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)
    return acc


def main():
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_p, X_test_p = preprocess(X_train, X_test)

    best_models = tune_models(X_train_p, y_train)

    cv_scores = cross_val_score(
        best_models["random_forest"]["model"], X_train_p, y_train, cv=5
    )
    print(f"\nRF 5-fold CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    ensemble = build_ensemble(best_models, X_train_p, y_train)

    print("\n--- Individual Best Models ---")
    for name, info in best_models.items():
        evaluate(info["model"], X_test_p, y_test, label=name)

    print("\n--- Ensemble (VotingClassifier) ---")
    evaluate(ensemble, X_test_p, y_test, label="VotingClassifier")


if __name__ == "__main__":
    main()
