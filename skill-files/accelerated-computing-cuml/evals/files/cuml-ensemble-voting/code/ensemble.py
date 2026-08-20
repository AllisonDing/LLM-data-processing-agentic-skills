# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voting ensemble of RF, LogReg, and KNN.

Generates a multi-class classification dataset, preprocesses with
StandardScaler and PCA, trains three individual models, combines them
into a VotingClassifier ensemble, and compares individual vs ensemble
performance.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def generate_data():
    X, y = make_classification(
        n_samples=12_000,
        n_features=45,
        n_informative=30,
        n_redundant=8,
        n_classes=5,
        n_clusters_per_class=2,
        class_sep=1.0,
        random_state=42,
    )
    return X.astype(np.float64), y


def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=25, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    print(f"PCA: {X_train_s.shape[1]} -> {X_train_p.shape[1]} dims "
          f"(explained: {pca.explained_variance_ratio_.sum():.4f})")
    return X_train_p, X_test_p


def build_models():
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=2,
        ),
        "logistic_regression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            multi_class="multinomial",
            n_jobs=2,
            random_state=42,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=9,
            weights="distance",
            metric="euclidean",
            n_jobs=2,
        ),
    }
    return models


def evaluate_individual(models, X_train, X_test, y_train, y_test):
    """Train and evaluate each model individually."""
    results = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                     scoring="accuracy", n_jobs=2)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "predictions": y_pred,
        }

        print(f"    Test accuracy: {acc:.4f}")
        print(f"    CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return results


def build_and_evaluate_ensemble(models, X_train, X_test, y_train, y_test):
    """Build VotingClassifier and evaluate."""
    estimators = [(name, model) for name, model in models.items()]

    # soft voting (uses predict_proba)
    ensemble_soft = VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=2,
    )
    ensemble_soft.fit(X_train, y_train)
    y_pred_soft = ensemble_soft.predict(X_test)
    acc_soft = accuracy_score(y_test, y_pred_soft)

    # hard voting
    ensemble_hard = VotingClassifier(
        estimators=estimators,
        voting="hard",
        n_jobs=2,
    )
    ensemble_hard.fit(X_train, y_train)
    y_pred_hard = ensemble_hard.predict(X_test)
    acc_hard = accuracy_score(y_test, y_pred_hard)

    print(f"\n  Soft voting accuracy: {acc_soft:.4f}")
    print(f"  Hard voting accuracy: {acc_hard:.4f}")

    return {
        "soft": {"model": ensemble_soft, "accuracy": acc_soft,
                 "predictions": y_pred_soft},
        "hard": {"model": ensemble_hard, "accuracy": acc_hard,
                 "predictions": y_pred_hard},
    }


def detailed_report(y_test, individual_results, ensemble_results):
    """Print detailed comparison."""
    print("\n--- Detailed Classification Reports ---")

    for name, res in individual_results.items():
        print(f"\n[{name}]")
        print(classification_report(y_test, res["predictions"]))

    print("\n[Ensemble (soft voting)]")
    print(classification_report(y_test, ensemble_results["soft"]["predictions"]))

    # comparison summary
    print("\n--- Accuracy Comparison ---")
    for name, res in individual_results.items():
        print(f"  {name:25s}: {res['accuracy']:.4f} "
              f"(CV: {res['cv_mean']:.4f} +/- {res['cv_std']:.4f})")
    print(f"  {'ensemble_soft':25s}: {ensemble_results['soft']['accuracy']:.4f}")
    print(f"  {'ensemble_hard':25s}: {ensemble_results['hard']['accuracy']:.4f}")

    # agreement analysis
    print("\n--- Model Agreement ---")
    preds = np.column_stack([
        res["predictions"] for res in individual_results.values()
    ])
    agreement = np.mean(np.all(preds == preds[:, [0]], axis=1))
    print(f"  All models agree: {agreement:.4f}")

    for i, (n1, r1) in enumerate(individual_results.items()):
        for j, (n2, r2) in enumerate(individual_results.items()):
            if j > i:
                pair_agree = np.mean(r1["predictions"] == r2["predictions"])
                print(f"  {n1} vs {n2}: {pair_agree:.4f}")


def main():
    print("=== Voting Ensemble Comparison ===\n")

    X, y = generate_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes")
    print(f"Class distribution: {np.bincount(y)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    print("Preprocessing...")
    X_train_p, X_test_p = preprocess(X_train, X_test)

    models = build_models()

    print("\n--- Individual Models ---")
    individual_results = evaluate_individual(
        models, X_train_p, X_test_p, y_train, y_test,
    )

    print("\n--- Ensemble (VotingClassifier) ---")
    ensemble_results = build_and_evaluate_ensemble(
        models, X_train_p, X_test_p, y_train, y_test,
    )

    detailed_report(y_test, individual_results, ensemble_results)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
