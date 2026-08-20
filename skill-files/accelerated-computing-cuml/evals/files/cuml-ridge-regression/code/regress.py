# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ridge regression with polynomial features and alpha tuning.

Generates a nonlinear regression dataset, creates polynomial interaction
features, standardizes, then tunes Ridge regression alpha via manual
cross-validation loop. Reports RMSE, MAE, and R-squared.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def generate_data():
    X, y, coef = make_regression(
        n_samples=10_000,
        n_features=15,
        n_informative=10,
        noise=20.0,
        coef=True,
        random_state=42,
    )
    # add some nonlinearity
    y += 0.5 * np.sum(X[:, :5] ** 2, axis=1)
    return X.astype(np.float64), y.astype(np.float64)


def create_polynomial_features(X_train, X_test, degree=2):
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    print(f"Polynomial features: {X_train.shape[1]} -> {X_train_poly.shape[1]} features")
    return X_train_poly, X_test_poly


def standardize(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s


def tune_alpha(X_train, y_train, alphas):
    """Manual cross-validation over alpha values."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for alpha in alphas:
        fold_rmses = []
        fold_r2s = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = Ridge(alpha=alpha, fit_intercept=True, solver="auto")
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            fold_rmses.append(rmse)
            fold_r2s.append(r2)

        mean_rmse = np.mean(fold_rmses)
        mean_r2 = np.mean(fold_r2s)
        results[alpha] = {"rmse": mean_rmse, "r2": mean_r2}
        print(f"  alpha={alpha:8.4f}: RMSE={mean_rmse:.4f}, R2={mean_r2:.4f}")

    return results


def final_evaluation(X_train, X_test, y_train, y_test, best_alpha):
    """Train on full training set with best alpha and evaluate."""
    model = Ridge(alpha=best_alpha, fit_intercept=True, solver="auto")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nFinal model (alpha={best_alpha}):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")

    # coefficient analysis
    coef = model.coef_
    print(f"  Number of coefficients: {len(coef)}")
    print(f"  Top 5 |coef|: {np.sort(np.abs(coef))[::-1][:5].round(4)}")
    print(f"  Intercept: {model.intercept_:.4f}")

    # residual stats
    residuals = y_test - y_pred
    print(f"  Residual mean: {residuals.mean():.4f}")
    print(f"  Residual std:  {residuals.std():.4f}")

    return model


def main():
    print("=== Ridge Regression with Polynomial Features ===\n")

    X, y = generate_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target range: [{y.min():.1f}, {y.max():.1f}]\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    print("Creating polynomial features...")
    X_train_poly, X_test_poly = create_polynomial_features(X_train, X_test)

    print("Standardizing...")
    X_train_s, X_test_s = standardize(X_train_poly, X_test_poly)

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    print(f"\nTuning alpha ({len(alphas)} values, 5-fold CV):")
    results = tune_alpha(X_train_s, y_train, alphas)

    best_alpha = min(results.keys(), key=lambda a: results[a]["rmse"])
    print(f"\nBest alpha: {best_alpha}")

    final_evaluation(X_train_s, X_test_s, y_train, y_test, best_alpha)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
