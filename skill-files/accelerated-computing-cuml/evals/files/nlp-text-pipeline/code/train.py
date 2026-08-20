# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text classification pipeline using TF-IDF and Logistic Regression.

Loads the 20 Newsgroups dataset, vectorizes with TF-IDF, trains a
LogisticRegression model inside a sklearn Pipeline, evaluates with
cross-validation, and prints a classification report on a held-out set.
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline


CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "sci.med",
    "soc.religion.christian",
    "talk.politics.guns",
    "rec.sport.baseball",
]


def load_data():
    dataset = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
        shuffle=True,
        random_state=42,
    )
    return dataset.data, dataset.target, dataset.target_names


def build_pipeline():
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        sublinear_tf=True,
        dtype=np.float64,
    )
    classifier = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        n_jobs=2,
    )
    return make_pipeline(vectorizer, classifier)


def evaluate_cross_val(pipeline, X, y):
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy", n_jobs=2)
    print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def train_and_report(pipeline, X_train, X_test, y_train, y_test, target_names):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))


def main():
    texts, labels, target_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = build_pipeline()

    evaluate_cross_val(pipeline, X_train, y_train)
    train_and_report(pipeline, X_train, X_test, y_train, y_test, target_names)


if __name__ == "__main__":
    main()
