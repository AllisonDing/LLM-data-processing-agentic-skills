# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small scikit-learn SVM grid with kernels that may not all map to GPU."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


@dataclass
class SVMRun:
    kernel: str
    c_value: float
    gamma: str | float = "scale"


@dataclass
class ExperimentData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass
class ReferenceResult:
    accuracy: float
    predictions: np.ndarray
    y_true: np.ndarray
    decision: np.ndarray | None


def make_experiment_data(n_samples: int = 20_000) -> ExperimentData:
    x, y = make_classification(
        n_samples=n_samples,
        n_features=32,
        n_informative=12,
        random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    return ExperimentData(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def sklearn_reference(config: SVMRun, data: ExperimentData) -> ReferenceResult:
    model = SVC(kernel=config.kernel, C=config.c_value, gamma=config.gamma)
    model.fit(data.x_train, data.y_train)
    predictions = model.predict(data.x_test)
    decision = None
    if hasattr(model, "decision_function"):
        decision = model.decision_function(data.x_test)
    return ReferenceResult(
        accuracy=float(accuracy_score(data.y_test, predictions)),
        predictions=np.asarray(predictions),
        y_true=np.asarray(data.y_test),
        decision=None if decision is None else np.asarray(decision),
    )


def _as_numpy(values) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        return np.asarray(values.to_numpy())
    try:
        import cupy as cp

        if isinstance(values, cp.ndarray):
            return cp.asnumpy(values)
    except Exception:
        pass
    return np.asarray(values)


def parity_report(
    reference: ReferenceResult,
    candidate_predictions,
    *,
    candidate_decision=None,
    accuracy_tolerance: float = 0.03,
    decision_atol: float = 0.25,
) -> dict[str, float | bool | None]:
    """Compare an accelerated candidate against the sklearn reference.

    SVM implementations can use different solvers and float precision, so this
    helper checks the behavior users observe: accuracy, predicted labels, and
    decision scores when both paths expose them. It accepts NumPy, CuPy, cuDF,
    or pandas-like outputs.
    """

    predictions = _as_numpy(candidate_predictions)
    candidate_accuracy = float(accuracy_score(reference.y_true, predictions))
    label_agreement = float(np.mean(predictions == reference.predictions))
    accuracy_delta = abs(candidate_accuracy - reference.accuracy)

    max_decision_delta = None
    decision_within_tolerance = None
    if reference.decision is not None and candidate_decision is not None:
        decision = _as_numpy(candidate_decision)
        if decision.shape == reference.decision.shape:
            max_decision_delta = float(np.max(np.abs(decision - reference.decision)))
            decision_within_tolerance = max_decision_delta <= decision_atol

    return {
        "reference_accuracy": reference.accuracy,
        "candidate_accuracy": candidate_accuracy,
        "candidate_label_agreement": label_agreement,
        "accuracy_delta": accuracy_delta,
        "accuracy_within_tolerance": accuracy_delta <= accuracy_tolerance,
        "max_decision_delta": max_decision_delta,
        "decision_within_tolerance": decision_within_tolerance,
    }


def run_experiment(config: SVMRun, n_samples: int = 20_000) -> float:
    data = make_experiment_data(n_samples)
    return sklearn_reference(config, data).accuracy


def main() -> None:
    configs = [
        SVMRun(kernel="rbf", c_value=1.0),
        SVMRun(kernel="linear", c_value=0.5),
        SVMRun(kernel="poly", c_value=1.0, gamma="scale"),
    ]
    for config in configs:
        print(config, f"accuracy={run_experiment(config):.4f}")


if __name__ == "__main__":
    main()
