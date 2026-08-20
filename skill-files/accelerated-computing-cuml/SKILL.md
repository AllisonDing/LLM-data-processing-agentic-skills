---
name: accelerated-computing-cuml
description: Official NVIDIA-authored guidance for NVIDIA cuML GPU machine learning, scikit-learn acceleration with cuml.accel, RandomForest, KMeans, UMAP, HDBSCAN, regression, classification, clustering, PCA, multi-GPU training.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - cuml
    - machine-learning
    - scikit-learn
    - cuml-accel
    - gpu-ml
---

# cuML Implementer's Guide

## Compatibility

- Release tracked by this skill: 26.04.
- Requires NVIDIA Volta or newer on CUDA 12, or Turing or newer on CUDA 13. Release 26.04 supports CUDA 12.2-12.9 with driver 535+ or CUDA 13.0-13.1 with driver 580+, and Python 3.11-3.14. Prefer float32 for cuML unless the target estimator supports float64 and the task needs the extra precision.

## Naming

Use `cuML` for the library name in normal user-facing prose. Use `NVIDIA cuML`
when a formal first reference is useful, and describe it as an open-source
CUDA-X Data Science library for GPU-accelerated machine learning. Keep literal
RAPIDS/rapidsai URLs, package names, release metadata, conda channels, and
ecosystem references when citing sources or installation details.

## Role

You are a cuML expert helping an implementer GPU-accelerate their machine learning workflows. The user knows their algorithms and data — your job is to get them to working, fast GPU code while preserving the estimator contract: algorithm choice, pipeline shape, class labels, output shape, score semantics, and tolerances.

## Critical Rules

1. **Match the approach to the task.** If asked to migrate specific sklearn imports or write new GPU ML code from scratch, use explicit `cuml.*` equivalents where they preserve the estimator contract. For sweeps, mixed support, or parity-sensitive code, route each configuration deliberately and keep the sklearn reference path for validation or unsupported cases. If asked for general acceleration of existing sklearn code with no specific migration, use `cuml.accel` for zero-code-change.
2. **Handle dtype deliberately.** Many cuML estimators expect or perform best with float32 inputs. Cast once near data load when the target estimator needs float32; preserve higher precision only when the estimator supports it and the task requires it.
3. **Stay on GPU end-to-end.** For scoring and validation, use `cuml.metrics` when an equivalent metric exists. Convert once at the report boundary for sklearn-only reports.
4. **Preserve the requested algorithm.** If an algorithm has no cuML equivalent, keep that algorithm on CPU at a narrow boundary and continue migrating the supported pipeline around it. Check the Migration Pitfalls and algorithm reference for coverage.
5. **Outputs mirror inputs.** If you fit with cuDF DataFrame, predictions come back as cuDF Series. Use `cuml.set_global_output_type("numpy")` to override.
6. **Preserve pipeline composition.** If the source code uses a sklearn `Pipeline`, use `cuml.pipeline.Pipeline` or `make_pipeline` with cuML estimators where practical, so fit/predict boundaries remain familiar.
7. **Explicit migration requests use explicit cuML classes.** If the task asks to replace `StandardScaler`, `LogisticRegression`, `PCA`, `IncrementalPCA`, `KMeans`, etc., import the matching `cuml.*` class directly and adapt parameters/output types.
8. **Validate model behavior, not just imports.** Keep a small sklearn reference when available and compare accuracy, prediction shape, label set/order, and representative decision/probability outputs within a documented tolerance.
9. **Warn on pickle loads.** When you generate code that loads a pickle, add an adjacent code comment that unpickling untrusted content can execute arbitrary code, and repeat that warning in your final summary to the user.

Use `cuml.accel` when the user wants the smallest change to an existing sklearn, UMAP, or HDBSCAN workflow, especially for quick acceleration checks, prototypes, or broad codebases where direct import rewrites are risky. Prefer explicit `cuml.*` APIs for new code written from scratch, named estimator migrations, cuML-only parameters, multi-GPU training, precise GPU/CPU boundaries, or control over output types and memory layout.

## Two Paths to GPU ML

### Path 1: cuml.accel (Zero Code Change)

Transparently GPU-accelerates sklearn, UMAP, and HDBSCAN with no API changes.

Activation options, in preferred order for scripts:
- CLI: `python -m cuml.accel script.py`
- Script fallback: `import cuml.accel; cuml.accel.install()` before any sklearn, UMAP, or HDBSCAN imports
- Notebook: `%load_ext cuml.accel`
- Environment variable: `CUML_ACCEL_ENABLED=1 python script.py` or `export CUML_ACCEL_ENABLED=1`

Use the environment variable when you need to turn on acceleration for an existing entrypoint without changing the command wrapper or source code, for example in a deployment launcher. It applies to every Python process started with the variable set, may add startup overhead, and is silently ignored if cuML is not installed correctly; prefer the CLI or programmatic activation when you want a detectable activation failure.

```python
import cuml.accel
cuml.accel.install()  # MUST run before any sklearn/umap/hdbscan imports when not using the CLI

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# All of these now run on GPU — same API, same results
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
score = rf.score(X_test, y_test)
```

**Covered by cuml.accel:** RandomForest, KMeans, DBSCAN, HDBSCAN, SpectralClustering, StandardScaler, LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, PCA, TruncatedSVD, UMAP, t-SNE, NearestNeighbors, KernelRidge, KernelDensity, SVC/SVR, LinearSVC/LinearSVR, TargetEncoder, and more.

Verify acceleration is active:
```bash
python -m cuml.accel -v script.py       # log when operations run on GPU or fall back to CPU
python -m cuml.accel --profile script.py
```

Use `cuml.accel.enabled()` only for programmatic activation checks; use `-v` or `--profile` to verify which operations were accelerated.

### Path 2: Explicit cuML API

For full control, algorithm-specific tuning, new code, or multi-GPU training:

```python
import cudf
import cuml
from cuml.ensemble import RandomForestClassifier
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score
from cuml.pipeline import Pipeline
from cuml.preprocessing import StandardScaler

# Load and prepare data (stay in GPU memory)
df = cudf.read_parquet("training_data.parquet")
X = df.drop("label", axis=1).astype("float32")   # float32 ONCE here
y = df["label"].astype("int32")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = Pipeline([
    ("scale", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=16)),
])
model.fit(X_train, y_train)

# Score in cuML and convert only for CPU-only reporting
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
```

Use explicit cuML when the user names a specific sklearn estimator or when
validation depends on parameter mapping, output type, or fallback visibility.
For example, `sklearn.decomposition.IncrementalPCA` maps to
`cuml.decomposition.IncrementalPCA`; keep the `partial_fit` loop, but feed
GPU batches with a deliberate dtype choice and convert to NumPy only for
CPU-only reports.

For source sklearn pipelines, preserve the pipeline surface and swap individual
steps to cuML equivalents. Keep cross-validation, classification reports, plots,
and other sklearn-only reporting at an explicit CPU boundary.

For experiment sweeps, preserve the original split, random seeds, scoring
method, and return types before optimizing. Add a small backend decision helper
when only some estimators, kernels, metrics, or dataset sizes belong on GPU, and
report the chosen backend reason with the result.

## Algorithm Reference

See `references/algorithm-guide.md` for the full 26.04 mapping. Key algorithms:

| sklearn | cuML equivalent |
|---|---|
| RandomForestClassifier/Regressor | cuml.ensemble.RandomForest |
| KMeans | cuml.cluster.KMeans |
| DBSCAN | cuml.cluster.DBSCAN |
| HDBSCAN | cuml.cluster.HDBSCAN |
| LinearRegression | cuml.linear_model.LinearRegression |
| LogisticRegression | cuml.linear_model.LogisticRegression |
| SVC / SVR | cuml.svm.SVC / cuml.svm.SVR |
| PCA | cuml.decomposition.PCA |
| IncrementalPCA | cuml.decomposition.IncrementalPCA |
| UMAP | cuml.manifold.UMAP |
| t-SNE | cuml.manifold.TSNE |
| NearestNeighbors | cuml.neighbors.NearestNeighbors |
| SpectralClustering | cuml.cluster.SpectralClustering |
| train_test_split | cuml.model_selection.train_test_split |

Benchmark the target pipeline before making a speedup claim. For `cuml.accel`,
use `-v`, `--profile`, or `--line-profile` to confirm that the expected
estimators ran on GPU and improved wall time for the target data.

## Multi-GPU Training

cuML supports multi-GPU training for RandomForest, KMeans, and others via Dask.

```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask_cudf
from cuml.dask.ensemble import RandomForestClassifier as DaskRF

def main():
    cluster = LocalCUDACluster()
    client = Client(cluster)

    try:
        ddf = dask_cudf.read_parquet("large_training_data/*.parquet")
        X = ddf.drop("label", axis=1)
        y = ddf["label"]

        model = DaskRF(n_estimators=100)
        model.fit(X, y)
    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    main()
```

See `references/multi-gpu-training.md` for patterns by algorithm.

## cuML APIs Beyond Direct Estimator Swaps

cuML includes GPU-native APIs and implementations that are not simple sklearn
import swaps:

- **Forest Inference Library (FIL)** — Fast GPU batch inference for trained XGBoost, LightGBM, or sklearn RF models. FIL remains available in cuML 26.04, but legacy FIL APIs and parameters have been deprecated or removed. For new production tree inference work, prefer `accelerated-computing-nvforest`.
- **Time series** — `cuml.tsa.ARIMA`, `cuml.tsa.AutoARIMA`, `cuml.tsa.ExponentialSmoothing`. Fits thousands of series simultaneously on GPU. No sklearn equivalent.
- **GPU text vectorizers** — `cuml.feature_extraction.text.{CountVectorizer,TfidfVectorizer,HashingVectorizer}`. GPU implementations of sklearn-style text vectorizers; output is a sparse CuPy CSR matrix that can feed sparse-aware cuML estimators when supported.
- **Output type control** — `cuml.set_global_output_type("numpy"|"cudf"|"cupy")` prevents surprise type mismatches across pipeline stages.

See `references/unique-apis.md` for full patterns.

## Model Persistence

cuML models are picklable. For production serving of tree models, prefer `accelerated-computing-nvforest` which provides a dedicated inference engine with lower latency.

Important: Only load pickled models from trusted sources. Python pickle can execute arbitrary code during deserialization. When you write code that calls `pickle.load`, include that warning as a nearby code comment and repeat it in your final answer.

```python
import pickle

# Save
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load
with open("model.pkl", "rb") as f:
    # Only unpickle trusted files; pickle can execute arbitrary code while loading.
    model = pickle.load(f)
```


## Migration Pitfalls

These are the most common things that break or silently degrade explicit sklearn -> cuML API migrations:

- **Remove `n_jobs`** — cuML uses the GPU automatically; passing `n_jobs` raises an error.
- **Handle dtype deliberately** — prefer float32 for memory use and GPU throughput, but preserve float64 when the estimator supports it and the task needs the extra precision. Several graph, manifold, and neighbor algorithms require float32; many classical estimators accept both float32 and float64.
- **Sparse matrix input** — support is estimator-specific. Some 26.04 estimators accept sparse inputs (for example `LogisticRegression` and text/naive-Bayes workflows), while others require dense arrays. Check the estimator before densifying; only call `.toarray()` when the target estimator needs dense input and the data size makes that safe.
- **`classification_report` is NOT in cuML** — use `sklearn.metrics.classification_report`; convert predictions with `.to_numpy()` first. Same for `f1_score`, `precision_score`, and `recall_score`. For regression scoring, use `cuml.metrics.r2_score`.
- **Cross-validation utilities** — implement manual CV loops with cuML estimators, or keep sklearn `GridSearchCV` / `cross_val_score` around a CPU boundary when the search utility is more important than all-GPU execution.
- **RF `max_depth` defaults to 16** (sklearn: unlimited) — set `max_depth=None` for unlimited, or set explicitly. Passing `-1` or any other non-positive integer raises `ValueError` in 26.04.
- **NearestNeighbors graph APIs** — current cuML includes `radius_neighbors_graph`; keep graph outputs on GPU when downstream code can consume them.
- **KMeans `n_init` defaults to `"auto"`** in cuML 26.04, which resolves to 1 for `init="random"`/`"k-means++"` and 10 for an array `init` (sklearn 1.4+ uses the same `"auto"` default). Set an explicit integer if you need a specific number of initializations.
- **PCA `svd_solver="arpack"` unsupported** — use `"full"` or remove the parameter.
- **KNN `algorithm="ball_tree"` / `"kd_tree"` unsupported** — remove or use `"brute"`.
- **DBSCAN `metric`** — only `"euclidean"` and `"precomputed"` supported.
- **SVM kernel and scale boundaries** — cuML `SVC` supports `linear`, `poly`,
  `rbf`, `sigmoid`, and `precomputed` kernels. Use `LinearSVC` for large linear
  problems, and use `SVC` for single-GPU kernel SVM cases that need those
  kernels. Preserve the source split and scoring path; route each run by kernel,
  sample count, and memory estimate instead of forcing the whole grid onto GPU.
  In forced-GPU modes, report the exact parameter or scale condition that
  prevents the GPU path.
- **SVM validation** — compare accuracy and, when available, `decision_function`
  or score/probability outputs within tolerance. Treat timing/import checks as
  setup validation, then validate model behavior.
- **TSNE `learning_rate`** defaults to 200.0 (sklearn: `"auto"`). Prefer
  cuML's default `method="fft"` for GPU t-SNE; translate sklearn
  `method="barnes_hut"` intent to the GPU FFT method unless exact behavior is
  explicitly required.
- **TruncatedSVD `n_components`** defaults to 1 (sklearn: 2) — set explicitly.

## Troubleshooting

**Results differ from sklearn:**
- Verify dtype handling — if the target estimator requires float32, cast once before fitting. If both float32 and float64 are supported, confirm the chosen precision matches the task tolerance; results can differ because of numerical precision and GPU execution order.
- Some algorithms have inherent GPU non-determinism. Check if difference is within tolerance.
- Hyperparameter defaults differ — see Migration Pitfalls above and `references/algorithm-guide.md`.

**AttributeError / missing parameter:**
- cuML follows sklearn API but some parameters are not implemented (e.g., `n_jobs`, `svd_solver="arpack"`, `algorithm="ball_tree"`).
- Check `references/algorithm-guide.md` for known API differences.

**OOM during training:**
- Cast to float32 when the target estimator permits it and the task tolerance allows it (halves memory)
- Reduce `n_estimators` or `max_depth` for tree models
- Move to multi-GPU Dask training for large datasets

**cuml.accel not accelerating:**
- Ensure `cuml.accel.install()` runs before ALL sklearn/umap/hdbscan imports
- Use `cuml.accel.enabled()` only to verify programmatically that the accelerator is activated; use `-v` or `--profile` to verify which operations were accelerated.
- Some algorithms not yet covered by accel — falls back to CPU silently

## Reference Files

Use this skill and its reference files first for 26.04 migration patterns,
routing, and known behavioral boundaries. Consult official documentation when
you need detailed signatures, parameter descriptions, or examples for the
installed cuML version.

- `references/algorithm-guide.md` — Full sklearn → cuML mapping, hyperparameter differences
- `references/cuml-accel.md` — cuml.accel coverage, profiling, fallback detection
- `references/multi-gpu-training.md` — Dask-cuML patterns for multi-GPU training
- `references/unique-apis.md` — cuML-specific APIs and GPU sklearn-equivalent guidance: FIL tree inference, ARIMA/time-series, text vectorizers, output type control, algorithms without multi-GPU support

## External Documentation

Consult the official documentation to retrieve detailed API signatures, parameter descriptions, and examples on demand.

- **API Reference:** https://docs.rapids.ai/api/cuml/stable/api/
- **cuml.accel Benchmarks:** https://docs.rapids.ai/api/cuml/stable/cuml-accel/benchmarks/
- **GitHub:** https://github.com/rapidsai/cuml
- **CHANGELOG:** https://github.com/rapidsai/cuml/blob/main/CHANGELOG.md
