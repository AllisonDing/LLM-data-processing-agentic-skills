# cuML APIs Beyond Direct Estimator Swaps

These capabilities are cuML-specific APIs or GPU-native implementations whose
behavior is worth checking directly rather than treating them as a simple
sklearn import swap.

---

## Forest Inference Library (FIL)

`cuml.fil.ForestInference` — GPU-accelerated inference for tree ensemble models. Use this instead of sklearn's `.predict()` for large-batch production inference of XGBoost, LightGBM, or RandomForest models.

**When to use FIL:** You trained a model with XGBoost/LightGBM/sklearn, and you need fast batch inference. Benchmark the target serving batch size before making a speedup claim.

**26.04 status:** The legacy FIL API (`output_class=`, `algo="BATCH_TREE_REORG"`, `precision="NATIVE"`, etc.) was removed before 26.04. The current API uses `is_classifier=`, `layout=`, and `precision='single'|'double'|None`. For new production tree inference work, prefer the `accelerated-computing-nvforest` skill and follow its exact `nvforest.*` APIs.

```python
from cuml.fil import ForestInference
import numpy as np

# Load from XGBoost (JSON or UBJ); model_type is auto-detected from the extension
fm = ForestInference.load("xgboost_model.json", is_classifier=True)

# Load from LightGBM
fm = ForestInference.load("lgbm_model.txt", is_classifier=True)

# Load from scikit-learn RandomForest (pickle)
import pickle

# Only unpickle trusted files; pickle can execute arbitrary code while loading.
with open("rf_model.pkl", "rb") as f:
    sklearn_rf = pickle.load(f)
fm = ForestInference.load_from_sklearn(sklearn_rf, is_classifier=True)

# Inference — cast inputs to match the model's precision (float32 by default)
X = np.random.random((100_000, 50)).astype(np.float32)
preds = fm.predict(X)             # classifiers: class labels (binary uses threshold); regressors: predictions
probs = fm.predict_proba(X)       # classifiers only: shape (n_samples, n_classes)
```

**Key parameters (26.04):**
- `is_classifier=False` — set `True` for classification models so `predict()` returns class labels and `predict_proba()` is available.
- `precision='single'` — inference precision. Use `'single'`, `'double'`, or `None` (use the model's native precision).
- `layout='depth_first'` — in-memory tree layout. Other options: `'breadth_first'`, `'layered'`. Try alternatives for the realistic batch size in performance-critical paths.
- `default_chunk_size=None` — default rows-per-chunk for `predict()`; on GPU, valid chunk sizes are powers of 2 from 1 to 32.
- `device_id=0` — target GPU for GPU execution.
- `threshold` is passed to `predict()`, not `load()`: binary classifiers treat probabilities above `threshold` (default 0.5) as positive.

**First call includes JIT compilation** — benchmark from the second call onward.

**Note:** For production tree inference, also consider `accelerated-computing-nvforest` which provides a standalone inference engine with overlapping model-loading concepts and more serving patterns. Check that skill for exact nvForest entrypoints and parameters.

---

## Time Series (No sklearn Equivalent)

cuML provides GPU-accelerated time series models that don't exist in sklearn at all.
For forecast horizons, `h` is the horizon parameter:

```python
forecast = model.forecast(h=10)   # equivalent: model.forecast(10)
```

### ARIMA

```python
from cuml.tsa import ARIMA

# Fit ARIMA(p, d, q) on multiple series simultaneously
# y: shape (n_timepoints, n_series) — all series fit in parallel on GPU
import cudf
y = cudf.DataFrame({"series_a": [...], "series_b": [...]})

model = ARIMA(endog=y, order=(2, 1, 2))
model.fit()

# Predict in-sample
in_sample = model.predict(start=0, end=len(y)-1)
```

### AutoARIMA

In cuML 26.04, `AutoARIMA` is configured in three steps: construct, then
`search(...)` to define the model space and pick the best order per series
via an information criterion, then `fit(...)`.

```python
from cuml.tsa.auto_arima import AutoARIMA

# `endog` is required; the constructor itself does not take search ranges.
model = AutoARIMA(endog=y)

# Define the search space. All of p, d, q, P, D, Q, s accept an int,
# a range, or any iterable. ic defaults to "aicc"; method defaults to "auto".
model.search(
    s=12,                 # seasonal period (None or 0 for non-seasonal)
    d=range(0, 2),
    D=range(0, 2),
    p=range(0, 4),
    q=range(0, 4),
    P=range(0, 3),
    Q=range(0, 3),
    ic="aicc",            # information criterion: "aic", "aicc", or "bic"
)

# Fit the per-series selected models.
model.fit()

# Inspect the selection (per-series summary of chosen orders and ICs).
model.summary()
```

### Exponential Smoothing (Holt-Winters)

cuML 26.04 implements Holt-Winters as additive or multiplicative
seasonal smoothing. There is no separate `trend=` parameter; the
seasonal model is selected via `seasonal="additive"` or
`seasonal="multiplicative"`.

```python
from cuml.tsa import ExponentialSmoothing

model = ExponentialSmoothing(
    y,                       # endog, positional
    seasonal="additive",     # "additive" or "multiplicative"
    seasonal_periods=12,     # season length (default 2)
    ts_num=1,                # number of time series in the batch
)
model.fit()
```

**Key difference from statsmodels:** cuML fits all series simultaneously on the GPU. If you have 10,000 product time series, cuML fits them all in one call — statsmodels requires a loop.

---

## GPU Text Vectorizers

GPU-accelerated text feature extraction. API mirrors sklearn's `CountVectorizer` and `TfidfVectorizer`.

```python
from cuml.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
import cudf

corpus = cudf.Series([
    "the quick brown fox",
    "the lazy dog",
    "fox and dog",
])

# CountVectorizer
vectorizer = CountVectorizer(max_features=10_000)
X = vectorizer.fit_transform(corpus)   # returns sparse cupy matrix

# TF-IDF
tfidf = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
X_tfidf = tfidf.fit_transform(corpus)

# HashingVectorizer (no vocabulary fitting needed)
hasher = HashingVectorizer(n_features=2**18)
X_hashed = hasher.transform(corpus)
```

**Output format:** All vectorizers return a sparse CuPy CSR matrix. Pass it
directly only to sparse-capable cuML estimators, such as
`cuml.naive_bayes.MultinomialNB` or estimators whose docs/tags confirm sparse
input support. For dense-only estimators, densify deliberately with `.toarray()`
only after checking memory.

**When to use GPU text processing:** Large corpora (>100K documents). Below that, sklearn's CPU vectorizer is fine.

For production migrations, run a tiny fit/predict smoke check with the selected
vectorizer and estimator before benchmarking the full corpus. That confirms the
installed cuML version accepts the sparse/dense representation you chose.

---

## Output Type Control

By default, cuML outputs match the input type: cuDF input → cuDF output, NumPy input → NumPy output. This can surprise users who mix input types across a pipeline.

```python
import cuml

# Set globally for the session
cuml.set_global_output_type("numpy")    # always return numpy arrays
cuml.set_global_output_type("cudf")     # always return cuDF Series/DataFrame
cuml.set_global_output_type("cupy")     # always return cupy arrays

# Or use as a context manager for a block
with cuml.using_output_type("numpy"):
    preds = model.predict(X_test)       # returns numpy here
    probs = model.predict_proba(X_test) # returns numpy here
# Outside block: reverts to previous setting
```

**When this matters:**
- Mixing cudf and numpy inputs across pipeline stages
- Passing cuML predictions to CPU or non-GPU libraries (sklearn metrics, matplotlib, etc.)
- Preventing accidental GPU→CPU copies mid-pipeline

---

## Multi-GPU Support Boundaries

For the canonical full/partial/no support table, including KernelRidge,
time-series models, DBSCAN, UMAP, HDBSCAN, and the SVC/SVR scaling note, see
[cuML Multi-GPU Training](multi-gpu-training.md#supported-algorithms-for-multi-gpu-training).

---

## Solver Classes (Direct Use)

cuML exposes solver algorithms as standalone classes for embedding in custom training loops:

```python
from cuml.solvers import CD, QN, SGD

# Coordinate Descent
solver = CD(loss="sigmoid", penalty="l2", alpha=0.001)
solver.fit(X, y)

# Quasi-Newton (good for logistic regression, L-BFGS)
solver = QN(loss="sigmoid", penalty="l2", linesearch_max_iter=50)
solver.fit(X, y)

# Stochastic Gradient Descent (mini-batch)
solver = SGD(loss="squared_loss", eta0=0.01, n_iter=1000)
solver.fit(X, y)
```

Typical use: embedding in meta-learners, custom regularization, or fine-grained convergence control.
