# cuml.accel — Zero-Code-Change Acceleration

## How It Works

`cuml.accel` patches the sklearn, umap-learn, and hdbscan module namespaces to route supported estimators to cuML GPU implementations. Unsupported estimators fall back to their CPU implementations silently by default; use `-v` or `--verbose` with the CLI to log GPU execution and CPU fallbacks.

Use `cuml.accel` when the user wants the smallest change to an existing sklearn, UMAP, or HDBSCAN workflow, especially for quick acceleration checks, prototypes, or broad codebases where direct import rewrites are risky. Prefer explicit `cuml.*` APIs for new code written from scratch, named estimator migrations, cuML-only parameters, multi-GPU training, precise GPU/CPU boundaries, or control over output types and memory layout.

## Activation

### Python script import

```python
# MUST be the first import in your script or notebook cell
import cuml.accel
cuml.accel.install()

# Then import sklearn/umap/hdbscan normally
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import umap
import hdbscan
```

### CLI

```bash
python -m cuml.accel script.py
```

### Environment variable

```bash
CUML_ACCEL_ENABLED=1 python script.py
export CUML_ACCEL_ENABLED=1
```

Use the environment variable when you need to enable acceleration for an existing entrypoint without changing the command wrapper or source code, for example in a deployment launcher. It applies to every Python process started with the variable set, may add startup overhead, and is silently ignored if cuML is not installed correctly; prefer the CLI or programmatic activation when you want a detectable activation failure.

### Notebook

```python
%load_ext cuml.accel
```

**Critical ordering**: Any sklearn, UMAP, or HDBSCAN import before activation will not be patched.

```python
# Verify activation
print(cuml.accel.enabled())  # True if accelerator is installed
```

## Coverage Matrix

| Library | Estimators Accelerated |
|---|---|
| sklearn.cluster | KMeans, DBSCAN, SpectralClustering |
| sklearn.decomposition | PCA, TruncatedSVD |
| sklearn.ensemble | RandomForestClassifier, RandomForestRegressor |
| sklearn.kernel_ridge | KernelRidge |
| sklearn.linear_model | LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet |
| sklearn.manifold | TSNE, SpectralEmbedding |
| sklearn.neighbors | NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor, KernelDensity |
| sklearn.preprocessing | StandardScaler, TargetEncoder |
| sklearn.svm | SVC, SVR, LinearSVC, LinearSVR |
| umap-learn | UMAP |
| hdbscan | HDBSCAN |

## Detecting Fallback

When an estimator falls back to CPU, it's silent. Detect it:

```bash
python -m cuml.accel -v script.py
python -m cuml.accel --verbose script.py
python -m cuml.accel --profile script.py
```

For code-level checks, profile the suspected block directly:

```python
import cuml.accel
cuml.accel.install()

with cuml.accel.profile():
    # run your ML code here
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5)
    km.fit(X)
```

## Profiling Acceleration

Use the CLI profiler when running a full script:

```bash
python -m cuml.accel --profile script.py
```

For focused timing inside Python:

```python
# Time comparison: with and without cuml.accel
import time

# Without acceleration
from sklearn.cluster import KMeans as SKLearnKMeans
t0 = time.perf_counter()
SKLearnKMeans(n_clusters=10).fit(X)
cpu_time = time.perf_counter() - t0

# With acceleration
import cuml.accel
cuml.accel.install()
from sklearn.cluster import KMeans as CuMLKMeans  # same import, now patched
t0 = time.perf_counter()
CuMLKMeans(n_clusters=10).fit(X)
gpu_time = time.perf_counter() - t0

print(f"Speedup: {cpu_time / gpu_time:.1f}x")
```

## Known Limitations

1. **Pipeline() mixed support**: pipelines with all-supported estimators work; mixed pipelines may fall back to CPU for unsupported steps.
2. **GridSearchCV and cross_val_score** are not GPU-accelerated — the CV loop itself runs on CPU, but the inner estimator uses GPU.
3. **Custom estimators and callbacks** are not intercepted.
4. **Array types**: list/tuple inputs only supported in accel mode; numpy/cuDF/CuPy all work.
5. **Parameter mapping**: `cuml.accel` preserves the sklearn API surface, but some parameters may be mapped to GPU-equivalent cuML behavior.

For full estimator and parameter fallback conditions, prefer the version-matched
limitations source for the installed cuML tag. For 26.04, that is:
`https://raw.githubusercontent.com/rapidsai/cuml/refs/tags/v26.04.00/docs/source/cuml-accel/limitations.rst`.

## Transition to Explicit cuML

When cuml.accel doesn't provide enough control:

```python
# cuml.accel path (convenient)
import cuml.accel
cuml.accel.install()
from sklearn.ensemble import RandomForestClassifier

# Explicit cuML path (full control)
from cuml.ensemble import RandomForestClassifier
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=16,
    n_streams=4,    # cuML-specific parallelism
)
```

The explicit API gives access to cuML-specific hyperparameters, multi-GPU training, and cuDF output types.
