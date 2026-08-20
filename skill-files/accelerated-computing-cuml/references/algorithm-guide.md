# cuML Algorithm Guide

## Full sklearn → cuML Mapping

### Supervised Learning

| sklearn | cuML | Notes |
|---|---|---|
| `LinearRegression` | `cuml.linear_model.LinearRegression` | Drop-in |
| `LogisticRegression` | `cuml.linear_model.LogisticRegression` | Drop-in; `max_iter` default may differ |
| `Ridge` | `cuml.linear_model.Ridge` | Drop-in |
| `Lasso` | `cuml.linear_model.Lasso` | Drop-in |
| `ElasticNet` | `cuml.linear_model.ElasticNet` | Drop-in |
| `RandomForestClassifier` | `cuml.ensemble.RandomForestClassifier` | See RF notes below |
| `RandomForestRegressor` | `cuml.ensemble.RandomForestRegressor` | See RF notes below |
| `SVC` | `cuml.svm.SVC` | `linear`, `poly`, `rbf`, `sigmoid`, `precomputed` kernels; single GPU |
| `SVR` | `cuml.svm.SVR` | `linear`, `poly`, `rbf`, `sigmoid`, `precomputed` kernels; single GPU |
| `KNeighborsClassifier` | `cuml.neighbors.KNeighborsClassifier` | Drop-in |
| `KNeighborsRegressor` | `cuml.neighbors.KNeighborsRegressor` | Drop-in |

### Unsupervised Learning

| sklearn/UMAP/HDBSCAN | cuML | Notes |
|---|---|---|
| `KMeans` | `cuml.cluster.KMeans` | Drop-in |
| `DBSCAN` | `cuml.cluster.DBSCAN` | Drop-in |
| `HDBSCAN` | `cuml.cluster.HDBSCAN` | Drop-in; benchmark target data |
| `SpectralClustering` | `cuml.cluster.SpectralClustering` | Drop-in for supported affinities |
| `umap.UMAP` | `cuml.manifold.UMAP` | Drop-in |
| `TSNE` | `cuml.manifold.TSNE` | Drop-in; benchmark target data |
| `PCA` | `cuml.decomposition.PCA` | Drop-in |
| `IncrementalPCA` | `cuml.decomposition.IncrementalPCA` | Supports `partial_fit`; feed GPU batches with a deliberate dtype choice |
| `TruncatedSVD` | `cuml.decomposition.TruncatedSVD` | Drop-in |
| `MiniBatchKMeans` | `cuml.cluster.KMeans` | Use KMeans directly (GPU doesn't need mini-batch) |
| `AgglomerativeClustering` | `cuml.cluster.AgglomerativeClustering` | Available in cuML; verify parameter compatibility before treating as drop-in |

### Preprocessing

| sklearn | cuML | Notes |
|---|---|---|
| `StandardScaler` | `cuml.preprocessing.StandardScaler` | Drop-in |
| `MinMaxScaler` | `cuml.preprocessing.MinMaxScaler` | Drop-in |
| `LabelEncoder` | `cuml.preprocessing.LabelEncoder` | Drop-in |
| `train_test_split` | `cuml.model_selection.train_test_split` | Drop-in |

### Model Selection

| sklearn | cuML | Notes |
|---|---|---|
| `train_test_split` | `cuml.model_selection.train_test_split` | Drop-in |
| `GridSearchCV` | Not available | Use manual loop or CPU sklearn GridSearchCV |
| `cross_val_score` | Not available | Use manual CV loop |

## Key Hyperparameter Differences

### RandomForest

| Parameter | sklearn | cuML | Action |
|---|---|---|---|
| `max_depth` | `None` (unlimited) | **16** | In 26.04, set `None` for unlimited or set explicitly. `-1` is **not** accepted in 26.04; verify installed behavior before using newer-version compatibility. |
| `n_jobs` | `-1` (all cores) | **Not supported** | Remove |
| `n_bins` | N/A | 128 | cuML-only; higher = more accurate but slower |

### LogisticRegression

| Parameter | sklearn | cuML | Action |
|---|---|---|---|
| `solver` | `"lbfgs"` | `"qn"` | cuML accepts only `solver="qn"`; any other value raises `ValueError`. OWL-QN is selected automatically when `penalty="l1"`. |
| `max_iter` | 100 | **1000** | cuML defaults higher |
| `n_jobs` | `None` | **Not supported** | Remove |

### KMeans

| Parameter | sklearn | cuML | Action |
|---|---|---|---|
| `n_init` | `"auto"` (sklearn 1.4+) | `"auto"` | Both default to `"auto"`. In cuML 26.04, `"auto"` resolves to 1 for `init="random"`/`"k-means++"` and 10 for an array `init`. Set an explicit integer for a specific count. |
| `algorithm` | `"lloyd"` | N/A | cuML always uses a GPU-optimized Lloyd variant |

### PCA

| Parameter | sklearn | cuML | Action |
|---|---|---|---|
| `svd_solver` | `"auto"` | `"auto"` | cuML supports `"full"`, `"jacobi"`, `"auto"`. **No `"arpack"` or `"randomized"`** (the cuML PCA docstring in 26.04 says `default='full'`, but the constructor actually defaults to `'auto'`; either value works). |

### IncrementalPCA

| Parameter / pattern | sklearn | cuML | Action |
|---|---|---|---|
| `partial_fit` chunks | NumPy batches | GPU array/cuDF/CuPy batches | Keep the loop; use float32 for memory/throughput unless the task needs supported float64 precision |
| `batch_size` | accepted | accepted | Preserve when meaningful; validate rows per batch are compatible |
| output arrays | NumPy | follows cuML output type | Convert deliberately for CPU-only metrics/printing |

### TruncatedSVD

| Parameter | sklearn | cuML | Action |
|---|---|---|---|
| `n_components` | 2 | **1** | Set explicitly |
| `algorithm` | `"randomized"` | `"full"` | cuML supports `"full"` and `"jacobi"` only |

### TSNE

| Parameter | sklearn | cuML | Action |
|---|---|---|---|
| `learning_rate` | `"auto"` | **200.0** | Set explicitly if needed |
| `method` | `"barnes_hut"` | `"fft"` | Use cuML's FFT GPU method for the same visualization intent |

### KNeighborsClassifier / KNeighborsRegressor / NearestNeighbors

| Parameter | sklearn | cuML | Action |
|---|---|---|---|
| `algorithm` | `"auto"` | `"brute"` | cuML supports `"brute"` and `"ivfflat"`. **No `"ball_tree"` or `"kd_tree"`** |
| `metric` | `"minkowski"` | `"euclidean"` | cuML supports `"euclidean"`, `"manhattan"`, `"chebyshev"`, `"minkowski"` |
| `n_jobs` | `None` | **Not supported** | Remove |

`NearestNeighbors.radius_neighbors_graph` is available in current cuML. Prefer
keeping the resulting graph on GPU for cuGraph, CuPy, or other GPU consumers;
convert to CPU only at a deliberate boundary.

### DBSCAN

| Parameter | sklearn | cuML | Action |
|---|---|---|---|
| `metric` | `"euclidean"` | `"euclidean"` | cuML only supports `"euclidean"` and `"precomputed"` |
| `n_jobs` | `None` | **Not supported** | Remove |

### Metrics Availability

Available in `cuml.metrics`: `accuracy_score`, `confusion_matrix`, `log_loss`, `roc_auc_score`, `precision_recall_curve`, `mean_squared_error`, `mean_absolute_error`, `r2_score`, plus clustering metrics under `cuml.metrics.cluster` such as `silhouette_score`, `adjusted_rand_score`, `homogeneity_score`, `completeness_score`, and `v_measure_score`.

Route classification reporting metrics to `sklearn.metrics`. After GPU prediction, convert only the final labels/predictions needed by the reporter (`cuDF -> .to_numpy()`, `CuPy -> .get()`), leaving training and inference on the GPU path.

### Sparse Input

Sparse input support is estimator-specific in cuML 26.04. Do not densify by
default: first check whether the target estimator accepts sparse inputs. Linear
models such as `LogisticRegression`, text vectorizer plus naive-Bayes workflows,
and selected preprocessing utilities support sparse inputs; many other
estimators still require dense arrays.

### Model Selection Availability

For `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`, and `KFold`, use a
manual GPU loop with cuML estimators or keep sklearn's utility around a CPU
boundary when the search wrapper matters more than end-to-end GPU residency.

### Numeric Precision

Prefer float32 for cuML memory use and GPU throughput, especially for graph,
manifold, and neighbor algorithms. Several of those paths require float32
(`HDBSCAN`, `UMAP`, `TSNE`, `NearestNeighbors`/KNN, spectral methods, and
agglomerative clustering). Many classical estimators accept both float32 and
float64, including linear models, `PCA`, `TruncatedSVD`, `KMeans`, `DBSCAN`,
`SVC`/`SVR`, `RandomForestClassifier`, and `RandomForestRegressor`. Preserve
float64 when the estimator supports it and the task needs the extra precision.

## cuml.accel Coverage

Algorithms covered by `cuml.accel` (zero-code-change acceleration):
- RandomForestClassifier, RandomForestRegressor
- KMeans, DBSCAN, SpectralClustering
- SVC, SVR, LinearSVC, LinearSVR
- LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
- KernelRidge
- PCA, TruncatedSVD
- TSNE, SpectralEmbedding
- UMAP (via umap-learn)
- HDBSCAN (via hdbscan package)
- NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor, KernelDensity
- StandardScaler, TargetEncoder

Use explicit cuML imports instead of relying on `cuml.accel` when the algorithm
is not covered by `cuml.accel`, such as AgglomerativeClustering or
MultinomialNB. Prefer explicit imports for source pipelines where you are
swapping individual sklearn steps to cuML, custom pipelines with `Pipeline()`,
or cases that need parameter mapping, output-type control, fallback visibility,
or scale/kernel routing.

## Output Type Behavior

```python
import cuml
import cudf
import numpy as np

# Input numpy → output numpy
X_np = np.random.random((1000, 10)).astype("float32")
model = cuml.cluster.KMeans(n_clusters=3)
model.fit(X_np)
labels = model.labels_   # numpy array

# Input cuDF → output cuDF Series
X_cudf = cudf.DataFrame({"a": range(1000), "b": range(1000)})
model.fit(X_cudf)
labels = model.labels_   # cuDF Series

```

For global or scoped output control, including CuPy output, see
[Output Type Control](unique-apis.md#output-type-control).
