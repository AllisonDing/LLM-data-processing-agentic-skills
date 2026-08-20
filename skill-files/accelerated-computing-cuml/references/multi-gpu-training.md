# cuML Multi-GPU Training

## Supported Algorithms for Multi-GPU Training

| Algorithm | Multi-GPU Support | Notes |
|---|---|---|
| RandomForest | Yes (dask) | Best multi-GPU scaling |
| KMeans | Yes (dask) | Good scaling |
| LinearRegression | Yes (dask) | Good scaling |
| LogisticRegression | Yes (dask) | Good scaling |
| DBSCAN | Partial (dask) | Use `cuml.dask.cluster.DBSCAN` for single-node multi-GPU workloads |
| HDBSCAN | Partial | Multi-GPU with `build_algo="nn_descent"` and `nnd_n_clusters > 1` |
| UMAP | Partial | Local multi-GPU fit with `build_algo="nn_descent"`, `build_kwds={"knn_n_clusters": n}`, and `device_ids`; Dask UMAP is distributed transform with a fitted local model |
| SVC/SVR | No | Single GPU only |
| LinearSVC/LinearSVR | No dask API | Scalable single-GPU alternatives for large linear SVM workloads |
| KernelRidge | No | Single GPU only |
| t-SNE | No | Single GPU only |
| ARIMA / AutoARIMA / ExponentialSmoothing | No | Multi-series batching on one GPU is supported; multi-GPU training is not |

For SVC/SVR on large data, subsample for kernel training or use cuML's
`LinearSVC`/`LinearSVR` when a linear model fits the problem.

## Setup

```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

def main():
    cluster = LocalCUDACluster()
    client = Client(cluster)

    try:
        # Run multi-GPU cuML work here.
        ...
    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    main()
```

## Multi-GPU RandomForest

```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask_cudf
from cuml.dask.ensemble import RandomForestClassifier

def main():
    cluster = LocalCUDACluster()
    client = Client(cluster)

    try:
        # Load distributed data
        ddf = dask_cudf.read_parquet("training_data/*.parquet")
        X = ddf.drop("label", axis=1).astype("float32")
        y = ddf["label"]

        # Train across all GPUs
        model = RandomForestClassifier(n_estimators=100, max_depth=16)
        model.fit(X, y)

        # Predict
        predictions = model.predict(X_test)  # X_test can be cuDF or dask_cudf
    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    main()
```

## Multi-GPU KMeans

```python
from cuml.dask.cluster import KMeans
import dask_cudf

ddf = dask_cudf.read_parquet("data/*.parquet").astype("float32")

km = KMeans(n_clusters=10)
km.fit(ddf)

# Get cluster assignments
labels = km.predict(ddf).compute()
```

## Multi-GPU Linear Models

```python
from cuml.dask.linear_model import LinearRegression, LogisticRegression
import dask_cudf

ddf = dask_cudf.read_parquet("data/*.parquet")
X = ddf[feature_cols].astype("float32")
y = ddf["target"].astype("float32")

model = LinearRegression()
model.fit(X, y)
```

## Multi-GPU UMAP

```python
from cuml.manifold import UMAP

reducer = UMAP(
    n_components=2,
    build_algo="nn_descent",
    build_kwds={"knn_n_clusters": 4},
    device_ids="all",
)
embedding = reducer.fit_transform(X)
```

For distributed transform, use `cuml.dask.manifold.UMAP` with a fitted
`cuml.UMAP` model:

```python
from cuml.dask.manifold import UMAP as DaskUMAP

distributed_reducer = DaskUMAP(model=reducer)
embedding = distributed_reducer.transform(X_dask).compute()
```

## Collecting Results

```python
# Collect distributed predictions to single GPU
local_predictions = model.predict(X_dask).compute()  # returns cuDF Series

# To pandas for downstream use
pd_predictions = local_predictions.to_pandas()
```

## Memory Management for Multi-GPU Training

```python
# Enable cuDF spilling to prevent OOM on workers
cluster = LocalCUDACluster(enable_cudf_spill=True)  # create inside main() in scripts

# Set explicit partition sizes to control per-worker memory
ddf = dask_cudf.read_parquet("data/*.parquet")
target_partitions = len(client.scheduler_info()["workers"]) * 4
ddf = ddf.repartition(npartitions=target_partitions)
```

## Persisting Model After Training

Only load pickled models from trusted sources. Python pickle can execute arbitrary code during deserialization. When generating code that calls `pickle.load`, include this warning as a nearby code comment and repeat it in the final summary.

```python
import pickle

# Save multi-GPU model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load and use for single-GPU inference
with open("model.pkl", "rb") as f:
    # Only unpickle trusted files; pickle can execute arbitrary code while loading.
    loaded_model = pickle.load(f)

# Or export to nvForest for production serving
# (train with cuML multi-GPU, serve with nvForest)
```
