# Commuting OD Matrix Generation — GPS Architecture Experiments

This repository extends the [Commuting OD Dataset](https://anonymous.4open.science/r/tmp_data-9362/metros.zip) with **GPS-based architecture experiments** for Origin-Destination flow prediction. It combines a GPS-GNN encoder (GPSConv + GINEConv + multi-head attention) with various decoders, positional encodings, loss functions, and normalization strategies.

## Repository Structure

```
models/
  shared/           # Canonical metrics and data loading (used by all models)
    metrics.py      # CPC, RMSE, MAE, MAPE, SMAPE, JSD, accuracy, etc. (17 metrics)
    data_load.py    # load_area_raw, construct_flat_features, load_graph_data, split helpers
  GPS/              # Our GPS-based model
    model.py        # GPSODModel, TransFlowerODModel, make_model()
    main.py         # train_single_city(), train_multi_city()
    config.py       # TrainingConfig dataclass (all experiment parameters)
    data_load.py    # GPS-specific data prep (PE computation, Huber weights, coords)
    loss.py         # compute_loss_for_city (CE, Huber, Focal, ZINB, Multitask, MAE)
    metrics.py      # predict_full_matrix(), evaluate_full_matrix()
    rle.py          # Relative Location Encoder
    lgbm_pipeline.py
  GMEL_GPS/         # GMEL architecture with GPS encoders + GBRT/LGBM decoder
  RF/               # Random Forest baseline
  SVR/              # Support Vector Regression baseline
  GBRT/             # Gradient Boosted Regression Trees baseline
  DGM/              # DeepGravity Model baseline
  GM_E/             # Gravity Model (exponential) baseline
  GM_P/             # Gravity Model (power-law) baseline
  GMEL/             # Graph Multi-task Embedding Learning baseline
  NetGAN/           # Network GAN baseline
  DiffODGen/        # Diffusion-based OD Generation
  WeDAN/            # WeDAN baseline
benchmarking/
  pipeline.py       # run_single_city_benchmark(), run_multi_city_benchmark()
  runners.py        # Thin dispatcher calling models/<name>/main.py
  gps_loader.py     # Load pre-trained GPS/LGBM/GMEL_GPS results
  config.py         # Benchmark IDs, baseline model list, result columns
  data_utils.py     # Shared data utilities for benchmark
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `gps_od.ipynb` | Train GPS models with different module combinations (PE type, decoder, loss, norm) |
| `lgbm_od.ipynb` | Train LGBM decoder on GPS embeddings |
| `benchmark.ipynb` | Compare GPS variants vs all baseline models from papers |
| `data_city_analysis.ipynb` | Analyze all `data/` areas, correlations, visualizations, and city type classification |

## Evaluation Modes

### Single-city
- City **48201** (Harris County, TX)
- Pair split supports **nonzero_pairs** and **all_pairs** modes with seed=42
- Zero-flow pairs are controlled by the chosen split mode rather than a separate training flag

### Multi-city
- 8 cities: 36061, 06059, 17031, 48201, 06075, 53033, 11001, 47157
- Cities split **4/2/2** (train/val/test), with New York City and Memphis fixed as the multi-city test areas

All models use identical data splits and the same canonical metrics from `models/shared/metrics.py`.

## GPS Model Configuration

All experiment parameters are controlled via `TrainingConfig` in `models/GPS/config.py`:

- **Encoder**: `gps` (GPSConv) or `mlp`
- **Decoder**: `transflower`, `bilinear`, `lgbm`, `gbrt`
- **PE type**: `rwpe`, `spe`, `rrwp`, `lape`, or `None`
- **Loss**: `ce`, `huber`, `focal`, `multitask`, `zinb`, `mae`
- **Norm**: `batch_norm` or `graph_norm`
- **RLE**: Relative Location Encoder (optional)
- **Destination sampling**: with/without zero pairs

## Metrics

The full metric suite (`cal_od_metrics`) computes 17 metrics:

| Metric | Description |
|--------|-------------|
| CPC | Common Parts Coefficient |
| RMSE, NRMSE, MAE, MAPE, SMAPE | Standard regression metrics |
| CPC/RMSE/MAE/MAPE/SMAPE_nonzero | Same metrics on nonzero entries only |
| accuracy | Fraction of correct zero/nonzero predictions |
| matrix_COS_similarity | Cosine similarity between flattened matrices |
| JSD_inflow / JSD_outflow / JSD_ODflow | Jensen-Shannon divergence on flow distributions |

## How to Run

### GPS experiments
```python
# In gps_od.ipynb — edit SC_CONFIGS dict, then run training cells
from models.GPS.main import train_single_city
from models.GPS.config import TrainingConfig

config = TrainingConfig(decoder_type='transflower', loss_type='ce', pe_type='rwpe')
result = train_single_city("my_run", "My Experiment", config)
```

### Baseline models
```python
# Each baseline has train() in models/<name>/main.py
from models.RF.main import train, evaluate
model = train(x_train, y_train)
metrics = evaluate(model, xs_test, ys_test)
```

### Full benchmark
```python
# In benchmark.ipynb — runs GPS + all baselines with unified metrics
from benchmarking.pipeline import run_single_city_benchmark
results, model_types = run_single_city_benchmark(gps_run_ids, lgbm_run_ids, "48201", data_path)
```

## Data

The dataset covers 3,233 areas across the United States (counties and metropolitan areas). Each area contains:

| File | Shape | Description |
|------|-------|-------------|
| `demos.npy` | (N, 97) | Demographics from American Community Survey |
| `pois.npy` | (N, 36) | POI category counts from OpenStreetMap |
| `adj.npy` | (N, N) | Adjacency matrix |
| `dis.npy` | (N, N) | Euclidean distance matrix |
| `od.npy` | (N, N) | Commuting OD matrix from LODES 2018 |

Metropolitan data: [download link](https://anonymous.4open.science/r/tmp_data-9362/metros.zip)

## Prerequisites

- Python 3.8+
- PyTorch 2.1+
- PyTorch Geometric
- DGL
- scikit-learn, numpy, scipy, pandas
- lightgbm (for LGBM decoder)
- tqdm

## Baseline Results (from original paper)

| Model | CPC | RMSE | NRMSE | JSD inflow | JSD outflow | JSD ODflow |
|-------|-----|------|-------|------------|-------------|------------|
| GM-P | 0.321 | 174.0 | 2.222 | 0.668 | 0.656 | 0.409 |
| GM-E | 0.329 | 162.9 | 2.080 | 0.652 | 0.637 | 0.422 |
| SVR | 0.420 | 95.4 | 1.218 | 0.417 | 0.555 | 0.410 |
| RF | 0.458 | 100.4 | 1.282 | 0.424 | 0.503 | 0.219 |
| GBRT | 0.461 | 91.0 | 1.620 | 0.424 | 0.491 | 0.233 |
| DGM | 0.431 | 92.9 | 1.186 | 0.469 | 0.561 | 0.230 |
| GMEL | 0.440 | 94.3 | 1.204 | 0.445 | 0.355 | 0.207 |
| NetGAN | 0.487 | 89.1 | 1.138 | 0.429 | 0.354 | 0.191 |
| DiffODGen | 0.532 | 74.6 | 0.953 | 0.324 | 0.270 | 0.149 |
| WeDAN | 0.593 | 68.6 | 0.876 | 0.291 | 0.269 | 0.147 |
