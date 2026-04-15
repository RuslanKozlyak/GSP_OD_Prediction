from pathlib import Path
import random
from dataclasses import replace
import numpy as np
import torch

from models.GPS.config import (
    FEATURE_PRESET,
    MULTI_CITY_IDS as GPS_MULTI_CITY_IDS,
    SINGLE_CITY_ID as GPS_SINGLE_CITY_ID,
    SINGLE_CITY_IDS as GPS_SINGLE_CITY_IDS,
    TrainingConfig,
    WEIGHTS_CPC_BEST_DIR,
    WEIGHTS_DIR,
    cleanup_gpu,
    device,
)
from .repeats import single_city_lgbm_run_id, single_city_run_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_PATH_WITH_LU = PROJECT_ROOT / "data_lu" / "data"
DATA_PATH = _DATA_PATH_WITH_LU if _DATA_PATH_WITH_LU.exists() else (PROJECT_ROOT / "data")
RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 42

# Per baseline training run. Set to None or <= 0 to disable.
BASELINE_TRAIN_TIMEOUT_SECONDS = int(1.5 * 60 * 60)

# Per-model timeout overrides (seconds). Models not listed fall back to
# BASELINE_TRAIN_TIMEOUT_SECONDS. GAN-based baselines require significantly
# more wall-clock time per epoch.
BASELINE_TRAIN_TIMEOUT_OVERRIDES = {
    "GAT_GAN_Orig": 6 * 60 * 60,
    "ODGN":         6 * 60 * 60,
}


def baseline_train_timeout_seconds(model_name):
    """Return per-model training timeout, falling back to the shared default."""
    return BASELINE_TRAIN_TIMEOUT_OVERRIDES.get(
        model_name, BASELINE_TRAIN_TIMEOUT_SECONDS
    )

GPS_BENCHMARK_IDS = [
    "SC_GG_CE_NORM_LAPE",
    "SC_GG_MULTITASK_RAW_LOG_LAPE",
    "SC_TF_CE_NORM_LAPE",
    "SC_GAT_GAN_ORIG_PAPER_RAW_NONE_NOPE",
    "SC_GG_GAN_CE_NORM_GN_LAPE",
    "SC_ODGN_PAPER",
]

GPS_MC_BENCHMARK_IDS = [
    "MC_GG_CE_NORM_LAPE",
    "MC_GG_MULTITASK_RAW_LOG_LAPE",
    "MC_TF_CE_NORM_LAPE",
    "MC_GAT_GAN_ORIG_PAPER_RAW_NONE_NOPE",
    "MC_GG_GAN_CE_NORM_GN_LAPE",
    "MC_ODGN_PAPER",
]

BASELINE_MODELS = [
    "RF",
    # "SVR",
    "GBRT",
    "DGM",
    "GM_E",
    "GM_P",
    "GMEL",
    "GMEL_LGBM",
    "TransFlowerOrig",
    "GAT_GAN_Orig",
    "ODGN",
]

FLAT_BASELINE_MODELS = ["RF", "SVR", "GBRT", "DGM", "GM_E", "GM_P"]
GRAPH_BASELINE_MODELS = ["GMEL", "GMEL_GBRT", "GMEL_LGBM", "NetGAN"]
GPS_BASELINE_MODELS = ["TransFlowerOrig", "GAT_GAN_Orig", "ODGN"]
SEPARABLE_BASELINE_MODELS = FLAT_BASELINE_MODELS + GRAPH_BASELINE_MODELS + GPS_BASELINE_MODELS

SINGLE_CITY_ID = GPS_SINGLE_CITY_ID
MULTI_CITY_IDS = list(GPS_MULTI_CITY_IDS)
SINGLE_CITY_IDS = list(GPS_SINGLE_CITY_IDS)
DEFAULT_FEATURE_MODE = "full" if FEATURE_PRESET == "all" else "reduced"

TRANSFLOWER_ORIG_CONFIG = TrainingConfig(
    encoder_type="mlp",
    decoder_type="transflower",
    loss_type="ce",
    prediction_mode="normalized",
    use_dest_sampling=False,
    pair_split_mode="nonzero_pairs",
    use_rle=True,
)

GAT_GAN_ORIG_CONFIG = TrainingConfig(
    encoder_type="gat",
    decoder_type="linear",
    training_mode="gan",
    gan_only=True,
    loss_type="mae",
    prediction_mode="raw",
    use_log_transform=False,
    pe_type=None,
    gps_norm_type="none",
    gnn_layers=3,
    gnn_heads=8,
    use_dest_sampling=False,
    pair_split_mode="nonzero_pairs",
    learning_rate=3e-4,
    discriminator_lr=3e-4,
    weight_decay=0.0,
    patience=200,
    adv_weight=1.0,
    gan_pretrain_epochs=0,
    gan_regularizer="clip",
    gan_clip_value=0.01,
    gan_gp_lambda=0.0,
    gan_n_critic=5,
    gan_n_critic_after_epoch=300,
    gan_n_critic_after=1,
    gan_noise_dim=0,
    gan_noise_dim_mode="match_input",
    gan_eval_num_samples=4,
    gan_use_supervised_monitoring=True,
    gat_use_edge_attr=True,
    pair_use_distance=True,
)

ODGN_BASELINE_CONFIG = replace(
    GAT_GAN_ORIG_CONFIG,
    decoder_type="gravity_guided",
)

GPS_BASELINE_CONFIGS = {
    "TransFlowerOrig": TRANSFLOWER_ORIG_CONFIG,
    "GAT_GAN_Orig": GAT_GAN_ORIG_CONFIG,
    "ODGN": ODGN_BASELINE_CONFIG,
}

FLAT_CHUNK_SIZE = 200
FLAT_SGD_EPOCHS = 5
FLAT_BATCH_SIZE = 10_000

# LinearSVR/liblinear can allocate a lot of RAM on multi-city flat OD matrices.
# Cap the multi-city training set before fitting so the subprocess cannot OOM
# the notebook/VS Code host. Set to None or <= 0 to use all pairs.
SVR_MULTI_CITY_MAX_TRAIN_SAMPLES = 250_000

# RF / GBRT fit the whole flat OD matrix via sklearn and will not finish in the
# default 1.5h baseline timeout for multi-city splits. Cap the training set
# analogously to SVR. Set to None or <= 0 to use all pairs.
RF_MULTI_CITY_MAX_TRAIN_SAMPLES = 250_000
GBRT_MULTI_CITY_MAX_TRAIN_SAMPLES = 250_000

FLAT_MULTI_CITY_TRAIN_CAPS = {
    "SVR":  SVR_MULTI_CITY_MAX_TRAIN_SAMPLES,
    "RF":   RF_MULTI_CITY_MAX_TRAIN_SAMPLES,
    "GBRT": GBRT_MULTI_CITY_MAX_TRAIN_SAMPLES,
}

# ─── Baseline model hyperparameters ──────────────────────────────────────────
BASELINE_HYPERPARAMS = {
    "RF": {
        "n_estimators": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "verbose": 1,
    },
    "SVR": {
        "C": 100,
        "loss": "squared_epsilon_insensitive",
        "max_iter": 10_000,
        "dual": False,
        "verbose": 1,
    },
    "GBRT": {
        "n_estimators": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "verbose": 1,
    },
    "DGM": {
        "lr": 3e-4,
        "batch_size": 10_000,
        "max_epochs": 10_000,
        "patience": 25,
        "grad_clip": 1.0,
        "verbose": 1,
    },
    "GM_E": {
        "lr": 1e-4,
        "batch_size": 10_000,
        "max_epochs": 10_000,
        "patience": 25,
        "verbose": 1,
    },
    "GM_P": {
        "lr": 1e-3,
        "batch_size": 10_000,
        "max_epochs": 10_000,
        "patience": 25,
        "verbose": 1,
    },
    "GMEL": {
        "encoder_lr": 3e-4,
        "encoder_max_epochs": 10_000,
        "encoder_patience": 25,
        "decoder_type": "gbrt",
        "verbose": 1,
        "lgbm_learning_rate": 0.05,
        "lgbm_num_leaves": 63,
        "lgbm_max_depth": 8,
        "lgbm_subsample": 0.8,
        "lgbm_colsample_bytree": 0.8,
        "lgbm_num_boost_round": 1000,
        "lgbm_early_stopping": 50,
        "lgbm_log_period": 100,
        "gbrt_n_estimators": 20,
        "gbrt_min_samples_split": 2,
        "gbrt_min_samples_leaf": 2,
    },
    "NetGAN": {
        "lr": 3e-4,
        "n_epochs": 2,
        "gp_lambda": 10,
        "batch_size": 128,
        "verbose": 1,
    },
}

def get_baseline_hyperparams(model_name):
    if model_name in ("GMEL_GBRT", "GMEL_LGBM"):
        params = dict(BASELINE_HYPERPARAMS.get("GMEL", {}))
        params["decoder_type"] = "lgbm" if model_name == "GMEL_LGBM" else "gbrt"
        return params
    return dict(BASELINE_HYPERPARAMS.get(model_name, {}))


_SPLIT_METRIC_NAMES = ("CPC", "MAE", "RMSE", "MAPE", "SMAPE", "NRMSE")
_RESULT_METRIC_COLUMNS = ["num_regions"]
for metric_name in _SPLIT_METRIC_NAMES:
    _RESULT_METRIC_COLUMNS.extend([
        f"{metric_name}_full",
        f"{metric_name}_nz",
        f"{metric_name}_test_full",
        f"{metric_name}_test_nz",
        f"{metric_name}_train_full",
        f"{metric_name}_val_full",
        f"{metric_name}_train_nz",
        f"{metric_name}_val_nz",
    ])
_RESULT_METRIC_COLUMNS.extend([
    "accuracy",
    "matrix_COS_similarity",
    "JSD_inflow",
    "JSD_outflow",
    "JSD_ODflow",
])
RESULT_COLUMNS = [
    * _RESULT_METRIC_COLUMNS,
]


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def baseline_single_city_run_id(model_name, city_id):
    return single_city_run_id(f"benchmark_{model_name}", city_id)


def baseline_multi_city_run_id(model_name):
    return f"benchmark_{model_name}__multi_city"


def baseline_artifact_paths(model_name, run_id):
    if model_name in ("RF", "SVR", "GBRT"):
        return [WEIGHTS_DIR / f"{run_id}.joblib"]
    if model_name in ("DGM", "GM_E", "GM_P", "NetGAN"):
        base_paths = [WEIGHTS_DIR / f"{run_id}.pt"]
        if model_name == "NetGAN":
            base_paths.append(WEIGHTS_DIR / f"{run_id}_meta.joblib")
        return base_paths
    if model_name in GPS_BASELINE_MODELS:
        return [
            WEIGHTS_DIR / f"{run_id}.pt",
            WEIGHTS_DIR / f"{run_id}.json",
        ]
    if model_name in ("GMEL", "GMEL_GBRT"):
        return [
            WEIGHTS_DIR / f"{run_id}.pt",
            WEIGHTS_DIR / f"{run_id}_gbrt.joblib",
            WEIGHTS_DIR / f"{run_id}_meta.joblib",
        ]
    if model_name == "GMEL_LGBM":
        return [
            WEIGHTS_DIR / f"{run_id}.pt",
            WEIGHTS_DIR / f"{run_id}_lgbm.lgbm",
            WEIGHTS_DIR / f"{run_id}_meta.joblib",
        ]
    raise ValueError(f"Unsupported baseline model for artifact lookup: {model_name}")


def has_trained_baseline_artifacts(model_name, run_id):
    return all(path.exists() for path in baseline_artifact_paths(model_name, run_id))



def trained_gps_run_ids(run_ids, weights_dir=WEIGHTS_DIR):
    return [run_id for run_id in run_ids if (weights_dir / f"{run_id}.pt").exists()]



def trained_lgbm_run_ids(run_ids):
    return [f"{run_id}_lgbm" for run_id in run_ids if (WEIGHTS_DIR / f"{run_id}_lgbm.lgbm").exists()]


def trained_single_city_gps_run_ids(run_ids, city_ids=None, weights_dir=WEIGHTS_DIR):
    city_ids = list(SINGLE_CITY_IDS if city_ids is None else city_ids)
    return [
        run_id for run_id in run_ids
        if all((weights_dir / f"{single_city_run_id(run_id, city_id)}.pt").exists() for city_id in city_ids)
    ]


def trained_single_city_lgbm_base_ids(run_ids, city_ids=None):
    city_ids = list(SINGLE_CITY_IDS if city_ids is None else city_ids)
    return [
        run_id for run_id in run_ids
        if all((WEIGHTS_DIR / f"{single_city_lgbm_run_id(run_id, city_id)}.lgbm").exists() for city_id in city_ids)
    ]


def trained_single_city_baseline_models(model_names=None, city_ids=None):
    city_ids = list(SINGLE_CITY_IDS if city_ids is None else city_ids)
    model_names = list(SEPARABLE_BASELINE_MODELS if model_names is None else model_names)
    return [
        model_name for model_name in model_names
        if all(
            has_trained_baseline_artifacts(
                model_name, baseline_single_city_run_id(model_name, city_id)
            )
            for city_id in city_ids
        )
    ]


def trained_multi_city_baseline_models(model_names=None):
    model_names = list(SEPARABLE_BASELINE_MODELS if model_names is None else model_names)
    return [
        model_name for model_name in model_names
        if has_trained_baseline_artifacts(
            model_name, baseline_multi_city_run_id(model_name)
        )
    ]


GMEL_GPS_BENCHMARK_IDS = [
    "GMEL_GPS_rwpe",
    "GMEL_GPS_lape",
    "GMEL_GPS_nope",
    "GMEL_GPS_rwpe_gn",
]


def trained_gmel_gps_run_ids():
    return [
        rid for rid in GMEL_GPS_BENCHMARK_IDS
        if (WEIGHTS_DIR / f"{rid}.pt").exists()
        and (WEIGHTS_DIR / f"{rid}_gbrt.joblib").exists()
    ]
