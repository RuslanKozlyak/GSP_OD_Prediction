from pathlib import Path
import random

import numpy as np
import torch

from models.GPS.config import TrainingConfig, WEIGHTS_CPC_BEST_DIR, WEIGHTS_DIR, cleanup_gpu, device
from .repeats import single_city_lgbm_run_id, single_city_run_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 42
INFERENCE_SEEDS = [42, 43, 44, 45, 46]

# Per baseline training run. Set to None or <= 0 to disable.
BASELINE_TRAIN_TIMEOUT_SECONDS = int(1.5 * 60 * 60)

GPS_BENCHMARK_IDS = [
    "SC_TF_CE",
]

GPS_MC_BENCHMARK_IDS = [
    "MC_BL_CE_lape_log_gn",
]

BASELINE_MODELS = [
    # "RF",
    # "SVR",
    # "GBRT",
    # "DGM",
    # "GM_E",
    # "GM_P",
    # "GMEL",
    # "GMEL_LGBM",
    # "NetGAN",
    # "DiffODGen",
    # "WeDAN",
    # "TransFlowerOrig",
]

FLAT_BASELINE_MODELS = ["RF", "SVR", "GBRT", "DGM", "GM_E", "GM_P"]
GRAPH_BASELINE_MODELS = ["GMEL", "GMEL_GBRT", "GMEL_LGBM", "NetGAN"]
SEPARABLE_BASELINE_MODELS = FLAT_BASELINE_MODELS + GRAPH_BASELINE_MODELS + ["TransFlowerOrig"]

SINGLE_CITY_ID = "48201"
MULTI_CITY_IDS = ["17031", "48201", "04013", "06073", "06059", "36047", "12086", "48113", "06065", "36081"]
SINGLE_CITY_IDS = [SINGLE_CITY_ID] + [cid for cid in MULTI_CITY_IDS if cid != SINGLE_CITY_ID][:2]

TRANSFLOWER_ORIG_CONFIG = TrainingConfig(
    encoder_type="mlp",
    decoder_type="transflower",
    loss_type="ce",
    prediction_mode="normalized",
    use_dest_sampling=False,
    use_rle=True,
)

FLAT_CHUNK_SIZE = 200
FLAT_SGD_EPOCHS = 5
FLAT_BATCH_SIZE = 10_000

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
        "max_iter": 10_000,
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
        "patience": 100,
        "grad_clip": 1.0,
        "verbose": 1,
    },
    "GM_E": {
        "lr": 1e-4,
        "batch_size": 10_000,
        "max_epochs": 10_000,
        "patience": 100,
        "verbose": 1,
    },
    "GM_P": {
        "lr": 1e-3,
        "batch_size": 10_000,
        "max_epochs": 10_000,
        "patience": 100,
        "verbose": 1,
    },
    "GMEL": {
        "encoder_lr": 3e-4,
        "encoder_max_epochs": 10_000,
        "encoder_patience": 100,
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


RESULT_COLUMNS = [
    "CPC",
    "CPC_std",
    "CPC_val",
    "CPC_val_std",
    "CPC_full",
    "CPC_full_std",
    "CPC_train_full",
    "CPC_train_full_std",
    "CPC_val_full",
    "CPC_val_full_std",
    "CPC_nonzero",
    "CPC_nonzero_std",
    "CPC_train_nz",
    "CPC_train_nz_std",
    "CPC_val_nz",
    "CPC_val_nz_std",
    "CPC_test",
    "CPC_test_std",
    "RMSE",
    "RMSE_std",
    "MAE",
    "MAE_std",
    "MAPE",
    "MAPE_std",
    "SMAPE",
    "SMAPE_std",
    "MAE_test",
    "MAE_test_std",
    "RMSE_test",
    "RMSE_test_std",
    "accuracy",
    "accuracy_std",
    "matrix_COS_similarity",
    "matrix_COS_similarity_std",
    "JSD_inflow",
    "JSD_inflow_std",
    "JSD_outflow",
    "JSD_outflow_std",
    "JSD_ODflow",
    "JSD_ODflow_std",
    "n_runs",
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
    if model_name in ("DGM", "GM_E", "GM_P", "NetGAN", "TransFlowerOrig"):
        base_paths = [WEIGHTS_DIR / f"{run_id}.pt"]
        if model_name == "TransFlowerOrig":
            base_paths.append(WEIGHTS_DIR / f"{run_id}.json")
        if model_name == "NetGAN":
            base_paths.append(WEIGHTS_DIR / f"{run_id}_meta.joblib")
        return base_paths
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
