from pathlib import Path
import random

import numpy as np
import torch

from models.GPS.config import TrainingConfig, WEIGHTS_DIR, cleanup_gpu, device
from .repeats import single_city_lgbm_run_id, single_city_run_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 42
INFERENCE_SEEDS = [42, 43, 44, 45, 46]

GPS_BENCHMARK_IDS = [
    "SC_TF_CE",
    "SC_TF_H",
    "SC_TF_CE_lape",
    "SC_TF_CE_gn",
    "SC_TF_focal",
    "SC_TF_CE_samp",
    "SC_TF_CE_nz",
    "SC_TF_CE_rle",
    "SC_TF_CE_lape_rle",
    "SC_TF_focal_rle",
    "SC_BL_CE",
    "SC_BL_H",
]

GPS_MC_BENCHMARK_IDS = [
    "MC_TF_CE",
    "MC_TF_H",
    "MC_TF_CE_lape",
    "MC_TF_focal",
    "MC_TF_CE_rle",
    "MC_BL_CE",
    "MC_BL_H",
]

BASELINE_MODELS = [
    # "RF",
    # "SVR",
    # "GBRT",
    # "DGM",
    # "GM_E",
    # "GM_P",
    "GMEL",
    "GMEL_LGBM",
    # "NetGAN",
    # "DiffODGen",
    # "WeDAN",
    "TransFlowerOrig",
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
RESULT_COLUMNS = [
    "CPC",
    "CPC_std",
    "CPC_val",
    "CPC_val_std",
    "CPC_full",
    "CPC_full_std",
    "CPC_nonzero",
    "CPC_nz",
    "CPC_nz_std",
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



def trained_gps_run_ids(run_ids):
    return [run_id for run_id in run_ids if (WEIGHTS_DIR / f"{run_id}.pt").exists()]



def trained_lgbm_run_ids(run_ids):
    return [f"{run_id}_lgbm" for run_id in run_ids if (WEIGHTS_DIR / f"{run_id}_lgbm.lgbm").exists()]


def trained_single_city_gps_run_ids(run_ids, city_ids=None):
    city_ids = list(SINGLE_CITY_IDS if city_ids is None else city_ids)
    return [
        run_id for run_id in run_ids
        if all((WEIGHTS_DIR / f"{single_city_run_id(run_id, city_id)}.pt").exists() for city_id in city_ids)
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
