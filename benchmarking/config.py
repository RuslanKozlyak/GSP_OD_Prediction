from pathlib import Path
import random

import numpy as np
import torch

from models.GPS.config import TrainingConfig, WEIGHTS_DIR, cleanup_gpu, device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 42

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
]

SINGLE_CITY_ID = "48201"
MULTI_CITY_IDS = ["17031", "48201", "04013", "06073", "06059", "36047", "12086", "48113", "06065", "36081"]

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
    "CPC_val",
    "CPC_full",
    "CPC_nonzero",
    "CPC_nz",
    "CPC_test",
    "RMSE",
    "MAE",
    "MAPE",
    "SMAPE",
    "MAE_test",
    "RMSE_test",
    "accuracy",
    "matrix_COS_similarity",
    "JSD_inflow",
    "JSD_outflow",
    "JSD_ODflow",
]


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def trained_gps_run_ids(run_ids):
    return [run_id for run_id in run_ids if (WEIGHTS_DIR / f"{run_id}.pt").exists()]



def trained_lgbm_run_ids(run_ids):
    return [f"{run_id}_lgbm" for run_id in run_ids if (WEIGHTS_DIR / f"{run_id}_lgbm.lgbm").exists()]


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
