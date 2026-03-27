import os
import csv
import warnings
from dataclasses import dataclass, field, replace, asdict
from typing import Optional, Literal
from datetime import datetime
from pathlib import Path

import torch

warnings.filterwarnings('ignore')

# ─── Paths (relative to project root) ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = str(PROJECT_ROOT / "data")
SHP_PATH = str(PROJECT_ROOT / "assets" / "Boundaries_Regions_within_Areas")
RESULTS_DIR = PROJECT_ROOT / "results"
WEIGHTS_DIR = RESULTS_DIR / "weights"
METRICS_CSV = RESULTS_DIR / "metrics.csv"

SINGLE_CITY_ID = "48201"
MULTI_CITY_IDS = ["17031","48201","04013","06073","06059","36047","12086","48113","06065","36081"]

# ─── Architecture ─────────────────────────────────────────────────────────────
HIDDEN_DIM = 64
PE_DIM = 8
PE_WALK_LEN = 20
GPS_HEADS = 4
GPS_LAYERS = 4
GPS_DROPOUT = 0.1
TF_HEADS = 4
TF_LAYERS = 2
TF_DROPOUT = 0.1

# ─── Training ─────────────────────────────────────────────────────────────────
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 30
ORIGIN_BATCH_SIZE = 32
DEST_BATCH_SIZE = 256
N_DEST_SAMPLE = 128
MC_EPOCHS = 30

# ─── Loss ─────────────────────────────────────────────────────────────────────
HUBER_DELTA = 1.0
HUBER_KDE_BW = 2.0
HUBER_MIN_PROB = 1e-4
LAMBDA_MAIN = 0.5
LAMBDA_SUB = 0.5
NORMALIZE_MULTITASK = True

# ─── Features ─────────────────────────────────────────────────────────────────
USE_LU_FEATURES = False
USE_JOBS_FEATURES = False

# ─── NaN protection ──────────────────────────────────────────────────────────
NAN_BATCH_THRESHOLD = 0.5

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class TrainingConfig:
    # fmt: off
    # ── Architecture ──────────────────────────────────────────────────────────
    encoder_type:       Literal['gps', 'mlp']                               = 'gps'
    decoder_type:       Literal['bilinear', 'transflower']                  = 'transflower'
    pe_type:            Literal['rwpe', 'spe', 'rrwp', 'lape']              = 'rwpe'
    gps_norm_type:      Literal['batch_norm', 'graph_norm', 'granola']      = 'batch_norm'
    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_type:          Literal['huber', 'ce', 'multitask', 'zinb', 'focal'] = 'huber'
    prediction_mode:    Literal['raw', 'normalized']                        = 'raw'
    use_log_transform:  bool  = False
    focal_gamma:        float = 2.0   # used only when loss_type='focal'
    # ── Destination sampling ──────────────────────────────────────────────────
    use_dest_sampling:  bool  = True
    n_dest_sample:      int   = N_DEST_SAMPLE
    include_zero_pairs: bool  = True
    zero_pair_ratio:    float = 0.3
    # ── Training schedule ─────────────────────────────────────────────────────
    epochs:             int   = EPOCHS
    learning_rate:      float = LEARNING_RATE
    patience:           int   = PATIENCE
    mc_epochs:          int   = MC_EPOCHS
    # ── RLE (Relative Location Encoder) ───────────────────────────────────────
    use_rle:            bool  = False
    rle_freq:           int   = 16
    rle_out_dim:        int   = 64
    rle_lambda_min:     float = 1.0
    rle_lambda_max:     float = 20000.0
    # fmt: on

    def __post_init__(self):
        _valid = {
            'encoder_type':    ('gps', 'mlp'),
            'decoder_type':    ('bilinear', 'transflower'),
            'pe_type':         ('rwpe', 'spe', 'rrwp', 'lape'),
            'gps_norm_type':   ('batch_norm', 'graph_norm', 'granola'),
            'loss_type':       ('huber', 'ce', 'multitask', 'zinb', 'focal'),
            'prediction_mode': ('raw', 'normalized'),
        }
        for attr, choices in _valid.items():
            val = getattr(self, attr)
            if val not in choices:
                raise ValueError(
                    f"TrainingConfig.{attr}={val!r} is invalid. "
                    f"Valid options: {choices}"
                )

    def describe(self):
        enc = 'MLP' if self.encoder_type == 'mlp' else 'GPS'
        parts = [f"{enc}+{self.decoder_type}+{self.loss_type}+{self.prediction_mode}",
                 f"pe={self.pe_type}", f"norm={self.gps_norm_type}"]
        if self.use_log_transform: parts.append("log")
        if self.use_rle: parts.append("RLE")
        if self.loss_type == 'focal': parts.append(f"γ={self.focal_gamma}")
        parts.append(f"zeros={self.include_zero_pairs} samp={self.use_dest_sampling}")
        return " | ".join(parts)


# ─── Result saving ────────────────────────────────────────────────────────────
def ensure_dirs():
    RESULTS_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)


def save_metrics_to_csv(run_id, run_name, config, metrics_full, metrics_nz,
                        metrics_test, n_params, epochs_trained, status='ok'):
    ensure_dirs()
    row = {
        'timestamp': datetime.now().isoformat(),
        'run_id': run_id, 'name': run_name, 'status': status,
        'decoder': config.decoder_type, 'loss_type': config.loss_type,
        'prediction_mode': config.prediction_mode,
        'pe_type': config.pe_type, 'gps_norm_type': config.gps_norm_type,
        'use_log_transform': config.use_log_transform,
        'n_params': n_params, 'epochs_trained': epochs_trained,
        'CPC_full': metrics_full['CPC'], 'CPC_nz': metrics_nz['CPC'],
        'CPC_test': metrics_test['CPC'],
        'MAE_full': metrics_full['MAE'], 'RMSE_full': metrics_full['RMSE'],
    }
    file_exists = METRICS_CSV.exists()
    with open(METRICS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(row)
    print(f"  -> Metrics saved to {METRICS_CSV}")


def save_model_weights(run_id, model, config=None):
    ensure_dirs()
    path = WEIGHTS_DIR / f"{run_id}.pt"
    torch.save(model.state_dict(), path)
    print(f"  -> Weights saved to {path}")
    if config is not None:
        import json
        cfg_path = WEIGHTS_DIR / f"{run_id}.json"
        with open(cfg_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        print(f"  -> Config  saved to {cfg_path}")


def load_model_config(run_id):
    """Load TrainingConfig saved alongside model weights. Returns None if not found."""
    import json
    cfg_path = WEIGHTS_DIR / f"{run_id}.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        return TrainingConfig(**json.load(f))
