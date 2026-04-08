import os
import csv
import json
import warnings
from dataclasses import dataclass, field, replace, asdict
from typing import Optional, Literal
from datetime import datetime
from pathlib import Path

import torch

warnings.filterwarnings('ignore')

# ─── Paths (relative to project root) ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = str(PROJECT_ROOT / "data_lu/data")
SHP_PATH = str(PROJECT_ROOT / "assets" / "Boundaries_Regions_within_Areas")
RESULTS_DIR = PROJECT_ROOT / "results"
WEIGHTS_DIR = RESULTS_DIR / "weights"
WEIGHTS_CPC_BEST_DIR = RESULTS_DIR / "weights_CPC_best"
METRICS_CSV = RESULTS_DIR / "metrics.csv"
METRICS_VAL_LOSS_CSV = RESULTS_DIR / "metrics_val_loss.csv"
METRICS_CPC_NZ_BEST_CSV = RESULTS_DIR / "metrics_cpc_nz_best.csv"
METRICS_RUNS_DIR = RESULTS_DIR / "metrics_runs"

SINGLE_CITY_ID = "48201"
MULTI_CITY_IDS = ["17031","48201","04013","06073","06059","36047","12086","48113","06065","36081", "32003", "42003"]
SINGLE_CITY_IDS = [SINGLE_CITY_ID] + [cid for cid in MULTI_CITY_IDS if cid != SINGLE_CITY_ID][:2]

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
EPOCHS = 200
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 30
ORIGIN_BATCH_SIZE = 32
DEST_BATCH_SIZE = 256
N_DEST_SAMPLE = 128
MC_EPOCHS = 200

# ─── Loss ─────────────────────────────────────────────────────────────────────
HUBER_KDE_BW = 2.0
HUBER_MIN_PROB = 1e-4

# ─── Features ─────────────────────────────────────────────────────────────────
USE_LU_FEATURES = True
USE_JOBS_FEATURES = True

# ─── NaN protection ──────────────────────────────────────────────────────────
NAN_BATCH_THRESHOLD = 0.5

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # НЕ МЕНЯТЬ CUDA:1


def cleanup_gpu():
    """Free GPU memory between model runs."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class TrainingConfig:
    # fmt: off
    # ── Architecture ──────────────────────────────────────────────────────────
    encoder_type:       Literal['gps', 'mlp', 'gat']                        = 'gps'
    decoder_type:       Literal['bilinear', 'linear', 'transflower', 'gravity_guided', 'lgbm', 'gbrt'] = 'transflower'
    pe_type:            Optional[Literal['rwpe', 'spe', 'rrwp', 'lape']]    = 'rwpe'
    gps_norm_type:      Literal['batch_norm', 'graph_norm']                 = 'batch_norm'
    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_type:          Literal['huber', 'ce', 'ce_old', 'multitask', 'zinb', 'focal', 'mae'] = 'huber'
    prediction_mode:    Literal['raw', 'normalized']                        = 'raw'
    use_log_transform:  bool  = False
    focal_gamma:        float = 2.0   # used only when loss_type='focal'
    huber_delta:        float = 1.0
    lambda_main:        float = 0.5
    lambda_sub:         float = 0.5
    normalize_multitask: bool = True
    # ── Destination sampling ──────────────────────────────────────────────────
    use_dest_sampling:  bool  = True
    n_dest_sample:      int   = N_DEST_SAMPLE
    include_zero_pairs: bool  = True
    zero_pair_ratio:    float = 0.3
    # ── Training schedule ─────────────────────────────────────────────────────
    epochs:             int   = EPOCHS
    learning_rate:      float = LEARNING_RATE
    weight_decay:       float = WEIGHT_DECAY
    patience:           int   = PATIENCE
    mc_epochs:          int   = MC_EPOCHS
    # GAN training (ODGN / GAT-GAN style adversarial topology loss)
    training_mode:      Literal['supervised', 'gan'] = 'supervised'
    gan_only:           bool  = False
    adv_weight:         float = 0.05
    discriminator_lr:   float = LEARNING_RATE
    gan_gp_lambda:      float = 10.0
    gan_n_critic:       int   = 1
    gan_pretrain_epochs: int  = 20
    gan_walk_len:       int   = 64
    gan_walk_batch_size: int  = 64
    gan_tau:            float = 1.0
    gan_disc_hidden_dim: int  = 64
    gan_disc_layers:    int   = 4
    gan_disc_dropout:   float = 0.05
    # ── RLE (Relative Location Encoder) ───────────────────────────────────────
    use_rle:            bool  = False
    rle_freq:           int   = 16
    rle_out_dim:        int   = 64
    rle_lambda_min:     float = 1.0
    rle_lambda_max:     float = 20000.0
    # ── GBRT / LGBM decoder (for GMEL_GPS mode) ─────────────────────────────
    gbrt_n_estimators:  int   = 20
    lgbm_n_estimators:  int   = 1000
    lgbm_num_leaves:    int   = 63
    # fmt: on

    def __post_init__(self):
        _valid = {
            'encoder_type':    ('gps', 'mlp', 'gat'),
            'decoder_type':    ('bilinear', 'linear', 'transflower', 'gravity_guided', 'lgbm', 'gbrt'),
            'pe_type':         ('rwpe', 'spe', 'rrwp', 'lape', None),
            'gps_norm_type':   ('batch_norm', 'graph_norm'),
            'loss_type':       ('huber', 'ce', 'ce_old', 'multitask', 'zinb', 'focal', 'mae'),
            'prediction_mode': ('raw', 'normalized'),
            'training_mode':   ('supervised', 'gan'),
        }
        for attr, choices in _valid.items():
            val = getattr(self, attr)
            if val not in choices:
                raise ValueError(
                    f"TrainingConfig.{attr}={val!r} is invalid. "
                    f"Valid options: {choices}"
                )
        if self.adv_weight < 0:
            raise ValueError("TrainingConfig.adv_weight must be non-negative")
        if self.gan_only and self.training_mode != 'gan':
            raise ValueError("TrainingConfig.gan_only=True requires training_mode='gan'")
        if self.discriminator_lr <= 0:
            raise ValueError("TrainingConfig.discriminator_lr must be positive")
        if self.gan_gp_lambda < 0:
            raise ValueError("TrainingConfig.gan_gp_lambda must be non-negative")
        if self.gan_n_critic < 1:
            raise ValueError("TrainingConfig.gan_n_critic must be >= 1")
        if self.gan_pretrain_epochs < 0:
            raise ValueError("TrainingConfig.gan_pretrain_epochs must be >= 0")
        if self.gan_walk_len < 1:
            raise ValueError("TrainingConfig.gan_walk_len must be >= 1")
        if self.gan_walk_batch_size < 1:
            raise ValueError("TrainingConfig.gan_walk_batch_size must be >= 1")
        if self.gan_tau <= 0:
            raise ValueError("TrainingConfig.gan_tau must be positive")
        if self.gan_disc_hidden_dim < 1:
            raise ValueError("TrainingConfig.gan_disc_hidden_dim must be >= 1")
        if self.gan_disc_layers < 1:
            raise ValueError("TrainingConfig.gan_disc_layers must be >= 1")
        if not 0 <= self.gan_disc_dropout < 1:
            raise ValueError("TrainingConfig.gan_disc_dropout must be in [0, 1)")

    def describe(self):
        enc = {'mlp': 'MLP', 'gat': 'GAT'}.get(self.encoder_type, 'GPS')
        pe_name = 'none' if self.pe_type is None else self.pe_type
        parts = [f"{enc}+{self.decoder_type}+{self.loss_type}+{self.prediction_mode}",
                 f"pe={pe_name}", f"norm={self.gps_norm_type}"]
        if self.use_log_transform:
            parts.append(
                "log_norm"
                if self.prediction_mode == 'normalized' and self.loss_type in ('huber', 'multitask', 'mae')
                else "log"
            )
        if self.use_rle: parts.append("RLE")
        if self.loss_type == 'focal': parts.append(f"γ={self.focal_gamma}")
        if self.training_mode == 'gan':
            parts.append(
                f"GAN adv={self.adv_weight:g} ncritic={self.gan_n_critic} "
                f"walk={self.gan_walk_batch_size}x{self.gan_walk_len}"
            )
            if self.gan_only:
                parts.append("gan_only")
        parts.append(f"zeros={self.include_zero_pairs} samp={self.use_dest_sampling}")
        return " | ".join(parts)


# ─── Result saving ────────────────────────────────────────────────────────────
def ensure_dirs():
    RESULTS_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    WEIGHTS_CPC_BEST_DIR.mkdir(exist_ok=True)
    METRICS_RUNS_DIR.mkdir(exist_ok=True)


def save_metrics_to_csv(run_id, run_name, config, metrics_full, metrics_nz,
                        metrics_test, n_params, epochs_trained, status='ok',
                        metrics_csv=None, run_suffix=None,
                        checkpoint_selection=None, selected_epoch=None,
                        selection_metric=None, selection_metric_value=None,
                        train_val_metrics=None):
    ensure_dirs()
    metrics_csv = Path(metrics_csv) if metrics_csv is not None else METRICS_CSV
    metrics_csv.parent.mkdir(exist_ok=True)
    train_val_metrics = train_val_metrics or {}
    row = {
        'timestamp': datetime.now().isoformat(),
        'run_id': run_id, 'name': run_name, 'status': status,
        'checkpoint_selection': checkpoint_selection,
        'selected_epoch': selected_epoch,
        'selection_metric': selection_metric,
        'selection_metric_value': selection_metric_value,
        'encoder_type': config.encoder_type,
        'decoder': config.decoder_type, 'loss_type': config.loss_type,
        'training_mode': config.training_mode,
        'gan_only': config.gan_only,
        'prediction_mode': config.prediction_mode,
        'pe_type': config.pe_type, 'gps_norm_type': config.gps_norm_type,
        'use_log_transform': config.use_log_transform,
        'use_dest_sampling': config.use_dest_sampling,
        'include_zero_pairs': config.include_zero_pairs,
        'zero_pair_ratio': config.zero_pair_ratio,
        'use_rle': config.use_rle,
        'learning_rate': config.learning_rate,
        'discriminator_lr': config.discriminator_lr,
        'weight_decay': config.weight_decay,
        'adv_weight': config.adv_weight,
        'gan_gp_lambda': config.gan_gp_lambda,
        'gan_n_critic': config.gan_n_critic,
        'gan_pretrain_epochs': config.gan_pretrain_epochs,
        'gan_walk_len': config.gan_walk_len,
        'gan_walk_batch_size': config.gan_walk_batch_size,
        'n_params': n_params, 'epochs_trained': epochs_trained,
        'CPC_full': metrics_full.get('CPC'), 'CPC_nz': metrics_nz.get('CPC'),
        'CPC_test': metrics_test.get('CPC'),
        'CPC_val': train_val_metrics.get('CPC_val_nz'),
        'CPC_train_full': train_val_metrics.get('CPC_train_full'),
        'CPC_val_full': train_val_metrics.get('CPC_val_full'),
        'CPC_train_nz': train_val_metrics.get('CPC_train_nz'),
        'CPC_val_nz': train_val_metrics.get('CPC_val_nz'),
        'MAE_full': metrics_full.get('MAE'), 'RMSE_full': metrics_full.get('RMSE'),
        # Full metric suite from cal_od_metrics
        'MAE_nz': metrics_nz.get('MAE'),
        'RMSE_nz': metrics_nz.get('RMSE'),
        'MAE_test': metrics_test.get('MAE'),
        'RMSE_test': metrics_test.get('RMSE'),
        'NRMSE_full': metrics_full.get('NRMSE'),
        'MAPE_full': metrics_full.get('MAPE'),
        'SMAPE_full': metrics_full.get('SMAPE'),
        'accuracy': metrics_full.get('accuracy'),
        'matrix_COS_similarity': metrics_full.get('matrix_COS_similarity'),
        'JSD_inflow': metrics_full.get('JSD_inflow'),
        'JSD_outflow': metrics_full.get('JSD_outflow'),
        'JSD_ODflow': metrics_full.get('JSD_ODflow'),
    }
    _append_metrics_row(metrics_csv, row)
    print(f"  -> Metrics saved to {metrics_csv}")
    suffix = f"__{run_suffix}" if run_suffix else ""
    metrics_path = METRICS_RUNS_DIR / f"{run_id}{suffix}.json"
    with open(metrics_path, 'w') as f:
        json.dump(row, f, indent=2)
    print(f"  -> Metrics saved to {metrics_path}")


def _append_metrics_row(metrics_csv, row):
    fieldnames = list(row.keys())
    if metrics_csv.exists() and metrics_csv.stat().st_size > 0:
        with open(metrics_csv, newline='') as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)
        if existing_fields and existing_fields != fieldnames:
            merged_fields = existing_fields + [
                key for key in fieldnames if key not in existing_fields
            ]
            with open(metrics_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=merged_fields)
                writer.writeheader()
                for old_row in existing_rows:
                    old_row.pop(None, None)
                    writer.writerow(old_row)
                writer.writerow(row)
            return

    file_exists = metrics_csv.exists() and metrics_csv.stat().st_size > 0
    with open(metrics_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_model_weights(run_id, model_or_state, config=None, weights_dir=WEIGHTS_DIR):
    ensure_dirs()
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(exist_ok=True)
    state_dict = model_or_state.state_dict() if hasattr(model_or_state, 'state_dict') else model_or_state
    path = weights_dir / f"{run_id}.pt"
    torch.save(state_dict, path)
    print(f"  -> Weights saved to {path}")
    if config is not None:
        cfg_path = weights_dir / f"{run_id}.json"
        with open(cfg_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        print(f"  -> Config  saved to {cfg_path}")


def load_model_config(run_id, weights_dir=WEIGHTS_DIR):
    """Load TrainingConfig saved alongside model weights. Returns None if not found."""
    cfg_path = Path(weights_dir) / f"{run_id}.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        return TrainingConfig(**json.load(f))
