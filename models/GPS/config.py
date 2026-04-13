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
_DATA_PATH_WITH_LU = PROJECT_ROOT / "data_lu" / "data"
_DATA_PATH_DEFAULT = PROJECT_ROOT / "data"
DATA_PATH = str(_DATA_PATH_WITH_LU if _DATA_PATH_WITH_LU.exists() else _DATA_PATH_DEFAULT)
SHP_PATH = str(PROJECT_ROOT / "assets" / "Boundaries_Regions_within_Areas")
RESULTS_DIR = PROJECT_ROOT / "results"
WEIGHTS_DIR = RESULTS_DIR / "weights"
WEIGHTS_CPC_BEST_DIR = RESULTS_DIR / "weights_CPC_best"
METRICS_CSV = RESULTS_DIR / "metrics.csv"
METRICS_VAL_LOSS_CSV = RESULTS_DIR / "metrics_val_loss.csv"
METRICS_CPC_NZ_BEST_CSV = RESULTS_DIR / "metrics_cpc_nz_best.csv"
METRICS_RUNS_DIR = RESULTS_DIR / "metrics_runs"

# NOTE: the local dataset does not contain 06037 (Los Angeles County), so we use
# 06059 as the available Greater Los Angeles proxy in the configured city set.
CITY_LABELS = {
    "36061": "New York City",
    "06059": "Los Angeles (06059 proxy)",
    "17031": "Chicago",
    "48201": "Houston",
    "06075": "San Francisco",
    "53033": "Seattle",
    "11001": "Washington D.C.",
    "47157": "Memphis",
}
# Legacy pre-8-city setup kept here for quick rollback:
# CITY_LABELS = {
#     "17031": "Chicago",
#     "48201": "Houston",
#     "04013": "Phoenix",
#     "06073": "San Diego",
#     "06059": "Los Angeles (06059 proxy)",
#     "36047": "Brooklyn",
#     "12086": "Miami-Dade",
#     "48113": "Dallas",
#     "06065": "Riverside",
#     "36081": "Queens",
#     "32003": "Las Vegas",
#     "42003": "Allegheny",
# }
# MULTI_CITY_IDS = list(CITY_LABELS.keys())
# SINGLE_CITY_IDS = ["48201", "17031", "04013"]
# SINGLE_CITY_ID = SINGLE_CITY_IDS[0]
# MULTI_CITY_VAL_IDS = ["06073", "12086"]
# MULTI_CITY_TEST_IDS = ["32003", "42003"]

MULTI_CITY_IDS = list(CITY_LABELS.keys())
SINGLE_CITY_IDS = ["36061", "53033", "47157"]
SINGLE_CITY_ID = SINGLE_CITY_IDS[0]
MULTI_CITY_VAL_IDS = ["06075", "11001"]
MULTI_CITY_TEST_IDS = ["36061", "47157"]


def split_configured_multi_city_ids(city_ids=None, val_city_ids=None, test_city_ids=None):
    ordered_city_ids = list(dict.fromkeys(MULTI_CITY_IDS if city_ids is None else city_ids))
    val_city_ids = list(MULTI_CITY_VAL_IDS if val_city_ids is None else val_city_ids)
    test_city_ids = list(MULTI_CITY_TEST_IDS if test_city_ids is None else test_city_ids)

    missing_val = [cid for cid in val_city_ids if cid not in ordered_city_ids]
    missing_test = [cid for cid in test_city_ids if cid not in ordered_city_ids]
    if missing_val or missing_test:
        raise ValueError(
            "Configured multi-city split refers to cities outside the provided city_ids: "
            f"val={missing_val}, test={missing_test}"
        )

    overlap = sorted(set(val_city_ids) & set(test_city_ids))
    if overlap:
        raise ValueError(f"Validation and test city sets must be disjoint, got overlap={overlap}")

    held_out = set(val_city_ids) | set(test_city_ids)
    train_city_ids = [cid for cid in ordered_city_ids if cid not in held_out]
    if not train_city_ids:
        raise ValueError("Configured multi-city split leaves no training cities")

    return ordered_city_ids, train_city_ids, val_city_ids, test_city_ids

# ─── Architecture ─────────────────────────────────────────────────────────────
HIDDEN_DIM = 64
PE_DIM = 8
PE_WALK_LEN = 20
GPS_HEADS = 4
GPS_LAYERS = 4
GPS_DROPOUT = 0.1
# ODGN paper hyperparameters (Rong et al. 2023, Section 4.1.4)
ODGN_GNN_LAYERS = 3   # "number of graph convolutional layers is set to 3"
ODGN_GNN_HEADS  = 8   # "number of heads is set to 8"
ODGN_NOISE_DIM  = 60  # "noise dimension is set to be 60, same as regional attributes"
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
FEATURE_PRESET = "all"  # switch to "reduced" to reproduce the previous demo subset
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
    gps_norm_type:      Literal['batch_norm', 'graph_norm', 'none']         = 'batch_norm'
    gnn_layers:         Optional[int] = None
    gnn_heads:          Optional[int] = None
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
    gan_regularizer:    Literal['gp', 'clip'] = 'gp'
    gan_clip_value:     float = 0.01
    gan_n_critic_after_epoch: int = 0
    gan_n_critic_after: int   = 1
    gan_noise_dim:      int   = 0
    gan_disc_hidden_dim: int  = 64
    gan_disc_layers:    int   = 4
    gan_disc_dropout:   float = 0.05
    gan_use_supervised_monitoring: bool = True
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
            'gps_norm_type':   ('batch_norm', 'graph_norm', 'none'),
            'loss_type':       ('huber', 'ce', 'ce_old', 'multitask', 'zinb', 'focal', 'mae'),
            'prediction_mode': ('raw', 'normalized'),
            'training_mode':   ('supervised', 'gan'),
            'gan_regularizer':  ('gp', 'clip'),
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
        if self.gnn_layers is not None and self.gnn_layers < 1:
            raise ValueError("TrainingConfig.gnn_layers must be >= 1 when set")
        if self.gnn_heads is not None and self.gnn_heads < 1:
            raise ValueError("TrainingConfig.gnn_heads must be >= 1 when set")
        if self.gan_clip_value <= 0:
            raise ValueError("TrainingConfig.gan_clip_value must be positive")
        if self.gan_n_critic_after_epoch < 0:
            raise ValueError("TrainingConfig.gan_n_critic_after_epoch must be >= 0")
        if self.gan_n_critic_after < 1:
            raise ValueError("TrainingConfig.gan_n_critic_after must be >= 1")
        if self.gan_noise_dim < 0:
            raise ValueError("TrainingConfig.gan_noise_dim must be >= 0")

    def describe(self):
        enc = {'mlp': 'MLP', 'gat': 'GAT'}.get(self.encoder_type, 'GPS')
        pe_name = 'none' if self.pe_type is None else self.pe_type
        parts = [f"{enc}+{self.decoder_type}+{self.loss_type}+{self.prediction_mode}",
                 f"pe={pe_name}", f"norm={self.gps_norm_type}"]
        if self.gnn_layers is not None or self.gnn_heads is not None:
            layers = self.gnn_layers if self.gnn_layers is not None else "default"
            heads = self.gnn_heads if self.gnn_heads is not None else "default"
            parts.append(f"gnn={layers}L/{heads}H")
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
            if self.gan_regularizer == 'clip':
                parts.append(f"WGAN-clip={self.gan_clip_value:g}")
            if self.gan_n_critic_after_epoch:
                parts.append(f"ncritic_after={self.gan_n_critic_after}@{self.gan_n_critic_after_epoch}")
            if self.gan_noise_dim:
                parts.append(f"noise={self.gan_noise_dim}")
            if self.gan_only:
                parts.append("gan_only")
            if not self.gan_use_supervised_monitoring:
                parts.append("supmon=off")
        parts.append(f"zeros={self.include_zero_pairs} samp={self.use_dest_sampling}")
        return " | ".join(parts)


# ─── Result saving ────────────────────────────────────────────────────────────
def ensure_dirs():
    RESULTS_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    WEIGHTS_CPC_BEST_DIR.mkdir(exist_ok=True)
    METRICS_RUNS_DIR.mkdir(exist_ok=True)


def save_metrics_to_csv(run_id, run_name, config, metrics,
                        n_params, epochs_trained, status='ok',
                        metrics_csv=None, run_suffix=None,
                        checkpoint_selection=None, selected_epoch=None,
                        selection_metric=None, selection_metric_value=None):
    """Save canonical metrics dict to CSV and JSON.

    ``metrics`` should be the output of canonical_od_metrics (possibly merged
    with train/val split metrics from masked_split_metrics).  All metric keys
    are written directly — no manual remapping.
    """
    ensure_dirs()
    metrics_csv = Path(metrics_csv) if metrics_csv is not None else METRICS_CSV
    metrics_csv.parent.mkdir(exist_ok=True)
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
        'gnn_layers': config.gnn_layers,
        'gnn_heads': config.gnn_heads,
        'gan_regularizer': config.gan_regularizer,
        'gan_clip_value': config.gan_clip_value,
        'gan_gp_lambda': config.gan_gp_lambda,
        'gan_n_critic': config.gan_n_critic,
        'gan_n_critic_after_epoch': config.gan_n_critic_after_epoch,
        'gan_n_critic_after': config.gan_n_critic_after,
        'gan_noise_dim': config.gan_noise_dim,
        'gan_pretrain_epochs': config.gan_pretrain_epochs,
        'gan_walk_len': config.gan_walk_len,
        'gan_walk_batch_size': config.gan_walk_batch_size,
        'n_params': n_params, 'epochs_trained': epochs_trained,
    }
    # Write all numeric metrics from the canonical dict directly
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            row[k] = v
    _append_metrics_row(metrics_csv, row)
    print(f"  -> Metrics saved to {metrics_csv}")
    suffix = f"__{run_suffix}" if run_suffix else ""
    metrics_path = METRICS_RUNS_DIR / f"{run_id}{suffix}.json"
    with open(metrics_path, 'w') as f:
        json.dump(row, f, indent=2, default=str)
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
