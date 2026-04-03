from dataclasses import dataclass
from typing import Optional

# Reuse all saving infrastructure and paths from GPS
from models.GPS.config import (
    RESULTS_DIR,
    WEIGHTS_DIR,
    METRICS_CSV,
    device,
    ensure_dirs,
    save_model_weights,
    save_metrics_to_csv,
)


@dataclass
class GmelGpsConfig:
    # ── Architecture ──────────────────────────────────────────────────────────
    hidden_dim:   int            = 64
    pe_dim:       int            = 8
    n_layers:     int            = 3
    n_heads:      int            = 4
    dropout:      float          = 0.1
    # ── Training schedule ─────────────────────────────────────────────────────
    max_epochs:   int            = 300
    patience:     int            = 20
    lr:           float          = 3e-4
    # ── GBRT decoder ──────────────────────────────────────────────────────────
    n_estimators: int            = 20
    # ── Fields required by GPS save_metrics_to_csv (duck-type compatibility) ──
    encoder_type:       str            = 'gmel_gps'
    decoder_type:       str            = 'gbrt'
    loss_type:          str            = 'multitask_mse'
    prediction_mode:    str            = 'raw'
    pe_type:            Optional[str]  = 'rwpe'
    gps_norm_type:      str            = 'batch_norm'
    use_log_transform:  bool           = False
    use_dest_sampling:  bool           = False
    include_zero_pairs: bool           = False
    zero_pair_ratio:    float          = 0.0
    use_rle:            bool           = False

    def describe(self):
        pe_name = 'none' if self.pe_type is None else self.pe_type
        return (f"GMEL_GPS hd={self.hidden_dim} pe={pe_name} "
                f"norm={self.gps_norm_type} layers={self.n_layers}")
