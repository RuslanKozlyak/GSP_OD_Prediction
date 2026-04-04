from datetime import datetime
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOSS_PLOTS_DIR = PROJECT_ROOT / "results" / "loss_plots"


def _slugify(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return value or "training"


def _normalize_series(values):
    if values is None:
        return []
    series = []
    for value in values:
        if value is None:
            series.append(np.nan)
            continue
        value = float(value)
        series.append(value if np.isfinite(value) else np.nan)
    return series


def save_loss_plot(train_losses=None, val_losses=None, title="Training Loss", save_path=None):
    train_series = _normalize_series(train_losses)
    val_series = _normalize_series(val_losses)
    if not train_series and not val_series:
        return None

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{_slugify(title)}_{timestamp}_loss.png"
        save_path = LOSS_PLOTS_DIR / filename
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    if train_series:
        ax.plot(range(1, len(train_series) + 1), train_series, label="train", linewidth=2)
    if val_series:
        ax.plot(range(1, len(val_series) + 1), val_series, label="val", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    if train_series or val_series:
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
