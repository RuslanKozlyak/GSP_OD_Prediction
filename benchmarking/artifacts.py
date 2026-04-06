"""Helpers for saving OD benchmark artifacts."""

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import RESULTS_DIR


BENCHMARK_ARTIFACTS_DIR = RESULTS_DIR / "benchmark_artifacts"


def _slugify(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return value or "artifact"


def _artifact_dir(run_id, city_id=None):
    path = BENCHMARK_ARTIFACTS_DIR / _slugify(run_id)
    if city_id is not None:
        path = path / f"city_{_slugify(city_id)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _seed_suffix(inference_seed):
    return "default" if inference_seed is None else f"seed_{_slugify(inference_seed)}"


def _as_matrix(matrix, name):
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    return arr


def _heatmap_vmax(pred_matrix, ground_truth_matrix):
    pred_max = float(np.nanmax(np.clip(pred_matrix, 0, None))) if pred_matrix.size else 0.0
    gt_max = float(np.nanmax(np.clip(ground_truth_matrix, 0, None))) if ground_truth_matrix.size else 0.0
    return max(np.log1p(max(pred_max, gt_max)), 1e-8)


def _save_heatmap(path, matrix, title, vmax):
    display = np.log1p(np.clip(matrix, 0, None))
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    image = ax.imshow(
        display,
        cmap="magma",
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Destination")
    ax.set_ylabel("Origin")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("log1p(flow)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_od_artifacts(run_id, pred_matrix, ground_truth_matrix, *, city_id=None,
                      inference_seed=None, model_name=None):
    """Persist raw OD matrices plus a heatmap for the predicted OD matrix."""
    pred_matrix = _as_matrix(pred_matrix, "pred_matrix")
    ground_truth_matrix = _as_matrix(ground_truth_matrix, "ground_truth_matrix")

    out_dir = _artifact_dir(run_id, city_id=city_id)
    seed_suffix = _seed_suffix(inference_seed)
    vmax = _heatmap_vmax(pred_matrix, ground_truth_matrix)

    gt_path = out_dir / "ground_truth.npy"
    pred_path = out_dir / f"prediction_{seed_suffix}.npy"
    heatmap_path = out_dir / f"prediction_{seed_suffix}_heatmap.png"

    np.save(gt_path, ground_truth_matrix)
    np.save(pred_path, pred_matrix)

    title_parts = [model_name or run_id]
    if city_id is not None:
        title_parts.append(f"city={city_id}")
    if inference_seed is not None:
        title_parts.append(f"seed={inference_seed}")
    title_parts.append("prediction")
    _save_heatmap(heatmap_path, pred_matrix, " | ".join(title_parts), vmax=vmax)

    print(f"  -> OD artifacts saved to {out_dir}")
    return {
        "dir": out_dir,
        "ground_truth_path": gt_path,
        "prediction_path": pred_path,
        "prediction_heatmap_path": heatmap_path,
    }
