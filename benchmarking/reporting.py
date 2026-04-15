import csv
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RESULT_COLUMNS, RESULTS_DIR


def results_to_dataframe(results_dict, model_types=None, sort_by="CPC_full"):
    if not results_dict:
        return pd.DataFrame()
    df = pd.DataFrame(results_dict).T
    if model_types is not None:
        df["model_type"] = df.index.map(lambda name: model_types.get(name, "?"))
    available_cols = [col for col in RESULT_COLUMNS if col in df.columns]
    if available_cols:
        tail_cols = ["model_type"] if "model_type" in df.columns else []
        df = df[available_cols + tail_cols]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    return df



def save_results_table(df, filename):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / filename
    df.to_csv(path)
    _save_benchmark_metrics_snapshot(df, filename)
    return path


def _benchmark_context_from_filename(filename):
    stem = Path(filename).stem.lower()
    if "single_city" in stem:
        split_scope = "single_city"
    elif "multi_city" in stem:
        split_scope = "multi_city"
    elif "combined" in stem:
        split_scope = "combined"
    else:
        return None

    if "cpc_best" in stem:
        checkpoint_selection = "cpc_best"
    elif "val_loss" in stem:
        checkpoint_selection = "val_loss"
    else:
        return None

    return split_scope, checkpoint_selection


def _append_csv_rows(csv_path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)
        if existing_fields and existing_fields != fieldnames:
            merged_fields = existing_fields + [
                key for key in fieldnames if key not in existing_fields
            ]
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=merged_fields)
                writer.writeheader()
                for old_row in existing_rows:
                    old_row.pop(None, None)
                    writer.writerow(old_row)
                for row in rows:
                    writer.writerow(row)
            return

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_benchmark_metrics_snapshot(df, filename):
    context = _benchmark_context_from_filename(filename)
    if context is None or df is None or df.empty:
        return None

    split_scope, checkpoint_selection = context
    timestamp = datetime.now().isoformat()
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    metrics_runs_dir = RESULTS_DIR / "benchmark_metrics_runs"
    metrics_runs_dir.mkdir(exist_ok=True)

    export_df = df.copy()
    export_df.index.name = "model_name"
    export_df = export_df.reset_index()
    rows = []
    for row in export_df.to_dict(orient="records"):
        normalized = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in row.items()
        }
        rows.append({
            "timestamp": timestamp,
            "source_table": filename,
            "split_scope": split_scope,
            "checkpoint_selection": checkpoint_selection,
            **normalized,
        })

    csv_path = RESULTS_DIR / f"benchmark_metrics_{split_scope}_{checkpoint_selection}.csv"
    _append_csv_rows(csv_path, rows)

    json_path = metrics_runs_dir / (
        f"{split_scope}_{checkpoint_selection}_{timestamp_slug}.json"
    )
    payload = {
        "timestamp": timestamp,
        "source_table": filename,
        "split_scope": split_scope,
        "checkpoint_selection": checkpoint_selection,
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"  -> Benchmark metrics saved to {csv_path}")
    print(f"  -> Benchmark metrics snapshot saved to {json_path}")
    return csv_path, json_path



def build_combined_summary(single_city_results, multi_city_results):
    summary_metrics = list(RESULT_COLUMNS)
    all_models = sorted(set(single_city_results) | set(multi_city_results))
    rows = []
    for model in all_models:
        row = {"Model": model}
        if model in single_city_results:
            sc = single_city_results[model]
            for metric in summary_metrics:
                row[f"SC_{metric}"] = sc.get(metric)
        if model in multi_city_results:
            mc = multi_city_results[model]
            for metric in summary_metrics:
                row[f"MC_{metric}"] = mc.get(metric)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")



def plot_comparison(results_dict, title, metrics_to_plot=None):
    if not results_dict:
        print("No results to plot.")
        return
    metrics_to_plot = metrics_to_plot or ["CPC_full", "MAE_full", "RMSE_full"]
    df = pd.DataFrame(results_dict).T
    available = [metric for metric in metrics_to_plot if metric in df.columns]
    if not available:
        print("No matching metrics found.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 6))
    if len(available) == 1:
        axes = [axes]

    colors = []
    for name in df.index:
        if (
            "GPS" in name
            or str(name).startswith(("SC_", "MC_"))
            or "ODGN" in str(name)
            or "_GAN_" in str(name)
        ):
            colors.append("#2196F3")
        elif name in ("DiffODGen", "WeDAN"):
            colors.append("#9C27B0")
        elif name in ("GMEL", "NetGAN"):
            colors.append("#FF9800")
        else:
            colors.append("#4CAF50")

    for idx, metric in enumerate(available):
        vals = df[metric].values
        axes[idx].barh(range(len(df)), vals, color=colors)
        axes[idx].set_yticks(range(len(df)))
        axes[idx].set_yticklabels(df.index, fontsize=9)
        axes[idx].set_xlabel(metric)
        axes[idx].set_title(metric)
        axes[idx].grid(axis="x", alpha=0.3)
        finite = np.isfinite(vals)
        if finite.any():
            maximize_metric = (
                metric.startswith("CPC")
                or metric in ("accuracy", "matrix_COS_similarity")
            )
            selector_vals = np.where(finite, vals, -np.inf)
            if not maximize_metric:
                selector_vals = np.where(finite, vals, np.inf)
            best_idx = int(
                np.argmax(selector_vals)
                if maximize_metric
                else np.argmin(selector_vals)
            )
            axes[idx].barh(best_idx, vals[best_idx], color="red", alpha=0.7)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
