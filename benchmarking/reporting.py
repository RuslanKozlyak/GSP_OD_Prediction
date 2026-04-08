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
    return path



def build_combined_summary(single_city_results, multi_city_results):
    all_models = sorted(set(single_city_results) | set(multi_city_results))
    rows = []
    for model in all_models:
        row = {"Model": model}
        if model in single_city_results:
            sc = single_city_results[model]
            row["SC_CPC_full"] = sc.get("CPC_full")
            row["SC_CPC_full_std"] = sc.get("CPC_full_std")
            row["SC_MAE_full"] = sc.get("MAE_full")
            row["SC_MAE_full_std"] = sc.get("MAE_full_std")
            row["SC_RMSE_full"] = sc.get("RMSE_full")
            row["SC_RMSE_full_std"] = sc.get("RMSE_full_std")
        if model in multi_city_results:
            mc = multi_city_results[model]
            row["MC_CPC_full"] = mc.get("CPC_full")
            row["MC_CPC_full_std"] = mc.get("CPC_full_std")
            row["MC_MAE_full"] = mc.get("MAE_full")
            row["MC_MAE_full_std"] = mc.get("MAE_full_std")
            row["MC_RMSE_full"] = mc.get("RMSE_full")
            row["MC_RMSE_full_std"] = mc.get("RMSE_full_std")
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
        if "GPS" in name:
            colors.append("#2196F3")
        elif name in ("DiffODGen", "WeDAN"):
            colors.append("#9C27B0")
        elif name in ("GMEL", "NetGAN"):
            colors.append("#FF9800")
        else:
            colors.append("#4CAF50")

    for idx, metric in enumerate(available):
        vals = df[metric].values
        err_col = f"{metric}_std"
        errs = df[err_col].values if err_col in df.columns else None
        axes[idx].barh(range(len(df)), vals, color=colors, xerr=errs, capsize=3 if errs is not None else 0)
        axes[idx].set_yticks(range(len(df)))
        axes[idx].set_yticklabels(df.index, fontsize=9)
        axes[idx].set_xlabel(metric)
        axes[idx].set_title(metric)
        axes[idx].grid(axis="x", alpha=0.3)
        best_idx = int(np.argmax(vals) if metric in ("CPC_full", "CPC_nz", "CPC_test", "accuracy", "matrix_COS_similarity") else np.argmin(vals))
        axes[idx].barh(best_idx, vals[best_idx], color="red", alpha=0.7)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
