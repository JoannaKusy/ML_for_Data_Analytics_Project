import os
import json
import tempfile
import numpy as np
import pandas as pd
import wandb

def extract_table_path_from_summary(summary_value):
    """Parses W&B summary metrics to isolate the raw path to a .table.json log."""
    if isinstance(summary_value, dict):
        path = summary_value.get("path")
        if isinstance(path, str) and path.endswith(".table.json"):
            return path
    return None

def download_predictions_table(run, key="predictions"):
    """Downloads a .table.json prediction artifact directly from a W&B run file structure."""
    candidate_paths = []
    summary_path = extract_table_path_from_summary(run.summary.get(key))
    if summary_path:
        candidate_paths.append(summary_path)

    try:
        for f in run.files():
            name = getattr(f, "name", "")
            if name.endswith(".table.json") and key.lower() in name.lower():
                candidate_paths.append(name)
    except Exception:
        pass

    seen = set()
    for rel_path in candidate_paths:
        if rel_path in seen:
            continue
        seen.add(rel_path)

        try:
            downloaded = run.file(rel_path).download(
                root=tempfile.gettempdir(),
                replace=True,
            )
            with open(downloaded.name, "r", encoding="utf-8") as fp:
                payload = json.load(fp)

            if isinstance(payload, dict) and "columns" in payload and "data" in payload:
                df = pd.DataFrame(payload["data"], columns=payload["columns"])
                return df, rel_path
        except Exception:
            continue
    return None, None

def calculate_table_metrics(df_pred, actual_col="actual_kWh", pred_col="predicted_kWh"):
    """Extracts numeric arrays, alignments, and calculates key forecast error indices."""
    actual_key = actual_col if actual_col in df_pred.columns else None
    pred_key = pred_col if pred_col in df_pred.columns else None

    if not actual_key or not pred_key:
        return np.nan, np.nan, np.nan

    y_true = pd.to_numeric(df_pred[actual_key], errors="coerce")
    y_pred = pd.to_numeric(df_pred[pred_key], errors="coerce")
    
    # Drop joint NaNs if they exist
    valid_mask = y_true.notna() & y_pred.notna()
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return np.nan, np.nan, np.nan

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    non_zero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
        if non_zero.any() else np.nan
    )
    
    return mae, rmse, mape

def compile_benchmark_table(entity, project, run_ids, predictions_key="predictions"):
    """Iterates through specific run IDs to fetch tables and generate a sorted DataFrame."""
    api = wandb.Api()
    table_rows = []

    for run_id in run_ids:
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
            df_pred, _ = download_predictions_table(run, key=predictions_key)
            
            if df_pred is None or df_pred.empty:
                print(f"Skipping {run_id}: No prediction table found.")
                continue

            mae, rmse, mape = calculate_table_metrics(df_pred)

            table_rows.append({
                "Run Name": run.name,
                "Run ID": run.id,
                "State": run.state,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE (%)": mape
            })
            print(f"Successfully processed: {run.name}")

        except Exception as e:
            print(f"Error handling run {run_id}: {e}")

    if len(table_rows) > 0:
        final_df = pd.DataFrame(table_rows)
        return final_df.sort_values(by="MAE", ascending=True).reset_index(drop=True)
    
    return pd.DataFrame(columns=["Run Name", "Run ID", "State", "MAE", "RMSE", "MAPE (%)"])