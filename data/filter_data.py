"""Module for cleaning, filtering, and validating environmental and occupancy datasets.

Merges environmental sensor streams with occupancy telemetry, applies physical 
guardrails, and isolates single-occupancy recovery events for model validation.
"""

import os
import pandas as pd
import numpy as np

# Automatically resolves the directory where filter_data.py lives (root folder)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def process_and_merge_datasets():
    data_dir = os.path.join(PROJECT_ROOT)
    filtered_path = os.path.join(data_dir, "5EnvSensor_filtered.csv")
    validated_path = os.path.join(data_dir, "5EnvSensor_validated.csv")
    occ_input_path = os.path.join(data_dir, "Occupancy_filtered.csv")
    merged_output_path = os.path.join(data_dir, "single_occupancy_decay.csv")

    if not os.path.exists(filtered_path):
        raise FileNotFoundError(f"Filtered dataset not found at '{filtered_path}'. Please ensure '5EnvSensor_filtered.csv' is in your data/ folder.")

    print(f"Loading filtered data from '{filtered_path}'...")
    df_filtered = pd.read_csv(filtered_path)
    
    # Standardize column names
    df_filtered.columns = [col.lower().strip() for col in df_filtered.columns]
    co2_col = [c for c in df_filtered.columns if "carbon_dioxide" in c or "co2" in c][0]
    time_col = [c for c in df_filtered.columns if "create" in c or "date" in c or "time" in c][0]
    
    # Parse timestamps and sort
    df_filtered[time_col] = pd.to_datetime(df_filtered[time_col], errors="coerce", utc=True)
    df_filtered = df_filtered.dropna(subset=[time_col, co2_col]).sort_values(time_col).reset_index(drop=True)

    # 1. Apply physical bounds (ASHRAE indoor air quality thresholds: 380 - 5000 ppm) 
    # and rolling median smoothing guardrails
    df_filtered = df_filtered[(df_filtered[co2_col] >= 380.0) & (df_filtered[co2_col] <= 5000.0)].copy()
    df_filtered["co2_smoothed"] = df_filtered[co2_col].rolling(window=5, min_periods=1, center=True).median()
    df_filtered[co2_col] = df_filtered["co2_smoothed"]
    df_filtered.drop(columns=["co2_smoothed"], inplace=True, errors="ignore")
    
    # Save validated environmental data
    os.makedirs(data_dir, exist_ok=True)
    df_filtered.to_csv(validated_path, index=False)
    print(f"[Success] Saved validated environmental data to '{validated_path}' ({len(df_filtered)} valid rows).")

    # 2. Merge with Occupancy Data to isolate 1-person occupancy decay events
    if os.path.exists(occ_input_path):
        print(f"Loading and merging occupancy telemetry from '{occ_input_path}'...")
        df_occ = pd.read_csv(occ_input_path)
        df_occ.columns = [c.lower().strip() for c in df_occ.columns]
        occ_time_col = [c for c in df_occ.columns if "collect" in c or "date" in c or "time" in c][0]
        
        df_occ[occ_time_col] = pd.to_datetime(df_occ[occ_time_col], errors="coerce", utc=True)
        df_occ = df_occ.dropna(subset=[occ_time_col]).sort_values(occ_time_col).reset_index(drop=True)

        # Merge using nearest timestamps
        df_merged = pd.merge_asof(
            df_filtered, 
            df_occ, 
            left_on=time_col, 
            right_on=occ_time_col, 
            direction="nearest"
        )

        # Filter specifically for single-occupancy events (headcount == 1 transitioning or steady)
        df_single_occ = df_merged[df_merged["headcount"] == 1].copy()
        
        if not df_single_occ.empty:
            df_single_occ.to_csv(merged_output_path, index=False)
            print(f"[Success] Isolated and saved {len(df_single_occ)} single-occupancy records to '{merged_output_path}'.\n")
        else:
            print("[Notice] No exact headcount == 1 rows found during merge.")
    else:
        print(f"[Notice] Occupancy file not found at '{occ_input_path}'. Skipping occupancy merge.")

if __name__ == "__main__":
    process_and_merge_datasets()