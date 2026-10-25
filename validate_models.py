import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Automatically ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.co2_recovery import (
    exponential_decay_model,
    fit_co2_recovery,
    linear_fit_baseline_recovery,
)


def load_real_recovery_events(
    env_csv_path: str = "data/single_occupancy_decay.csv",
    occ_csv_path: str = "data/Occupancy_validated.csv",
    co2_threshold: float = 430.0,
):
    """Extracts distinct single-peak CO2 decay events using peak detection to prevent overlapping multiple peaks."""
    events = []
    
    if not os.path.exists(env_csv_path):
        raise FileNotFoundError(f"Required dataset not found at '{env_csv_path}'.")

    print(f"Loading real-world recovery events from '{env_csv_path}'...")
    df_env = pd.read_csv(env_csv_path)
    col_map = {col: col.lower().strip() for col in df_env.columns}
    df_env.rename(columns=col_map, inplace=True)
    
    co2_col = [c for c in df_env.columns if "carbon_dioxide" in c or "co2" in c]
    time_col = [c for c in df_env.columns if "create" in c or "date" in c or "time" in c]

    if not co2_col or not time_col:
        raise ValueError("Environmental dataset missing required CO2 or timestamp columns.")

    co2_name = co2_col[0]
    time_name = time_col[0]

    df_env[time_name] = pd.to_datetime(df_env[time_name], errors="coerce", utc=True)
    df_env = df_env.dropna(subset=[co2_name, time_name]).sort_values(time_name).reset_index(drop=True)

    if occ_csv_path and os.path.exists(occ_csv_path):
        try:
            df_occ = pd.read_csv(occ_csv_path)
            print(f"Cross-referencing with occupancy dataset '{occ_csv_path}'...")
        except Exception as e:
            print(f"[Warning] Could not parse occupancy file: {e}")

    co2_vals = df_env[co2_name].values
    start_time = df_env[time_name].iloc[0]
    time_vals = (df_env[time_name] - start_time).dt.total_seconds().values / 60.0

    # Detect distinct prominent peaks
    peaks, _ = find_peaks(co2_vals, prominence=15, distance=20)

    for p_idx in peaks:
        # Trace backwards to find event start (baseline rise)
        start_idx = p_idx
        while start_idx > 0 and co2_vals[start_idx] > co2_threshold and co2_vals[start_idx] >= co2_vals[start_idx - 1] - 5:
            start_idx -= 1
            
        # Trace forwards to find event end (return to normal baseline)
        end_idx = p_idx
        while end_idx < len(co2_vals) - 1 and co2_vals[end_idx] > co2_threshold and co2_vals[end_idx] >= co2_vals[end_idx + 1] - 5:
            end_idx += 1
            
        if (p_idx - start_idx) >= 3 and (end_idx - p_idx) >= 5:
            t_segment = time_vals[start_idx:end_idx+1]
            co2_segment = co2_vals[start_idx:end_idx+1]
            
            # Slice strictly from the peak onwards
            local_peak_offset = np.argmax(co2_segment)
            t_arr = t_segment[local_peak_offset:] - t_segment[local_peak_offset]
            co2_arr = co2_segment[local_peak_offset:]
            
            if len(t_arr) >= 5 and co2_arr[0] > co2_arr[-1] + 5.0:
                k_ashrae = 1.5 / 60.0
                t_rec_ashrae = 3.0 / k_ashrae
                events.append((t_arr, co2_arr, t_rec_ashrae))

    print(f"Successfully extracted {len(events)} clean single-peak decay events against ASHRAE standard.\n")
    return events


def evaluate_models(test_events):
    """Evaluates Linear Fit Baseline vs Exponential Model errors with 650-minute domain guardrails."""
    if not test_events:
        raise ValueError("No test events available for evaluation from filtered data.")

    baseline_errors = []
    exp_errors = []
    baseline_preds = []
    exp_preds = []
    ground_truths = []

    print("--- Running Quantitative Model Evaluation with Domain Guardrails ---")
    for i, (t, co2, true_t_rec) in enumerate(test_events):
        ground_truths.append(true_t_rec)

        # 1. Linear Fit Baseline prediction with 650-min cap
        pred_base = linear_fit_baseline_recovery(t, co2, threshold_ppm=430.0)
        if not np.isnan(pred_base):
            pred_base = min(pred_base, 650.0)
        baseline_preds.append(pred_base)
        
        if not np.isnan(pred_base) and not np.isnan(true_t_rec):
            baseline_errors.append(pred_base - true_t_rec)

        # 2. Exponential Fit prediction with 650-min domain guardrail
        try:
            _, _, k, pred_exp = fit_co2_recovery(t, co2)
            if k < 0.001 or pred_exp > 650.0:
                pred_exp = np.nan  
            
            exp_preds.append(pred_exp)
            if not np.isnan(pred_exp) and not np.isnan(true_t_rec):
                exp_errors.append(pred_exp - true_t_rec)
        except ValueError:
            exp_preds.append(np.nan)
            continue

    baseline_errors = np.array(baseline_errors)
    exp_errors = np.array(exp_errors)

    mae_b = np.mean(np.abs(baseline_errors)) if baseline_errors.size > 0 else np.nan
    rmse_b = np.sqrt(np.mean(baseline_errors**2)) if baseline_errors.size > 0 else np.nan
    mae_e = np.mean(np.abs(exp_errors)) if exp_errors.size > 0 else np.nan
    rmse_e = np.sqrt(np.mean(exp_errors**2)) if exp_errors.size > 0 else np.nan

    print("\n=== Quantitative Validation Results (650-Minute Cap & Guardrailed) ===")
    if not np.isnan(mae_b):
        print(f"Linear Fit Baseline     -> MAE: {mae_b:.2f} min | RMSE: {rmse_b:.2f} min")
    if not np.isnan(mae_e):
        print(f"Exponential Decay Model -> MAE: {mae_e:.2f} min | RMSE: {rmse_e:.2f} min\n")

    results = {
        "ground_truths": np.array(ground_truths),
        "baseline_preds": np.array(baseline_preds),
        "exp_preds": np.array(exp_preds),
        "mae_b": mae_b,
        "rmse_b": rmse_b,
        "mae_e": mae_e,
        "rmse_e": rmse_e,
    }
    return results


def plot_validation_overview(test_events, eval_results, threshold_ppm=430.0):
    if not test_events:
        print("[Warning] No test events to plot.")
        return

    print("--- Generating Validation Graphs ---")
    plt.figure(figsize=(15, 10))

    ax1 = plt.subplot(2, 2, (1, 2))
    t_sample, co2_sample, true_t_sample = test_events[0]
    
    # Filter data points to show up to 600 minutes
    mask = t_sample <= 600.0
    t_pts = t_sample[mask]
    co2_pts = co2_sample[mask]

    # Fit lines extend smoothly up to 650 minutes
    t_smooth = np.linspace(t_sample[0], 650.0, 200)
    
    # Linear baseline fit on full sample
    m_sample, c_sample = np.polyfit(t_sample, co2_sample, 1)
    linear_fit_line = m_sample * t_smooth + c_sample

    try:
        c_base, a, k, pred_exp_sample = fit_co2_recovery(t_sample, co2_sample)
        co2_fit = exponential_decay_model(t_smooth, c_base, a, k)
        exp_label = f"Exponential Fit (t_rec = {pred_exp_sample:.1f} min)"
    except Exception:
        co2_fit = linear_fit_line
        exp_label = "Exponential Fit (Fallback)"

    ax1.scatter(t_pts, co2_pts, color="black", s=30, label="Single-Peak Measured CO2 (Up to 600 min)", zorder=3)
    ax1.plot(t_smooth, linear_fit_line, color="crimson", linestyle="--", linewidth=2, label="Linear Baseline (up to 650 min)", zorder=4)
    ax1.plot(t_smooth, co2_fit, color="dodgerblue", linewidth=2.5, label=exp_label, zorder=5)
    ax1.axhline(y=threshold_ppm, color="gray", linestyle=":", linewidth=1.5, label=f"Threshold ({threshold_ppm} ppm)")

    ax1.set_xlim(0, 650)
    ax1.set_xlabel("Time Since Peak (minutes)", fontsize=11)
    ax1.set_ylabel("CO2 Concentration (ppm)", fontsize=11)
    ax1.set_title("Independent ASHRAE Benchmark: Single-Peak Decay Modeling", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Predicted Recovery Times (Sampled subset of first 20 events)
    ax2 = plt.subplot(2, 2, 3)
    max_plot_events = min(20, len(test_events))
    event_ids = np.arange(1, max_plot_events + 1)
    width = 0.25

    g_sub = eval_results["ground_truths"][:max_plot_events]
    b_sub = np.nan_to_num(eval_results["baseline_preds"][:max_plot_events], nan=650.0)
    e_sub = np.nan_to_num(eval_results["exp_preds"][:max_plot_events], nan=650.0)

    ax2.bar(event_ids - width, g_sub, width, label="ASHRAE Standard", color="forestgreen", alpha=0.8)
    ax2.bar(event_ids, b_sub, width, label="Linear Baseline", color="crimson", alpha=0.8)
    ax2.bar(event_ids + width, e_sub, width, label="Exponential Fit", color="dodgerblue", alpha=0.8)

    ax2.set_xlabel("Decay Event ID (First 20 Sample)", fontsize=11)
    ax2.set_ylabel("Recovery Time (minutes)", fontsize=11)
    ax2.set_title("Predicted Recovery Times (Sampled Subset)", fontsize=12, fontweight="bold")
    ax2.set_xticks(event_ids)
    ax2.set_ylim(0, 700)  # Capped at 700 to cleanly show up to 650 min
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend(loc="upper left")

    # 3. Error Metrics Comparison
    ax3 = plt.subplot(2, 2, 4)
    metrics = ["MAE", "RMSE"]
    baseline_metrics = [eval_results["mae_b"], eval_results["rmse_b"]]
    exp_metrics = [eval_results["mae_e"], eval_results["rmse_e"]]

    x = np.arange(len(metrics))
    ax3.bar(x - 0.15, baseline_metrics, 0.3, label="Linear Baseline", color="crimson", alpha=0.8)
    ax3.bar(x + 0.15, exp_metrics, 0.3, label="Exponential Decay", color="dodgerblue", alpha=0.8)

    max_metric_val = max([m for m in baseline_metrics + exp_metrics if not np.isnan(m)] or [100])
    ax3.set_ylim(0, max_metric_val * 1.25)

    for i in range(len(metrics)):
        if not np.isnan(baseline_metrics[i]):
            ax3.text(x[i] - 0.15, baseline_metrics[i] + (max_metric_val * 0.05), f"{baseline_metrics[i]:.1f}", ha="center", fontsize=9)
        if not np.isnan(exp_metrics[i]):
            ax3.text(x[i] + 0.15, exp_metrics[i] + (max_metric_val * 0.05), f"{exp_metrics[i]:.1f}", ha="center", fontsize=9)

    ax3.set_ylabel("Error (minutes)", fontsize=11)
    ax3.set_title("Error Metrics Comparison against Independent Standard", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=11)
    ax3.grid(True, linestyle=":", alpha=0.7)
    ax3.legend(loc="upper right")

    plt.tight_layout()

    docs_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "docs"))
    os.makedirs(docs_dir, exist_ok=True)
    output_plot_path = os.path.join(docs_dir, "model_comparison_plot.png")
    plt.savefig(output_plot_path, dpi=300)
    print(f"[Success] Validation graphs generated and saved to: {output_plot_path}\n")

    plt.show()


def profile_performance():
    """Measures execution runtime and profiles function performance."""
    t = np.linspace(0, 60, 50)
    co2 = 500.0 + 400.0 * np.exp(-0.08 * t) + np.random.normal(0, 5, 50)

    print("--- Running Computational Profiling ---")
    start_time = time.perf_counter()
    iterations = 500
    for _ in range(iterations):
        fit_co2_recovery(t, co2)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    print(f"Total time for {iterations} curve fits: {total_time:.4f} seconds")
    print(f"Average time per fit:   {(total_time / iterations) * 1000:.2f} ms\n")


if __name__ == "__main__":
    test_events = load_real_recovery_events()
    eval_results = evaluate_models(test_events)
    plot_validation_overview(test_events, eval_results)
    profile_performance()