"""Module for CO2 recovery and exponential decay modeling.

This module provides baseline threshold-detection, linear baseline fits, 
and non-linear curve fitting methods to analyze CO2 concentration decay over time.
"""

import warnings
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning


def baseline_recovery_time(
    t: np.ndarray, co2: np.ndarray, threshold_ppm: float = 600.0
) -> float:
  """Estimates recovery time using a simple baseline threshold-crossing method."""
  if t.size == 0 or co2.size == 0:
    raise ValueError("Input arrays 't' and 'co2' must not be empty.")
  if t.shape != co2.shape:
    raise ValueError("Input arrays 't' and 'co2' must have the same shape.")

  below_threshold = np.where(co2 <= threshold_ppm)[0]
  if below_threshold.size > 0:
    idx = below_threshold[0]
    return float(t[idx])
  return np.nan


def linear_fit_baseline_recovery(
    t: np.ndarray, co2: np.ndarray, threshold_ppm: float = 600.0
) -> float:
  """Estimates recovery time using a simple linear regression baseline (y = mt + c)."""
  if t.size < 2 or co2.size < 2:
    raise ValueError("At least 2 data points are required for linear fit.")
  if t.shape != co2.shape:
    raise ValueError("Time 't' and 'co2' arrays must have identical dimensions.")

  # Fit first-order polynomial (linear regression: m*t + c)
  m, c = np.polyfit(t, co2, 1)

  # If slope is >= 0, CO2 is not decaying downwards toward the threshold
  if m >= 0:
    return np.nan

  # Solve for t when C(t) = threshold_ppm
  t_rec_linear = (threshold_ppm - c) / m
  return float(max(0.0, t_rec_linear))


def exponential_decay_model(
    t: np.ndarray, c_baseline: float, a: float, k: float
) -> np.ndarray:
  """Computes expected CO2 concentration using an exponential decay model."""
  return c_baseline + a * np.exp(-k * t)


def fit_co2_recovery(
    t: np.ndarray, co2: np.ndarray
) -> tuple[float, float, float, float]:
  """Fits an exponential decay model to CO2 recovery event data using non-linear least squares."""
  if t.size < 3 or co2.size < 3:
    raise ValueError("At least 3 data points are required for curve fitting.")
  if t.shape != co2.shape:
    raise ValueError("Time 't' and 'co2' arrays must have identical dimensions.")

  c_baseline_guess = float(co2[-1])
  a_guess = float(co2[0] - c_baseline_guess)
  if a_guess <= 0:
    a_guess = float(np.max(co2) - np.min(co2))
  k_guess = 0.05

  p0 = [c_baseline_guess, a_guess, k_guess]
  bounds = ([-100.0, -100.0, 1e-7], [np.inf, np.inf, 20.0])

  try:
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=OptimizeWarning)
      warnings.simplefilter("ignore", category=UserWarning)
      popt, _ = curve_fit(
          exponential_decay_model, t, co2, p0=p0, bounds=bounds, maxfev=10000
      )
    c_base, a, k = popt
  except Exception:
    try:
      p0_fallback = [float(np.min(co2)), float(np.ptp(co2)), 0.01]
      popt, _ = curve_fit(
          exponential_decay_model, t, co2, p0=p0_fallback, bounds=bounds, maxfev=15000
      )
      c_base, a, k = popt
    except Exception as e:
      raise ValueError(f"Curve fitting failed to converge: {str(e)}")

  t_rec = 3.0 / k if k > 0 else np.nan
  return float(c_base), float(a), float(k), float(t_rec)