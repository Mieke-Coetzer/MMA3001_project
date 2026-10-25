"""Unit tests for CO2 recovery and decay models using pytest."""

import os
import sys

# Dynamically add the project root to Python's path so 'src' can be found
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import numpy as np
import pytest
from src.co2_recovery import (
    baseline_recovery_time,
    exponential_decay_model,
    fit_co2_recovery,
)


def test_baseline_recovery_time():
  """Tests if the baseline model correctly identifies when CO2 drops below threshold."""
  t = np.array([0, 10, 20, 30])
  co2 = np.array([800, 700, 590, 550])
  # Should cross 600 ppm at index 2 (elapsed time = 20 minutes)
  rec_time = baseline_recovery_time(t, co2, threshold_ppm=600.0)
  assert rec_time == 20.0


def test_exponential_decay_model():
  """Tests the mathematical output of the exponential decay equation."""
  t = np.array([0, 10])
  res = exponential_decay_model(t, c_baseline=500.0, a=400.0, k=0.1)
  assert len(res) == 2
  assert np.isclose(res[0], 900.0)  # 500 + 400 * exp(0) = 900


def test_fit_co2_recovery_synthetic():
  """Tests curve fitting accuracy using synthetic data with known parameters."""
  t = np.linspace(0, 60, 50)
  co2_true = 500.0 + 400.0 * np.exp(-0.1 * t)
  c_base, a, k, t_rec = fit_co2_recovery(t, co2_true)

  assert np.isclose(c_base, 500.0, rtol=0.1)
  assert np.isclose(k, 0.1, rtol=0.1)


def test_fit_co2_recovery_insufficient_data():
  """Tests that insufficient data points correctly trigger a ValueError."""
  t = np.array([5])
  co2 = np.array([800])
  with pytest.raises(ValueError):
    fit_co2_recovery(t, co2)