# Indoor CO2 Recovery Time Estimation After Occupancy

**Unit:** MMA3001 – Numerical Methods and Machine Learning  
**Project Context:** Monash Smart Infrastructure Occupancy and Environmental Sensor Data (Dataset 2)  
**Language:** Python 3.10+  

---
## 1. Project Overview & Engineering Context

In indoor environments, high CO2 concentrations accumulated during occupancy adversely impact cognitive function and indoor air quality (IAQ). Efficient facility management and HVAC control require accurate predictions of **CO2 recovery time**, which is the time elapsed from the end of room occupancy until indoor CO2 concentrations return to an unoccupied baseline.

This project implements a reproducible computational workflow to model post-occupancy indoor CO2 dissipation. It compares a primary **Non-Linear Exponential Decay Model** against a naive **Linear Decay Baseline**.

### Scope Limitations & Validity Rules
- **Empirical Regression Scope:** The exponential model provides an empirical fit to time-series sensor observations rather than a direct physical measurement of mechanical air exchange rates (ACH).
- **Single-Peak Event Slicing:** To prevent model corruption during active occupancy buildup, decay windows are extracted strictly from peak concentration onwards using peak detection.
- **Domain Guardrails:** Function outputs enforce physical constraints, capping maximum estimated recovery time at **600.0 minutes (10 hours)** and rejecting non-decaying data.

---
## 2. System Inputs & Outputs

| Specification | Primary Input Data | Primary Output Specification |
| :--- | :--- | :--- |
| **Physical Parameter** | Indoor CO2 Concentration & Time | Bounded Recovery Duration |
| **Physical Units** | Concentration in Parts Per Million (ppm); Time (min) | Elapsed time (min) |
| **Expected Domain** | Concentration: 350 to 2000 ppm; Elapsed time >= 0 | 0 <= Recovery time <= 600 |
| **Error Handling** | Missing timestamps or non-decaying arrays return `np.nan`. | Unconverged fits raise `RuntimeError` or yield `np.nan`. |

---
## 3. Quantitative Results & Performance

Evaluated against the independent ASHRAE benchmark across single-peak decay events:
* **Exponential Decay Model:** MAE = **313.5 min** | RMSE = **430.2 min**
* **Linear Fit Baseline:** MAE = **389.0 min** | RMSE = **492.6 min**
* **Execution Performance:** ~1.34 ms per curve fit (optimized via `scipy.optimize.curve_fit`).

---

### Repository Structure
MMA3001_project/
	data/
    filter_data.py
    filter_env_data.py
    filter_occ_data.py
		5EnvSensor_filtered.csv
		5EnvSensor_validated.csv
		Occupancy_filtered.csv
		single_occupancy_decay.csv
	docs/
		co2_recovery.html
		index.html
		model_comparison_plot.png
	src/
		co2_recovery.py
	tests/
		test_recovery.py
		test_report.txt
	validate_models.py
	README.md

---
### Installation & Prerequisites
Ensure you have Python 3.10+ installed. Clone this repository and install the required numerical and data science dependencies:

### License
This project is open-source and available under the [MIT License](LICENSE).
