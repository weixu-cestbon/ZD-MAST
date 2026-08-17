# Manuscript v6 analysis addendum

The review-facing utilities introduced in v1.0.0-rc1 and retained in v1.0.0-rc2 address bounded methodological
questions without changing the historical task panel or selecting a model from
target performance.

| Utility | Purpose | Target-label rule |
|---|---|---|
| `build_task_universe_selection_audit_v1.py` | Performance-independent support funnel for all organism-drug combinations | No model fitting |
| `run_calendar_year_rolling_v1.py` | Named-year patient-purged rolling-origin evaluation | Test year isolated |
| `run_same_test_training_history_v1.py` | Compare training histories on one fixed test | Test isolated |
| `run_training_history_size_matched_v2.py` | Match sample size and class composition across histories | Test isolated |
| `run_current_contract_logistic_sensitivity_v1.py` | Logistic-regression model-dependence sensitivity | Same frozen splits |
| `run_site_a_representation_comparison_v1.py` | Compare intensity6000 and peak_presence6000 on identical Site A samples | No target-domain feature selection |
| `run_site_b_peak_extraction_sensitivity_v1.py` | Label-free peak-extraction parameter sensitivity | Target labels evaluation only |
| `build_revision_v6_statistical_addendum.py` | Calibration, ECE, support precision and median-gap bootstrap from frozen predictions | Post-processing only |

Scripts that can write row-level predictions do so only when explicitly
requested. Those outputs are ignored by Git and are not part of the public
repository or companion data release.

The current ZD-MAST-A VITEK 2 console configuration was audited after cohort
closure as VITEK 2 Systems version 9, Client/Core 9.02.4.531, AES components
2.0.0 and Myla Connector 1.0.0.1. These values are metadata for the audited
current configuration; they are not assigned retrospectively to individual
historical AST rows. The exact ZD-MAST-B software build is not available in the
public release metadata.
