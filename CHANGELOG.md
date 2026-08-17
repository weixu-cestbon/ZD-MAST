# Changelog

## v1.0.0-rc1 - 2026-08-17

### Added

- Clean latest-state import of the validated ZD-MAST feature builders,
  validators, reproducibility commands and expected aggregate results.
- Frozen temporal, cross-platform, task-universe, training-history,
  logistic-sensitivity, representation-comparison and calibration/ECE utilities.
- Public tests for reusable `zd_mast` modules.

### Changed

- Corrected public platform metadata to bioMérieux VITEK MS for ZD-MAST-A and
  Autobio Autof MS600 for ZD-MAST-B.
- Set the companion dataset pointer to reserved Zenodo DOI
  `10.5281/zenodo.21927440` with CC0 1.0 data licensing.
- Prepared the repository for Apache-2.0 release at
  `https://github.com/weixu-cestbon/ZD-MAST`.

### Evidence boundary

- The ten tasks remain a historical evaluation panel.
- The 2026 ZD-MAST-A test is retrospective out-of-time.
- ZD-MAST-B source-only evaluation is a cross-platform external stress test.
- Analyses using ZD-MAST-B labels are joint-development or adaptation
  sensitivities, not independent external validation.
- No patient-level data, private predictions, original identifiers, exact
  private timestamps, raw hospital exports, credentials or licensed CLSI source
  tables are included.
