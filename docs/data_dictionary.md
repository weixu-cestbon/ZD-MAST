# Data dictionary

Public IDs are release-only random pseudonyms and are not derived from patient or hospital identifiers. `historical_sir` is the reported laboratory S/I/R category; `binary_s_vs_ir` maps S=0 and I/R=1; `binary_si_vs_r` maps S/I=0 and R=1. Year is the coarsest time field retained for descriptive use; exact collection and approval timestamps are not released. The split file publishes task-level train/validation/test assignments without dates.

The ten-task panel is a historical project evaluation panel assembled during development, not a prospectively prespecified global task selection.

## M100 and measurement layers

`zd_mast_ast_labels_m100_v1.0.0.parquet` contains parallel historical and harmonized Site A labels. `zd_mast_ast_measurements_v1.0.0.parquet` contains de-identified MIC/zone measurement values and operators for linked core-task rows; Site B harmonized fields are intentionally not asserted by the primary analysis. `zd_mast_m100_breakpoints_core_v1.0.0.csv` contains only the ten derived task-level breakpoints required for reproducibility and does not reproduce CLSI source tables.
