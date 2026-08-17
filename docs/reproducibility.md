# Reproducibility status

## Exact public reruns

| Analysis | Public input | Status |
|---|---|---|
| Site A primary temporal Protocol B | intensity6000, labels, frozen folds | 180/180 manuscript-facing fields reproduced |
| Patient-disjoint and episode-first sensitivity | intensity6000, public groups, frozen primary parameters | 370/380 exact plus 10 documented descriptive metadata corrections |
| Historical versus M100 agreement | parallel public labels | 120/120 fields reproduced |
| Site A to Site B source-only stress test | peak_presence6000, public labels, frozen source split | 90/90 fields reproduced |

## Frozen but not one-command public reruns

Workflow-era model sensitivity, Staphylococcus supplementary reassessment,
label-using pooling/site-balancing branches and third-party public benchmark
adapters remain frozen aggregate analyses. They are not represented as complete
one-command reproductions in this candidate.

## Reproduction rule

Model selection, calibration and threshold selection use development data only.
The Site B target labels are used only for final evaluation in the source-only
stress test. Outputs that use target labels for training are named adaptation or
joint-development simulations.

## Feature-level reconstruction

Both sites are rebuilt in the published metadata order, not lexicographic file
name order. `intensity6000` uses the common 2,000-20,000 Da, 3-Da maximum-bin,
square-root and L2 contract. ZD-MAST-A `peak_presence6000` uses nonzero bins from
the sparse peak-list-like export. ZD-MAST-B applies the frozen label-free
dense-profile peak extraction before 3-Da presence binning and sample-level OR
aggregation. The rebuild command writes matrix-level equivalence metrics against
the deposited arrays.
