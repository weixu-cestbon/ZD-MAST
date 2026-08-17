# Data governance

ZD-MAST public identifiers are random release identifiers. They are not hashes
of hospital identifiers and do not encode dates, wards, organisms or outcomes.
Private identifier maps remain in the controlled institutional environment.

Public dates are represented through approved year, period or frozen split
fields rather than exact private timestamps. Patient-cluster and episode IDs are
release-specific random identifiers used only to reproduce grouped analyses.

The historical laboratory S/I/R label and the parallel M100-harmonized label are
separate fields. Neither overwrites the other. Conflicting or unsupported labels
remain explicitly flagged.

The two sites share the laboratory-confirmed VITEK 2 phenotypic AST workflow, but
their MALDI platforms, spectral export pipelines, laboratory environments and
isolate populations differ. Cross-site performance must therefore be interpreted
as domain transportability, not as a pure instrument-only effect.
