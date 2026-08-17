# Public repository scope

This is a clean latest-state repository. It intentionally does not import the
commit history of earlier exploratory or manuscript-integration repositories.

Included:

- feature reconstruction from de-identified open peak tables;
- release schema, checksum and privacy validation;
- frozen temporal and cross-platform analysis utilities;
- aggregate expected results used for regression tests;
- public configuration and documentation;
- unit tests.

Excluded:

- manuscript drafts and peer-review materials;
- patient-level data or private prediction rows;
- original identifiers and linkage maps;
- exact private timestamps and hospital exports;
- vendor-native binaries and proprietary databases;
- model binaries, credentials and private server paths;
- licensed CLSI source tables;
- third-party public benchmark datasets.

The historical analysis identifier is retained only as provenance metadata.
Git commit dates in this repository describe the clean import and subsequent
public-code changes, not the dates of the historical analyses.
