# ZD-MAST v1.0.0-rc1 code validation report

Status: **PASS FOR PRIVATE GITHUB RELEASE-CANDIDATE REVIEW**

## Code validation

- Python: 3.11.15 in an isolated clean-room virtual environment.
- Test result: 70 passed, 0 failed.
- Python source compilation: PASS.
- CLI construction and documented command contracts: PASS.
- Public repository privacy audit: PASS, 0 findings.
- Forbidden clinical/vendor file types in the repository: 0.
- Absolute private paths and private IPv4 addresses in tracked text: 0.
- Files larger than 50 MiB: 0.

## Frozen public-data contract

- De-identified sample metadata rows: 64,611.
- Open peak tables/spectrum rows: 79,713.
- ZD-MAST-A open peak tables: 70,336.
- ZD-MAST-B open peak tables: 9,377.
- De-identified AST measurement rows: 1,117,585.
- Sample-organism-antimicrobial label rows: 1,117,719.
- Released samples linked to labels: 62,434.
- Historical ten-task sample-task rows after the release-level uniqueness rule: 59,978.
- Original identifiers, exact private timestamps and hospital export files: not included.

These public-data counts describe the release package. They are not interchangeable
with manuscript-specific development, out-of-time, cross-platform or silent-run
analysis denominators.

## Remaining publication action

The code candidate is suitable for a new private GitHub repository. Public release
should follow publication of the companion Zenodo record and checksum verification
of the hosted archives. Ethics, data-sharing and vendor-permission determinations
remain documented outside the public repository.
