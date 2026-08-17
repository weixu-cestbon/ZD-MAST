"""Command-line entry point for ZD-MAST utilities."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .provenance import validate_source_manifest
from .release import validate_release


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zd-mast")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-data", help="validate a de-identified feature release")
    validate.add_argument("release_root", type=Path)
    compare = subparsers.add_parser("compare-results", help="verify frozen manuscript source checksums")
    compare.add_argument("project_root", type=Path)
    compare.add_argument(
        "--manifest",
        type=Path,
        default=Path("checksums/manuscript_source_artifact_manifest_v1.csv"),
    )
    reproduce = subparsers.add_parser("reproduce", help="rebuild the checksum-verified submission package")
    reproduce.add_argument("project_root", type=Path)
    reproduce.add_argument("--skip-docx", action="store_true")
    rebuild = subparsers.add_parser(
        "rebuild-features",
        help="rebuild one site's frozen 6000-feature matrices from open peak tables",
    )
    rebuild.add_argument("project_root", type=Path)
    rebuild.add_argument("release_root", type=Path)
    rebuild.add_argument("site_id", choices=("ZD-MAST-A", "ZD-MAST-B"))
    rebuild.add_argument("output_root", type=Path)
    rebuild.add_argument("--workers", type=int, default=16)
    cross_platform = subparsers.add_parser(
        "reproduce-cross-platform",
        help="rerun the frozen Site A to Site B source-only stress test",
    )
    cross_platform.add_argument("project_root", type=Path)
    cross_platform.add_argument("release_root", type=Path)
    cross_platform.add_argument("output_root", type=Path)
    primary = subparsers.add_parser(
        "reproduce-primary",
        help="rerun the frozen ten-task Site A temporal analysis",
    )
    primary.add_argument("project_root", type=Path)
    primary.add_argument("release_root", type=Path)
    primary.add_argument("output_root", type=Path)
    primary.add_argument("--threads", type=int, default=4)
    primary.add_argument("--bootstrap", type=int, default=1000)
    patient = subparsers.add_parser(
        "reproduce-patient-episode",
        help="rerun patient-disjoint and episode-first sensitivities",
    )
    patient.add_argument("project_root", type=Path)
    patient.add_argument("release_root", type=Path)
    patient.add_argument("primary_metrics", type=Path)
    patient.add_argument("output_root", type=Path)
    patient.add_argument("--threads", type=int, default=4)
    patient.add_argument("--bootstrap", type=int, default=1000)
    m100 = subparsers.add_parser(
        "reproduce-m100-agreement",
        help="recompute historical versus M100 parallel-label agreement",
    )
    m100.add_argument("project_root", type=Path)
    m100.add_argument("release_root", type=Path)
    m100.add_argument("output_root", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-data":
        result = validate_release(args.release_root)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    elif args.command == "compare-results":
        root = args.project_root.resolve()
        manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
        result = validate_source_manifest(root, manifest)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        if result.status != "PASS":
            raise SystemExit(1)
    elif args.command == "reproduce":
        root = args.project_root.resolve()
        command = [sys.executable, str(root / "scripts/reproduce_submission_package.py"), "--root", str(root)]
        if args.skip_docx:
            command.append("--skip-docx")
        subprocess.run(command, cwd=root, check=True)
    elif args.command == "rebuild-features":
        root = args.project_root.resolve()
        command = [
            sys.executable,
            str(root / "scripts/rebuild_feature_release_from_open_peak_tables.py"),
            "--release-root",
            str(args.release_root.resolve()),
            "--site-id",
            args.site_id,
            "--output-dir",
            str(args.output_root.resolve()),
            "--workers",
            str(args.workers),
        ]
        subprocess.run(command, cwd=root, check=True)
    elif args.command == "reproduce-cross-platform":
        root = args.project_root.resolve()
        command = [
            sys.executable,
            str(root / "scripts/reproduce_cross_platform_source_only.py"),
            "--release-root",
            str(args.release_root.resolve()),
            "--frozen-results",
            str(
                root
                / "results/tables/legacy_manuscript_tables/jiangbei/jiangbei_peakpresence_external_results_v1.csv"
            ),
            "--output",
            str(args.output_root.resolve()),
        ]
        subprocess.run(command, cwd=root, check=True)
    elif args.command == "reproduce-primary":
        root = args.project_root.resolve()
        command = [
            sys.executable,
            str(root / "scripts/reproduce_primary_protocol_b.py"),
            "--release-root",
            str(args.release_root.resolve()),
            "--output-dir",
            str(args.output_root.resolve()),
            "--threads",
            str(args.threads),
            "--bootstrap",
            str(args.bootstrap),
            "--frozen-metrics",
            str(
                root
                / "results/audits/primary_protocol_b_reproduction_v1/zd_mast_primary_protocol_b_reproduced_metrics_v1.0.0.csv"
            ),
        ]
        subprocess.run(command, cwd=root, check=True)
    elif args.command == "reproduce-patient-episode":
        root = args.project_root.resolve()
        command = [
            sys.executable,
            str(root / "scripts/reproduce_patient_episode_sensitivity.py"),
            "--release-root",
            str(args.release_root.resolve()),
            "--primary-metrics",
            str(args.primary_metrics.resolve()),
            "--output-dir",
            str(args.output_root.resolve()),
            "--threads",
            str(args.threads),
            "--bootstrap",
            str(args.bootstrap),
            "--frozen-sensitivity",
            str(
                root
                / "results/audits/patient_episode_sensitivity_reproduction_v1/zd_mast_patient_episode_sensitivity_metrics_v1.0.0.csv"
            ),
        ]
        subprocess.run(command, cwd=root, check=True)
    elif args.command == "reproduce-m100-agreement":
        root = args.project_root.resolve()
        command = [
            sys.executable,
            str(root / "scripts/reproduce_m100_label_agreement.py"),
            "--release-root",
            str(args.release_root.resolve()),
            "--output-dir",
            str(args.output_root.resolve()),
            "--frozen-agreement",
            str(
                root
                / "results/audits/m100_label_agreement_public_rc13_v1/zd_mast_m100_label_agreement_reproduced_v1.0.0.csv"
            ),
        ]
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
