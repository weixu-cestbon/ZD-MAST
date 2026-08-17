from pathlib import Path

from zd_mast.cli import build_parser


def test_cross_platform_cli_contract() -> None:
    args = build_parser().parse_args(
        ["reproduce-cross-platform", ".", "/release", "/output"]
    )
    assert args.command == "reproduce-cross-platform"
    assert args.project_root == Path(".")
    assert args.release_root == Path("/release")
    assert args.output_root == Path("/output")


def test_rebuild_features_cli_contract() -> None:
    args = build_parser().parse_args(
        ["rebuild-features", ".", "/release", "ZD-MAST-B", "/output", "--workers", "24"]
    )
    assert args.command == "rebuild-features"
    assert args.site_id == "ZD-MAST-B"
    assert args.workers == 24


def test_primary_cli_contract() -> None:
    args = build_parser().parse_args(
        ["reproduce-primary", ".", "/release", "/output", "--threads", "8"]
    )
    assert args.command == "reproduce-primary"
    assert args.project_root == Path(".")
    assert args.release_root == Path("/release")
    assert args.output_root == Path("/output")
    assert args.threads == 8


def test_patient_episode_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "reproduce-patient-episode",
            ".",
            "/release",
            "/primary.csv",
            "/output",
        ]
    )
    assert args.command == "reproduce-patient-episode"
    assert args.primary_metrics == Path("/primary.csv")


def test_m100_cli_contract() -> None:
    args = build_parser().parse_args(
        ["reproduce-m100-agreement", ".", "/release", "/output"]
    )
    assert args.command == "reproduce-m100-agreement"
    assert args.release_root == Path("/release")
