#!/usr/bin/env python3
"""Audit a candidate public repository for sensitive paths and unsafe files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".mzml",
    ".spectrum",
    ".db",
    ".sqlite",
    ".pkl",
    ".joblib",
    ".pt",
    ".pth",
    ".ckpt",
    ".parquet",
    ".feather",
    ".h5",
    ".hdf5",
    ".xlsx",
    ".xls",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".cff",
    ".csv",
    ".gitignore",
    ".gitattributes",
    ".json",
    ".md",
    ".py",
    ".sha256",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_PATTERNS = {
    "mac_absolute_home": re.compile(r"/" + r"Users/[A-Za-z0-9._-]+/"),
    "linux_absolute_home": re.compile(r"/" + r"home/[A-Za-z0-9._-]+/"),
    "private_ipv4": re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),
    "windows_absolute_path": re.compile(r"\b[A-Za-z]:\\[^\s]+"),
}


def iter_files(root: Path) -> list[Path]:
    excluded_directories = {
        ".git",
        ".mamba",
        ".conda",
        ".pytest_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
    }
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not (set(path.relative_to(root).parts) & excluded_directories)
        and not any(part.endswith(".egg-info") for part in path.relative_to(root).parts)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-bytes", type=int, default=50 * 1024 * 1024)
    args = parser.parse_args()
    root = args.root.resolve()
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError("VERSION must not be empty")
    output = args.output or root / f"manifests/public_repository_privacy_audit_{version}.csv"
    findings: list[dict[str, object]] = []

    for path in iter_files(root):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            findings.append({"path": relative, "finding": "forbidden_suffix", "detail": path.suffix})
        if size > args.max_bytes:
            findings.append({"path": relative, "finding": "large_file", "detail": size})
        suffix = path.suffix.casefold() or path.name.casefold()
        if suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append({"path": relative, "finding": "non_utf8_text", "detail": "decode_failed"})
            continue
        for name, pattern in PRIVATE_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                findings.append(
                    {"path": relative, "finding": name, "detail": f"match_count={len(matches)}"}
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "finding", "detail"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(findings)
    status = "PASS" if not findings else "FAIL"
    print(f"public_repository_privacy_audit={status} findings={len(findings)} output={output}")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
