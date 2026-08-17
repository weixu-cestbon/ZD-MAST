#!/usr/bin/env python3
"""Build a deterministic SHA-256 manifest for the public code candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError("VERSION must not be empty")
    output = root / f"manifests/public_code_file_manifest_{version}.csv"
    checksum_output = root / f"manifests/public_code_checksums_{version}.sha256"
    excluded = {output.resolve(), checksum_output.resolve()}
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
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not (set(path.relative_to(root).parts) & excluded_directories)
        and not any(part.endswith(".egg-info") for part in path.relative_to(root).parts)
        and path.resolve() not in excluded
    )
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "size_bytes", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    checksum_output.write_text(
        "".join(f"{row['sha256']}  {row['relative_path']}\n" for row in rows),
        encoding="utf-8",
    )
    print(f"manifest_files={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
