#!/usr/bin/env python3
"""Audit open peak tables against released spectrum-level feature matrices."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from zd_mast.features import bin_spectrum, dense_profile_peak_presence, peak_presence


def read_peak_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter="\t", skiprows=1, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"expected exactly two columns: {path}")
    return values[:, 0], values[:, 1]


def rebuild_one(path: Path, semantics: str) -> tuple[np.ndarray, np.ndarray]:
    mz, intensity = read_peak_table(path)
    intensity_vector = bin_spectrum(mz, intensity)
    if semantics == "mzml_export_peak_list_like":
        presence_vector = peak_presence(intensity_vector)
    elif semantics == "converted_dense_profile_txt":
        presence_vector = dense_profile_peak_presence(mz, intensity)
    else:
        raise ValueError(f"unsupported spectrum semantics: {semantics}")
    return intensity_vector, presence_vector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--site-id", choices=("ZD-MAST-A", "ZD-MAST-B"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    feature_dirs = sorted(args.release_root.glob("feature-release-*"))
    peak_dirs = sorted(args.release_root.glob("peak-table-release-*"))
    if len(feature_dirs) != 1 or len(peak_dirs) != 1:
        raise ValueError("release root must contain one feature and one peak-table release")
    feature, peak = feature_dirs[0], peak_dirs[0]
    site_code = "a" if args.site_id == "ZD-MAST-A" else "b"
    table_dir = peak / f"zd_mast_{site_code}_open_peak_tables"
    metadata = pd.read_csv(feature / "zd_mast_spectrum_metadata_public_v1.0.0.csv")
    metadata = metadata.loc[metadata["site_id"].eq(args.site_id)].reset_index(drop=True)
    paths = [table_dir / f"{value}.tsv" for value in metadata["public_spectrum_id"]]
    semantics = metadata["source_semantics"].astype(str).tolist()

    released_intensity = np.load(
        feature / f"zd_mast_{site_code}_spectrum_level_intensity6000_v1.0.0.npy",
        mmap_mode="r",
    )
    released_presence = np.load(
        feature / f"zd_mast_{site_code}_spectrum_level_peak_presence6000_v1.0.0.npy",
        mmap_mode="r",
    )
    if released_intensity.shape != (len(metadata), 6000):
        raise ValueError("spectrum metadata and released intensity matrix disagree")

    def work(item: tuple[Path, str]) -> tuple[np.ndarray, np.ndarray]:
        return rebuild_one(*item)

    sample_metadata = pd.read_csv(feature / "zd_mast_sample_metadata_public_v1.0.0.csv")
    sample_metadata = sample_metadata.loc[
        sample_metadata["site_id"].eq(args.site_id)
    ].sort_values("feature_row").reset_index(drop=True)
    expected_sample_rows = np.arange(len(sample_metadata), dtype=np.int64)
    if not np.array_equal(
        sample_metadata["feature_row"].to_numpy(dtype=np.int64),
        expected_sample_rows,
    ):
        raise ValueError("sample feature_row is not contiguous")
    sample_row_map = sample_metadata.set_index("public_sample_id")["feature_row"].astype(int)
    sample_intensity_sum = np.zeros((len(sample_metadata), 6000), dtype=np.float32)
    sample_presence = np.zeros((len(sample_metadata), 6000), dtype=np.uint8)
    sample_counts = np.zeros(len(sample_metadata), dtype=np.uint32)

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        rebuilt = executor.map(work, zip(paths, semantics, strict=True))
        for index, (intensity_vector, presence_vector) in enumerate(rebuilt):
            sample_id = str(metadata.loc[index, "public_sample_id"])
            sample_row = int(sample_row_map.loc[sample_id])
            sample_intensity_sum[sample_row] += intensity_vector
            sample_presence[sample_row] |= presence_vector
            sample_counts[sample_row] += 1
            intensity_max_abs = float(
                np.max(np.abs(intensity_vector - released_intensity[index]))
            )
            presence_difference_n = int(
                np.count_nonzero(presence_vector != released_presence[index])
            )
            rows.append(
                {
                    "feature_row": index,
                    "public_spectrum_id": metadata.loc[index, "public_spectrum_id"],
                    "source_semantics": semantics[index],
                    "released_presence_n": int(released_presence[index].sum()),
                    "rebuilt_presence_n": int(presence_vector.sum()),
                    "intensity_max_abs_difference": intensity_max_abs,
                    "presence_difference_n": presence_difference_n,
                    "intensity_status": "PASS" if intensity_max_abs <= 5e-7 else "FAIL",
                    "presence_status": "PASS" if presence_difference_n == 0 else "FAIL",
                }
            )

    audit = pd.DataFrame(rows)
    audit.to_csv(args.output_dir / "spectrum_feature_equivalence.csv", index=False)
    if np.any(sample_counts == 0):
        raise ValueError("sample metadata contains rows without public spectra")
    sample_intensity = sample_intensity_sum / sample_counts[:, None].astype(np.float32)
    norms = np.linalg.norm(sample_intensity, axis=1)
    nonzero = norms > 0
    sample_intensity[nonzero] /= norms[nonzero, None]
    sample_intensity[~nonzero] = 0.0
    released_sample_intensity = np.load(
        feature / f"zd_mast_{site_code}_sample_level_intensity6000_v1.0.0.npy",
        mmap_mode="r",
    )
    released_sample_presence = np.load(
        feature / f"zd_mast_{site_code}_sample_level_peak_presence6000_v1.0.0.npy",
        mmap_mode="r",
    )
    sample_rows: list[dict[str, object]] = []
    for index in range(len(sample_metadata)):
        intensity_max_abs = float(
            np.max(np.abs(sample_intensity[index] - released_sample_intensity[index]))
        )
        presence_difference_n = int(
            np.count_nonzero(sample_presence[index] != released_sample_presence[index])
        )
        sample_rows.append(
            {
                "feature_row": index,
                "public_sample_id": sample_metadata.loc[index, "public_sample_id"],
                "spectrum_n": int(sample_counts[index]),
                "released_presence_n": int(released_sample_presence[index].sum()),
                "rebuilt_presence_n": int(sample_presence[index].sum()),
                "intensity_max_abs_difference": intensity_max_abs,
                "presence_difference_n": presence_difference_n,
                "intensity_status": "PASS" if intensity_max_abs <= 5e-7 else "FAIL",
                "presence_status": "PASS" if presence_difference_n == 0 else "FAIL",
            }
        )
    sample_audit = pd.DataFrame(sample_rows)
    sample_audit.to_csv(args.output_dir / "sample_feature_equivalence.csv", index=False)
    summary = {
        "site_id": args.site_id,
        "spectrum_n": int(len(audit)),
        "intensity_pass_n": int(audit["intensity_status"].eq("PASS").sum()),
        "intensity_fail_n": int(audit["intensity_status"].eq("FAIL").sum()),
        "intensity_max_abs_difference": float(audit["intensity_max_abs_difference"].max()),
        "presence_pass_n": int(audit["presence_status"].eq("PASS").sum()),
        "presence_fail_n": int(audit["presence_status"].eq("FAIL").sum()),
        "released_zero_presence_rows": int(audit["released_presence_n"].eq(0).sum()),
        "rebuilt_zero_presence_rows": int(audit["rebuilt_presence_n"].eq(0).sum()),
        "sample_n": int(len(sample_audit)),
        "sample_intensity_pass_n": int(sample_audit["intensity_status"].eq("PASS").sum()),
        "sample_intensity_fail_n": int(sample_audit["intensity_status"].eq("FAIL").sum()),
        "sample_intensity_max_abs_difference": float(
            sample_audit["intensity_max_abs_difference"].max()
        ),
        "sample_presence_pass_n": int(sample_audit["presence_status"].eq("PASS").sum()),
        "sample_presence_fail_n": int(sample_audit["presence_status"].eq("FAIL").sum()),
        "released_zero_presence_samples": int(
            sample_audit["released_presence_n"].eq(0).sum()
        ),
        "rebuilt_zero_presence_samples": int(
            sample_audit["rebuilt_presence_n"].eq(0).sum()
        ),
        "uses_ast_labels": False,
    }
    (args.output_dir / "equivalence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["intensity_fail_n"] == 0 and summary["sample_intensity_fail_n"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
