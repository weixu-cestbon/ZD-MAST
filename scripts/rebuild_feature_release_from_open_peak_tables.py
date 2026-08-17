#!/usr/bin/env python3
"""Rebuild one site's spectrum- and sample-level ZD-MAST feature matrices."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from zd_mast.features import bin_spectrum, dense_profile_peak_presence, peak_presence


N_FEATURES = 6000


def read_peak_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter="\t", skiprows=1, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"expected exactly two columns: {path}")
    return values[:, 0], values[:, 1]


def vectorize(path: Path, semantics: str) -> tuple[np.ndarray, np.ndarray]:
    mz, intensity = read_peak_table(path)
    intensity_vector = bin_spectrum(mz, intensity)
    if semantics == "mzml_export_peak_list_like":
        presence_vector = peak_presence(intensity_vector)
    elif semantics == "converted_dense_profile_txt":
        presence_vector = dense_profile_peak_presence(mz, intensity)
    else:
        raise ValueError(f"unsupported spectrum semantics: {semantics}")
    return intensity_vector, presence_vector


def require_contiguous_feature_rows(frame: pd.DataFrame, name: str) -> None:
    expected = np.arange(len(frame), dtype=np.int64)
    observed = frame["feature_row"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError(f"{name} feature_row is not contiguous and ordered")


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

    spectrum_meta = pd.read_csv(feature / "zd_mast_spectrum_metadata_public_v1.0.0.csv")
    spectrum_meta = spectrum_meta.loc[
        spectrum_meta["site_id"].eq(args.site_id)
    ].sort_values("feature_row").reset_index(drop=True)
    sample_meta = pd.read_csv(feature / "zd_mast_sample_metadata_public_v1.0.0.csv")
    sample_meta = sample_meta.loc[
        sample_meta["site_id"].eq(args.site_id)
    ].sort_values("feature_row").reset_index(drop=True)
    require_contiguous_feature_rows(spectrum_meta, "spectrum metadata")
    require_contiguous_feature_rows(sample_meta, "sample metadata")
    if spectrum_meta["public_spectrum_id"].duplicated().any():
        raise ValueError("duplicate public spectrum IDs")
    if sample_meta["public_sample_id"].duplicated().any():
        raise ValueError("duplicate public sample IDs")

    spectrum_intensity = np.lib.format.open_memmap(
        args.output_dir / f"zd_mast_{site_code}_spectrum_level_intensity6000_rebuilt.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(spectrum_meta), N_FEATURES),
    )
    spectrum_presence = np.lib.format.open_memmap(
        args.output_dir / f"zd_mast_{site_code}_spectrum_level_peak_presence6000_rebuilt.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(spectrum_meta), N_FEATURES),
    )
    sample_intensity = np.lib.format.open_memmap(
        args.output_dir / f"zd_mast_{site_code}_sample_level_intensity6000_rebuilt.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(sample_meta), N_FEATURES),
    )
    sample_presence = np.lib.format.open_memmap(
        args.output_dir / f"zd_mast_{site_code}_sample_level_peak_presence6000_rebuilt.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(sample_meta), N_FEATURES),
    )
    sample_intensity[:] = 0
    sample_presence[:] = 0
    sample_counts = np.zeros(len(sample_meta), dtype=np.uint32)
    sample_row_map = sample_meta.set_index("public_sample_id")["feature_row"].astype(int)

    paths = [table_dir / f"{value}.tsv" for value in spectrum_meta["public_spectrum_id"]]
    if not all(path.exists() for path in paths):
        missing = [str(path) for path in paths if not path.exists()][:10]
        raise FileNotFoundError(f"missing public peak tables: {missing}")
    semantics = spectrum_meta["source_semantics"].astype(str).tolist()

    def work(item: tuple[Path, str]) -> tuple[np.ndarray, np.ndarray]:
        return vectorize(*item)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        rebuilt = executor.map(work, zip(paths, semantics, strict=True))
        for row, (intensity_vector, presence_vector) in enumerate(rebuilt):
            spectrum_intensity[row] = intensity_vector
            spectrum_presence[row] = presence_vector
            sample_id = str(spectrum_meta.loc[row, "public_sample_id"])
            sample_row = int(sample_row_map.loc[sample_id])
            sample_intensity[sample_row] += intensity_vector
            sample_presence[sample_row] |= presence_vector
            sample_counts[sample_row] += 1
    spectrum_intensity.flush()
    spectrum_presence.flush()
    if np.any(sample_counts == 0):
        raise ValueError(f"samples without spectra: {int(np.count_nonzero(sample_counts == 0))}")

    for start in range(0, len(sample_meta), 2048):
        stop = min(start + 2048, len(sample_meta))
        block = np.asarray(sample_intensity[start:stop], dtype=np.float32)
        block /= sample_counts[start:stop, None].astype(np.float32)
        norms = np.linalg.norm(block, axis=1)
        nonzero = norms > 0
        block[nonzero] /= norms[nonzero, None]
        block[~nonzero] = 0.0
        sample_intensity[start:stop] = block
    sample_intensity.flush()
    sample_presence.flush()

    released = {
        "spectrum_intensity": np.load(
            feature / f"zd_mast_{site_code}_spectrum_level_intensity6000_v1.0.0.npy",
            mmap_mode="r",
        ),
        "spectrum_presence": np.load(
            feature / f"zd_mast_{site_code}_spectrum_level_peak_presence6000_v1.0.0.npy",
            mmap_mode="r",
        ),
        "sample_intensity": np.load(
            feature / f"zd_mast_{site_code}_sample_level_intensity6000_v1.0.0.npy",
            mmap_mode="r",
        ),
        "sample_presence": np.load(
            feature / f"zd_mast_{site_code}_sample_level_peak_presence6000_v1.0.0.npy",
            mmap_mode="r",
        ),
    }
    rebuilt_matrix = {
        "spectrum_intensity": spectrum_intensity,
        "spectrum_presence": spectrum_presence,
        "sample_intensity": sample_intensity,
        "sample_presence": sample_presence,
    }
    rows: list[dict[str, object]] = []
    for name in released:
        is_presence = name.endswith("presence")
        max_abs = float(np.max(np.abs(rebuilt_matrix[name] - released[name])))
        different_rows = int(
            np.count_nonzero(np.any(rebuilt_matrix[name] != released[name], axis=1))
        )
        status = (
            "PASS"
            if (is_presence and different_rows == 0)
            or (not is_presence and max_abs <= 5e-7)
            else "FAIL"
        )
        rows.append(
            {
                "site_id": args.site_id,
                "matrix": name,
                "row_n": int(released[name].shape[0]),
                "feature_n": int(released[name].shape[1]),
                "max_abs_difference": max_abs,
                "different_row_n": different_rows,
                "status": status,
            }
        )
    equivalence = pd.DataFrame(rows)
    equivalence.to_csv(args.output_dir / "feature_rebuild_equivalence.csv", index=False)
    summary = {
        "site_id": args.site_id,
        "spectrum_n": int(len(spectrum_meta)),
        "sample_n": int(len(sample_meta)),
        "n_features": N_FEATURES,
        "spectra_per_sample_min": int(sample_counts.min()),
        "spectra_per_sample_median": float(np.median(sample_counts)),
        "spectra_per_sample_max": int(sample_counts.max()),
        "equivalence_gate": "PASS" if equivalence["status"].eq("PASS").all() else "FAIL",
        "uses_ast_labels": False,
    }
    (args.output_dir / "feature_rebuild_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["equivalence_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
