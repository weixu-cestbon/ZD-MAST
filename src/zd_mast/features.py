"""Frozen feature construction for ZD-MAST spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter


SAVGOL_WINDOW = 15
SAVGOL_POLYORDER = 3
BASELINE_SIGMA_POINTS = 150.0
PROMINENCE_NOISE_MULTIPLIER = 3.0
MIN_PEAK_DISTANCE_POINTS = 5


@dataclass(frozen=True)
class FeatureSchema:
    """Definition of the canonical 6,000-bin feature space."""

    mz_min: float = 2000.0
    mz_max: float = 20000.0
    bin_width: float = 3.0

    @property
    def n_features(self) -> int:
        return int(round((self.mz_max - self.mz_min) / self.bin_width))


@dataclass(frozen=True)
class DensePeakParameters:
    """Label-free peak extraction settings for converted dense profiles."""

    savgol_window: int = SAVGOL_WINDOW
    savgol_polyorder: int = SAVGOL_POLYORDER
    baseline_sigma_points: float = BASELINE_SIGMA_POINTS
    prominence_noise_multiplier: float = PROMINENCE_NOISE_MULTIPLIER
    min_peak_distance_points: int = MIN_PEAK_DISTANCE_POINTS

    def validate(self) -> None:
        if self.savgol_window < 3 or self.savgol_window % 2 == 0:
            raise ValueError("Savitzky-Golay window must be odd and at least 3")
        if self.savgol_polyorder < 0 or self.savgol_polyorder >= self.savgol_window:
            raise ValueError("Savitzky-Golay polynomial order is invalid")
        if self.baseline_sigma_points <= 0:
            raise ValueError("Baseline sigma must be positive")
        if self.prominence_noise_multiplier <= 0:
            raise ValueError("Prominence multiplier must be positive")
        if self.min_peak_distance_points < 1:
            raise ValueError("Minimum peak distance must be at least 1")


def l2_normalize(values: np.ndarray) -> np.ndarray:
    """Return a float32 L2-normalized copy, preserving all-zero vectors."""

    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm):
        raise ValueError("Feature vector contains non-finite values")
    if norm == 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def bin_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    schema: FeatureSchema | None = None,
) -> np.ndarray:
    """Build the canonical intensity6000 vector from one spectrum.

    Values outside ``[mz_min, mz_max)`` are excluded. Multiple observations in
    one bin are aggregated by maximum intensity, followed by square-root and
    spectrum-level L2 normalization.
    """

    schema = schema or FeatureSchema()
    mz_array = np.asarray(mz, dtype=np.float64)
    intensity_array = np.asarray(intensity, dtype=np.float64)
    if mz_array.ndim != 1 or intensity_array.ndim != 1:
        raise ValueError("m/z and intensity must be one-dimensional")
    if mz_array.shape != intensity_array.shape:
        raise ValueError("m/z and intensity lengths differ")
    if not np.isfinite(mz_array).all() or not np.isfinite(intensity_array).all():
        raise ValueError("Spectrum contains non-finite values")
    if (intensity_array < 0).any():
        raise ValueError("Intensity values must be non-negative")

    output = np.zeros(schema.n_features, dtype=np.float64)
    keep = (mz_array >= schema.mz_min) & (mz_array < schema.mz_max)
    if keep.any():
        indices = np.floor((mz_array[keep] - schema.mz_min) / schema.bin_width).astype(np.int64)
        np.maximum.at(output, indices, intensity_array[keep])
    np.sqrt(output, out=output)
    return l2_normalize(output)


def aggregate_replicates(replicates: np.ndarray) -> np.ndarray:
    """Average spectrum-level vectors and apply sample-level L2 normalization."""

    matrix = np.asarray(replicates, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Replicate matrix must be two-dimensional")
    if matrix.shape[0] == 0:
        raise ValueError("At least one replicate is required")
    if not np.isfinite(matrix).all():
        raise ValueError("Replicate matrix contains non-finite values")
    return l2_normalize(matrix.mean(axis=0))


def peak_presence(intensity_features: np.ndarray) -> np.ndarray:
    """Convert sparse-spectrum intensity features to binary peak presence.

    This operation is valid for ZD-MAST-A peak-list-like exports. ZD-MAST-B
    converted dense profiles require :func:`dense_profile_peak_presence` so
    baseline-positive bins are not misclassified as detected peaks.
    """

    matrix = np.asarray(intensity_features)
    return (matrix > 0).astype(np.uint8)


def robust_sigma(values: np.ndarray) -> float:
    """Return the median-absolute-deviation noise estimate."""

    array = np.asarray(values, dtype=np.float64)
    value = float(1.4826 * np.median(np.abs(array - np.median(array))))
    return value if np.isfinite(value) and value > 0 else 1.0


def dense_profile_peak_presence(
    mz: np.ndarray,
    intensity: np.ndarray,
    schema: FeatureSchema | None = None,
    parameters: DensePeakParameters | None = None,
) -> np.ndarray:
    """Extract peak-presence features from a converted dense profile.

    Defaults reproduce the release-candidate Site B representation. Passing an
    explicit parameter object supports prespecified sensitivity analyses; it
    must never be selected by target AST performance.
    """

    schema = schema or FeatureSchema()
    parameters = parameters or DensePeakParameters()
    parameters.validate()
    mz_array = np.asarray(mz, dtype=np.float64)
    intensity_array = np.asarray(intensity, dtype=np.float64)
    if mz_array.ndim != 1 or intensity_array.ndim != 1:
        raise ValueError("m/z and intensity must be one-dimensional")
    if mz_array.shape != intensity_array.shape:
        raise ValueError("m/z and intensity lengths differ")
    if len(mz_array) < parameters.savgol_window:
        raise ValueError("Dense profile is shorter than the smoothing window")
    if not np.isfinite(mz_array).all() or not np.isfinite(intensity_array).all():
        raise ValueError("Dense profile contains non-finite values")
    if not np.all(np.diff(mz_array) > 0):
        raise ValueError("Dense-profile m/z values must be strictly increasing")

    smooth = savgol_filter(
        intensity_array,
        parameters.savgol_window,
        parameters.savgol_polyorder,
        mode="interp",
    )
    residual = smooth - gaussian_filter1d(
        smooth,
        parameters.baseline_sigma_points,
        mode="nearest",
    )
    peak_indices, _ = find_peaks(
        residual,
        prominence=(
            parameters.prominence_noise_multiplier * robust_sigma(residual)
        ),
        distance=parameters.min_peak_distance_points,
    )
    selected_mz = mz_array[peak_indices]
    keep = (selected_mz >= schema.mz_min) & (selected_mz < schema.mz_max)
    bins = np.floor(
        (selected_mz[keep] - schema.mz_min) / schema.bin_width
    ).astype(np.int64)
    output = np.zeros(schema.n_features, dtype=np.uint8)
    output[np.unique(bins)] = 1
    return output
