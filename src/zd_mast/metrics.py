"""Metrics and validation-derived threshold policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class BinaryOperatingPoint:
    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float


def _operating_points(y_true: np.ndarray, probability: np.ndarray) -> list[BinaryOperatingPoint]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("Labels and probabilities must be aligned one-dimensional arrays")
    if not set(np.unique(y)).issubset({0, 1}) or np.unique(y).size != 2:
        raise ValueError("Both binary classes are required")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")

    thresholds = np.unique(np.concatenate(([0.0], p, [1.0, np.nextafter(1.0, 2.0)])))
    points: list[BinaryOperatingPoint] = []
    for threshold in thresholds:
        predicted = p >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        tn = int(np.sum(~predicted & (y == 0)))
        fp = int(np.sum(predicted & (y == 0)))
        fn = int(np.sum(~predicted & (y == 1)))
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp) if tp + fp else float("nan")
        npv = tn / (tn + fn) if tn + fn else float("nan")
        points.append(BinaryOperatingPoint(float(threshold), sensitivity, specificity, ppv, npv))
    return points


def youden_threshold(y_true: np.ndarray, probability: np.ndarray) -> BinaryOperatingPoint:
    """Choose the validation threshold maximizing Youden's J statistic."""

    points = _operating_points(y_true, probability)
    return max(points, key=lambda point: (point.sensitivity + point.specificity - 1.0, point.threshold))


def threshold_at_min_sensitivity(
    y_true: np.ndarray,
    probability: np.ndarray,
    target: float = 0.90,
) -> BinaryOperatingPoint:
    """Choose the highest threshold meeting a minimum validation sensitivity."""

    eligible = [point for point in _operating_points(y_true, probability) if point.sensitivity >= target]
    if not eligible:
        raise ValueError("No threshold meets the requested sensitivity")
    return max(eligible, key=lambda point: (point.threshold, point.specificity))


def threshold_at_min_specificity(
    y_true: np.ndarray,
    probability: np.ndarray,
    target: float = 0.90,
) -> BinaryOperatingPoint:
    """Choose the lowest threshold meeting a minimum validation specificity."""

    eligible = [point for point in _operating_points(y_true, probability) if point.specificity >= target]
    if not eligible:
        raise ValueError("No threshold meets the requested specificity")
    return min(eligible, key=lambda point: (point.threshold, -point.sensitivity))


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    """Calculate equal-width expected calibration error."""

    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (p >= lower) & (p < upper if index < bins - 1 else p <= upper)
        if mask.any():
            error += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error) if total else float("nan")


def discrimination_summary(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Return ranking, calibration, and prevalence metrics."""

    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    prevalence = float(y.mean())
    auprc = float(average_precision_score(y, p))
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": auprc,
        "auprc_baseline": prevalence,
        "auprc_lift": auprc / prevalence if prevalence > 0 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }

