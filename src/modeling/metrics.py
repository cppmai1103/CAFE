"""Calibration metrics for Phase 1 baselines B0/B1/B3, split out of
calibrate_ner_confidence.py so the metric computations (independent of model fitting or
plotting) have one place to live.

Brier score is sklearn's own brier_score_loss (mean squared error between predicted
probability and the binary label_reliable outcome) -- re-exported here so callers only
need one import site for both metrics.

Expected Calibration Error (ECE): bins predictions into n_bins confidence buckets and
takes a size-weighted average of |avg_confidence - empirical_accuracy| across bins -- a
well-calibrated score means a bucket's average predicted probability should roughly equal
how often candidates in that bucket are actually reliable.

Maximum Calibration Error (MCE): the single worst bin's |avg_confidence - accuracy|, i.e.
max instead of ECE's size-weighted average -- unlike ECE, one small-but-badly-calibrated
bin can't get diluted away by the rest of the distribution being well-calibrated. Computed
from the same bin table ECE already builds (maximum_calibration_error_from_bins), so
compute_metrics_table gets it for free without rebinning.

AUROC and E-AURC measure DISCRIMINATION (does higher confidence rank reliable candidates
above unreliable ones?) rather than CALIBRATION (does the confidence value itself equal
the true probability?) -- a score can discriminate well while being badly calibrated (e.g.
scaled/shifted), or vice versa, so these are a different axis from Brier/ECE/MCE above.
Standard selective-classification definitions (Geifman & El-Yaniv 2017, "Selective
Classification for Deep Neural Networks"; Corbiere et al. 2019, "Addressing Failure
Prediction by Learning Model Confidence"):

- AUROC: sklearn's roc_auc_score(correct, confidences) -- treats "is this candidate
  reliable" as the positive class and confidence as the ranking score. Higher is better
  (1.0 = confidence perfectly ranks every reliable candidate above every unreliable one,
  0.5 = no better than random).

- AURC (area_under_risk_coverage_curve): sort candidates by confidence descending; at
  coverage c = k/n (the k most confident candidates), risk(c) is the error rate among
  just those k; AURC is the average of risk(c) over every k = 1..n. Lower is better (a
  confidence score that puts errors last keeps risk near 0 across most of the coverage
  range).

- E-AURC (excess_aurc): AURC minus the oracle's AURC (same overall error rate p, but
  sorted by TRUE correctness instead of confidence) -- isolates how much worse this
  score's ranking is than the best possible ranking at that error rate, since AURC alone
  conflates ranking quality with the base error rate. Oracle AURC has the closed form
  p + (1 - p) * ln(1 - p) (same references as above). Lower is better, 0 = oracle-optimal
  ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

__all__ = [
    "brier_score_loss",
    "roc_curve",
    "expected_calibration_error",
    "maximum_calibration_error_from_bins",
    "compute_metrics_table",
    "auroc",
    "risk_coverage_curve",
    "area_under_risk_coverage_curve",
    "excess_aurc",
]


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "avg_confidence": np.nan, "accuracy": np.nan})
            continue
        avg_conf = confidences[in_bin].mean()
        acc = correct[in_bin].mean()
        ece += (count / n) * abs(avg_conf - acc)
        rows.append({"bin_lo": lo, "bin_hi": hi, "count": count, "avg_confidence": avg_conf, "accuracy": acc})
    return ece, pd.DataFrame(rows)


def maximum_calibration_error_from_bins(bins_df: pd.DataFrame) -> float:
    """MCE from an already-computed ECE bin table: the largest |avg_confidence -
    accuracy| across bins that actually had candidates (count > 0), ignoring empty bins'
    NaN rows. 0.0 if every bin was empty (shouldn't happen with real data)."""
    gaps = (bins_df["avg_confidence"] - bins_df["accuracy"]).abs().dropna()
    return float(gaps.max()) if len(gaps) else 0.0


def auroc(confidences: np.ndarray, correct: np.ndarray) -> float:
    """sklearn's roc_auc_score(correct, confidences) -- see module docstring. Requires
    both classes (0 and 1) present in `correct`."""
    return float(roc_auc_score(correct, confidences))


def risk_coverage_curve(confidences: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    """One row per candidate, sorted by confidence descending: coverage = k/n (the top-k
    most confident candidates kept), risk = error rate among just those k -- see module
    docstring. area_under_risk_coverage_curve is just this curve's mean risk."""
    order = np.argsort(-confidences)
    sorted_correct = correct[order]
    n = len(sorted_correct)
    k = np.arange(1, n + 1)
    cumulative_errors = np.cumsum(1 - sorted_correct)
    return pd.DataFrame({"coverage": k / n, "risk": cumulative_errors / k})


def area_under_risk_coverage_curve(confidences: np.ndarray, correct: np.ndarray) -> float:
    """AURC -- see module docstring. The mean risk over risk_coverage_curve."""
    return float(risk_coverage_curve(confidences, correct)["risk"].mean())


def excess_aurc(confidences: np.ndarray, correct: np.ndarray) -> float:
    """E-AURC = AURC - oracle_AURC -- see module docstring. oracle_AURC has the closed
    form p + (1 - p) * ln(1 - p), where p is the overall error rate (handles p=0 and p=1
    as the limiting cases 0 and 1 respectively, since (1-p)*ln(1-p) -> 0 as p -> 1)."""
    aurc = area_under_risk_coverage_curve(confidences, correct)
    p = 1.0 - float(correct.mean())
    if p <= 0.0:
        oracle_aurc = 0.0
    elif p >= 1.0:
        oracle_aurc = 1.0
    else:
        oracle_aurc = p + (1.0 - p) * np.log(1.0 - p)
    return float(aurc - oracle_aurc)


def compute_metrics_table(
    labels: np.ndarray, b0_scores: np.ndarray, b1_scores: np.ndarray, b3_scores: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Brier score + ECE + MCE for B0/B1/B3 against the same binary labels, plus each
    baseline's ECE bin table (needed downstream for the sigmoid-fit and reliability
    diagram plots). Returns (metrics_df, b0_bins, b1_bins, b3_bins)."""
    b0_ece, b0_bins = expected_calibration_error(b0_scores, labels)
    b1_ece, b1_bins = expected_calibration_error(b1_scores, labels)
    b3_ece, b3_bins = expected_calibration_error(b3_scores, labels)
    metrics_df = pd.DataFrame(
        {
            "metric": ["Brier score", "ECE", "MCE"],
            "B0_raw_ner_score": [
                brier_score_loss(labels, b0_scores), b0_ece, maximum_calibration_error_from_bins(b0_bins),
            ],
            "B1_platt_calibrated": [
                brier_score_loss(labels, b1_scores), b1_ece, maximum_calibration_error_from_bins(b1_bins),
            ],
            "B3_logistic_regression": [
                brier_score_loss(labels, b3_scores), b3_ece, maximum_calibration_error_from_bins(b3_bins),
            ],
        }
    )
    return metrics_df, b0_bins, b1_bins, b3_bins
