"""Train Phase 1 baseline B3: logistic regression over all implemented manual features
(docs/phase1_manual.md SS11), for comparison against B0 (raw ner_score) and B1 (Platt-
calibrated ner_score) in calibrate_ner_confidence.py.

Unlike B0/B1, which use only ner_score, B3 combines ner_score with every other manual
feature from the feature groups actually implemented so far -- SS4.1 (NER evidence),
SS4.2 (OCR span evidence), SS4.3 (context evidence). SS4.4/4.5/4.6 are marked "(optional:
not implement now)" in the manual, so their features aren't available to join in here.

Feature matrix (see NUMERIC_FEATURES/BOOLEAN_FEATURES/CATEGORICAL_FEATURES below):
    - SS4.1: ner_score, span_length_tokens, span_length_characters, predicted_entity_type
      (one-hot), sentence_chunked
    - SS4.2: span_ocr_mean, span_low_conf_word_fraction, span_first_word_ocr,
      span_last_word_ocr, sentence_ocr_mean, document_ocr_mean, ocr_correct
    - SS4.3: left_context_ocr_mean_10, right_context_ocr_mean_10, context_ocr_min_10,
      context_low_conf_word_fraction_10, sentence_length, context_window_length
top1_top2_type_margin and type_entropy (also SS4.1) aren't included -- GLiNER2 scores
each entity type as an independent sigmoid (see extract_ner_features.py's own docstring
caveat), so there's no shared softmax distribution across types to compute a margin or
entropy from.

Missing values: many of these are None for a candidate whose span is pure punctuation, or
near a sentence boundary (see extract_ocr_features.py / extract_context_features.py).
Each numeric feature with any missing values gets a companion <feature>_missing
indicator column (1.0 if missing), and NaNs are filled with that feature's expert_train
median -- computed on expert_train only (build_feature_matrix's `medians` argument) and
reused as-is on every other split, so no split's own distribution leaks into another
split's imputation.

Fit split: expert_train (docs/phase1_manual.md SS6.1 -- "expert_train: for training
experts"; B3 is trained like an expert would be, just over the combined feature set
rather than one family). No class-balancing is applied (e.g. no class_weight="balanced")
-- B3's whole purpose is to produce a probability that means what it says, and balancing
would change what P(label_reliable=1 | x) represents, defeating that purpose.

Usage:
    python train_b3_logistic_regression.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyze_ocr_context_features import (
    DEFAULT_CONTEXT_FEATURES,
    DEFAULT_NER_FEATURES,
    DEFAULT_OCR_FEATURES,
    DEFAULT_TRAIN_DATA,
    build_gold_spans,
    label_reliability,
    load_candidates,
)
from extract_ner_features import LABELS

NUMERIC_FEATURES = [
    "ner_score",
    "span_length_tokens",
    "span_length_characters",
    "span_ocr_mean",
    "span_low_conf_word_fraction",
    "span_first_word_ocr",
    "span_last_word_ocr",
    "sentence_ocr_mean",
    "document_ocr_mean",
    "left_context_ocr_mean_10",
    "right_context_ocr_mean_10",
    "context_ocr_min_10",
    "context_low_conf_word_fraction_10",
    "sentence_length",
    "context_window_length",
]
BOOLEAN_FEATURES = ["ocr_correct", "sentence_chunked"]
CATEGORICAL_FEATURES = ["predicted_entity_type"]


def _as_float_bool(series: pd.Series) -> pd.Series:
    """CSV round-tripped True/False (possibly as the strings "True"/"False") -> 1.0/0.0."""
    return series.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})


def build_feature_matrix(
    candidates_df: pd.DataFrame, fit_stats: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Build B3's feature matrix from a joined candidates table (see
    analyze_ocr_context_features.load_candidates). If `fit_stats` is None, both the
    imputation medians AND which features get a `_missing` indicator column are derived
    from candidates_df itself (the expert_train call site). Otherwise the given
    `fit_stats` (computed on expert_train) are reused as-is on every other split --
    medians, so no other split's own distribution leaks into its imputation, and *which*
    features get a `_missing` column, so a feature that happens to have zero NaNs in one
    split (e.g. test) but not another (e.g. expert_train) can't silently change the
    output's column count/order and break the fitted model's predict_proba."""
    out = pd.DataFrame(index=candidates_df.index)

    if fit_stats is None:
        fit_stats = {
            "medians": {feat: candidates_df[feat].median() for feat in NUMERIC_FEATURES},
            "missing_features": [feat for feat in NUMERIC_FEATURES if candidates_df[feat].isna().any()],
        }

    for feat in NUMERIC_FEATURES:
        values = candidates_df[feat]
        if feat in fit_stats["missing_features"]:
            out[f"{feat}_missing"] = values.isna().astype(float)
        out[feat] = values.fillna(fit_stats["medians"][feat])

    for feat in BOOLEAN_FEATURES:
        out[feat] = _as_float_bool(candidates_df[feat]).fillna(0.0)

    for feat in CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(
            pd.Categorical(candidates_df[feat], categories=LABELS), prefix=feat, dtype=float
        )
        dummies.index = candidates_df.index
        out = pd.concat([out, dummies], axis=1)

    return out, fit_stats


def fit_b3_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Standardize then fit plain logistic regression -- see module docstring for why
    class-balancing is deliberately skipped."""
    model = Pipeline([("scale", StandardScaler()), ("logreg", LogisticRegression(max_iter=2000))])
    model.fit(X_train, y_train)
    return model


def load_labeled_candidates(train_data_path: str, ner_path: str, ocr_path: str, context_path: str) -> pd.DataFrame:
    """train data + joined candidates, with label_reliable and split columns attached --
    shared entry point for this script and calibrate_ner_confidence.py."""
    train_df = pd.read_csv(train_data_path, dtype={"TOKEN": str, "MISC": str})
    train_df["token_id"] = train_df["token_id"].astype(int)
    gold_spans = build_gold_spans(train_df)
    doc_to_split = train_df.drop_duplicates("document_id").set_index("document_id")["split"].to_dict()

    candidates_df = load_candidates(Path(ner_path), Path(ocr_path), Path(context_path))
    candidates_df["label_reliable"] = label_reliability(candidates_df, gold_spans)
    candidates_df["split"] = candidates_df["document_id"].map(doc_to_split)
    return candidates_df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA))
    parser.add_argument("--ner-features", default=str(DEFAULT_NER_FEATURES))
    parser.add_argument("--ocr-features", default=str(DEFAULT_OCR_FEATURES))
    parser.add_argument("--context-features", default=str(DEFAULT_CONTEXT_FEATURES))
    args = parser.parse_args()

    print("=== Step 1: Load train data, gold spans, and joined candidate features ===")
    candidates_df = load_labeled_candidates(args.train_data, args.ner_features, args.ocr_features, args.context_features)
    print(f"{len(candidates_df)} candidates")

    print("=== Step 2: Build B3 feature matrix (expert_train medians + missing-indicator set) ===")
    train_mask = candidates_df["split"] == "expert_train"
    X_train, fit_stats = build_feature_matrix(candidates_df[train_mask])
    y_train = candidates_df.loc[train_mask, "label_reliable"].astype(int)
    print(f"{X_train.shape[1]} features, {len(X_train)} expert_train candidates")

    print("=== Step 3: Fit B3 logistic regression on expert_train ===")
    model = fit_b3_model(X_train, y_train)
    coefs = pd.Series(model.named_steps["logreg"].coef_[0], index=X_train.columns)
    print("Top coefficients (standardized scale, sorted by |coefficient|):")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15).to_string())

    print("=== Step 4: Score every split ===")
    for split in ["expert_train", "gate_train", "calibration", "test"]:
        mask = candidates_df["split"] == split
        X, _ = build_feature_matrix(candidates_df[mask], fit_stats=fit_stats)
        scores = model.predict_proba(X)[:, 1]
        y = candidates_df.loc[mask, "label_reliable"].astype(int)
        print(f"{split}: {len(X)} candidates, mean B3 score {scores.mean():.4f}, positive rate {y.mean():.4%}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
