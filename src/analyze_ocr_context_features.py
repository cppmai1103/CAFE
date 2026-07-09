"""Analyze OCR span evidence (ocr_features.csv), context evidence (context_features.csv),
and raw token-level OCR-quality signals (train data), and plot the results.

Per docs/phase1_manual.md SS3, a candidate (a predicted span + type from ner_features.csv)
is "reliable" (label_reliable = 1) iff it exactly matches a gold entity: same
start_token_id/end_token_id and the same type as NE-COARSE-LIT, closed into spans. This
script builds that gold-match label and reports (console tables only, no plot) whether OCR
quality -- of the span itself, or of the text around it -- predicts reliability.

ner_features.csv, ocr_features.csv, and context_features.csv are one row per candidate,
in the same order (verified: identical document_id/sentence_id/start_token_id/end_token_id
across all three) -- so they're joined positionally, no key merge needed.

Reliability tables (printed, no plot):
    1. Does the span's own OCR quality (ocr_correct) predict reliability?
    2. Does span_low_conf_word_fraction (finer-grained than ocr_correct) predict it?
    3. Does the OCR quality of the surrounding context (10 tokens each side) predict it,
       even though the context tokens aren't part of the span at all?

Plots, from the token-level train data:
    4. How many tokens does the OCR-QA bloom filter mark known (True) / unknown (False) /
       not-applicable-punctuation (None)?                          -> bar chart
    5. Distribution of document_ocr_mean, one value per document.  -> histogram
    6. Distribution of sentence_ocr_mean, one value per sentence.  -> histogram

Usage:
    python analyze_ocr_context_features.py
    python analyze_ocr_context_features.py --figures-dir /tmp/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_TRAIN_DATA = DATA_DIR / "hipe2020_train_fr_train_data.csv"
DEFAULT_NER_FEATURES = DATA_DIR / "ner_features.csv"
DEFAULT_OCR_FEATURES = DATA_DIR / "ocr_features.csv"
DEFAULT_CONTEXT_FEATURES = DATA_DIR / "context_features.csv"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent / "figures"

# HIPE's coarse bare types map 1:1 onto GLiNER2's predicted_entity_type values.
_TYPE_MAP = {"pers": "PERS", "loc": "LOC", "org": "ORG", "time": "TIME", "prod": "PROD"}

CATEGORICAL_BLUE = "#2a78d6"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"


def gold_type(tag: str) -> str | None:
    """Normalize a gold NE-COARSE-LIT bare type to GLiNER's scheme, or None for "O"/an
    out-of-scope subtype (e.g. a component tag)."""
    if pd.isna(tag) or tag == "O":
        return None
    raw_type = tag.split("-", 1)[1]
    return _TYPE_MAP.get(raw_type.split(".", 1)[0])


def build_gold_spans(train_df: pd.DataFrame) -> dict[tuple, str]:
    """Close NE-COARSE-LIT's per-token IOB2 tags into gold spans, keyed by
    (document_id, start_token_id, end_token_id) -> entity_type, for exact-match lookup
    against candidate spans."""
    tags_df = train_df[["document_id", "token_id", "NE-COARSE-LIT"]].rename(columns={"NE-COARSE-LIT": "tag"})

    spans: dict[tuple, str] = {}
    current: dict | None = None

    def flush():
        nonlocal current
        if current is not None:
            spans[(current["document_id"], current["start"], current["end"])] = current["type"]
        current = None

    for row in tqdm(tags_df.itertuples(index=False), total=len(tags_df), desc="Closing gold spans", unit="token"):
        prefix = None if row.tag == "O" else row.tag.split("-", 1)[0]
        etype = gold_type(row.tag)
        same_doc = current is not None and current["document_id"] == row.document_id
        continues = same_doc and prefix == "I" and etype == current["type"]

        if not continues:
            flush()
        if etype is None:
            continue
        if continues:
            current["end"] = row.token_id
        else:
            current = {"document_id": row.document_id, "start": row.token_id, "end": row.token_id, "type": etype}

    flush()
    return spans


def load_candidates(ner_path: Path, ocr_path: Path, context_path: Path) -> pd.DataFrame:
    ner_df = pd.read_csv(ner_path)
    ocr_df = pd.read_csv(ocr_path)
    context_df = pd.read_csv(context_path)

    key_cols = ["document_id", "sentence_id", "start_token_id", "end_token_id"]
    if not ner_df[key_cols].equals(ocr_df[key_cols]) or not ner_df[key_cols].equals(context_df[key_cols]):
        raise ValueError("ner_features.csv, ocr_features.csv, and context_features.csv are not row-aligned")

    ocr_only = ocr_df.drop(columns=key_cols + ["span_text"])
    context_only = context_df.drop(columns=key_cols + ["sentence_ocr_mean"])
    return pd.concat([ner_df, ocr_only, context_only], axis=1)


def label_reliability(candidates_df: pd.DataFrame, gold_spans: dict[tuple, str]) -> pd.Series:
    tqdm.pandas(desc="Matching candidates against gold spans", unit="candidate")

    def is_reliable(row) -> bool:
        if pd.isna(row["start_token_id"]) or pd.isna(row["end_token_id"]):
            return False
        key = (row["document_id"], int(row["start_token_id"]), int(row["end_token_id"]))
        return gold_spans.get(key) == row["predicted_entity_type"]

    return candidates_df.progress_apply(is_reliable, axis=1)


def reliability_by_bucket(df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    grouped = df.groupby(bucket_col, observed=True)["label_reliable"]
    return pd.DataFrame({"n_candidates": grouped.size(), "reliability_rate": grouped.mean()}).reset_index()


def plot_dictionary_score_counts(train_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of how many tokens the OCR-QA bloom filter marked known (True) / unknown
    (False) / not-applicable-punctuation (None)."""
    dictionary_score = train_df["dictionary_score"]
    labels = ["Known (True)", "Unknown (False)", "N/A -- punctuation (None)"]
    values = [int((dictionary_score == True).sum()), int((dictionary_score == False).sum()), int(dictionary_score.isna().sum())]
    colors = [STATUS_GOOD, STATUS_CRITICAL, MUTED_INK]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], fontsize=9, color=MUTED_INK, padding=3)

    ax.set_ylabel("Token count", color=PRIMARY_INK)
    ax.set_title("Token-level dictionary_score (OCR-QA bloom filter)", color=PRIMARY_INK)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)


def plot_ocr_mean_distribution(values: pd.Series, out_path: Path, title: str, xlabel: str) -> None:
    # Auto-range to where the data actually lives -- these means cluster tight near 1.0,
    # so a fixed (0, 1) range would waste most of the chart on empty space.
    lo, hi = values.min(), values.max()
    pad = max((hi - lo) * 0.05, 0.005)
    x_min, x_max = max(0.0, lo - pad), min(1.0, hi + pad)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    ax.hist(values, bins=30, range=(x_min, x_max), color=CATEGORICAL_BLUE, edgecolor=CHART_SURFACE, linewidth=0.5)

    ax.set_ylabel("Count", color=PRIMARY_INK)
    ax.set_xlabel(xlabel, color=PRIMARY_INK)
    ax.set_title(title, color=PRIMARY_INK)
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA), help="Token-level train data CSV (gold labels)")
    parser.add_argument("--ner-features", default=str(DEFAULT_NER_FEATURES), help="ner_features.csv")
    parser.add_argument("--ocr-features", default=str(DEFAULT_OCR_FEATURES), help="ocr_features.csv")
    parser.add_argument("--context-features", default=str(DEFAULT_CONTEXT_FEATURES), help="context_features.csv")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save plots into")
    args = parser.parse_args()

    print("=== Step 1: Load train data and close gold spans ===")
    train_df = pd.read_csv(args.train_data, dtype={"TOKEN": str, "MISC": str})
    train_df["token_id"] = train_df["token_id"].astype(int)
    gold_spans = build_gold_spans(train_df)
    print(f"{len(gold_spans)} gold entity spans")

    print("=== Step 2: Load and join candidate feature tables ===")
    print(f"Loading {args.ner_features}, {args.ocr_features}, {args.context_features}")
    candidates_df = load_candidates(Path(args.ner_features), Path(args.ocr_features), Path(args.context_features))
    print(f"{len(candidates_df)} candidates")

    print("=== Step 3: Label each candidate reliable/unreliable against gold ===")
    candidates_df["label_reliable"] = label_reliability(candidates_df, gold_spans)
    overall_rate = candidates_df["label_reliable"].mean()
    print(f"Overall reliability rate: {overall_rate:.4%} ({candidates_df['label_reliable'].sum()} / {len(candidates_df)})")

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 4: Reliability vs span OCR correctness (ocr_correct) ===")
    ocr_correct_df = candidates_df.dropna(subset=["ocr_correct"]).copy()
    ocr_correct_df["ocr_correct"] = ocr_correct_df["ocr_correct"].map({True: "Correct", False: "Has OCR error", "True": "Correct", "False": "Has OCR error"})
    ocr_summary = reliability_by_bucket(ocr_correct_df, "ocr_correct")
    print(ocr_summary.to_string(index=False))

    print("=== Step 5: Reliability vs span_low_conf_word_fraction (finer-grained) ===")
    frac_df = candidates_df.dropna(subset=["span_low_conf_word_fraction"]).copy()
    bins = [-0.01, 0.0, 0.34, 0.67, 1.0]
    bin_labels = ["0% (all known)", "1-34%", "35-67%", "68-100%"]
    frac_df["low_conf_bucket"] = pd.cut(frac_df["span_low_conf_word_fraction"], bins=bins, labels=bin_labels)
    frac_summary = reliability_by_bucket(frac_df, "low_conf_bucket")
    print(frac_summary.to_string(index=False))

    print("=== Step 6: Reliability vs surrounding context OCR quality (10 tokens each side) ===")
    context_df = candidates_df.dropna(subset=["context_low_conf_word_fraction_10"]).copy()
    context_df["context_bucket"] = pd.cut(
        context_df["context_low_conf_word_fraction_10"], bins=bins, labels=bin_labels
    )
    context_summary = reliability_by_bucket(context_df, "context_bucket")
    print(context_summary.to_string(index=False))

    print("=== Step 7: Token-level dictionary_score counts (True/False/None) ===")
    dictionary_score = train_df["dictionary_score"]
    print(
        f"Known (True): {int((dictionary_score == True).sum()):,}  "
        f"Unknown (False): {int((dictionary_score == False).sum()):,}  "
        f"N/A (None): {int(dictionary_score.isna().sum()):,}"
    )
    plot_dictionary_score_counts(train_df, figures_dir / "dictionary_score_counts.png")
    print(f"Saved {figures_dir / 'dictionary_score_counts.png'}")

    print("=== Step 8: Distribution of document_ocr_mean (one value per document) ===")
    document_means = train_df.drop_duplicates("document_id")["document_ocr_mean"].dropna()
    print(document_means.describe())
    plot_ocr_mean_distribution(
        document_means, figures_dir / "document_ocr_mean_distribution.png",
        "Distribution of document_ocr_mean (one value per document)", "document_ocr_mean",
    )
    print(f"Saved {figures_dir / 'document_ocr_mean_distribution.png'}")

    print("=== Step 9: Distribution of sentence_ocr_mean (one value per sentence) ===")
    sentence_means = train_df.drop_duplicates(["document_id", "sentence_id"])["sentence_ocr_mean"].dropna()
    print(sentence_means.describe())
    plot_ocr_mean_distribution(
        sentence_means, figures_dir / "sentence_ocr_mean_distribution.png",
        "Distribution of sentence_ocr_mean (one value per sentence)", "sentence_ocr_mean",
    )
    print(f"Saved {figures_dir / 'sentence_ocr_mean_distribution.png'}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
