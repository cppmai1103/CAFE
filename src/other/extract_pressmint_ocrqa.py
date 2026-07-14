"""Apply the OCR-QA bloom filter (see preprocessing_data.py) to every word in a PressMint
TEI XML newspaper page, and plot the resulting dictionary_score distribution.

PressMint TEI format (inferred from the file itself, e.g.
PressMint-FR_1920-01-01-LeTemps-BnF-0243877.xml): a <TEI>/<text>/<body> containing <p>
paragraphs, each holding raw OCR text interleaved with <lb facs="..."/> line-break
markers -- there's no pre-tokenized word layer like HIPE's TSV, so words here are just
whitespace-split substrings of that raw text. Paragraph text is reconstructed via
itertext() (concatenating both a <p>'s own text and each <lb/> child's tail text, in
document order) so words split across a line break aren't accidentally merged -- the
source XML's own pretty-printed newlines land in each <lb/>'s tail text and act as the
separator.

Reuses get_bloomfilter/word_is_known from preprocessing/ocr_dictionary_check.py rather
than reimplementing OCR-QA scoring -- same French bloom filter, same normalization
(NFKC + lowercase + digits->0 + strip punctuation). Garbled OCR sequences (this file has
plenty, e.g. repeated mojibake for a corrupted replacement character) aren't cleaned up
before scoring -- they're exactly the kind of token OCR-QA is meant to flag as unknown.

Output: figures/<file-stem>_ocrqa_distribution.png -- bar chart of how many words the
bloom filter marked known (True) / unknown (False) / not-applicable-punctuation (None).

Usage:
    python src/other/extract_pressmint_ocrqa.py
    python src/other/extract_pressmint_ocrqa.py --xml-path /path/to/other-page.xml
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.ocr_dictionary_check import BLOOM_FILENAME, BLOOM_MODEL_ID, get_bloomfilter, word_is_known

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_XML = REPO_ROOT / "PressMint-FR_1920-01-01-LeTemps-BnF-0243877.xml"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures"

STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
MUTED_INK = "#898781"
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
GRIDLINE = "#e1e0d9"


def extract_words(xml_path: Path) -> list[str]:
    """Every whitespace-delimited word in the TEI file's <body>, in document order."""
    tree = ET.parse(xml_path)
    body = tree.getroot().find(".//tei:body", TEI_NS)
    words = []
    for p in body.findall(".//tei:p", TEI_NS):
        text = "".join(p.itertext())
        words.extend(text.split())
    return words


def plot_ocrqa_distribution(scores: list[Optional[bool]], out_path: Path, title: str) -> None:
    """Bar chart of how many words scored known (True) / unknown (False) /
    not-applicable-punctuation (None), same convention as
    analyze_ocr_context_features.py's plot_dictionary_score_counts."""
    labels = ["Known (True)", "Unknown (False)", "N/A -- punctuation (None)"]
    values = [
        sum(1 for s in scores if s is True),
        sum(1 for s in scores if s is False),
        sum(1 for s in scores if s is None),
    ]
    colors = [STATUS_GOOD, STATUS_CRITICAL, MUTED_INK]
    total = len(scores)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(
        bars, labels=[f"{v:,} ({v / total:.1%})" for v in values], fontsize=9, color=MUTED_INK, padding=3
    )

    ax.set_ylabel("Word count", color=PRIMARY_INK)
    ax.set_title(title, color=PRIMARY_INK)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml-path", default=str(DEFAULT_XML), help="PressMint TEI XML file to analyze")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the plot into")
    args = parser.parse_args()

    xml_path = Path(args.xml_path)

    print("=== Step 1: Extract words from the TEI XML ===")
    words = extract_words(xml_path)
    print(f"{len(words)} words extracted from {xml_path.name}")

    print("=== Step 2: Load French OCR-quality bloom filter ===")
    bloom_filter = get_bloomfilter(BLOOM_MODEL_ID, BLOOM_FILENAME)

    print("=== Step 3: Score every word ===")
    scores = [word_is_known(w, bloom_filter) for w in tqdm(words, desc="Scoring words", unit="word")]
    known = sum(1 for s in scores if s is True)
    unknown = sum(1 for s in scores if s is False)
    na = sum(1 for s in scores if s is None)
    print(
        f"Known: {known:,} ({known / len(scores):.2%})  "
        f"Unknown: {unknown:,} ({unknown / len(scores):.2%})  "
        f"N/A: {na:,} ({na / len(scores):.2%})"
    )

    print("=== Step 4: Plot distribution ===")
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"{xml_path.stem}_ocrqa_distribution.png"
    plot_ocrqa_distribution(scores, out_path, title=f"OCR-QA word distribution: {xml_path.name}")
    print(f"Saved {out_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
