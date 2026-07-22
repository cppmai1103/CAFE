"""OCR-quality dictionary-membership check, split out of preprocessing_data.py so the
bloom filter normalization/scoring logic can be reused independently of the HIPE-specific
loading/splitting pipeline (see other/extract_pressmint_ocrqa.py, which reuses
get_bloomfilter/word_is_known on a different, non-HIPE source).

dictionary_score is True/False/None per token, from the impresso
OCR-quality-assessment-unigram bloom filter for the token's language (a list of known
word forms built from Wikipedia + lexicons -- see bloom_filename_for/BLOOM_LANGUAGES for
which languages are available): True = known word, False = unknown (likely OCR error, or
a rare/proper name -- the filter can't tell the two apart), None = punctuation (not
scoreable). "Known" is a proxy for correct OCR, not a verified fact -- there is no
continuous OCR confidence anywhere in the HIPE data, so this 0/1 dictionary-membership
signal stands in for it throughout.

sentence_ocr_mean / document_ocr_mean are the mean dictionary_score (as 0/1) across all
scoreable (non-punctuation) tokens in the row's sentence / document respectively -- the
same value repeated on every row of that sentence / document. Punctuation tokens
(dictionary_score is None) are excluded from both, so a comma or period can't dilute them.
"""

from __future__ import annotations

import unicodedata
from typing import Optional

import pandas as pd
from huggingface_hub import hf_hub_download
from pybloomfilter import BloomFilter
from tqdm import tqdm

BLOOM_MODEL_ID = "impresso-project/OCR-quality-assessment-unigram"
BLOOM_LANGUAGES = ("fr", "de")
DEFAULT_BLOOM_LANGUAGE = "fr"


def bloom_filename_for(language: str) -> str:
    """impresso's per-language bloom filter filename, e.g. "fr" ->
    "ocrqa-wp_v1.0.6-fr.bloom" -- same v1.0.6 release used for every BLOOM_LANGUAGES
    entry, confirmed present on the model repo for both."""
    if language not in BLOOM_LANGUAGES:
        raise ValueError(f"language must be one of {BLOOM_LANGUAGES}, got {language!r}")
    return f"ocrqa-wp_v1.0.6-{language}.bloom"


BLOOM_FILENAME = bloom_filename_for(DEFAULT_BLOOM_LANGUAGE)  # backward-compat default (French)

# Normalization table as documented on the bloom filter's model card
# (NFKC + lowercase + digits->'0' + strip punctuation). Ported from
# notebook/hipe_ocr_ner_extraction.ipynb.
QUOTES_PUNCT = "„•<>!\"#%&'’"
ASCII_PUNCT = "()*,./:;?"
BRACKETS_SPECIAL = "[]\\~_{}"
UNICODE_PUNCT = "\xa1\xab\xb7\xbb\xbf"
DASH_CARET = "—^`"
SPECIAL_SYMBOLS = "\xa6\xa7\xa3="
HYPHEN = "-"
DIGITS = "0123456789"

NORMALIZATION_TABLE = str.maketrans(
    {char: " " for char in (QUOTES_PUNCT + ASCII_PUNCT + BRACKETS_SPECIAL + UNICODE_PUNCT + DASH_CARET + SPECIAL_SYMBOLS + HYPHEN)}
    | {char: "0" for char in DIGITS}
)


def normalize_text(s: str, unicode_normalize: Optional[str] = "NFKC") -> str:
    if unicode_normalize:
        s = unicodedata.normalize(unicode_normalize, s).lower()
    return s.translate(NORMALIZATION_TABLE)


def get_bloomfilter(model_id: str, filename: str) -> BloomFilter:
    return BloomFilter.open(hf_hub_download(repo_id=model_id, filename=filename))


def word_is_known(token: str, bloom_filter: BloomFilter) -> Optional[bool]:
    """True: token is a known French word (bloom filter hit).
    False: token normalized to a non-empty string but was not found (likely OCR error,
    or a rare/proper name).
    None: token normalized to nothing (pure punctuation/whitespace token) -- not
    applicable."""
    normalized = normalize_text(token).strip()
    if not normalized:
        return None
    sub_tokens = normalized.split()
    return all(t in bloom_filter for t in sub_tokens)


def compute_dictionary_score(df: pd.DataFrame, bloom_filter: BloomFilter) -> pd.DataFrame:
    """Append dictionary_score (True/False/None) to the token-level table: True = known
    French word, False = unknown (likely OCR error, or a rare/proper name -- the bloom
    filter can't tell the two apart), None = punctuation/not scoreable."""
    out = df.copy()
    tqdm.pandas(desc="Scoring token dictionary membership", unit="token")
    out["dictionary_score"] = out["TOKEN"].progress_apply(lambda t: word_is_known(t, bloom_filter))
    return out


def compute_ocr_means(df: pd.DataFrame) -> pd.DataFrame:
    """Append sentence_ocr_mean / document_ocr_mean to the token-level table: the mean
    dictionary_score (as 0/1) across all scoreable (non-punctuation) tokens in the row's
    sentence / document. Punctuation tokens (dictionary_score is None) are excluded
    entirely from both means -- sum(known) / count(scoreable) -- rather than counted as
    "known", so they can't dilute the result."""
    scoreable = df[df["dictionary_score"].notna()].copy()
    scoreable["dictionary_score"] = scoreable["dictionary_score"].astype(float)

    sentence_mean = (
        scoreable.groupby(["document_id", "sentence_id"])["dictionary_score"]
        .mean()
        .rename("sentence_ocr_mean")
        .reset_index()
    )
    document_mean = (
        scoreable.groupby("document_id")["dictionary_score"].mean().rename("document_ocr_mean").reset_index()
    )

    out = df.merge(sentence_mean, on=["document_id", "sentence_id"], how="left")
    out = out.merge(document_mean, on="document_id", how="left")
    return out
