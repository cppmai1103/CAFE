"""Integer vocabularies for Phase 2's side-channel embeddings (docs/new_phase2.md SS6).
Embedding layers need integer ids, never None -- every flag/type value used anywhere in
Phase 2 must resolve to one of these.

Simple (first-version) vocabularies, per SS6.1/6.2's "recommended first version" -- richer
variants (OOV_CAPITALIZED vs OOV_LOWERCASE, START/END target boundaries) are listed in the
doc as later ablations, not implemented here yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gliner.extract_ner_features import LABELS

# SS6.1: dictionary/OCR-quality flag per token. Maps directly from
# preprocessing/ocr_dictionary_check.py's dictionary_score (True/False/None):
#   True -> GOOD, False -> BAD, None (punctuation, not scoreable) -> PUNCT.
# SPECIAL is for [CLS]/[SEP]/padding introduced at the subword-tokenization stage (SS9).
# No UNKNOWN entry -- dictionary_score only ever produces those 3 cases, so it would be
# dead code; add it back if a richer dictionary vocab (SS6.1's OOV_CAPITALIZED/
# MIXED_ALPHA_DIGIT/... later version) is implemented.
DICT_FLAG_VOCAB = {
    "PAD": 0,
    "SPECIAL": 1,
    "GOOD": 2,
    "BAD": 3,
    "PUNCT": 4,
}

# SS6.2: is this token part of the candidate's target span? Simple 3-way version
# (SPECIAL/OUTSIDE/INSIDE_TARGET) recommended first; START/INSIDE/END is a later ablation.
TARGET_FLAG_VOCAB = {
    "PAD": 0,
    "SPECIAL": 1,
    "OUTSIDE": 2,
    "INSIDE_TARGET": 3,
}

# SS6.3: predicted entity type. Reuses the project's existing HIPE label scheme
# (gliner/extract_ner_features.LABELS = PERS/LOC/ORG/TIME/PROD) rather than the doc's
# illustrative PERSON/LOCATION/ORGANIZATION/OTHER example, so Phase 2 candidates (which
# come from the same GLiNER2 extraction as Phase 1) use the same type vocabulary
# throughout the project.
ENTITY_TYPE_VOCAB = {label: i for i, label in enumerate(LABELS)}
