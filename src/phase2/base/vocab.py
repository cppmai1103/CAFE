"""Integer vocabularies for Phase 2's side-channel embeddings (docs/phase2_learned_features.md SS6).
Embedding layers need integer ids, never None -- every flag/type value used anywhere in
Phase 2 must resolve to one of these.

Simple (first-version) vocabularies, per SS6.1/6.2's "recommended first version" -- richer
variants (OOV_CAPITALIZED vs OOV_LOWERCASE, START/END target boundaries) are listed in the
doc as later ablations, not implemented here yet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ner.gliner.extract_ner_features import LABELS

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


def entity_type_vocab_for(labels: list[str]) -> dict[str, int]:
    """Same construction as ENTITY_TYPE_VOCAB above, for a caller-supplied label list --
    lets Phase2Model/Phase2WindowDataset be built for a different NER source's own
    tagset (e.g. a corpus whose gold types don't match the standard HIPE-2022 5-type
    scheme), without touching the default ENTITY_TYPE_VOCAB every other caller still
    relies on."""
    return {label: i for i, label in enumerate(labels)}


def entity_type_vocab_from_file(labels_file: str | Path) -> dict[str, int]:
    """Same {TYPE: prompt wording} JSON file gliner/extract_ner_features.py's
    --labels-file reads (see that file's load_label_map() docstring, e.g.
    src/ner/gliner/labels.json or test/ajmc/labels.json) -- only the keys (TYPE codes) are
    used here, since Phase 2's vocab doesn't need the GLiNER2 prompt wording, just the
    type list. One JSON file per NER source drives both the extraction prompt AND the
    Phase 2 type vocab, instead of two separate --labels-file/--labels flags that could
    drift out of sync with each other."""
    with open(labels_file) as f:
        label_map = json.load(f)
    return entity_type_vocab_for(list(label_map.keys()))
