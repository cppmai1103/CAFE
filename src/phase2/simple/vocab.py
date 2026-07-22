"""Literal marker text for phase2_simple's prompt-style input (see model.py's module
docstring): the candidate's type/confidence metadata is written directly into the token
sequence as bracketed tags, instead of phase2's separate trainable side-channel embeddings
(DictFlagEmb/TargetFlagEmb/TypeEmb/ScoreMLP). No integer vocab is needed here -- the
frozen encoder's own tokenizer handles these tag strings like any other text.

TYPE_DISPLAY_NAME maps the project's existing short entity-type codes (gliner/
extract_ner_features.LABELS, reused by phase2/vocab.py's ENTITY_TYPE_VOCAB) to plain
English words, on the theory that a frozen pretrained LM's subword vocabulary already
carries a much stronger semantic prior for "Person"/"Location" than for the abbreviated
codes "PERS"/"LOC".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ner.gliner.extract_ner_features import LABELS

ENTITY_OPEN = "[Entity]"
ENTITY_CLOSE = "[\\Entity]"
TYPE_OPEN = "[Type]"
TYPE_CLOSE = "[\\Type]"
CONFIDENCE_OPEN = "[Confidence]"
CONFIDENCE_CLOSE = "[\\Confidence]"

TYPE_DISPLAY_NAME = {
    "PERS": "Person",
    "LOC": "Location",
    "ORG": "Organization",
    "TIME": "Time",
    "PROD": "Product",
}
assert set(TYPE_DISPLAY_NAME) == set(LABELS), f"TYPE_DISPLAY_NAME is missing/stale vs gliner LABELS={LABELS}"


def type_display_name_for(label_map: dict[str, str]) -> dict[str, str]:
    """Same idea as TYPE_DISPLAY_NAME above, from a caller-supplied {TYPE: prompt wording}
    map -- lets phase2_simple's marker-text tags use a different NER source's own tagset
    (e.g. ajmc's PERS/WORK/LOC/OBJECT/DATE/SCOPE) instead of the standard HIPE-2022 5-type
    scheme, without touching the default TYPE_DISPLAY_NAME every other caller still relies
    on. Capitalizes each value's first letter -- label_map's wording is written lowercase
    for GLiNER2's own extraction prompt (e.g. "human creation"), text_b wants the same
    "Person"/"Location"-style capitalization TYPE_DISPLAY_NAME already uses."""
    return {code: wording[0].upper() + wording[1:] for code, wording in label_map.items()}


def type_display_name_from_file(labels_file: str | Path) -> dict[str, str]:
    """Same {TYPE: prompt wording} JSON file gliner/extract_ner_features.py's --labels-file
    reads (see that file's load_label_map() docstring, e.g. src/ner/gliner/labels.json or
    test/ajmc/labels.json) -- reused directly as the display-name source instead of
    hand-writing a second English-word table that could drift out of sync with it."""
    with open(labels_file) as f:
        label_map = json.load(f)
    return type_display_name_for(label_map)
