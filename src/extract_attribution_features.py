"""Extract Phase 1 SS4.6 attribution evidence: per-token importance scores for each NER
candidate decision, via either Integrated Gradients or LIME (--method).

Model: the same GLiNER2 entity extractor used by extract_ner_features.py (see
GLINER_MODEL_ID/LABELS there). Both interpretation methods come from Captum
(https://captum.ai/):

  --method ig (default): LayerIntegratedGradients, hooked on the encoder's embedding
      layer (encoder.embeddings -- GLiNER2's encoder is a DeBERTa-v2 backbone, confirmed
      via its encoder_config). Gradient-based; needs the score computation below to be
      differentiable.
  --method lime: Lime, perturbing whole GLiNER2 words at a time (via a feature_mask
      built from the same word/subword ranges IG uses) and fitting a local linear
      surrogate (plain unregularized SkLearnLinearRegression -- Captum's own default,
      SkLearnLasso(alpha=1.0), shrinks every coefficient to exactly 0.0 at this scale:
      a handful of binary word-presence features fit against small sigmoid-score
      fluctuations, which an L1 penalty of 1.0 dominates completely). Black-box; only
      needs forward passes, no gradients -- useful as a sanity check against the IG
      numbers since the two methods make different assumptions (local linearity in
      embedding space vs. local linearity in word presence/absence space).

For each NER candidate (span, predicted_entity_type) in ner_features.csv, this computes
how much each token in the candidate's sentence contributed to that specific decision's
sigmoid confidence score, relative to a baseline where the sentence's text tokens are
replaced with the tokenizer's pad token (the schema/instruction prompt is kept identical
between baseline and input, so attribution mass reflects the sentence text, not the
fixed prompt). Positive values mean a token pushed the score up; negative, down.

Why this replays GLiNER2's forward pass by hand rather than calling extract_entities():
both Captum methods need a plain function from input_ids to a single scalar score (IG
additionally needs it differentiable, and needs to hook the exact submodule that turns
input_ids into embeddings). GLiNER2's own inference path (batch_extract) runs under
torch.inference_mode() and returns already-thresholded, non-differentiable results, so
this module reconstructs the same score computation GLiNER2._extract_span_result performs
internally (compute_span_rep -> count_pred/count_embed -> einsum -> sigmoid), reading out
one specific (entity_type, span) score instead of every span above threshold. The same
forward function is reused for both methods -- it's agnostic to whether the leading batch
dimension it receives is IG's interpolation steps or LIME's perturbed samples.

GLiNER2's own word-splitting (extractor.processor.word_splitter, the same regex
extract_ner_features.py's chunker uses) is reused so a candidate's
start_token_id/end_token_id (train-data token ids) can be located inside the model's own
tokenization by character-offset overlap -- same convention as
extract_ner_features.py's token_id_range, just run in the opposite direction (every
GLiNER2 word in the sentence is mapped back to the train-data token(s) it overlaps).

Cost: one interpretation run per candidate (IG: n_steps forward+backward passes through
the GLiNER2 encoder, batched into a single call; LIME: n_samples forward-only passes,
plus fitting a linear model per candidate) -- this is by far the most expensive step in
the pipeline. Use --limit for smoke testing and --n-steps/--n-samples to trade off
attribution fidelity for speed.

Output: attribution_features.csv -- document_id, sentence_id, entity_token_ids,
entity_text, predicted_entity, importance_scores. One row per candidate decision.
entity_token_ids is the JSON list of train-data token_ids the model labeled as the
entity (the start_token_id..end_token_id range from ner_features.csv); entity_text is
that span's surface text (same field as ner_features.csv's entity_text), kept here for
readability. Together they disambiguate rows when a sentence holds several candidates,
including several of the same predicted_entity type. importance_scores is a JSON object
mapping every train-data token_id in the candidate's sentence to [token,
importance_score] for that token's contribution to this specific decision, ordered by
importance_score descending (highest first) -- e.g. {"13": ["Poincar6", 0.83], "12":
["Raymond", 0.41], ...}. Candidates whose span can't be represented in the model's span
enumeration (width >= max_width) are skipped and counted in the final "skipped" log
line.

Usage:
    pip install -r requirements.txt
    python preprocessing_data.py
    python extract_ner_features.py
    python extract_attribution_features.py --method ig
    python extract_attribution_features.py --method lime --n-samples 200
    python extract_attribution_features.py --limit 20 --out /tmp/smoke_attribution.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import LayerIntegratedGradients, Lime
from gliner2 import GLiNER2
from tqdm import tqdm

from preprocessing_data import DEFAULT_OUT as DEFAULT_TRAIN_DATA
from extract_ner_features import (
    DEFAULT_OUT as DEFAULT_NER_FEATURES,
    GLINER_MODEL_ID,
    LABELS,
    build_sentence_texts,
)

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_OUT = DATA_DIR / "hipe2020_train_fr_attribution_features.csv"

METHODS = ["ig", "lime"]
N_STEPS = 20  # Integrated Gradients interpolation steps
N_SAMPLES = 200  # LIME perturbed samples


def prepare_sentence(extractor: GLiNER2, sentence_text: str, device: str) -> dict:
    """Collate one sentence against the fixed 5-label entities schema, and pull out
    everything the forward pass needs that does NOT depend on which candidate is being
    attributed: input_ids/attention_mask, the entities schema's special-token positions
    ([P] + one per label, in field order), the text-word gather indices, and each
    GLiNER2 word's character span within sentence_text."""
    schema = extractor.create_schema().entities(LABELS)
    batch = extractor.processor.collate_fn_inference([(sentence_text, schema)])
    batch = batch.to(device)

    schema_tokens = batch.schema_tokens_list[0][0]
    field_names = [schema_tokens[j + 1] for j in range(len(schema_tokens) - 1) if schema_tokens[j] == "[E]"]

    n_words = batch.text_word_counts[0]
    return {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "field_names": field_names,
        "schema_positions": torch.as_tensor(batch.schema_special_indices[0][0], device=device),
        "text_word_indices": batch.text_word_indices[0, :n_words],
        "word_spans": list(zip(batch.start_mappings[0], batch.end_mappings[0])),
    }


def word_subword_ranges(text_word_indices: torch.Tensor, seq_len: int) -> list[tuple[int, int]]:
    """[start, end) subword-position range for each GLiNER2 word, derived from the
    first-subword gather positions -- a word's subwords are contiguous and in order, so
    they run from its own first-subword position up to the next word's."""
    starts = text_word_indices.tolist()
    ends = starts[1:] + [seq_len]
    return list(zip(starts, ends))


def locate_span_in_words(
    word_spans: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int | None, int | None]:
    """GLiNER2 word-index (start, end inclusive) overlapping [char_start, char_end)."""
    overlapping = [wi for wi, (w_start, w_end) in enumerate(word_spans) if w_end > char_start and w_start < char_end]
    if not overlapping:
        return None, None
    return min(overlapping), max(overlapping)


def map_words_to_tokens(
    word_spans: list[tuple[int, int]], token_spans: list[tuple[int, int, int]]
) -> dict[int, list[int]]:
    """token_spans: (token_id, char_start, char_end) for this sentence's train-data
    tokens. Returns token_id -> list of GLiNER2 word indices overlapping that token
    (almost always exactly one, since both use the same word-boundary regex)."""
    return {
        token_id: [wi for wi, (w_start, w_end) in enumerate(word_spans) if w_end > t_start and w_start < t_end]
        for token_id, t_start, t_end in token_spans
    }


def build_forward_func(extractor: GLiNER2, schema_positions, text_word_indices, type_idx: int, target_start: int, target_width: int):
    """fn(input_ids, attention_mask) -> (batch,) tensor reproducing
    GLiNER2._extract_span_result's score computation for one fixed (entity type, span).
    Batch-agnostic: works whether the leading dimension is IG's interpolation steps or
    LIME's perturbed samples -- Captum supplies it either way."""

    def forward_func(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = extractor.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        n_steps = hidden.shape[0]

        token_embs_list = [hidden[i, text_word_indices, :] for i in range(n_steps)]
        span_results = extractor.compute_span_rep_batched(token_embs_list)
        schema_embs = hidden[:, schema_positions, :]

        scores = []
        for i in range(n_steps):
            count_logits = extractor.count_pred(schema_embs[i, 0].unsqueeze(0))
            pred_count = max(int(count_logits.argmax(dim=1).item()), 1)
            struct_proj = extractor.count_embed(schema_embs[i, 1:], pred_count)
            span_scores = torch.sigmoid(
                torch.einsum("lkd,bpd->bplk", span_results[i]["span_rep"], struct_proj)
            )
            scores.append(span_scores[0, type_idx, target_start, target_width])
        return torch.stack(scores)

    return forward_func


def word_baseline_and_ranges(prep: dict, extractor: GLiNER2) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """baseline_input_ids: input_ids with every text-segment subword replaced by the
    tokenizer's pad token (schema/prompt positions left untouched, so baseline == input
    there and perturbing/interpolating them is a no-op either method uses). ranges: the
    [start, end) subword range of each GLiNER2 word, see word_subword_ranges."""
    input_ids = prep["input_ids"]
    seq_len = input_ids.shape[1]
    ranges = word_subword_ranges(prep["text_word_indices"], seq_len)

    is_text = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
    for start, end in ranges:
        is_text[start:end] = True

    pad_id = extractor.processor.tokenizer.pad_token_id
    baseline_input_ids = torch.where(is_text.unsqueeze(0), torch.full_like(input_ids, pad_id), input_ids)
    return baseline_input_ids, ranges


def attribute_candidate_ig(
    extractor: GLiNER2, prep: dict, type_idx: int, target_start: int, target_width: int, n_steps: int
) -> torch.Tensor | None:
    """Runs Integrated Gradients for one (sentence, entity_type, span) decision. Returns
    a 1-D tensor of per-GLiNER2-word importance scores (subword attributions summed per
    word), or None if the span can't be represented in the model's span enumeration."""
    if target_width >= extractor.max_width:
        return None

    baseline_input_ids, ranges = word_baseline_and_ranges(prep, extractor)

    forward_func = build_forward_func(
        extractor, prep["schema_positions"], prep["text_word_indices"], type_idx, target_start, target_width
    )
    lig = LayerIntegratedGradients(forward_func, extractor.encoder.embeddings)

    attributions = lig.attribute(
        inputs=prep["input_ids"],
        baselines=baseline_input_ids,
        additional_forward_args=(prep["attention_mask"],),
        n_steps=n_steps,
    )  # (1, seq_len, hidden)

    subword_scores = attributions.sum(dim=-1).squeeze(0)  # (seq_len,)
    return torch.stack([subword_scores[start:end].sum() for start, end in ranges])


def attribute_candidate_lime(
    extractor: GLiNER2, prep: dict, type_idx: int, target_start: int, target_width: int, n_samples: int
) -> torch.Tensor | None:
    """Runs LIME for one (sentence, entity_type, span) decision, perturbing whole
    GLiNER2 words at a time (every subword of a word shares one feature_mask id, so LIME
    always turns a word fully on or off, never a partial word). Returns a 1-D tensor of
    per-GLiNER2-word importance scores (that word's fitted linear coefficient), or None
    if the span can't be represented in the model's span enumeration."""
    if target_width >= extractor.max_width:
        return None

    input_ids = prep["input_ids"]
    baseline_input_ids, ranges = word_baseline_and_ranges(prep, extractor)
    seq_len = input_ids.shape[1]

    # Non-text (schema/prompt) positions all share one dummy feature id -- baseline ==
    # input there, so whether LIME turns that "feature" on or off never changes the
    # forward_func output, and its fitted coefficient is simply not read out below.
    feature_mask = torch.full((1, seq_len), len(ranges), dtype=torch.long, device=input_ids.device)
    for word_idx, (start, end) in enumerate(ranges):
        feature_mask[0, start:end] = word_idx

    forward_func = build_forward_func(
        extractor, prep["schema_positions"], prep["text_word_indices"], type_idx, target_start, target_width
    )
    # Captum's Lime defaults to SkLearnLasso(alpha=1.0) as the local surrogate model.
    # That L1 penalty is far too strong for this scale of problem (~10-30 binary
    # word-presence features fit against small sigmoid-score fluctuations) -- it
    # shrinks every coefficient to exactly 0.0 rather than partially regularizing them.
    # An unregularized linear fit reports the surrogate model's true local slopes.
    lime = Lime(forward_func, interpretable_model=SkLearnLinearRegression())

    attributions = lime.attribute(
        inputs=input_ids,
        baselines=baseline_input_ids,
        additional_forward_args=(prep["attention_mask"],),
        feature_mask=feature_mask,
        n_samples=n_samples,
    )  # (1, seq_len) -- every position of a word already shares that word's coefficient

    first_positions = torch.as_tensor([start for start, _ in ranges], device=attributions.device)
    return attributions[0, first_positions]


def attribute_candidate(
    method: str,
    extractor: GLiNER2,
    prep: dict,
    type_idx: int,
    target_start: int,
    target_width: int,
    n_steps: int,
    n_samples: int,
) -> torch.Tensor | None:
    if method == "ig":
        return attribute_candidate_ig(extractor, prep, type_idx, target_start, target_width, n_steps)
    return attribute_candidate_lime(extractor, prep, type_idx, target_start, target_width, n_samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA), help="Train data CSV produced by preprocessing_data.py")
    parser.add_argument("--ner-features", default=str(DEFAULT_NER_FEATURES), help="NER features CSV produced by extract_ner_features.py")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Attribution features output CSV path")
    parser.add_argument("--method", choices=METHODS, default="ig", help="Interpretation method")
    parser.add_argument("--n-steps", type=int, default=N_STEPS, help="Integrated Gradients interpolation steps (--method ig)")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help="LIME perturbed samples (--method lime)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N candidates (smoke test)")
    args = parser.parse_args()

    print("=== Step 1: Load train data and NER features ===")
    tokens_df = pd.read_csv(args.train_data, dtype={"TOKEN": str, "MISC": str})
    tokens_df["MISC"] = tokens_df["MISC"].fillna("_")
    ner_df = pd.read_csv(args.ner_features)
    if args.limit is not None:
        ner_df = ner_df.head(args.limit)
    ner_df = ner_df.sort_values(["document_id", "sentence_id"]).reset_index(drop=True)
    print(f"{tokens_df.shape[0]} tokens, {ner_df.shape[0]} candidates")

    print("=== Step 2: Load GLiNER2 model ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {GLINER_MODEL_ID} on {device}")
    extractor = GLiNER2.from_pretrained(GLINER_MODEL_ID, map_location=device)
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad_(False)

    print("=== Step 3: Build sentence text and train-token char spans ===")
    sentences_df, token_spans_df = build_sentence_texts(tokens_df)
    sentence_text_by_key = dict(
        zip(zip(sentences_df["document_id"], sentences_df["sentence_id"]), sentences_df["sentence_text"])
    )
    token_spans_by_sentence: dict[tuple, list[tuple[int, int, int]]] = {
        key: list(zip(g["token_id"], g["start"], g["end"]))
        for key, g in token_spans_df.groupby(["document_id", "sentence_id"])
    }
    token_text_by_doc_token = dict(zip(zip(tokens_df["document_id"], tokens_df["token_id"]), tokens_df["TOKEN"]))

    print(f"=== Step 4: Compute attributions (method={args.method}) ===")
    records = []
    skipped = 0
    current_key = None
    prep = None
    word_to_token: dict[int, list[int]] = {}

    for cand in tqdm(ner_df.to_dict("records"), total=len(ner_df), desc="Computing attribution features", unit="candidate"):
        key = (cand["document_id"], cand["sentence_id"])
        if key != current_key:
            current_key = key
            sentence_text = sentence_text_by_key.get(key)
            spans = token_spans_by_sentence.get(key, [])
            if sentence_text is None or not spans:
                prep = None
                skipped += 1
                continue
            prep = prepare_sentence(extractor, sentence_text, device)
            word_to_token = map_words_to_tokens(prep["word_spans"], spans)

        if prep is None:
            skipped += 1
            continue

        start_tid, end_tid = cand["start_token_id"], cand["end_token_id"]
        predicted_type = cand["predicted_entity_type"]
        if pd.isna(start_tid) or pd.isna(end_tid) or predicted_type not in prep["field_names"]:
            skipped += 1
            continue

        token_spans_lookup = {tid: (s, e) for tid, s, e in token_spans_by_sentence[key]}
        char_start = token_spans_lookup[int(start_tid)][0]
        char_end = token_spans_lookup[int(end_tid)][1]
        target_start, target_end = locate_span_in_words(prep["word_spans"], char_start, char_end)
        if target_start is None:
            skipped += 1
            continue
        target_width = target_end - target_start

        type_idx = prep["field_names"].index(predicted_type)
        word_scores = attribute_candidate(
            args.method, extractor, prep, type_idx, target_start, target_width, args.n_steps, args.n_samples
        )
        if word_scores is None:
            skipped += 1
            continue

        doc_id = cand["document_id"]
        token_importances = []
        for token_id, word_indices in word_to_token.items():
            if not word_indices:
                continue
            importance = sum(word_scores[wi].item() for wi in word_indices) / len(word_indices)
            token_importances.append((token_id, token_text_by_doc_token[(doc_id, token_id)], importance))

        token_importances.sort(key=lambda t: t[2], reverse=True)
        importance_scores = {token_id: [token, importance] for token_id, token, importance in token_importances}

        records.append(
            {
                "document_id": doc_id,
                "sentence_id": cand["sentence_id"],
                "entity_token_ids": json.dumps(list(range(int(start_tid), int(end_tid) + 1))),
                "entity_text": cand["entity_text"],
                "predicted_entity": predicted_type,
                "importance_scores": json.dumps(importance_scores),
            }
        )

    print(f"{len(records)} rows written, {skipped} candidates skipped")

    print("=== Step 5: Save attribution features ===")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Saved attribution features to {out_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
