"""phase2_simple: a "prompt"-style alternative to phase2/model.py's Phase2Model, built to
compare against it directly (same candidates, same frozen-encoder philosophy, same
document-level split) with a much smaller/simpler trainable surface.

phase2/model.py injects the candidate's metadata (dictionary-quality flag, target-span
location, entity type, NER score) as separate TRAINABLE EMBEDDINGS added into
inputs_embeds, alongside a hand-built classifier that concatenates pooled text with a
TypeEmb lookup and a ScoreMLP([p, logit(p), 1-p]) branch. This module instead writes the
type and confidence directly into the token sequence as TEXT, using the frozen encoder's
own pretrained vocabulary (no new embedding tables, no resize_token_embeddings):

    text_a = window context (unmarked, target embedded naturally)
    text_b = [Entity] <span words> [\\Entity] [Type] <type word> [\\Type]
             [Confidence] <confidence, e.g. "0.98"> [\\Confidence]
    input  = tokenizer(text_a, text_b)  -- see tokenize_windows.py for the exact scheme
    H = FrozenEncoder(input)
    v_c = concat(h_cls, h_span, h_sep, h_entity, h_type, h_confidence)
    final_logit = MLP(v_c)
    reliability = sigmoid(final_logit)

where h_span is the mean over the span word(s)' subwords (text_b's copy of the target,
not text_a's), and h_sep/h_entity are each a SINGLE token's hidden state -- the first
subword of the separator / "[Entity]" tag. This is the standard "marker token as
feature" trick: a bidirectional encoder's self-attention lets a tag token's hidden state
summarize whatever it introduces (the tag itself plus, causally-unconstrained, the value
that follows it). h_entity is deliberately kept this way rather than mean-pooled like
h_span -- content-inside-its-own-brackets for [Entity] IS the span text, so that would
just duplicate h_span as a second, redundant input to the classifier.

type_confidence_pool controls how h_type/h_confidence are computed (two modes, see
variant_name()):
    "one" (default)  -- same marker-token trick as h_entity: the first subword of the
        "[Type]"/"[Confidence]" tag itself, not the value.
    "average"         -- mean-pool over the VALUE's own subwords instead ("Location",
        "0.87"), no tag/bracket tokens included. Empirically the tags are longer than
        their values in subwords ("[Type]"+"[\\Type]" = 7 subwords vs. "Location" = 1;
        "[Confidence]"+"[\\Confidence]" = 11 subwords vs. "0.87" = 2-3), so bracket-
        inclusive pooling would dilute the value with mostly-identical-looking bracket
        punctuation -- "average" mode therefore never includes the brackets, only the
        value.

The ENTIRE frozen encoder stays frozen exactly like phase2 (requires_grad=False, forward
pass NOT under torch.no_grad() -- except here there's no reason gradients need to flow
*through* the encoder at all, since there are no trainable embeddings feeding into it; see
forward() below, it's simple inference). The classifier MLP head is the model's only
trainable component -- no side embeddings, no lambda_dict/lambda_target, no per-type
embedding table, no ScoreMLP. This is what "simple" refers to in this folder's name.

Usage (smoke test -- builds a small real batch via dataset.py and runs one forward pass):
    python src/phase2/simple/model.py
    python src/phase2/simple/model.py --batch-size 4 --encoder-name xlm-roberta-base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DEFAULT_ENCODER_NAME = "bert-base-multilingual-cased"
DEFAULT_HEAD_HIDDEN = 512
DEFAULT_HEAD_DROPOUT = 0.1
DEFAULT_TYPE_CONFIDENCE_POOL = "one"
TYPE_CONFIDENCE_POOL_CHOICES = ("one", "average")


KNOWN_ENCODER_SHORT_NAMES = {
    "bert-base-multilingual-cased": "mbert",
}


def encoder_short_name(encoder_name: str) -> str:
    """Same convention as phase2/model.py's encoder_short_name -- 'camembert-base' ->
    'camembert', 'xlm-roberta-base' -> 'xlm-roberta', and known ids with no clean '-base'
    suffix (KNOWN_ENCODER_SHORT_NAMES) get a hand-picked name instead, e.g.
    'bert-base-multilingual-cased' -> 'mbert'."""
    name = encoder_name.split("/")[-1]
    if name in KNOWN_ENCODER_SHORT_NAMES:
        return KNOWN_ENCODER_SHORT_NAMES[name]
    if name.endswith("-base"):
        name = name[: -len("-base")]
    return name


def variant_name(encoder_name: str = DEFAULT_ENCODER_NAME, type_confidence_pool: str = DEFAULT_TYPE_CONFIDENCE_POOL) -> str:
    """<encoder>_simple_mlp -- e.g. 'camembert_simple_mlp', to compare directly against
    phase2/model.py's '<encoder>_mlp' for the same encoder. type_confidence_pool="one"
    (the original/default mode) adds no suffix; "average" adds "_average", e.g.
    'camembert_simple_mlp_average'."""
    base = f"{encoder_short_name(encoder_name)}_simple_mlp"
    return base if type_confidence_pool == "one" else f"{base}_{type_confidence_pool}"


class Phase2SimpleModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER_NAME,
        head_hidden: int = DEFAULT_HEAD_HIDDEN,
        head_dropout: float = DEFAULT_HEAD_DROPOUT,
        type_confidence_pool: str = DEFAULT_TYPE_CONFIDENCE_POOL,
    ):
        super().__init__()
        if type_confidence_pool not in TYPE_CONFIDENCE_POOL_CHOICES:
            raise ValueError(f"type_confidence_pool must be one of {TYPE_CONFIDENCE_POOL_CHOICES}, got {type_confidence_pool!r}")
        self.encoder_name = encoder_name
        self.head_hidden = head_hidden
        self.head_dropout = head_dropout
        self.type_confidence_pool = type_confidence_pool

        # dtype=torch.float32 explicit -- some checkpoints (e.g. microsoft/mdeberta-v3-base)
        # ship fp16 weights and load natively as fp16 otherwise (see phase2/model.py's own
        # fix for the same issue).
        self.encoder = AutoModel.from_pretrained(encoder_name, dtype=torch.float32)
        for p in self.encoder.parameters():
            p.requires_grad = False
        hidden_size = self.encoder.config.hidden_size

        # Only trainable component: v_c = concat(h_cls, h_span, h_sep, h_entity, h_type,
        # h_confidence), 6 pooled hidden_size-wide vectors.
        self.classifier = nn.Sequential(
            nn.Linear(6 * hidden_size, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def config(self) -> dict:
        return {
            "encoder_name": self.encoder_name,
            "head_hidden": self.head_hidden,
            "head_dropout": self.head_dropout,
            "type_confidence_pool": self.type_confidence_pool,
        }

    def variant_name(self) -> str:
        return variant_name(self.encoder_name, self.type_confidence_pool)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_breakdown(self) -> list[tuple[str, str, bool, int]]:
        def n(mod) -> int:
            return sum(p.numel() for p in mod.parameters())

        hidden_size = self.encoder.config.hidden_size
        v_dim = 6 * hidden_size
        return [
            ("classifier (MLP head)", f"Linear({v_dim},{self.head_hidden})->ReLU->Dropout->Linear({self.head_hidden},1)", True, n(self.classifier)),
            (f"encoder (frozen {self.encoder_name}: full transformer, no side embeddings)", f"hidden_size={hidden_size}", False, n(self.encoder)),
        ]

    @staticmethod
    def _mean_pool(H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over positions where mask==1 (same pattern for h_span/h_type/h_confidence
        in "average" mode -- mask is 0/1 float [B, L])."""
        summed = (H * mask.unsqueeze(-1)).sum(dim=1)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return summed / count

    def forward(
        self,
        input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor,
        sep_pos: torch.Tensor, entity_pos: torch.Tensor, type_pos: torch.Tensor, confidence_pos: torch.Tensor,
        span_mask: torch.Tensor, type_value_mask: torch.Tensor, confidence_value_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns final_logit [B] (pre-sigmoid -- pair with BCEWithLogitsLoss)."""
        outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state  # [B, L, hidden]

        batch_idx = torch.arange(H.size(0), device=H.device)
        h_cls = H[:, 0, :]
        h_sep = H[batch_idx, sep_pos, :]
        h_entity = H[batch_idx, entity_pos, :]
        h_span = self._mean_pool(H, span_mask)

        if self.type_confidence_pool == "average":
            h_type = self._mean_pool(H, type_value_mask)
            h_confidence = self._mean_pool(H, confidence_value_mask)
        else:  # "one"
            h_type = H[batch_idx, type_pos, :]
            h_confidence = H[batch_idx, confidence_pos, :]

        v_c = torch.cat([h_cls, h_span, h_sep, h_entity, h_type, h_confidence], dim=-1)
        return self.classifier(v_c).squeeze(-1)


def save_checkpoint(model: Phase2SimpleModel, path: str | Path) -> None:
    """Only the classifier head is trainable -- the frozen encoder is exactly
    reproducible from config()'s encoder_name, so it's excluded from the checkpoint."""
    trainable_state = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
    torch.save({"state_dict": trainable_state, "config": model.config()}, path)


def load_model(path: str | Path, device: str = "cpu") -> Phase2SimpleModel:
    checkpoint = torch.load(path, map_location=device)
    model = Phase2SimpleModel(**checkpoint["config"])
    result = model.load_state_dict(checkpoint["state_dict"], strict=False)
    assert not result.unexpected_keys, f"unexpected keys in checkpoint: {result.unexpected_keys}"
    assert all(k.startswith("encoder.") for k in result.missing_keys), f"missing non-encoder keys: {result.missing_keys}"
    model.to(device)
    model.eval()
    return model


def print_parameter_breakdown(model: Phase2SimpleModel) -> None:
    rows = model.parameter_breakdown()
    trainable_rows = [r for r in rows if r[2]]
    frozen_rows = [r for r in rows if not r[2]]

    print("Trainable parameters:")
    for name, shape, _, count in trainable_rows:
        print(f"  {name:<55} {shape:<45} {count:>12,}")
    print(f"  {'TOTAL TRAINABLE':<55} {'':<45} {sum(c for *_, c in trainable_rows):>12,}")

    print("Frozen:")
    for name, shape, _, count in frozen_rows:
        print(f"  {name:<55} {shape:<45} {count:>12,}")
    print(f"  {'TOTAL FROZEN':<55} {'':<45} {sum(c for *_, c in frozen_rows):>12,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER_NAME)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--type-confidence-pool", default=DEFAULT_TYPE_CONFIDENCE_POOL, choices=TYPE_CONFIDENCE_POOL_CHOICES)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from phase2.base.build_candidate_windows import DEFAULT_OUT as DEFAULT_WINDOWS
    from phase2.simple.dataset import Phase2SimpleWindowDataset

    print("=== Step 1: Build a small real batch (see dataset.py) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    dataset = Phase2SimpleWindowDataset(DEFAULT_WINDOWS, tokenizer, split="val")
    batch = dataset.collate([dataset[i] for i in range(args.batch_size)])
    print(f"batch input_ids shape: {tuple(batch['input_ids'].shape)}")

    print("=== Step 2: Build model and run one forward pass ===")
    model = Phase2SimpleModel(encoder_name=args.encoder_name, type_confidence_pool=args.type_confidence_pool)
    print(f"variant: {model.variant_name()}")
    print_parameter_breakdown(model)

    logits = model(
        batch["input_ids"], batch["token_type_ids"], batch["attention_mask"],
        batch["sep_pos"], batch["entity_pos"], batch["type_pos"], batch["confidence_pos"],
        batch["span_mask"], batch["type_value_mask"], batch["confidence_value_mask"],
    )
    print(f"final_logit shape: {tuple(logits.shape)}, values: {logits.tolist()}")
    print(f"reliability_score (sigmoid): {torch.sigmoid(logits).tolist()}")

    print("=== Step 3: Verify gradients reach the classifier only (encoder stays frozen) ===")
    loss = logits.sum()
    loss.backward()
    print(f"classifier grad is not None: {model.classifier[0].weight.grad is not None}")
    print(f"encoder param .grad is None (frozen, correctly not accumulated): {next(model.encoder.parameters()).grad is None}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
