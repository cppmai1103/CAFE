"""Phase 2's first model (docs/phase2_learned_features.md SS34 "Minimal first model" / SS35 "Final
Phase 2 formula", simple-pooling + MLP-head variant only -- no target-aware attention,
no latent MoE, both later additions per SS22/SS25):

    E_i = TokenEmb_i + lambda_dict * DictFlagEmb_i + lambda_target * TargetFlagEmb_i
    H = FrozenEncoder(E)
    v_text = concat(H_CLS, H_first_target, mean(H_target), H_last_target)
    v_c = concat(v_text, TypeEmb(predicted_type), ScoreMLP([p, logit(p), 1-p]))
    final_logit = MLP(v_c)
    reliability = sigmoid(final_logit)

SS11's important caveat: the encoder is frozen (requires_grad=False on every encoder
parameter) but its forward pass must NOT run under torch.no_grad() -- gradients still
need to flow *through* the frozen computation to reach inputs_embeds, which is where
DictFlagEmb/TargetFlagEmb/lambda_dict/lambda_target actually live. Freezing via
requires_grad=False (rather than no_grad()) is exactly what makes both true at once: the
frozen weights themselves never accumulate .grad (saving memory, and the optimizer only
ever sees trainable_parameters()), but the graph connecting inputs_embeds -> outputs is
still fully differentiable.

Ablations (docs/phase2_learned_features.md SS31: "remove NER score / remove dictionary flags / remove
target flag embeddings"; entity type is the same kind of span-level metadata as NER score,
so it gets the same treatment): use_ner_score/use_type/use_dict_flag/use_target_flag each
default True (the full model). Setting one False drops exactly that component:
    use_dict_flag=False   -- inputs_embeds skips the lambda_dict*DictFlagEmb term
    use_target_flag=False -- inputs_embeds skips the lambda_target*TargetFlagEmb term
        (target_flag_ids is STILL required and still used for pooling -- knowing which
        subwords are the target span is structural, not a side-channel embedding choice;
        this ablation only removes telling the ENCODER about the target via embeddings)
    use_type=False        -- v_c drops type_emb, classifier's input dim shrinks by d_type
    use_ner_score=False   -- v_c drops score_emb, classifier's input dim shrinks by d_score

A second, independent ablation dimension -- score_features -- doesn't remove the NER-score
component, it simplifies what ScoreMLP sees (only meaningful when use_ner_score=True):
    "full" (default)  -- ScoreMLP input is [p, logit(p), 1-p], a Linear(3, d_score) first layer
    "p_logit_only"     -- ScoreMLP input is [p, logit(p)], a Linear(2, d_score) first layer
    "logit_only"       -- ScoreMLP input is [logit(p)] only, Linear(1, d_score)
    "p_only"           -- ScoreMLP input is [p] only, Linear(1, d_score)
    "binned"           -- no ScoreMLP at all: ner_score is bucketed into N_SCORE_BINS (10)
        equal-width bins (0-0.1, 0.1-0.2, ..., 0.9-1.0, right edge inclusive for the last
        bin only -- floor(p * 10) clamped to [0, 9]) and looked up in a plain
        Embedding(N_SCORE_BINS, d_score) table (ScoreEmb), same output width as every
        other score_features choice so it drops into v_c unchanged. Trades ScoreMLP's
        smooth, shared-slope function of p for one independently-learned vector per decile
        -- tests whether the model actually needs fine-grained continuous confidence, or
        whether "which confidence decile" is already enough signal.
p and 1-p are linearly redundant with each other for a first Linear layer (1-p = -p + 1,
a fixed affine reparameterization), so "full"'s only genuinely new information over
"p_only" is logit(p) -- "p_logit_only" drops just the redundant 1-p term and should, in
theory, carry exactly the same information as "full" (nothing a linear layer can't
already reconstruct from [p, logit(p)] alone); comparing it against "full" empirically
checks whether that redundancy is actually harmless in practice, or whether having 1-p
spelled out explicitly makes optimization measurably easier despite carrying no new
information. "p_only" is included separately to see whether the raw probability alone
forces the (small, 32-unit) first layer to relearn that nonlinear near-0/near-1 stretch
from data instead of getting it for free from logit(p).

variant_name() below turns whichever combination is active into the checkpoint/CSV naming
convention train.py/evaluate.py use by default -- e.g. "camembert_mlp_ner_logit_only" or
"camembert_mlp_ner_p_only" for the two score_features ablations, composable with the
use_*=False ones (e.g. "camembert_mlp_without_type_ner_p_only").

Usage (smoke test -- builds a small real batch via dataset.py and runs one forward pass):
    python src/phase2/base/model.py
    python src/phase2/base/model.py --batch-size 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase2.base.vocab import DICT_FLAG_VOCAB, ENTITY_TYPE_VOCAB, TARGET_FLAG_VOCAB

DEFAULT_ENCODER_NAME = "bert-base-multilingual-cased"
DEFAULT_D_TYPE = 32
DEFAULT_D_SCORE = 32
DEFAULT_HEAD_HIDDEN = 512
DEFAULT_HEAD_DROPOUT = 0.1
DEFAULT_SCORE_FEATURES = "full"
SCORE_FEATURES_CHOICES = ("full", "p_logit_only", "logit_only", "p_only", "binned")
N_SCORE_BINS = 10


def score_input_dim_of(score_features: str) -> int:
    """ScoreMLP's first-layer input width for a given score_features choice -- shared by
    __init__ (building the layer) and parameter_breakdown (reporting its shape), so the
    two can never drift apart. Not called for "binned" -- that variant has no ScoreMLP,
    see score_bin_of / N_SCORE_BINS instead."""
    return {"full": 3, "p_logit_only": 2, "logit_only": 1, "p_only": 1}[score_features]


def score_bin_of(ner_score: torch.Tensor) -> torch.Tensor:
    """ner_score in [0, 1] -> integer bin id in [0, N_SCORE_BINS) for score_features=
    "binned"'s ScoreEmb lookup: N_SCORE_BINS equal-width bins (0-0.1, 0.1-0.2, ...,
    0.9-1.0). floor(p * N_SCORE_BINS) puts every bin's left edge in that bin and its right
    edge in the next one, EXCEPT p==1.0 (floor(10)=10, out of range) -- clamp folds that
    single point into the last bin, so 0.9-1.0 is the only right-inclusive bin."""
    return (ner_score * N_SCORE_BINS).long().clamp(0, N_SCORE_BINS - 1)


KNOWN_ENCODER_SHORT_NAMES = {
    "bert-base-multilingual-cased": "mbert",
}


def encoder_short_name(encoder_name: str) -> str:
    """HF model id -> filename-safe short token, e.g. 'camembert-base' -> 'camembert',
    'xlm-roberta-base' -> 'xlm-roberta'. Strips any 'org/' prefix first, then checks
    KNOWN_ENCODER_SHORT_NAMES for a hand-picked name (needed for ids with no clean
    '-base' suffix to strip, e.g. 'bert-base-multilingual-cased' -> 'mbert' -- without
    this it would keep the full, unwieldy HF id in every checkpoint/variant name), and
    finally falls back to stripping a trailing '-base' if present, else keeping the id
    as-is."""
    name = encoder_name.split("/")[-1]
    if name in KNOWN_ENCODER_SHORT_NAMES:
        return KNOWN_ENCODER_SHORT_NAMES[name]
    if name.endswith("-base"):
        name = name[: -len("-base")]
    return name


def variant_name(
    encoder_name: str = DEFAULT_ENCODER_NAME,
    use_ner_score: bool = True, use_type: bool = True, use_dict_flag: bool = True, use_target_flag: bool = True,
    score_features: str = DEFAULT_SCORE_FEATURES,
) -> str:
    """<encoder>_mlp naming convention for ablations -- the full model (all four use_*
    True, score_features="full") is just "<encoder>_mlp" (e.g. "camembert_mlp",
    "xlm-roberta_mlp" -- see encoder_short_name); each disabled component gets appended in
    a fixed order, e.g. "camembert_mlp_without_ner_score" or
    "camembert_mlp_without_ner_score_type". score_features (only meaningful when
    use_ner_score=True) adds its own suffix, e.g. "camembert_mlp_ner_logit_only" or,
    composed with a use_*=False ablation, "camembert_mlp_without_type_ner_p_only"."""
    base = f"{encoder_short_name(encoder_name)}_mlp"
    ablated = []
    if not use_ner_score:
        ablated.append("ner_score")
    if not use_type:
        ablated.append("type")
    if not use_dict_flag:
        ablated.append("dict_flag")
    if not use_target_flag:
        ablated.append("target_flag")
    name = base + "_without_" + "_".join(ablated) if ablated else base
    if use_ner_score and score_features != "full":
        name += f"_ner_{score_features}"
    return name


class Phase2Model(nn.Module):
    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER_NAME,
        d_type: int = DEFAULT_D_TYPE,
        d_score: int = DEFAULT_D_SCORE,
        head_hidden: int = DEFAULT_HEAD_HIDDEN,
        head_dropout: float = DEFAULT_HEAD_DROPOUT,
        use_ner_score: bool = True,
        use_type: bool = True,
        use_dict_flag: bool = True,
        use_target_flag: bool = True,
        score_features: str = DEFAULT_SCORE_FEATURES,
        entity_type_vocab: dict[str, int] | None = None,
    ):
        super().__init__()
        if score_features not in SCORE_FEATURES_CHOICES:
            raise ValueError(f"score_features must be one of {SCORE_FEATURES_CHOICES}, got {score_features!r}")
        self.encoder_name = encoder_name
        self.d_type = d_type
        self.d_score = d_score
        self.head_hidden = head_hidden
        self.head_dropout = head_dropout
        self.use_ner_score = use_ner_score
        self.use_type = use_type
        self.use_dict_flag = use_dict_flag
        self.use_target_flag = use_target_flag
        self.score_features = score_features
        # None (the default, and what every existing checkpoint's saved config()
        # implicitly means, since this key didn't exist before) -> the project's
        # standard HIPE-2022 5-type scheme. A caller-supplied vocab (see phase2/train.py's
        # --labels) lets this model be trained on a different NER source's own tagset --
        # e.g. ajmc's PERS/WORK/LOC/OBJECT/DATE/SCOPE, which doesn't map onto the
        # standard 5 types.
        self.entity_type_vocab = entity_type_vocab if entity_type_vocab is not None else ENTITY_TYPE_VOCAB

        # dtype=torch.float32 is explicit, not a no-op default: some checkpoints (e.g.
        # microsoft/mdeberta-v3-base) ship fp16 weights and transformers loads them
        # natively as fp16 unless told otherwise, which then crashes inside the frozen
        # encoder's own LayerNorm once inputs_embeds gets promoted to fp32 by adding the
        # (fp32-initialized) side embeddings -- "expected scalar type Float but found Half".
        self.encoder = AutoModel.from_pretrained(encoder_name, dtype=torch.float32)
        for p in self.encoder.parameters():
            p.requires_grad = False
        hidden_size = self.encoder.config.hidden_size

        # SS12: new, trainable side embeddings -- small init scale (lambda_* start at
        # 0.1) so they don't strongly disrupt the frozen pretrained representation early
        # in training. Only created if their ablation flag is on.
        if self.use_dict_flag:
            self.dict_flag_embedding = nn.Embedding(len(DICT_FLAG_VOCAB), hidden_size)
            self.lambda_dict = nn.Parameter(torch.tensor(0.1))
        if self.use_target_flag:
            self.target_flag_embedding = nn.Embedding(len(TARGET_FLAG_VOCAB), hidden_size)
            self.lambda_target = nn.Parameter(torch.tensor(0.1))

        # SS18/19: span-level metadata.
        if self.use_type:
            self.type_embedding = nn.Embedding(len(self.entity_type_vocab), d_type)
        if self.use_ner_score:
            if self.score_features == "binned":
                self.score_embedding = nn.Embedding(N_SCORE_BINS, d_score)
            else:
                score_input_dim = score_input_dim_of(self.score_features)
                self.score_mlp = nn.Sequential(
                    nn.Linear(score_input_dim, d_score),
                    nn.ReLU(),
                    nn.Linear(d_score, d_score),
                )

        # SS21: v_c = concat(h_cls, h_first, h_span, h_last, [type_emb], [score_emb]) --
        # classifier's input width shrinks by d_type/d_score for whichever is disabled.
        v_dim = 4 * hidden_size + (d_type if self.use_type else 0) + (d_score if self.use_ner_score else 0)
        self.classifier = nn.Sequential(
            nn.Linear(v_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def config(self) -> dict:
        """Hyperparameters needed to reconstruct this architecture -- see
        save_checkpoint/load_model below. entity_type_vocab is always included (even
        when it's just the standard default) so the checkpoint is fully self-describing
        -- load_model() never has to guess which vocab a given checkpoint was trained
        with."""
        return {
            "encoder_name": self.encoder_name,
            "d_type": self.d_type,
            "d_score": self.d_score,
            "head_hidden": self.head_hidden,
            "head_dropout": self.head_dropout,
            "use_ner_score": self.use_ner_score,
            "use_type": self.use_type,
            "use_dict_flag": self.use_dict_flag,
            "use_target_flag": self.use_target_flag,
            "score_features": self.score_features,
            "entity_type_vocab": self.entity_type_vocab,
        }

    def variant_name(self) -> str:
        return variant_name(self.encoder_name, self.use_ner_score, self.use_type, self.use_dict_flag, self.use_target_flag, self.score_features)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_breakdown(self) -> list[tuple[str, str, bool, int]]:
        """(component_name, shape, is_trainable, param_count) for every named piece of
        the model that's actually present -- ablated-away components (use_X=False)
        simply don't have rows, since they don't exist as submodules at all. See
        train.py's Step 2 print / this module's main() for how it's displayed."""

        def n(mod) -> int:
            return sum(p.numel() for p in mod.parameters())

        hidden_size = self.encoder.config.hidden_size
        rows: list[tuple[str, str, bool, int]] = []
        if self.use_dict_flag:
            rows.append(("dict_flag_embedding (DictFlagEmb)", f"Embedding{tuple(self.dict_flag_embedding.weight.shape)}", True, n(self.dict_flag_embedding)))
            rows.append(("lambda_dict", "scalar", True, self.lambda_dict.numel()))
        if self.use_target_flag:
            rows.append(("target_flag_embedding (TargetFlagEmb)", f"Embedding{tuple(self.target_flag_embedding.weight.shape)}", True, n(self.target_flag_embedding)))
            rows.append(("lambda_target", "scalar", True, self.lambda_target.numel()))
        if self.use_type:
            rows.append(("type_embedding (TypeEmb)", f"Embedding{tuple(self.type_embedding.weight.shape)}", True, n(self.type_embedding)))
        if self.use_ner_score:
            if self.score_features == "binned":
                rows.append(("score_embedding (ScoreEmb, binned into 10)", f"Embedding{tuple(self.score_embedding.weight.shape)}", True, n(self.score_embedding)))
            else:
                score_input_dim = score_input_dim_of(self.score_features)
                rows.append((f"score_mlp (ScoreMLP, score_features={self.score_features})", f"Linear({score_input_dim},{self.d_score})->ReLU->Linear({self.d_score},{self.d_score})", True, n(self.score_mlp)))
        v_dim = 4 * hidden_size + (self.d_type if self.use_type else 0) + (self.d_score if self.use_ner_score else 0)
        rows.append(("classifier (MLP head)", f"Linear({v_dim},{self.head_hidden})->ReLU->Dropout->Linear({self.head_hidden},1)", True, n(self.classifier)))
        rows.append((f"encoder (frozen {self.encoder_name}: token/position/token-type text embeddings + transformer layers)", f"hidden_size={hidden_size}", False, n(self.encoder)))
        return rows

    def forward(
        self,
        input_ids: torch.Tensor, dict_flag_ids: torch.Tensor, target_flag_ids: torch.Tensor,
        attention_mask: torch.Tensor, entity_type_id: torch.Tensor, ner_score: torch.Tensor,
    ) -> torch.Tensor:
        """Returns final_logit [B] (pre-sigmoid -- pair with BCEWithLogitsLoss)."""
        # SS14: E_i = TokenEmb_i + lambda_dict * DictFlagEmb_i + lambda_target * TargetFlagEmb_i
        # (the last two terms only if their ablation flag is on).
        inputs_embeds = self.encoder.get_input_embeddings()(input_ids)
        if self.use_dict_flag:
            inputs_embeds = inputs_embeds + self.lambda_dict * self.dict_flag_embedding(dict_flag_ids)
        if self.use_target_flag:
            inputs_embeds = inputs_embeds + self.lambda_target * self.target_flag_embedding(target_flag_ids)

        # SS11: no torch.no_grad() here -- see module docstring.
        outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        H = outputs.last_hidden_state  # [B, L, hidden]

        h_cls = H[:, 0, :]

        # SS17: mean/first/last pooling over INSIDE_TARGET positions. target_flag_ids is
        # always used here regardless of use_target_flag -- locating the target span is
        # structural, not the side-channel embedding this ablation flag controls.
        inside_mask = (target_flag_ids == TARGET_FLAG_VOCAB["INSIDE_TARGET"]).float()
        span_sum = (H * inside_mask.unsqueeze(-1)).sum(dim=1)
        span_len = inside_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_span = span_sum / span_len

        batch_idx = torch.arange(H.size(0), device=H.device)
        first_pos = inside_mask.argmax(dim=1)  # first index where inside_mask==1 (argmax picks the first max on ties)
        last_pos = H.size(1) - 1 - inside_mask.flip(dims=[1]).argmax(dim=1)
        h_first = H[batch_idx, first_pos, :]
        h_last = H[batch_idx, last_pos, :]

        v_parts = [h_cls, h_first, h_span, h_last]

        # SS18/19: type + NER-score metadata, each only if its ablation flag is on.
        if self.use_type:
            v_parts.append(self.type_embedding(entity_type_id))
        if self.use_ner_score:
            if self.score_features == "binned":
                v_parts.append(self.score_embedding(score_bin_of(ner_score)))
            else:
                p = ner_score.clamp(1e-5, 1 - 1e-5)
                logit_p = torch.log(p / (1 - p))
                if self.score_features == "logit_only":
                    score_input = logit_p.unsqueeze(-1)
                elif self.score_features == "p_only":
                    score_input = p.unsqueeze(-1)
                elif self.score_features == "p_logit_only":
                    score_input = torch.stack([p, logit_p], dim=-1)
                else:  # "full"
                    score_input = torch.stack([p, logit_p, 1 - p], dim=-1)
                v_parts.append(self.score_mlp(score_input))

        v_c = torch.cat(v_parts, dim=-1)
        return self.classifier(v_c).squeeze(-1)


def save_checkpoint(model: Phase2Model, path: str | Path) -> None:
    """Only saves the trainable parameters -- the frozen encoder's ~110M weights are
    exactly reproducible from config()'s encoder_name via AutoModel.from_pretrained, so
    including them here would bloat the checkpoint ~70x for no benefit (449MB -> ~6MB)."""
    trainable_state = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
    torch.save({"state_dict": trainable_state, "config": model.config()}, path)


def load_model(path: str | Path, device: str = "cpu") -> Phase2Model:
    checkpoint = torch.load(path, map_location=device)
    model = Phase2Model(**checkpoint["config"])  # rebuilds the frozen encoder from encoder_name
    result = model.load_state_dict(checkpoint["state_dict"], strict=False)
    assert not result.unexpected_keys, f"unexpected keys in checkpoint: {result.unexpected_keys}"
    assert all(k.startswith("encoder.") for k in result.missing_keys), f"missing non-encoder keys: {result.missing_keys}"
    model.to(device)
    model.eval()
    return model


def print_parameter_breakdown(model: Phase2Model) -> None:
    """Prints model.parameter_breakdown() as two grouped tables -- trainable (new
    components, small) then frozen (the pretrained encoder, large) -- with each
    component's shape formula alongside its parameter count."""
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
    parser.add_argument("--no-ner-score", action="store_true", help="Ablation: drop the NER-score embedding")
    parser.add_argument("--no-type", action="store_true", help="Ablation: drop the entity-type embedding")
    parser.add_argument("--no-dict-flag", action="store_true", help="Ablation: drop the dictionary-flag side embedding")
    parser.add_argument("--no-target-flag", action="store_true", help="Ablation: drop the target-flag side embedding")
    parser.add_argument("--score-features", default=DEFAULT_SCORE_FEATURES, choices=SCORE_FEATURES_CHOICES, help="Ablation: simplify ScoreMLP's input (only matters if NER score isn't dropped)")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from phase2.base.build_candidate_windows import DEFAULT_OUT as DEFAULT_WINDOWS
    from phase2.base.dataset import Phase2WindowDataset

    print("=== Step 1: Build a small real batch (see dataset.py) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    dataset = Phase2WindowDataset(DEFAULT_WINDOWS, tokenizer, split="val")
    batch = dataset.collate([dataset[i] for i in range(args.batch_size)])
    print(f"batch input_ids shape: {tuple(batch['input_ids'].shape)}")

    print("=== Step 2: Build model and run one forward pass ===")
    model = Phase2Model(
        encoder_name=args.encoder_name,
        use_ner_score=not args.no_ner_score, use_type=not args.no_type,
        use_dict_flag=not args.no_dict_flag, use_target_flag=not args.no_target_flag,
        score_features=args.score_features,
    )
    print(f"variant: {model.variant_name()}")
    print_parameter_breakdown(model)

    logits = model(
        batch["input_ids"], batch["dict_flag_ids"], batch["target_flag_ids"],
        batch["attention_mask"], batch["entity_type_id"], batch["ner_score"],
    )
    print(f"final_logit shape: {tuple(logits.shape)}, values: {logits.tolist()}")
    print(f"reliability_score (sigmoid): {torch.sigmoid(logits).tolist()}")

    print("=== Step 3: Verify gradients reach the frozen encoder's inputs (SS11) ===")
    loss = logits.sum()
    loss.backward()
    if model.use_dict_flag:
        print(f"lambda_dict.grad: {model.lambda_dict.grad.item():.6f}")
        print(f"dict_flag_embedding.weight.grad is not None: {model.dict_flag_embedding.weight.grad is not None}")
    if model.use_target_flag:
        print(f"lambda_target.grad: {model.lambda_target.grad.item():.6f}")
    print(f"encoder param .grad is None (frozen, correctly not accumulated): {next(model.encoder.parameters()).grad is None}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
