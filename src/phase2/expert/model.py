"""phase2_expert: docs/phase2_learned_features.md SS25 "Latent MoE head" / SS26 "MoE pseudo-code" --
same backbone as phase2/model.py (frozen encoder + DictFlagEmb/TargetFlagEmb side
embeddings + simple h_cls/h_first/h_span/h_last pooling + TypeEmb + ScoreMLP -> v_c, SEE
phase2/model.py's own module docstring for that half, unchanged here), but v_c's final
head is replaced: instead of one MLP classifier, K "latent" experts each score v_c
independently, and a gate softmax-weights their outputs:

    z_k = Expert_k(v_c)              for k in 1..K, each a small MLP -> 1 logit
    alpha = softmax(Gate(v_c))       -- v_c also decides how much to trust each expert
    final_logit = sum_k alpha_k * z_k
    reliability = sigmoid(final_logit)

SS25's important caveat, kept verbatim: "For pure Phase 2, experts are latent and
unnamed. Do not manually define TEXT expert, DICT expert, NER-score expert." -- all K
experts receive the exact same v_c and are architecturally identical at init; any
specialization has to emerge from training, not from being wired to different inputs.

Unlike phase2/model.py, there are no use_ner_score/use_type/use_dict_flag/use_target_flag/
score_features ablation flags here -- this is the one full-featured backbone (matching
phase2's default "camembert_mlp" configuration) with the MoE head swapped in, not a
second ablation matrix. variant_name() -> "<encoder>_experts", e.g. "camembert_experts",
to compare directly against phase2/model.py's "<encoder>_mlp" for the same encoder (see
compare.py in this folder).

Usage (smoke test -- builds a small real batch via phase2.dataset and runs one forward pass):
    python src/phase2/expert/model.py
    python src/phase2/expert/model.py --batch-size 4 --num-experts 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase2.base.model import encoder_short_name
from phase2.base.vocab import DICT_FLAG_VOCAB, ENTITY_TYPE_VOCAB, TARGET_FLAG_VOCAB

DEFAULT_ENCODER_NAME = "bert-base-multilingual-cased"
DEFAULT_D_TYPE = 32
DEFAULT_D_SCORE = 32
DEFAULT_NUM_EXPERTS = 4
DEFAULT_EXPERT_HIDDEN = 64
DEFAULT_GATE_HIDDEN = 128
DEFAULT_EXPERT_DROPOUT = 0.1


def variant_name(encoder_name: str = DEFAULT_ENCODER_NAME) -> str:
    """<encoder>_experts -- e.g. 'camembert_experts', to compare directly against
    phase2/model.py's '<encoder>_mlp' for the same encoder."""
    return f"{encoder_short_name(encoder_name)}_experts"


class Phase2ExpertModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER_NAME,
        d_type: int = DEFAULT_D_TYPE,
        d_score: int = DEFAULT_D_SCORE,
        num_experts: int = DEFAULT_NUM_EXPERTS,
        expert_hidden: int = DEFAULT_EXPERT_HIDDEN,
        gate_hidden: int = DEFAULT_GATE_HIDDEN,
        expert_dropout: float = DEFAULT_EXPERT_DROPOUT,
        entity_type_vocab: dict[str, int] | None = None,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.d_type = d_type
        self.d_score = d_score
        self.num_experts = num_experts
        self.expert_hidden = expert_hidden
        self.gate_hidden = gate_hidden
        self.expert_dropout = expert_dropout
        # None (default) -> standard HIPE-2022 5-type vocab, same convention as
        # phase2/model.py's Phase2Model -- see that file's own docstring for why.
        self.entity_type_vocab = entity_type_vocab if entity_type_vocab is not None else ENTITY_TYPE_VOCAB

        # dtype=torch.float32 explicit -- some checkpoints (e.g. microsoft/mdeberta-v3-base)
        # ship fp16 weights and load natively as fp16 otherwise (see phase2/model.py's own
        # fix for the same issue).
        self.encoder = AutoModel.from_pretrained(encoder_name, dtype=torch.float32)
        for p in self.encoder.parameters():
            p.requires_grad = False
        hidden_size = self.encoder.config.hidden_size

        # SS12: new, trainable side embeddings -- identical to phase2/model.py's full
        # (no ablations) configuration.
        self.dict_flag_embedding = nn.Embedding(len(DICT_FLAG_VOCAB), hidden_size)
        self.lambda_dict = nn.Parameter(torch.tensor(0.1))
        self.target_flag_embedding = nn.Embedding(len(TARGET_FLAG_VOCAB), hidden_size)
        self.lambda_target = nn.Parameter(torch.tensor(0.1))

        # SS18/19: span-level metadata.
        self.type_embedding = nn.Embedding(len(self.entity_type_vocab), d_type)
        self.score_mlp = nn.Sequential(
            nn.Linear(3, d_score),
            nn.ReLU(),
            nn.Linear(d_score, d_score),
        )

        # SS21: v_c = concat(h_cls, h_first, h_span, h_last, type_emb, score_emb).
        v_dim = 4 * hidden_size + d_type + d_score

        # SS25/26: K latent experts (architecturally identical, no manual specialization)
        # + a gate that softmax-weights them, both fed the same v_c.
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(v_dim, expert_hidden),
                nn.ReLU(),
                nn.Dropout(expert_dropout),
                nn.Linear(expert_hidden, 1),
            )
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(v_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, num_experts),
        )

    def config(self) -> dict:
        return {
            "encoder_name": self.encoder_name,
            "d_type": self.d_type,
            "d_score": self.d_score,
            "num_experts": self.num_experts,
            "expert_hidden": self.expert_hidden,
            "gate_hidden": self.gate_hidden,
            "expert_dropout": self.expert_dropout,
            "entity_type_vocab": self.entity_type_vocab,
        }

    def variant_name(self) -> str:
        return variant_name(self.encoder_name)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_breakdown(self) -> list[tuple[str, str, bool, int]]:
        def n(mod) -> int:
            return sum(p.numel() for p in mod.parameters())

        hidden_size = self.encoder.config.hidden_size
        v_dim = 4 * hidden_size + self.d_type + self.d_score
        rows: list[tuple[str, str, bool, int]] = [
            ("dict_flag_embedding (DictFlagEmb)", f"Embedding{tuple(self.dict_flag_embedding.weight.shape)}", True, n(self.dict_flag_embedding)),
            ("lambda_dict", "scalar", True, self.lambda_dict.numel()),
            ("target_flag_embedding (TargetFlagEmb)", f"Embedding{tuple(self.target_flag_embedding.weight.shape)}", True, n(self.target_flag_embedding)),
            ("lambda_target", "scalar", True, self.lambda_target.numel()),
            ("type_embedding (TypeEmb)", f"Embedding{tuple(self.type_embedding.weight.shape)}", True, n(self.type_embedding)),
            ("score_mlp (ScoreMLP, full)", f"Linear(3,{self.d_score})->ReLU->Linear({self.d_score},{self.d_score})", True, n(self.score_mlp)),
        ]
        for k, expert in enumerate(self.experts):
            rows.append((f"experts[{k}] (latent Expert_{k})", f"Linear({v_dim},{self.expert_hidden})->ReLU->Dropout->Linear({self.expert_hidden},1)", True, n(expert)))
        rows.append(("gate (softmax over K experts)", f"Linear({v_dim},{self.gate_hidden})->ReLU->Linear({self.gate_hidden},{self.num_experts})", True, n(self.gate)))
        rows.append((f"encoder (frozen {self.encoder_name}: token/position/token-type text embeddings + transformer layers)", f"hidden_size={hidden_size}", False, n(self.encoder)))
        return rows

    def forward(
        self,
        input_ids: torch.Tensor, dict_flag_ids: torch.Tensor, target_flag_ids: torch.Tensor,
        attention_mask: torch.Tensor, entity_type_id: torch.Tensor, ner_score: torch.Tensor,
        return_alpha: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Returns final_logit [B] (pre-sigmoid -- pair with BCEWithLogitsLoss), or
        (final_logit, alpha [B, K]) if return_alpha -- the gate's per-expert softmax
        weights, for inspecting which expert(s) actually drove a given candidate's score
        (see analyze_experts.py). Same base 6-argument signature as phase2/model.py's
        Phase2Model.forward -- a drop-in replacement for phase2.dataset's batch shape
        (see train.py/evaluate.py); return_alpha defaults False so train.py's existing
        single-tensor forward() calls are unaffected."""
        # SS14: E_i = TokenEmb_i + lambda_dict * DictFlagEmb_i + lambda_target * TargetFlagEmb_i
        inputs_embeds = self.encoder.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds + self.lambda_dict * self.dict_flag_embedding(dict_flag_ids)
        inputs_embeds = inputs_embeds + self.lambda_target * self.target_flag_embedding(target_flag_ids)

        # SS11: no torch.no_grad() here -- gradients must flow through the frozen
        # encoder's computation to reach the side embeddings above.
        outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        H = outputs.last_hidden_state  # [B, L, hidden]

        h_cls = H[:, 0, :]

        # SS17: mean/first/last pooling over INSIDE_TARGET positions.
        inside_mask = (target_flag_ids == TARGET_FLAG_VOCAB["INSIDE_TARGET"]).float()
        span_sum = (H * inside_mask.unsqueeze(-1)).sum(dim=1)
        span_len = inside_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_span = span_sum / span_len

        batch_idx = torch.arange(H.size(0), device=H.device)
        first_pos = inside_mask.argmax(dim=1)
        last_pos = H.size(1) - 1 - inside_mask.flip(dims=[1]).argmax(dim=1)
        h_first = H[batch_idx, first_pos, :]
        h_last = H[batch_idx, last_pos, :]

        # SS18/19: type + NER-score metadata.
        type_emb = self.type_embedding(entity_type_id)
        p = ner_score.clamp(1e-5, 1 - 1e-5)
        logit_p = torch.log(p / (1 - p))
        score_input = torch.stack([p, logit_p, 1 - p], dim=-1)
        score_emb = self.score_mlp(score_input)

        v_c = torch.cat([h_cls, h_first, h_span, h_last, type_emb, score_emb], dim=-1)

        # SS25/26: K latent experts + softmax gate, all fed the same v_c.
        z = torch.stack([expert(v_c).squeeze(-1) for expert in self.experts], dim=-1)  # [B, K]
        alpha = torch.softmax(self.gate(v_c), dim=-1)  # [B, K]
        final_logit = (alpha * z).sum(dim=-1)
        if return_alpha:
            return final_logit, alpha
        return final_logit


def load_balance_loss(alpha: torch.Tensor) -> torch.Tensor:
    """Shazeer et al. 2017 / Switch-Transformer-style load-balancing auxiliary loss:
    penalizes the gate for concentrating a batch's total weight on few experts, so a
    lucky early lead for one expert doesn't snowball into permanent gate collapse (the
    unused experts get ~no gradient once their alpha is ~0, so they never improve enough
    to compete back -- see analyze_experts.py, which is what surfaced this happening in
    practice). 0 when every expert gets an equal share of the batch's total alpha
    (importance), grows towards K-1 as it concentrates onto a single expert.

    importance = alpha.sum(dim=0)          # [K], total gate weight each expert got
    loss = K * sum((importance / importance.sum())^2) - 1

    Meant to be added to the main BCE loss during training only (train.py's
    --lambda-balance), not folded into the val loss used for early stopping/model
    selection -- that should stay a pure measure of task performance."""
    importance = alpha.sum(dim=0)  # [K]
    k = alpha.size(-1)
    return k * (importance / importance.sum()).pow(2).sum() - 1


def save_checkpoint(model: Phase2ExpertModel, path: str | Path) -> None:
    """Only saves the trainable parameters -- the frozen encoder is exactly reproducible
    from config()'s encoder_name via AutoModel.from_pretrained."""
    trainable_state = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
    torch.save({"state_dict": trainable_state, "config": model.config()}, path)


def load_model(path: str | Path, device: str = "cpu") -> Phase2ExpertModel:
    checkpoint = torch.load(path, map_location=device)
    model = Phase2ExpertModel(**checkpoint["config"])
    result = model.load_state_dict(checkpoint["state_dict"], strict=False)
    assert not result.unexpected_keys, f"unexpected keys in checkpoint: {result.unexpected_keys}"
    assert all(k.startswith("encoder.") for k in result.missing_keys), f"missing non-encoder keys: {result.missing_keys}"
    model.to(device)
    model.eval()
    return model


def print_parameter_breakdown(model: Phase2ExpertModel) -> None:
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
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from phase2.base.build_candidate_windows import DEFAULT_OUT as DEFAULT_WINDOWS
    from phase2.base.dataset import Phase2WindowDataset

    print("=== Step 1: Build a small real batch (see phase2/dataset.py, reused as-is) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    dataset = Phase2WindowDataset(DEFAULT_WINDOWS, tokenizer, split="val")
    batch = dataset.collate([dataset[i] for i in range(args.batch_size)])
    print(f"batch input_ids shape: {tuple(batch['input_ids'].shape)}")

    print("=== Step 2: Build model and run one forward pass ===")
    model = Phase2ExpertModel(encoder_name=args.encoder_name, num_experts=args.num_experts)
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
    print(f"lambda_dict.grad: {model.lambda_dict.grad.item():.6f}")
    print(f"lambda_target.grad: {model.lambda_target.grad.item():.6f}")
    print(f"gate weight grad is not None: {model.gate[0].weight.grad is not None}")
    print(f"encoder param .grad is None (frozen, correctly not accumulated): {next(model.encoder.parameters()).grad is None}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
