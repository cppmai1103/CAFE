import sys, json, torch
sys.path.insert(0, 'src')
from transformers import AutoTokenizer
from phase2.tokenize_windows import tokenize_candidate_window
from phase2.vocab import DICT_FLAG_VOCAB, TARGET_FLAG_VOCAB, ENTITY_TYPE_VOCAB
from phase2.model import Phase2Model

with open('data_phase2/phase2_candidate_windows.jsonl') as f:
    rec = next(json.loads(l) for l in f if json.loads(l)['candidate_id'] == 'EXP-1998-06-03-a-i0180__s0__1-2')

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained('camembert-base')
result = tokenize_candidate_window(tok, rec['window_tokens'], rec['dict_flags'], rec['target_start_window'], rec['target_end_window'])
subwords = tok.convert_ids_to_tokens(result['input_ids'])
dict_names = {v: k for k, v in DICT_FLAG_VOCAB.items()}
target_names = {v: k for k, v in TARGET_FLAG_VOCAB.items()}

model = Phase2Model()
model.eval()
input_ids = torch.tensor([result['input_ids']])
dict_flag_ids = torch.tensor([result['dict_flag_ids']])
target_flag_ids = torch.tensor([result['target_flag_ids']])
attention_mask = torch.tensor([result['attention_mask']])
entity_type_id = torch.tensor([ENTITY_TYPE_VOCAB[rec['predicted_type']]])
ner_score_t = torch.tensor([rec['ner_score']])
with torch.no_grad():
    logit = model(input_ids, dict_flag_ids, target_flag_ids, attention_mask, entity_type_id, ner_score_t)
reliability = torch.sigmoid(logit).item()

FLAG_COLOR = {"SPECIAL": "#898781", "GOOD": "#0ca30c", "BAD": "#e34948", "PUNCT": "#898781"}
TARGET_COLOR = {"SPECIAL": "#898781", "OUTSIDE": "#898781", "INSIDE_TARGET": "#8e5cd9"}

rows = []
for sw, d, t in zip(subwords, result["dict_flag_ids"], result["target_flag_ids"]):
    dname, tname = dict_names[d], target_names[t]
    sw_disp = sw.replace("▁", "_").replace("<", "&lt;").replace(">", "&gt;")
    bg = "#f3e9fc" if tname == "INSIDE_TARGET" else "#fcfcfb"
    rows.append(
        f'<TR><TD BGCOLOR="{bg}"><FONT FACE="Courier">{sw_disp}</FONT></TD>'
        f'<TD BGCOLOR="{bg}"><FONT COLOR="{FLAG_COLOR[dname]}">{dname}</FONT></TD>'
        f'<TD BGCOLOR="{bg}"><FONT COLOR="{TARGET_COLOR[tname]}">{tname}</FONT></TD></TR>'
    )
table_rows = "\n".join(rows)

p = max(min(rec["ner_score"], 1 - 1e-5), 1e-5)
import math
logit_p = math.log(p / (1 - p))

dot = f'''digraph Phase2Inference {{
    rankdir=TB;
    bgcolor="#fcfcfb";
    fontname="Helvetica";
    node [fontname="Helvetica", fontsize=11, style="filled,rounded", shape=box, penwidth=1.2, fixedsize=false];
    edge [fontname="Helvetica", fontsize=9, color="#898781", fontcolor="#52514e"];

    labelloc="t";
    label="Phase 2 inference flow -- one real candidate, step by step\\n(candidate_id={rec['candidate_id']}, untrained checkpoint -- see caption)";
    fontsize=16;
    fontcolor="#0b0b0b";

    node [shape=box, style="filled,rounded", fillcolor="#fcfcfb", color="#898781", fontcolor="#0b0b0b"];
    raw [label="STEP 0: raw candidate (label_reliability_type_only.csv)\\ldocument_id={rec['document_id']}, sentence_id={rec['sentence_id']}\\lstart_token_id={rec['start_token_id']}, end_token_id={rec['end_token_id']} (inclusive)\\lspan_text={rec['span_text']!r}   predicted_type={rec['predicted_type']}\\lner_score={rec['ner_score']:.4f}   label_reliable={rec['label_reliable']} (ground truth)\\l"];

    node [fillcolor="#eaf2fc", color="#2a78d6"];
    window [label="STEP 1: build_candidate_windows.py -- word-level window (+/-16, clipped at doc start)\\lwindow_tokens ({len(rec['window_tokens'])} words):\\l{' '.join(rec['window_tokens'])}\\ltarget_start_window={rec['target_start_window']}  target_end_window={rec['target_end_window']}\\ldict_flags: mostly GOOD; 'repoussant' -> BAD (not in French dictionary, OCR-suspicious)\\l"];

    node [shape=none, fillcolor="#fcfcfb"];
    tokens [label=<
<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="2" CELLPADDING="3" BGCOLOR="#fcfcfb" COLOR="#8e5cd9">
<TR><TD COLSPAN="3"><B>STEP 2: tokenize_windows.py -- subword alignment ({result['n_subwords']} subwords, no truncation needed)</B></TD></TR>
<TR><TD><B>subword</B></TD><TD><B>dict_flag</B></TD><TD><B>target_flag</B></TD></TR>
{table_rows}
<TR><TD COLSPAN="3">"Menendez" -&gt; 3 subwords (Men/end/ez), all inherit INSIDE_TARGET from their word.<BR/>"repoussant" -&gt; 3 subwords, all inherit BAD (not a French dictionary word here).</TD></TR>
</TABLE>>];

    node [shape=box, style="filled,rounded", fillcolor="#0b0b0b", fontcolor="white", color="#0b0b0b"];
    embeds [label="STEP 3: model.py -- inputs_embeds = word_emb + lambda_dict*DictFlagEmb + lambda_target*TargetFlagEmb\\lshape: [1, {result['n_subwords']}, 768]"];

    node [fillcolor="#e34948", fontcolor="white", color="#e34948"];
    encoder [label="STEP 4: frozen CamemBERT encoder(inputs_embeds, attention_mask)\\l(+ position/token-type embeddings added internally, SS11)\\lH = last_hidden_state, shape [1, {result['n_subwords']}, 768]"];

    node [fillcolor="#8e5cd9", fontcolor="white", color="#8e5cd9"];
    pooling [label="STEP 5: pool over the 4 INSIDE_TARGET subwords (_Pedro, _Men, end, ez)\\lh_cls = H[:,0,:]            (<s>)\\lh_first = H at _Pedro        (first target subword)\\lh_span = mean(H over all 4 INSIDE_TARGET subwords)\\lh_last = H at ez             (last target subword)\\lv_text = concat(h_cls, h_first, h_span, h_last)  shape [1, 3072]"];

    node [fillcolor="#0ca30c", fontcolor="white", color="#0ca30c"];
    metadata [label="STEP 6: metadata embeddings\\lentity_type_id = ENTITY_TYPE_VOCAB['PERS'] = {ENTITY_TYPE_VOCAB['PERS']}  -&gt;  TypeEmb  -&gt;  type_emb [1, 32]\\lner_score={rec['ner_score']:.4f}  -&gt;  [p, logit(p), 1-p] = [{p:.4f}, {logit_p:.3f}, {1-p:.4f}]  -&gt;  ScoreMLP  -&gt;  score_emb [1, 32]"];

    node [fillcolor="#0b0b0b", fontcolor="white", color="#0b0b0b"];
    vc [label="STEP 7: v_c = concat(v_text, type_emb, score_emb)   shape [1, 3136]"];

    node [fillcolor="#0ca30c", fontcolor="white", color="#0ca30c"];
    classifier [label="STEP 8: classifier(v_c) = Linear-&gt;ReLU-&gt;Dropout(0.1)-&gt;Linear\\lfinal_logit = {logit.item():.4f}"];

    node [shape=oval, fillcolor="#fdf1e2", color="#e8871e", fontcolor="#0b0b0b"];
    result_node [label="STEP 9: reliability_score = sigmoid(final_logit) = {reliability:.4f}\\l(this checkpoint is UNTRAINED -- random init -- so ~0.5 is expected;\\lground truth label_reliable={rec['label_reliable']} is shown in STEP 0 for reference only)"];

    raw -> window -> tokens -> embeds -> encoder -> pooling -> metadata -> vc -> classifier -> result_node;
}}
'''
with open(".scratch/inference_flow.dot", "w") as f:
    f.write(dot)
print("wrote .scratch/inference_flow.dot")
