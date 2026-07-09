# CAFE Project Overview and Two-Phase Experimental Plan

## 1. Project goal

The goal of CAFE is to estimate a reliable confidence score for each named-entity candidate produced by an off-the-shelf NER model on OCR-derived historical newspapers.

For each candidate entity `c`, the system outputs is :

```text
r(c) in [0, 1]
```

The output is candidate-level, not sentence-level or document-level. A sentence may contain several entities, and each entity receives its own score.

Example:

```json
{
  "candidate_id": "issue_1908_04_17_page_3_cand_0021",
  "context": "Le president Raymond Poincar6 arriva a Geneve.",
  "span_text": "Raymond Poincar6",
  "predicted_type": "PERSON",
  "ner_score": 0.94,
  "reliability_score": 0.72
}
```

Interpretation:

```text
The original NER model is confident, but because the OCR form is suspicious, the reliability verifier estimates only 0.72 reliability.
```

## 2. Main research question

Historical newspaper digitization often provides OCR text, page layout, word-level OCR confidence, and facsimile links, but it does not necessarily provide named entities. The project asks:

```text
How can we enrich OCR-derived historical newspapers with named entities while estimating whether each automatic annotation is reliable?
```

The central problem is that raw NER confidence is not enough. A model can output a high score for a wrong candidate when OCR noise, layout artifacts, broken lines, or historical spelling variation mislead the model.

The project therefore learns a second-stage reliability model:

```text
NER candidate + context + OCR evidence -> reliability score
```

## 3. Prediction unit and input unit

The prediction unit is one NER candidate:

```text
candidate = span + predicted entity type + original NER confidence score + source context
```

The text input can be a sentence or a local window. The target candidate is marked explicitly inside the context.


## 4. Shared candidate dataset for both phases

Both Phase 1 and Phase 2 should use the same candidate dataset and evaluation protocol. This makes the comparison fair.

### 4.1 Candidate generation

Run one or more NER systems over the OCR-derived text.

Possible candidate sources:

```text
GLiNER2
French NER model
MELHISSA or another historical entity-linking system
Gazetteer-based candidate generation
Optional LLM-based candidate generation for a small subset
```

For each predicted candidate, store:

```text
candidate_id
source_document_id
raw context text
candidate span start/end offsets
candidate surface string
predicted entity type
NER confidence score
```

### 4.2 Extract OCR quality in word-level 

Store:

```text
word-level OCR confidence values
```

### 4.3 Context construction

Recommended default:

```text
64 tokens before candidate + candidate span + 64 tokens after candidate
```

Alternative context units to compare:

```text
sentence containing the candidate
fixed token window, such as +/- 64 or +/- 128 tokens
```

For historical OCR, a fixed local window is often safer than relying only on sentence segmentation.

### 4.4 Annotation target

Compare label predicted by model and grountruth label:

```text
label_reliable = 1 if the candidate is match with groundtruth
label_reliable = 0 otherwise
```

## 4. Shared splits

Use separate splits for experts, gates, calibration, and final testing.

For Phase 1, use:

```text
expert_train: 50 percent
gate_train: 20 percent
calibration: 10 percent
test: 20 percent
```

For Phase 2, use:

```text
train: 70 percent
calibration: 10 percent
test: 20 percent
```

For early stopping, split `train` into `train` and `dev`, or use cross-validation if data is limited.

The train, calibration, and test set should be the same in two phases. 

## 5. Two-phase research design

The project is divided into two phases.

### Phase 1: Manual experts plus learned gate

Phase 1 is interpretable and controlled.

Pipeline:

```text
manual feature extraction
-> human-defined evidence experts
-> train experts separately
-> freeze experts
-> compute expert logits
-> train adaptive gate
-> calibrated reliability score
```

Main purpose:

```text
Show which evidence families matter: NER confidence, OCR quality, context, correction stability.
```

Expected contribution:

```text
A transparent reliability model that learns adaptive weights instead of using fixed hand-written weights.
```

### Phase 2: Latent extraction plus jointly trained experts and gate

Phase 2 reduces manual feature engineering.

Pipeline:

```text
context with marked entity span
+ token-level OCR confidence embeddings
+ NER score/type embeddings
-> language model encoder
-> latent experts
-> gate
-> calibrated reliability score
```

Main purpose:

```text
Let the model learn useful evidence representations from hidden states instead of manually defining every feature and every expert.
```

Expected contribution:

```text
A neural OCR-aware span-targeted verifier with latent mixture-of-experts reliability scoring.
```

### Shared expert-gate formula:  

For the first set of experiments, both phases use the same high-level reliability-scoring architecture. The difference between the two phases is not the final scoring mechanism, but how the evidence representation is obtained. Two phases should have the same number of experts. 

A generic architecture has the following structure:

Expert_j:

| Step | Phase 1 | Phase 2 |
|---|---|---|
| Input | Manual feature group `X_j` | Candidate representation `h_c` |
| Layer 1 | `Linear(d_j, 32)` | `Linear(hidden_dim, 256)` |
| Activation | `ReLU` | `ReLU` |
| Regularization | `Dropout(0.1)` | `Dropout(0.1)` |
| Layer 2 | `Linear(32, 1)` | `Linear(256, 1)` |
| Output | `z_j` | `z_j` | 

Gate: 

| Step | Phase 1 | Phase 2 |
|---|---|---|
| Input | Routing features `q` | Candidate representation `h_c` |
| Layer 1 | `Linear(d_gate, 32)` | `Linear(hidden_dim, 128)` |
| Activation | `ReLU` | `ReLU` |
| Layer 2 | `Linear(32, number_of_experts)` | `Linear(128, 1)` |
| Normalization | `Softmax` | `Softmax` |
| Output | `alpha` | `alpha` |

Experts of phase 1 can consider to try simpler baselines with logistic regression or gradient boosting later. 


### Phase comparison

| Aspect | Phase 1 | Phase 2 |
|---|---|---|
| Feature extraction | Manual aggregate features | Learned from hidden states plus raw metadata |
| Experts | Human-defined evidence experts | Latent neural experts |
| Gate | Trained after experts | Trained jointly with experts |
| Interpretability | High | Medium; requires gate/expert analysis |
| Data need | Lower | Higher |

## 6. Shared evaluation metrics

The expected output is a probability-like reliability score, so evaluation should focus on calibration and selective prediction, not only accuracy.

Report:

```text
Accuracy
Precision / recall / F1 for reliable vs unreliable candidates
Brier score
Expected Calibration Error, ECE
Adaptive Calibration Error, optional
Reliability diagram
Risk-coverage curve
Area under risk-coverage curve, AURC
Precision at selected coverage levels
Coverage at target precision levels
Error rate among automatically accepted candidates
```

Recommended selective evaluation:

```text
Accept candidate if r(c) >= threshold
Reject or review candidate otherwise
```

Report results at thresholds such as:

```text
0.50, 0.70, 0.80, 0.90, 0.95
```

Also choose thresholds on the calibration set for target error budgets:

```text
accepted error <= 5 percent
accepted error <= 10 percent
```

## 7. Shared calibration step

After either phase, calibrate the final score on the calibration split.

Possible calibration methods:

```text
Platt scaling / sigmoid calibration
isotonic regression
```

Final output should contain both raw and calibrated scores:

```json
{
  "candidate_id": "issue_1908_04_17_page_3_cand_0021",
  "span_text": "Raymond Poincar6",
  "predicted_type": "PERSON",
  "ner_score": 0.94,
  "raw_reliability_score": 0.78,
  "calibrated_reliability_score": 0.72,
  "decision": "uncertain_or_review"
}
```


## 11. Main claims the experiments should support

The experiments should support these claims:

```text
1. Raw NER confidence is not reliable enough for OCR-derived historical newspapers by do the calibration on it and compare the result with our method. 
3. Adaptive evidence fusion is better than fixed-weight scoring.
4. Latent LM-based extraction can reduce manual feature engineering while preserving or improving reliability.
```
