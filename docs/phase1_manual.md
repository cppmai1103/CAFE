# CAFE-TEI Phase 1: Manual Feature Extraction, Human-Defined Experts, and Learned Gate

## 1. Phase 1 objective

Phase 1 builds an interpretable reliability estimator for NER candidates.

The expected output for each candidate `c` is:

```text
r(c) in [0, 1]
```

Phase 1 uses manually extracted features and human-defined evidence experts. Each expert is trained separately on one evidence family. After that, the experts are frozen and a gate is trained to learn adaptive weights over expert outputs.

Overall pipeline:

```text
candidate examples
-> manual feature extraction
-> train evidence experts separately
-> freeze experts
-> compute expert logits
-> train adaptive gate
-> calibrate final score
-> output reliable confidence score
```

The main research purpose is to show that fixed weights are insufficient and that the importance of evidence changes by candidate condition.

Example:

```text
High NER confidence + high OCR confidence + body text:
    gate can trust the NER expert more.

High NER confidence + low OCR confidence + unstable after correction:
    gate should trust OCR and stability experts more.

High NER confidence + high OCR confidence + page header:
    gate should trust layout expert more.
```

## 2. Input data

Each training instance is one predicted NER candidate.

Required fields:

```text
candidate_id
context tokens
span start/end inside the context
span text
predicted entity type
NER confidence score
source model
label_reliable
word-level OCR confidence score
```

Optional fields:

```text
OCR-corrected text
entity-linking candidate and score
gazetteer matches
other NER model predictions
Integrated Gradients attribution values
```

## 3. Output target

Start with one binary target:

```text
label_reliable = 1 if the candidate is reliable (a predicted entity map with ground-truth)
label_reliable = 0 otherwise
```

The model output is:

```text
r(c) = P(label_reliable = 1 | candidate evidence)
```


## 4. Manual feature groups

Split features into evidence families. Each evidence family will train one expert.

### 4.1 NER evidence features

Purpose:

```text
Measure how confident and internally consistent the original NER prediction is.
```

Features:

```text
ner_score
top1_top2_type_margin
type_entropy
predicted_entity_type
span_length_tokens
span_length_characters
```

### 4.2 OCR span evidence features

Purpose:

```text
Measure whether the entity surface itself is visually reliable.
```

For the OCR confidences of words inside the candidate span, compute:

```text
span_ocr_mean
span_low_conf_word_fraction
span_first_word_ocr
span_last_word_ocr
sentence_ocr_mean
documnet_ocr_mean
```

Important intuition:

```text
mean OCR confidence may hide one corrupted word;
minimum OCR confidence can detect one risky token inside the entity.
```

Example:

```text
Raymond     WC = 0.97
Poincar6    WC = 0.42

span_ocr_mean = 0.695
span_ocr_min = 0.42
span_low_conf_word_fraction = 0.5
```

### 4.3 Context evidence features

Purpose:

```text
Measure whether the surrounding context supports the predicted entity type.
```

Features:

```text
left_context_ocr_mean_10
right_context_ocr_mean_10
context_ocr_min_10
context_low_conf_word_fraction_10
sentence_ocr_mean
sentence_ocr_min
sentence_length
context_window_length
```

Important distinction:

```text
Low span OCR confidence often affects span/boundary reliability.
Low context OCR confidence often affects type or link reliability.
```


### 4.4 OCR-correction stability features (optional: not implement now)

Purpose:

```text
Measure whether the NER candidate is stable under OCR correction.
```

Run NER on both:

```text
raw OCR text
OCR-corrected text
```

Then compare raw and corrected candidates.

Features:

```text
same_span_after_correction
span_IoU_raw_vs_corrected
surface_edit_distance_raw_corrected
ner_score_raw
ner_score_corrected
ner_score_delta
candidate_disappears_after_correction
candidate_appears_only_after_correction
```


Example 1:

```text
Raw OCR:       Raymond Poincar6 -> PERSON, score 0.94
Corrected OCR: Raymond Poincare -> PERSON, score 0.97

Interpretation: surface is noisy, but candidate is stable.
```

Example 2:

```text
Raw OCR:       La Commisston -> ORG, score 0.91
Corrected OCR: la commission -> no entity

Interpretation: raw OCR probably created a fake entity.
```

### 4.5 Linking and gazetteer evidence features (optional: not implement now)

Purpose:

```text
Measure whether external lexical or linking evidence supports the candidate.
```

Features:

```text
NEL_score
top1_top2_link_margin
number_of_candidate_links
gazetteer_exact_match
gazetteer_fuzzy_match
historical_variant_match
link_candidate_ambiguity
entity_frequency_in_corpus
place_name_match
person_name_match
organization_name_match
```

### 4.6 Attribution evidence features (optional: not implement now)

Purpose:

```text
Measure whether the NER model relies on trustworthy or noisy tokens.
```

Features:

```text
IG_mass_on_entity_span
IG_mass_on_left_context
IG_mass_on_right_context
IG_mass_on_low_ocr_tokens
IG_entropy
IG_mass_on_punctuation
IG_mass_on_digits
```

Attribution-weighted OCR risk:

```text
attr_ocr_risk = sum(abs(attribution_t) * (1 - ocr_conf_t)) / sum(abs(attribution_t))
```

Interpretation:

```text
High attr_ocr_risk means the NER decision depends strongly on low-confidence OCR tokens.
```

## 5. Human-defined experts

Train one expert for each feature group.

Recommended experts:

```text
E_NER: NER confidence expert
E_OCR: OCR span expert
E_CONTEXT: context expert
E_STABILITY: correction-stability expert (optional)
E_LINK: linking/gazetteer expert (optional)
E_ATTR: attribution expert (optional)
```

Each expert predicts the same target:

```text
P(label_reliable = 1 | its own feature group)
```

For candidate `i` and expert `j`:

```text
z_i_j = expert_j_logit(x_i_j)
p_i_j = sigmoid(z_i_j)
```

where:

```text
x_i_j = features for expert j
z_i_j = expert logit
p_i_j = expert probability
```

## 6. Training experts separately

### 6.1 Data split for Phase 1

Use:

```text
expert_train: for training experts
gate_train: for training the gate
calibration: for score calibration 
test: final evaluation only
```

Recommended split:

```text
expert_train: 50 percent
gate_train: 20 percent
calibration: 10 percent
test: 20 percent
```

If the dataset is small, use K-fold out-of-fold predictions for gate training.

### 6.2 Model choices for experts

A clean first implementation:

```text
All experts are small MLPs:
input -> Linear(d, 32) -> ReLU -> Dropout -> Linear(32, 1)
```

Interpretable alternative:

```text
Use logistic regression for each expert first.
```

Other choices: 

```text
logistic regression
small MLP
gradient-boosted decision trees
```

### 6.3 Expert loss

Each expert is trained with binary cross-entropy:

```text
loss_j = BCE(sigmoid(z_i_j), label_reliable_i)
```

Train each expert only on its own feature family.

### 6.4 Save expert outputs

After training, freeze all experts.

For every candidate in `gate_train`, `calibration`, and `test`, compute:

```text
z_NER
z_OCR
z_CONTEXT
z_LAYOUT
z_STABILITY (optional)
z_LINK (optional)
z_ATTR (optional)
```

Store expert outputs in a table:

```json
{
  "candidate_id": "issue_1908_04_17_page_3_cand_0021",
  "label_reliable": 1,
  "z_NER": 2.94,
  "z_OCR": -0.62,
  "z_CONTEXT": 1.39,
  "z_LAYOUT": 2.20,
  "z_STABILITY": 1.10,
  "z_LINK": 0.41
  ...
}
```

Use logits rather than probabilities for the final weighted sum.

## 7. Gate input: routing features

The gate should learn which expert to trust under each candidate condition.

Gate features:

```text
predicted_entity_type
span_ocr
context_ocr
span_length
page_ocr_quality
```

The gate input for candidate `i` is:

```text
q_i = routing_features_i
```

The gate output is a vector of weights over experts:

```text
alpha_i = softmax(gate(q_i))
```

Example: 

```text
alpha_i = [alpha_NER, alpha_OCR, alpha_CONTEXT, alpha_STABILITY, ...]
```

The weights sum to 1.

## 8. Final score computation

For candidate `i`, the experts produce logits:

```text
z_i = [z_NER, z_OCR, z_CONTEXT, z_STABILITY, ...]
```

The gate produces adaptive weights:

```text
alpha_i = [alpha_NER, alpha_OCR, alpha_CONTEXT, alpha_STABILITY, ...]
```

Final logit:

```text
s_i = b + sum_j alpha_i_j * z_i_j
```

Final reliability score:

```text
r_i = sigmoid(s_i)
```

This is the final output before calibration.

## 9. Training the gate

### 9.1 Two-stage training

Phase 1 uses two-stage training:

```text
Stage A: train experts separately.
Stage B: freeze experts and train gate.
```

The gate is trained on `gate_train`.

For each gate-training candidate:

```text
1. Load frozen expert logits z_i.
2. Load routing features q_i.
3. Gate computes alpha_i = softmax(gate(q_i)).
4. Compute final logit s_i = b + sum_j alpha_i_j * z_i_j.
5. Compute r_i = sigmoid(s_i).
6. Optimize BCE(r_i, label_reliable_i).
```

### 9.2 Gate model

Start with a small MLP:

```text
routing features
-> Linear(d_gate, 32)
-> ReLU
-> Linear(32, number_of_experts)
-> softmax
```

The gate does not need gold expert weights. It learns weights because the final reliability prediction must match the gold label.

### 9.3 Why the gate learns adaptive weights

If many examples have:

```text
high NER score
low OCR confidence
unstable after correction
gold label = unreliable
```

then putting high weight on the NER expert will produce errors. The gate learns to reduce NER weight and increase OCR/stability weight for similar routing conditions.

If other examples have:

```text
low OCR confidence
stable after correction
strong linking/gazetteer evidence
gold label = reliable
```

then the gate learns that low OCR alone is not enough to reject the candidate.

## 10. Calibration

After gate training, calibrate final scores on the calibration split.

Inputs to calibration:

```text
raw reliability score r_i
label_reliable_i
```

Methods:

```text
Platt scaling / sigmoid calibration
isotonic regression
```

Output:

```text
calibrated_r_i in [0, 1]
```

Use calibrated scores for final reporting and TEI decisions.

## 11. Phase 1 baselines

Compare against:

```text
B0: raw NER confidence
B1: calibrated raw NER confidence
B2: adaptive gate over expert logits, main Phase 1 method 
B3: logistic regression over all manual features 
B4: product of experts with fixed weights (optional)
B5: best single expert (optional)
B6: average of expert logits (optional)
B7: global learned weights over expert logits (optional)

```

## 12. Expected Phase 1 output file

For each candidate, output:

```json
{
  "candidate_id": "issue_1908_04_17_page_3_cand_0021",
  "span_text": "Raymond Poincar6",
  "predicted_type": "PERSON",
  "ner_score": 0.94,
  "expert_logits": {
    "ner": 2.94,
    "ocr": -0.62,
    "context": 1.39,
    "layout": 2.20,
    "stability": 1.10,
    "link": 0.41
  },
  "gate_weights": {
    "ner": 0.14,
    "ocr": 0.24,
    "context": 0.12,
    "layout": 0.06,
    "stability": 0.28,
    "link": 0.16
  },
  "raw_reliability_score": 0.78,
  "calibrated_reliability_score": 0.72,
}
```

## 13. Phase 1 success criteria

Phase 1 is successful if:

```text
1. Adaptive gate outperforms raw NER confidence.
2. Reliability scores are better calibrated than raw NER scores.
```
