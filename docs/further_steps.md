Optional further steps

## 1. Gate variants to test

### G0: fixed global weights baseline

```text
Use one learned or manually set weight vector for all candidates.
```

This is a baseline, not the final method.

### G1: entity-type-specific gate

Learn different expert weights for:

```text
PERSON
LOCATION
ORGANIZATION
OTHER
```

### G2: OCR-bin gate

Learn different weights for:

```text
low OCR
medium OCR
high OCR
```

### G3: OCR x NER confidence gate

Condition weights on the 2D grid:

```text
NER confidence: low / medium / high
OCR confidence: low / medium / high
```

This is easy to visualize and good for analysis.

### G4: layout-conditioned gate

Learn weights based on:

```text
body
title
header
footer
margin
advertisement
table
```

### G5: softmax MLP gate

Use the MLP gate over routing features. This should be the main Phase 1 model.

### G6: sparse top-k gate

Use only the top 1 or top 2 experts per candidate:

```text
alpha = softmax(gate(q))
keep top-k weights
set others to 0
renormalize
```

This improves interpretability.

## 2. Preventing gate collapse

The gate may learn to trust only one expert, usually the NER expert. Prevent this with:

### 2.1 Expert dropout

During gate training, randomly hide some expert logits:

```text
drop z_NER sometimes
drop z_OCR sometimes
drop z_LINK sometimes
```

### 2.2 Load-balancing regularization

Add a small penalty if the average gate distribution collapses:

```text
loss = BCE + lambda * sum_j (mean_alpha_j - 1/K)^2
```

where `K` is the number of experts.

Use a small `lambda`; the goal is not equal weights everywhere, but avoiding total collapse.

### 2.3 Gate entropy monitoring

Track the average entropy of gate weights:

```text
low entropy everywhere may mean hard routing or collapse
very high entropy everywhere may mean no specialization
```

## 12. Phase 1 ablations (optional)

Ablate evidence groups:

```text
Full Phase 1 model
minus OCR expert
minus context expert
minus layout expert
minus stability expert
minus linking expert
minus attribution expert, if used
minus adaptive gate, use fixed weights
```

Report:

```text
Brier score
ECE
risk-coverage AURC
precision at selected coverage
accepted error rate at target threshold
```

## 15. Phase 1 analysis

### 15.1 Gate weight analysis

Report average gate weights by condition:

```text
low OCR span
high OCR span
header/footer
body text
PERSON
LOCATION
ORGANIZATION
stable after correction
unstable after correction
```

Example table:

```text
Condition: low OCR span
NER: 0.14
OCR: 0.27
CONTEXT: 0.10
LAYOUT: 0.08
STABILITY: 0.25
LINK: 0.16
```

### 15.2 Error analysis

For false positives and false negatives, report whether the error is associated with:

```text
OCR-corrupted entity surface
broken line or boundary problem
noisy context
layout artifact
wrong entity type
ambiguous entity surface
link/gazetteer ambiguity
```

### 15.3 OCR and NER interaction analysis

Create a 2D heatmap:

```text
x-axis: NER confidence bin
y-axis: OCR confidence bin
cell value: empirical error rate
```

This directly tests whether high NER confidence remains risky in low-OCR regions.