# Dependency Engine Evaluation

- **Dataset:** `data\aerospace_requirements.csv` (64 requirements)
- **Gold edges:** `data\aerospace_requirements_gold_edges.csv` (42 labeled, 8 objectively verifiable)
- **Cluster labels:** module
- **Generated:** 2026-09-04T07:13:11+00:00
- **spaCy available:** True

## Summary

| Configuration | Edges | Detection F1 | Typed F1 | Undirected F1 |
|---|---|---|---|---|
| regex | 20 | 0.387 | 0.161 | 0.452 |
| spacy | 20 | 0.387 | 0.161 | 0.452 |
| spacy + near-match@0.75 | 20 | 0.387 | 0.194 | 0.452 |

## regex

| Axis | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| detection (any edge) | 0.600 | 0.286 | 0.387 | 12 | 8 | 30 | 42 |
| typed (correct label) | 0.250 | 0.119 | 0.161 | 5 | 15 | 37 | 42 |
| undirected (ignore arrow) | 0.700 | 0.333 | 0.452 | 14 | 6 | 28 | 42 |

**Objectively verifiable subset** — gold edges where the target's text literally cites the source ID. These labels are reproducible from the text alone and need no trust in the annotator.

Recall only: precision is not meaningful against a subset, since edges correctly found outside it would count as false positives.

| Axis | Recall | Found | Missed | Support |
|---|---|---|---|---|
| detection | 1.000 | 8 | 0 | 8 |
| typed | 0.375 | 3 | 5 | 8 |

### By relation type

| Relation | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| data | 0.667 | 0.083 | 0.148 | 2 | 1 | 22 | 24 |
| hierarchical | 0.000 | 0.000 | 0.000 | 0 | 1 | 1 | 1 |
| reference | 0.333 | 1.000 | 0.500 | 2 | 4 | 0 | 2 |
| sequential | 0.100 | 0.067 | 0.080 | 1 | 9 | 14 | 15 |

### Error breakdown (missed 30, spurious 8, mislabeled 7)

**Missed gold edges**

- `REQ-001 -> REQ-004` (data)
- `REQ-001 -> REQ-006` (data)
- `REQ-001 -> REQ-008` (data)
- `REQ-003 -> REQ-002` (hierarchical)
- `REQ-017 -> REQ-021` (sequential)
- `REQ-017 -> REQ-024` (data)
- `REQ-018 -> REQ-017` (sequential)
- `REQ-020 -> REQ-040` (sequential)
- `REQ-022 -> REQ-017` (sequential)
- `REQ-026 -> REQ-013` (sequential)
- `REQ-026 -> REQ-030` (sequential)
- `REQ-034 -> REQ-036` (data)
- `REQ-035 -> REQ-055` (data)
- `REQ-037 -> REQ-033` (sequential)
- `REQ-041 -> REQ-046` (data)
- `REQ-042 -> REQ-041` (data)
- `REQ-044 -> REQ-045` (data)
- `REQ-044 -> REQ-048` (data)
- `REQ-046 -> REQ-054` (data)
- `REQ-047 -> REQ-041` (data)
- `REQ-051 -> REQ-053` (sequential)
- `REQ-053 -> REQ-026` (sequential)
- `REQ-054 -> REQ-055` (data)
- `REQ-054 -> REQ-056` (sequential)
- `REQ-057 -> REQ-058` (data)

**Wrong relation label**

- `REQ-001 -> REQ-002`: predicted *sequential*, gold *data*
- `REQ-009 -> REQ-014`: predicted *sequential*, gold *data*
- `REQ-017 -> REQ-019`: predicted *reference*, gold *data*
- `REQ-025 -> REQ-026`: predicted *reference*, gold *data*
- `REQ-033 -> REQ-034`: predicted *reference*, gold *data*
- `REQ-041 -> REQ-043`: predicted *reference*, gold *data*
- `REQ-052 -> REQ-035`: predicted *sequential*, gold *data*


## spacy

| Axis | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| detection (any edge) | 0.600 | 0.286 | 0.387 | 12 | 8 | 30 | 42 |
| typed (correct label) | 0.250 | 0.119 | 0.161 | 5 | 15 | 37 | 42 |
| undirected (ignore arrow) | 0.700 | 0.333 | 0.452 | 14 | 6 | 28 | 42 |

**Objectively verifiable subset** — gold edges where the target's text literally cites the source ID. These labels are reproducible from the text alone and need no trust in the annotator.

Recall only: precision is not meaningful against a subset, since edges correctly found outside it would count as false positives.

| Axis | Recall | Found | Missed | Support |
|---|---|---|---|---|
| detection | 1.000 | 8 | 0 | 8 |
| typed | 0.375 | 3 | 5 | 8 |

### By relation type

| Relation | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| data | 0.667 | 0.083 | 0.148 | 2 | 1 | 22 | 24 |
| hierarchical | 0.000 | 0.000 | 0.000 | 0 | 1 | 1 | 1 |
| reference | 0.333 | 1.000 | 0.500 | 2 | 4 | 0 | 2 |
| sequential | 0.100 | 0.067 | 0.080 | 1 | 9 | 14 | 15 |

### Error breakdown (missed 30, spurious 8, mislabeled 7)

**Missed gold edges**

- `REQ-001 -> REQ-004` (data)
- `REQ-001 -> REQ-006` (data)
- `REQ-001 -> REQ-008` (data)
- `REQ-003 -> REQ-002` (hierarchical)
- `REQ-017 -> REQ-021` (sequential)
- `REQ-017 -> REQ-024` (data)
- `REQ-018 -> REQ-017` (sequential)
- `REQ-020 -> REQ-040` (sequential)
- `REQ-022 -> REQ-017` (sequential)
- `REQ-026 -> REQ-013` (sequential)
- `REQ-026 -> REQ-030` (sequential)
- `REQ-034 -> REQ-036` (data)
- `REQ-035 -> REQ-055` (data)
- `REQ-037 -> REQ-033` (sequential)
- `REQ-041 -> REQ-046` (data)
- `REQ-042 -> REQ-041` (data)
- `REQ-044 -> REQ-045` (data)
- `REQ-044 -> REQ-048` (data)
- `REQ-046 -> REQ-054` (data)
- `REQ-047 -> REQ-041` (data)
- `REQ-051 -> REQ-053` (sequential)
- `REQ-053 -> REQ-026` (sequential)
- `REQ-054 -> REQ-055` (data)
- `REQ-054 -> REQ-056` (sequential)
- `REQ-057 -> REQ-058` (data)

**Wrong relation label**

- `REQ-001 -> REQ-002`: predicted *sequential*, gold *data*
- `REQ-009 -> REQ-014`: predicted *sequential*, gold *data*
- `REQ-017 -> REQ-019`: predicted *reference*, gold *data*
- `REQ-025 -> REQ-026`: predicted *reference*, gold *data*
- `REQ-033 -> REQ-034`: predicted *reference*, gold *data*
- `REQ-041 -> REQ-043`: predicted *reference*, gold *data*
- `REQ-052 -> REQ-035`: predicted *sequential*, gold *data*


## spacy + near-match@0.75

| Axis | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| detection (any edge) | 0.600 | 0.286 | 0.387 | 12 | 8 | 30 | 42 |
| typed (correct label) | 0.300 | 0.143 | 0.194 | 6 | 14 | 36 | 42 |
| undirected (ignore arrow) | 0.700 | 0.333 | 0.452 | 14 | 6 | 28 | 42 |

**Objectively verifiable subset** — gold edges where the target's text literally cites the source ID. These labels are reproducible from the text alone and need no trust in the annotator.

Recall only: precision is not meaningful against a subset, since edges correctly found outside it would count as false positives.

| Axis | Recall | Found | Missed | Support |
|---|---|---|---|---|
| detection | 1.000 | 8 | 0 | 8 |
| typed | 0.500 | 4 | 4 | 8 |

### By relation type

| Relation | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| data | 0.750 | 0.125 | 0.214 | 3 | 1 | 21 | 24 |
| hierarchical | 0.000 | 0.000 | 0.000 | 0 | 1 | 1 | 1 |
| reference | 0.400 | 1.000 | 0.571 | 2 | 3 | 0 | 2 |
| sequential | 0.100 | 0.067 | 0.080 | 1 | 9 | 14 | 15 |

### Error breakdown (missed 30, spurious 8, mislabeled 6)

**Missed gold edges**

- `REQ-001 -> REQ-004` (data)
- `REQ-001 -> REQ-006` (data)
- `REQ-001 -> REQ-008` (data)
- `REQ-003 -> REQ-002` (hierarchical)
- `REQ-017 -> REQ-021` (sequential)
- `REQ-017 -> REQ-024` (data)
- `REQ-018 -> REQ-017` (sequential)
- `REQ-020 -> REQ-040` (sequential)
- `REQ-022 -> REQ-017` (sequential)
- `REQ-026 -> REQ-013` (sequential)
- `REQ-026 -> REQ-030` (sequential)
- `REQ-034 -> REQ-036` (data)
- `REQ-035 -> REQ-055` (data)
- `REQ-037 -> REQ-033` (sequential)
- `REQ-041 -> REQ-046` (data)
- `REQ-042 -> REQ-041` (data)
- `REQ-044 -> REQ-045` (data)
- `REQ-044 -> REQ-048` (data)
- `REQ-046 -> REQ-054` (data)
- `REQ-047 -> REQ-041` (data)
- `REQ-051 -> REQ-053` (sequential)
- `REQ-053 -> REQ-026` (sequential)
- `REQ-054 -> REQ-055` (data)
- `REQ-054 -> REQ-056` (sequential)
- `REQ-057 -> REQ-058` (data)

**Wrong relation label**

- `REQ-001 -> REQ-002`: predicted *sequential*, gold *data*
- `REQ-009 -> REQ-014`: predicted *sequential*, gold *data*
- `REQ-025 -> REQ-026`: predicted *reference*, gold *data*
- `REQ-033 -> REQ-034`: predicted *reference*, gold *data*
- `REQ-041 -> REQ-043`: predicted *reference*, gold *data*
- `REQ-052 -> REQ-035`: predicted *sequential*, gold *data*

**Artifact synonym merges**

- `telemetry frames` -> `telemetry` (similarity 0.753)

