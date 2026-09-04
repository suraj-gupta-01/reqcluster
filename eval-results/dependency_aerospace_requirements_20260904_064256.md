# Dependency Engine Evaluation

- **Dataset:** `data\aerospace_requirements.csv` (64 requirements)
- **Gold edges:** `data\aerospace_requirements_gold_edges.csv` (42 labeled)
- **Cluster labels:** module
- **Generated:** 2026-09-04T06:42:56+00:00
- **spaCy available:** True

## Summary

| Configuration | Edges | Detection F1 | Typed F1 | Undirected F1 |
|---|---|---|---|---|
| regex | 31 | 0.329 | 0.110 | 0.411 |
| spacy | 31 | 0.329 | 0.110 | 0.411 |
| spacy + near-match@0.75 | 31 | 0.329 | 0.110 | 0.411 |

## regex

| Axis | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| detection (any edge) | 0.387 | 0.286 | 0.329 | 12 | 19 | 30 | 42 |
| typed (correct label) | 0.129 | 0.095 | 0.110 | 4 | 27 | 38 | 42 |
| undirected (ignore arrow) | 0.484 | 0.357 | 0.411 | 15 | 16 | 27 | 42 |

### By relation type

| Relation | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| data | 0.500 | 0.042 | 0.077 | 1 | 1 | 23 | 24 |
| hierarchical | 0.000 | 0.000 | 0.000 | 0 | 1 | 1 | 1 |
| reference | 0.250 | 1.000 | 0.400 | 2 | 6 | 0 | 2 |
| sequential | 0.050 | 0.067 | 0.057 | 1 | 19 | 14 | 15 |

### Error breakdown (missed 30, spurious 19, mislabeled 8)

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

- `REQ-001 -> REQ-002`: predicted *reference*, gold *data*
- `REQ-009 -> REQ-011`: predicted *reference*, gold *data*
- `REQ-009 -> REQ-014`: predicted *sequential*, gold *data*
- `REQ-017 -> REQ-019`: predicted *reference*, gold *data*
- `REQ-025 -> REQ-026`: predicted *reference*, gold *data*
- `REQ-033 -> REQ-034`: predicted *reference*, gold *data*
- `REQ-041 -> REQ-043`: predicted *reference*, gold *data*
- `REQ-052 -> REQ-035`: predicted *sequential*, gold *data*


## spacy

| Axis | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| detection (any edge) | 0.387 | 0.286 | 0.329 | 12 | 19 | 30 | 42 |
| typed (correct label) | 0.129 | 0.095 | 0.110 | 4 | 27 | 38 | 42 |
| undirected (ignore arrow) | 0.484 | 0.357 | 0.411 | 15 | 16 | 27 | 42 |

### By relation type

| Relation | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| data | 0.500 | 0.042 | 0.077 | 1 | 1 | 23 | 24 |
| hierarchical | 0.000 | 0.000 | 0.000 | 0 | 1 | 1 | 1 |
| reference | 0.250 | 1.000 | 0.400 | 2 | 6 | 0 | 2 |
| sequential | 0.050 | 0.067 | 0.057 | 1 | 19 | 14 | 15 |

### Error breakdown (missed 30, spurious 19, mislabeled 8)

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

- `REQ-001 -> REQ-002`: predicted *reference*, gold *data*
- `REQ-009 -> REQ-011`: predicted *reference*, gold *data*
- `REQ-009 -> REQ-014`: predicted *sequential*, gold *data*
- `REQ-017 -> REQ-019`: predicted *reference*, gold *data*
- `REQ-025 -> REQ-026`: predicted *reference*, gold *data*
- `REQ-033 -> REQ-034`: predicted *reference*, gold *data*
- `REQ-041 -> REQ-043`: predicted *reference*, gold *data*
- `REQ-052 -> REQ-035`: predicted *sequential*, gold *data*


## spacy + near-match@0.75

| Axis | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| detection (any edge) | 0.387 | 0.286 | 0.329 | 12 | 19 | 30 | 42 |
| typed (correct label) | 0.129 | 0.095 | 0.110 | 4 | 27 | 38 | 42 |
| undirected (ignore arrow) | 0.484 | 0.357 | 0.411 | 15 | 16 | 27 | 42 |

### By relation type

| Relation | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| data | 0.500 | 0.042 | 0.077 | 1 | 1 | 23 | 24 |
| hierarchical | 0.000 | 0.000 | 0.000 | 0 | 1 | 1 | 1 |
| reference | 0.250 | 1.000 | 0.400 | 2 | 6 | 0 | 2 |
| sequential | 0.050 | 0.067 | 0.057 | 1 | 19 | 14 | 15 |

### Error breakdown (missed 30, spurious 19, mislabeled 8)

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

- `REQ-001 -> REQ-002`: predicted *reference*, gold *data*
- `REQ-009 -> REQ-011`: predicted *reference*, gold *data*
- `REQ-009 -> REQ-014`: predicted *sequential*, gold *data*
- `REQ-017 -> REQ-019`: predicted *reference*, gold *data*
- `REQ-025 -> REQ-026`: predicted *reference*, gold *data*
- `REQ-033 -> REQ-034`: predicted *reference*, gold *data*
- `REQ-041 -> REQ-043`: predicted *reference*, gold *data*
- `REQ-052 -> REQ-035`: predicted *sequential*, gold *data*

**Artifact synonym merges**

- `telemetry frames` -> `telemetry` (similarity 0.753)

