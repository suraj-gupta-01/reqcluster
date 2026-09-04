# Dependency Engine Evaluation

- **Dataset:** `data\dependency_adversarial_set.csv` (38 requirements)
- **Gold edges:** `data\dependency_adversarial_set_gold_edges.csv` (22 labeled)
- **Cluster labels:** module
- **Generated:** 2026-09-04T06:43:16+00:00
- **spaCy available:** True

## Summary

| Configuration | Edges | Detection F1 | Typed F1 | Undirected F1 |
|---|---|---|---|---|
| regex | 5 | 0.296 | 0.222 | 0.370 |
| spacy | 17 | 0.821 | 0.821 | 0.872 |
| spacy + near-match@0.75 | 24 | 0.913 | 0.913 | 0.957 |

## regex

| Axis | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| detection (any edge) | 0.800 | 0.182 | 0.296 | 4 | 1 | 18 | 22 |
| typed (correct label) | 0.600 | 0.136 | 0.222 | 3 | 2 | 19 | 22 |
| undirected (ignore arrow) | 1.000 | 0.227 | 0.370 | 5 | 0 | 17 | 22 |

### By relation type

| Relation | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| data | 1.000 | 0.136 | 0.240 | 3 | 0 | 19 | 22 |
| hierarchical | 0.000 | 0.000 | 0.000 | 0 | 1 | 0 | 0 |
| sequential | 0.000 | 0.000 | 0.000 | 0 | 1 | 0 | 0 |

### Error breakdown (missed 18, spurious 1, mislabeled 1)

**Missed gold edges**

- `ADV-001 -> ADV-002` (data)
- `ADV-005 -> ADV-006` (data)
- `ADV-007 -> ADV-008` (data)
- `ADV-009 -> ADV-010` (data)
- `ADV-011 -> ADV-013` (data)
- `ADV-014 -> ADV-016` (data)
- `ADV-014 -> ADV-017` (data)
- `ADV-018 -> ADV-020` (data)
- `ADV-021 -> ADV-022` (data)
- `ADV-023 -> ADV-024` (data)
- `ADV-025 -> ADV-026` (data)
- `ADV-027 -> ADV-028` (data)
- `ADV-029 -> ADV-030` (data)
- `ADV-031 -> ADV-032` (data)
- `ADV-033 -> ADV-034` (data)
- `ADV-033 -> ADV-035` (data)
- `ADV-036 -> ADV-037` (data)
- `ADV-036 -> ADV-038` (data)

**Wrong relation label**

- `ADV-003 -> ADV-004`: predicted *hierarchical*, gold *data*


## spacy

| Axis | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| detection (any edge) | 0.941 | 0.727 | 0.821 | 16 | 1 | 6 | 22 |
| typed (correct label) | 0.941 | 0.727 | 0.821 | 16 | 1 | 6 | 22 |
| undirected (ignore arrow) | 1.000 | 0.773 | 0.872 | 17 | 0 | 5 | 22 |

### By relation type

| Relation | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| data | 1.000 | 0.727 | 0.842 | 16 | 0 | 6 | 22 |
| sequential | 0.000 | 0.000 | 0.000 | 0 | 1 | 0 | 0 |

### Error breakdown (missed 6, spurious 1, mislabeled 0)

**Missed gold edges**

- `ADV-021 -> ADV-022` (data)
- `ADV-023 -> ADV-024` (data)
- `ADV-025 -> ADV-026` (data)
- `ADV-027 -> ADV-028` (data)
- `ADV-029 -> ADV-030` (data)
- `ADV-031 -> ADV-032` (data)


## spacy + near-match@0.75

| Axis | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| detection (any edge) | 0.875 | 0.955 | 0.913 | 21 | 3 | 1 | 22 |
| typed (correct label) | 0.875 | 0.955 | 0.913 | 21 | 3 | 1 | 22 |
| undirected (ignore arrow) | 0.917 | 1.000 | 0.957 | 22 | 2 | 0 | 22 |

### By relation type

| Relation | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| data | 0.913 | 0.955 | 0.933 | 21 | 2 | 1 | 22 |
| sequential | 0.000 | 0.000 | 0.000 | 0 | 1 | 0 | 0 |

### Error breakdown (missed 1, spurious 3, mislabeled 0)

**Missed gold edges**

- `ADV-023 -> ADV-024` (data)

**Artifact synonym merges**

- `altitude report` -> `altitude data` (similarity 0.803)
- `baseplate temperature reading` -> `baseplate temp value` (similarity 0.837)
- `downlink telemetry packet` -> `downlink telemetry frame` (similarity 0.757)
- `fuel quantity estimate` -> `fuel quantity value` (similarity 0.805)
- `obstacle distance measurement` -> `obstacle range measurement` (similarity 0.886)
- `rotor speed measurement` -> `rotor rpm reading` (similarity 0.807)

