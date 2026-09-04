# Dependency Engine Evaluation

- **Dataset:** `data\dependency_adversarial_set.csv` (38 requirements)
- **Gold edges:** `data\dependency_adversarial_set_gold_edges.csv` (22 labeled, 0 objectively verifiable)
- **Cluster labels:** module
- **Generated:** 2026-09-04T07:13:30+00:00
- **spaCy available:** True

## Summary

| Configuration | Edges | Detection F1 | Typed F1 | Undirected F1 |
|---|---|---|---|---|
| regex | 4 | 0.308 | 0.231 | 0.308 |
| spacy | 16 | 0.842 | 0.842 | 0.842 |
| spacy + near-match@0.75 | 23 | 0.933 | 0.933 | 0.933 |

## regex

| Axis | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| detection (any edge) | 1.000 | 0.182 | 0.308 | 4 | 0 | 18 | 22 |
| typed (correct label) | 0.750 | 0.136 | 0.231 | 3 | 1 | 19 | 22 |
| undirected (ignore arrow) | 1.000 | 0.182 | 0.308 | 4 | 0 | 18 | 22 |

### By relation type

| Relation | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| data | 1.000 | 0.136 | 0.240 | 3 | 0 | 19 | 22 |
| hierarchical | 0.000 | 0.000 | 0.000 | 0 | 1 | 0 | 0 |

### Error breakdown (missed 18, spurious 0, mislabeled 1)

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
| detection (any edge) | 1.000 | 0.727 | 0.842 | 16 | 0 | 6 | 22 |
| typed (correct label) | 1.000 | 0.727 | 0.842 | 16 | 0 | 6 | 22 |
| undirected (ignore arrow) | 1.000 | 0.727 | 0.842 | 16 | 0 | 6 | 22 |

### By relation type

| Relation | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| data | 1.000 | 0.727 | 0.842 | 16 | 0 | 6 | 22 |

### Error breakdown (missed 6, spurious 0, mislabeled 0)

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
| detection (any edge) | 0.913 | 0.955 | 0.933 | 21 | 2 | 1 | 22 |
| typed (correct label) | 0.913 | 0.955 | 0.933 | 21 | 2 | 1 | 22 |
| undirected (ignore arrow) | 0.913 | 0.955 | 0.933 | 21 | 2 | 1 | 22 |

### By relation type

| Relation | P | R | F1 | TP | FP | FN | Support |
|---|---|---|---|---|---|---|---|
| data | 0.913 | 0.955 | 0.933 | 21 | 2 | 1 | 22 |

### Error breakdown (missed 1, spurious 2, mislabeled 0)

**Missed gold edges**

- `ADV-023 -> ADV-024` (data)

**Artifact synonym merges**

- `altitude report` -> `altitude data` (similarity 0.803)
- `baseplate temperature reading` -> `baseplate temp value` (similarity 0.837)
- `downlink telemetry packet` -> `downlink telemetry frame` (similarity 0.757)
- `fuel quantity estimate` -> `fuel quantity value` (similarity 0.805)
- `obstacle distance measurement` -> `obstacle range measurement` (similarity 0.886)
- `rotor speed measurement` -> `rotor rpm reading` (similarity 0.807)

