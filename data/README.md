# Test datasets

## Bundled

| File | Rows | Notes |
|------|------|-------|
| `sample_requirements.csv` | 119 | Original Phase-1 sample, 6 domains. |
| `aerospace_requirements.csv` | 64 | 8 subsystems with intentional dependency structure (explicit `REQ-xxx` cross-references, sequential preconditions, producer/consumer data links). Best for demoing the dependency tree and MBSE export. Regenerate with `python data/generate_aerospace_dataset.py`. |

Upload either file on the **Upload** page, run clustering, then try
**Dependency Tree**, **Refinement**, **Active Learning**, and **Export**.

## Real-world public datasets (recommended for evaluation)

These are established requirements-engineering research datasets. They are
research-licensed; use for academic / design-a-thon evaluation and cite the
sources.

| Dataset | What it provides | Use for | Source |
|---------|------------------|---------|--------|
| **PROMISE NFR** | 625 labelled requirements (functional + 11 NFR classes) | Clustering quality vs. ground-truth labels | tera-PROMISE repository |
| **PURE** (Ferrari et al., 2017) | 79 real public SRS documents, thousands of requirements incl. aerospace/defense | Scale + realism, 500-requirement runs | Zenodo: "PURE: A Dataset of Public Requirements Documents" |
| **Dronology** | UAV requirements **with trace links** | Validating the dependency tree against ground-truth dependencies | dronology.info / Notre Dame |
| **PROMISE CM1 / MODIS / CCHIT** | Requirement-to-requirement trace matrices | Dependency-edge precision/recall | tera-PROMISE traceability datasets |

To use one: export the requirements to a CSV with at least a `text` column
(optionally `id`, `module`, `section`) and upload it. The preprocessor
auto-detects common column-name variants (`requirement`, `description`, etc.).

## Hard evaluation sets (`generate_eval_datasets.py`)

Deterministic (seeded, stdlib + numpy) generator; `uv run python data/generate_eval_datasets.py` regenerates and self-checks all three. Mixed sentence structures, shared cross-module vocabulary, ~10% terse and ~5% long requirements, ~18% cross-module borrowing so classes overlap lexically (MiniLM + KMeans k=24 gives ARI about 0.5, not 0.94).

| File | Rows | Columns |
|------|------|---------|
| `multidomain_requirements.csv` | ~1200 | `id,text,domain,module` - automotive / medical / rail, 24 unbalanced modules (35-69 rows) |
| `dependency_gold_requirements.csv` + `dependency_gold_edges.csv` | ~400 reqs, ~214 edges | `id,text,module,section`; edges `source_id,target_id,relation` with relation in data/reference/refines/sequence (source = producer/referenced/parent/predecessor). Includes ~30% no-dependency distractors and artifact synonyms. `module` here is the dependency kind, not a topic label. |
| `adversarial_requirements.csv` | 600 | `id,text,module,perturbation,source_id,contradicts_id`; perturbation in paraphrase/near_duplicate/typo/contradiction/noise; noise rows have module `__noise__` |
