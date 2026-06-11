# Napkin AI prompt — ReqCluster pipeline diagram

Paste the block below into **Napkin AI** (napkin.ai) and pick a **flowchart /
process** visual style. It describes the whole ReqCluster pipeline so it fits on
**one screen**. Tip: choose a horizontal (left-to-right) layout, dark theme, teal
accent.

---

## Prompt (copy everything below)

Create a single-screen, left-to-right **pipeline flowchart** of "ReqCluster", an
AI requirements-clustering system. Use a clean dark theme with a teal accent. Group
it into three horizontal lanes: **Ingest → Clustering Pipeline → Intelligence &
Output**, with a thin **Data Layer** strip underneath. Use rounded boxes for steps
and arrows for flow.

Lane 1 — Ingest:
1. Upload CSV / XLSX of requirements
2. Preprocess (clean, de-duplicate, validate)

Lane 2 — Clustering Pipeline (the core flow, left to right):
3. SBERT embeddings (all-MiniLM-L6-v2, 384-dim) — runs on GPU
4. UMAP dimensionality reduction (384 → 10-D for clustering, and → 2-D for the plot)
5. HDBSCAN density clustering (auto cluster count, marks outliers as noise)
6. c-TF-IDF labeling (names each cluster by its distinctive keywords)
7. ANN similarity graph (hnswlib, O(N log N))

Lane 3 — Intelligence & Output (these act on the clustered result; optional):
8. LLM Enrichment (expand each requirement — local Ollama model or offline mock)
9. Refinement (suggest merge / split of clusters)
10. Dependency tree (infer requirement-to-requirement dependencies)
11. Active learning (human corrections fed back into a constrained re-cluster)
12. Export (PDF report, ReqIF, SysML/XMI, Jama, CSV)

Data Layer (strip under everything, connected to the pipeline):
- PostgreSQL (stores sessions, requirements, clusters, graph)
- Redis (per-text embedding cache — re-runs skip re-embedding)
- React + Plotly dashboard (the UI that drives and visualizes all of the above)

Show the main flow as a straight arrow chain 1 → 2 → 3 → 4 → 5 → 6 → 7, then a fan-
out from step 7 into the Intelligence steps 8-12. Keep it readable on one screen;
short labels, no long sentences inside the boxes.

---

## Shorter alternative (if the above is too dense for one screen)

Create a clean left-to-right flowchart titled "ReqCluster pipeline":
**Upload → Preprocess → SBERT embeddings (GPU) → UMAP → HDBSCAN clustering →
c-TF-IDF labels → similarity graph → Dashboard.**
Below it, a small box labeled "Data: PostgreSQL + Redis cache".
To the right, a branch from "Dashboard" to four chips: "Enrichment", "Refinement",
"Dependency tree", "Export". Dark theme, teal accent, rounded boxes, minimal text.
