# ReqCluster — Architecture & Flow Reference

A complete, diagram-driven explanation of the platform: system tiers, the full
ML pipeline, every phase (1 through 5 + the Honeywell DP5 deliverables), the
data model, the API surface, and the module dependency graph.

> Edge convention in dependency diagrams: `A --> B` means *A is a prerequisite
> of B* (B depends on A).

---

## 1. System architecture (three tiers)

```mermaid
flowchart TB
    subgraph T3["Tier 3 - Application"]
        UI["React SPA (Vite, Tailwind, Plotly)"]
        NAV["Sidebar: Workspace + Intelligence"]
    end
    subgraph T2["Tier 2 - API + ML"]
        API["FastAPI router (/api)"]
        SVC["Service layer:\nenrichment · refinement · feedback\ndependency · active_learning · export"]
        CORE["ML core:\nembeddings · reduction · clustering\nlabeling · graph · dependency_tree"]
        LLM["llm_services:\nproviders · enrichment · refinement\nvocabulary · quality · feedback_analyst"]
    end
    subgraph T1["Tier 1 - Data"]
        PRE["Preprocessing (CSV/XLSX)"]
        DB[("SQLite via SQLAlchemy")]
        CACHE[("Embedding cache (.npy)")]
    end

    UI --> API
    NAV --> UI
    API --> SVC
    SVC --> CORE
    SVC --> LLM
    SVC --> DB
    CORE --> CACHE
    API --> PRE
    PRE --> DB
```

---

## 2. End-to-end workflow (what a user does)

```mermaid
flowchart LR
    A["Upload CSV/XLSX"] --> B["Preprocess + persist requirements"]
    B --> C{"Enrich?\n(optional)"}
    C -- yes --> D["LLM enrichment\n(base/enriched/hybrid)"]
    C -- no --> E["Cluster pipeline"]
    D --> E
    E --> F["Overview · Scatter · Graph · Requirements"]
    F --> G["Dependency Tree (DP5)"]
    F --> H["Refinement (merge/split)"]
    F --> I["Review Queue (corrections)"]
    I --> J["Active Learning\n(apply constraints)"]
    J --> E
    F --> K["MBSE Export\nReqIF · SysML · Jama · CSV"]
    G --> K
```

---

## 3. The core ML pipeline (Phase 1, `core/pipeline.py`)

```mermaid
flowchart TB
    T["texts[]"] --> EMB["SBERT embeddings\nall-MiniLM-L6-v2 -> (N,384)\ncontent-hash cache"]
    EMB --> RED["UMAP reduction\n384 -> 10D (cluster)\n384 -> 2D (viz)"]
    RED --> CL["HDBSCAN\nmin_cluster_size = max(5, N/50)\n-1 = noise"]
    CL --> LAB["c-TF-IDF labeling\nunigram+bigram, stopword filter\ntop keywords -> label"]
    EMB --> GR["Similarity graph\ncosine > threshold\ncap 500 nodes / 2000 edges"]
    CL --> GR
    LAB --> OUT["Persist: clusters,\nrequirement assignments, graph"]
    GR --> OUT
    RED --> OUT

    EMB -. "embedding_mode = enriched/hybrid" .-> DOM["domain_embeddings\n(original + enriched context)"]
    DOM --> RED
```

Progress is reported step-by-step (`embedding -> umap -> clustering -> labeling
-> graph -> done`) and polled by the UI; the heavy work runs in a threadpool so
the event loop stays responsive.

---

## 4. Phase 2 — LLM semantic enrichment

```mermaid
sequenceDiagram
    participant UI
    participant API as POST /api/enrich
    participant SVC as enrichment_service
    participant LLM as llm_services.enrichment
    participant P as provider (mock/local/openai)
    participant DB

    UI->>API: {session_id, provider, mode}
    API->>SVC: run_and_persist_enrichment
    SVC->>SVC: extract domain vocabulary
    SVC->>SVC: check cache (provider+model+hash+prompt_version)
    SVC->>LLM: enrich_requirements(missing only)
    LLM->>P: expand_requirement (async, bounded retries)
    P-->>LLM: ParsedExpansion (strict JSON, bounded)
    LLM-->>SVC: batch result + quality report
    SVC->>DB: upsert EnrichedRequirement rows
    SVC-->>UI: status, quality, domain vocab, warnings
```

Clustering can then use `base`, `enriched`, or `hybrid` embeddings;
`enable_embedding_comparison` and `run_ablation` produce comparison metrics.

---

## 5. Phase 3 — ClusterLLM refinement

```mermaid
flowchart TB
    GEN["POST /api/suggestions/generate"] --> EMB["embeddings + 10D reduce"]
    EMB --> MS["merge_suggest\ncentroid similarity + silhouette delta"]
    EMB --> SS["split_suggest\nspread + GMM bimodality + silhouette"]
    EMB --> CO["coherence scoring (deterministic)"]
    EMB --> RP["representatives (closest to centroid)"]
    MS --> PROV["refinement provider\nmock template OR on-prem LLM"]
    SS --> PROV
    RP --> PROV
    PROV --> PERSIST["RefinementSuggestion rows (pending)"]
    PERSIST --> APPLY{"POST /api/suggestions/apply"}
    APPLY -- accept merge --> M["reassign B->A, re-label, drop B"]
    APPLY -- accept split --> SP["split by GMM sub-labels, new cluster"]
    APPLY -- reject --> R["status=rejected"]
    M --> AUD["RefinementAuditLog (before/after)"]
    SP --> AUD
    R --> AUD
```

---

## 6. Phase 4 + Phase 5 — Human-in-the-loop and active-learning loop

```mermaid
flowchart TB
    SUB["POST /api/feedback/submit\nmanual cluster correction"] --> MUT["move requirement,\nadjust cluster sizes"]
    MUT --> CON["feedback_bridge:\nmust-link to new-cluster reps\ncannot-link to old-cluster reps"]
    CON --> CP[("ConstraintPair")]
    SUB --> FC[("FeedbackCorrection (pending)")]
    FC --> REV{"POST /api/feedback/review"}
    REV -- approved --> KEEP["correction stands"]
    REV -- rejected --> ROLL["revert assignment + sizes,\ndelete constraint pairs"]

    KEEP --> AL["POST /api/cluster/constrained"]
    CP --> AL
    AL --> ENF["constrained_clustering:\nmust-link union-find merge\ncannot-link repair"]
    ENF --> RELABEL["rebuild clusters + label"]
    RELABEL --> QUAL["quality_tracker:\nsilhouette, noise rate, n_clusters"]
    QUAL --> ITER[("ClusteringIteration")]
    QUAL --> CONFLICT["detect_constraints_conflicts\n(union-find vs cannot-link)"]

    UNC["GET /api/active-learning/queue\nuncertainty sampling"] --> SUB
```

The loop closes: corrections become constraints, constraints re-shape the
clustering, quality is tracked across iterations, and the uncertainty queue
feeds the next round of corrections.

---

## 7. DP5 — Dependency tree inference

```mermaid
flowchart TB
    IN["requirements + embeddings + labels"] --> XREF["explicit REQ-id references\n(weight 0.95)"]
    IN --> CAND["semantic candidates\ntop-k by cosine"]
    CAND --> DATA["data: output-verb -> input-verb\nover shared salient term"]
    CAND --> SEQ["sequential: precondition cue\n(once/after/upon)"]
    CAND --> HIER["hierarchical: same cluster,\ngeneral -> specific"]
    XREF --> DAG["break cycles (greedy by weight)"]
    DATA --> DAG
    SEQ --> DAG
    HIER --> DAG
    DAG --> LVL["assign levels (longest path)"]
    LVL --> RAT["rationale document:\ngrouping (LLM/mock) + edge justifications"]
    RAT --> STORE[("DependencyTree")]
```

---

## 8. MBSE export (Phase 5)

```mermaid
flowchart LR
    REQ["GET /api/export/{fmt}"] --> GATHER["export_service._gather\nreqs + clusters + dependency edges"]
    GATHER --> RF["reqif_exporter\nReqIF 1.2 XML"]
    GATHER --> XMI["sysml_xmi_exporter\nXMI 2.5.1 / UML 2.5"]
    GATHER --> JAMA["jama_connector\nREST item bundle"]
    GATHER --> CSV["CSV"]
    RF --> DOORS["DOORS Next · ReqView · Polarion"]
    XMI --> PAP["Papyrus · MagicDraw"]
    JAMA --> JC["Jama Connect"]
```

---

## 9. Data model (SQLAlchemy)

```mermaid
erDiagram
    SESSION ||--o{ REQUIREMENT : has
    SESSION ||--o{ CLUSTER : has
    SESSION ||--|| GRAPH : has
    SESSION ||--o{ ENRICHED_REQUIREMENT : has
    SESSION ||--o{ REFINEMENT_SUGGESTION : has
    SESSION ||--o{ REFINEMENT_AUDIT_LOG : has
    SESSION ||--o{ FEEDBACK_CORRECTION : has
    SESSION ||--o{ CONSTRAINT_PAIR : has
    SESSION ||--o{ CLUSTERING_ITERATION : has
    SESSION ||--|| DEPENDENCY_TREE : has
    FEEDBACK_CORRECTION ||--o{ CONSTRAINT_PAIR : generates

    SESSION {
        int id PK
        string status
        int total_requirements
        int total_clusters
        int noise_count
    }
    REQUIREMENT {
        int id PK
        int session_id FK
        string req_id
        text text
        int cluster_id
        float membership_prob
        float umap_x
        float umap_y
        bool is_noise
    }
    CLUSTER {
        int id PK
        int cluster_id
        string label
        json keywords
        int size
    }
    DEPENDENCY_TREE {
        int id PK
        json nodes
        json edges
        json rationale
    }
    CONSTRAINT_PAIR {
        int id PK
        int requirement_a_id
        int requirement_b_id
        string constraint_type
        int feedback_id FK
    }
    CLUSTERING_ITERATION {
        int id PK
        int iteration
        float silhouette
        float noise_rate
        int points_moved
    }
```

---

## 10. API surface

```mermaid
flowchart LR
    subgraph Phase1["Phase 1"]
        u1["POST /upload"]
        u2["POST /cluster"]
        u3["GET /progress/{id}"]
        u4["GET /sessions · /clusters · /graph · /requirements"]
    end
    subgraph Phase2["Phase 2"]
        e1["POST /enrich"]
        e2["GET /enrich/status · /enrich/results"]
    end
    subgraph Phase3["Phase 3"]
        r1["POST /suggestions/generate · /suggestions/apply"]
        r2["GET /suggestions · /suggestions/audit"]
    end
    subgraph Phase4["Phase 4"]
        f1["POST /feedback/submit · /feedback/review"]
        f2["GET /feedback/queue · /feedback/constraints · /feedback/export"]
    end
    subgraph Phase5["Phase 5 / DP5"]
        d1["POST /dependencies/generate"]
        d2["GET /dependencies"]
        a1["POST /cluster/constrained"]
        a2["GET /active-learning/queue · /quality/history"]
        x1["GET /export/{reqif|sysml|jama|csv}"]
    end
```

---

## 11. Backend module dependency graph

```mermaid
flowchart TB
    routes["api/routes.py"] --> es["services/enrichment_service"]
    routes --> rs["services/refinement_service"]
    routes --> fs["services/feedback_service"]
    routes --> ds["services/dependency_service"]
    routes --> als["services/active_learning_service"]
    routes --> xs["services/export_service"]
    routes --> pipe["core/pipeline"]
    routes --> pre["core/preprocessing"]

    pipe --> emb["core/embeddings"]
    pipe --> dom["core/domain_embeddings"]
    pipe --> red["core/reduction"]
    pipe --> clu["core/clustering"]
    pipe --> lab["core/labeling"]
    pipe --> grp["core/graph"]
    pipe --> abl["core/ablation"]
    pipe --> cmp["core/embedding_comparison"]

    ds --> deptree["core/dependency_tree"]
    ds --> reps["core/representatives"]
    ds --> llmref["llm_services/refinement"]
    rs --> ms["core/merge_suggest"]
    rs --> ss["core/split_suggest"]
    rs --> reps
    rs --> llmref
    als --> cc["core/constrained_clustering"]
    als --> alm["core/active_learning"]
    fs --> fb["core/feedback_bridge"]
    fs --> fa["llm_services/feedback_analyst"]
    es --> llmenr["llm_services/enrichment"]
    llmenr --> prov["llm_services/providers"]
    llmref --> prov
    xs --> rfx["export/reqif_exporter"]
    xs --> sxm["export/sysml_xmi_exporter"]
    xs --> jmc["export/jama_connector"]

    deptree --> emb
    dom --> emb
```

---

## 12. Provider model (offline-first)

```mermaid
flowchart LR
    REQ["enrichment / refinement / dependency rationale"] --> SEL{"provider_name"}
    SEL -- mock --> MOCK["deterministic, offline\n(default)"]
    SEL -- local --> LOCAL["Ollama / local HTTP\n(e.g. Qwen)"]
    SEL -- openai --> OAI["OpenAI-compatible gateway"]
    LOCAL -. error/unconfigured .-> MOCK
    OAI -. error/unconfigured .-> MOCK
```

Everything runs offline by default; LLM output is strictly parsed and bounded,
and any provider failure falls back to the deterministic path.

---

## 13. Code-review knowledge graph

A graphify-generated knowledge graph of the backend (519 nodes, 994 edges, 31
labelled communities) is checked in under `graphify-out/`:

- `graphify-out/graph.html` — interactive graph, open in any browser.
- `graphify-out/GRAPH_REPORT.md` — audit report (god nodes, surprising
  connections, suggested questions).
- `graphify-out/graph.json` — raw graph data (GraphRAG-ready).

The communities recovered by the graph match the intended module boundaries
(LLM provider & enrichment, domain embeddings & ablation, MBSE export,
dependency tree, refinement, constrained clustering, preprocessing, etc.),
which is a good structural-cohesion signal. The most-connected "god node" is
`normalize_plain_text()` (the shared sanitizer used across the LLM layer).
