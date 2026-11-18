# 🎤 ReqCluster — Full Pitch Deck Script
### Honeywell Hackathon · Problem Statement DP5

> **How to use this doc:** Each slide has a title, what to show, and exactly what to say (natural spoken language). Bold lines = key punchy statements you want to land. Italics = delivery tips.

---

## ⏱️ Timing Guide
| Segment | Slides | Time |
|---|---|---|
| Hook + Problem | 1–2 | 2 min |
| Solution Overview | 3 | 1.5 min |
| Live Demo / Pipeline | 4 | 3 min |
| Intelligence Features | 5 | 2.5 min |
| Architecture + Scale | 6–7 | 2 min |
| Results + Tests | 8 | 1 min |
| Closing + Q&A | 9–10 | 2 min |
| **Total** | | **~14 min** |

---

---

# SLIDE 1 — The Hook

## 🖥️ What to show
Title card: **"ReqCluster — AI-Assisted Requirements Engineering"**  
Subtitle: *Built for Honeywell Hackathon · Problem DP5*

---

## 🎤 What to say

*Open with a pause. Let the room settle. Then:*

> "Imagine you're a systems engineer at an aerospace company.
> You've just received a 500-row Excel sheet.
> Every row is a requirement. They came from five different teams, written over three years.
> Your job? Sort them by meaning, figure out which ones depend on which, and write a rationale for every group.
>
> **Your deadline is tomorrow.**"

*Pause.*

> "That job — today — takes 8 to 12 hours. Per 500 requirements.
> And two engineers doing it separately will only agree on the groupings about 60% of the time.
>
> **We built ReqCluster to fix that.**"

---

---

# SLIDE 2 — The Problem

## 🖥️ What to show
Three columns side-by-side:
- **Scale**: "500 → 50,000 requirements per program"
- **Time**: "8–12 hours per 500, manual"
- **Quality**: "~60% inter-analyst agreement (Cohen's kappa)"

Below: "Existing tools each solve one slice and stop there."

| Tool | What it does | What it misses |
|---|---|---|
| DOORS / Jama | Store and version requirements | No clustering |
| k-means / LDA | Cluster text | Needs preset count, ignores domain meaning |
| LLM prompting | Summarize requirements | Non-deterministic, manual, outside RE workflow |

---

## 🎤 What to say

> "Let me give you the real numbers.
>
> A serious aerospace or automotive program carries **500 to 50,000 requirements**, written by many teams over many years.
> Sorting them by hand is slow — **8 to 12 hours per 500 requirements**.
> It's inconsistent — **two experienced analysts only agree about 60% of the time**, measured by Cohen's kappa.
> And cross-cutting links — requirements that belong to multiple domains — get missed entirely.
>
> The tools that exist today each solve one piece of this puzzle and stop there.
> DOORS and Jama are great at storing requirements. They don't cluster anything.
> k-means needs you to tell it how many groups to make — in advance — which you never know.
> And ChatGPT? It's non-deterministic, it lives outside your RE workflow, and you can't send Honeywell IP to a cloud API.
>
> **ReqCluster is the missing middle layer. Automatic, explainable, repeatable, human-supervised — and it exports back into the tools you already use.**"

---

---

# SLIDE 3 — The Solution

## 🖥️ What to show
The pipeline diagram from the README:

```
CSV / XLSX → preprocess → SBERT embeddings → UMAP → HDBSCAN → c-TF-IDF labels
                                 |                                     |
                                 +──────────> similarity graph <───────+
                                                    |
                  dependency tree + rationale · dashboard · export
```

---

## 🎤 What to say

> "Here's what ReqCluster does in one picture.
>
> You upload a CSV or Excel file. Any format — we handle the column name variations automatically.
>
> Step one: **SBERT embeddings**. We use a sentence transformer model to turn every requirement into a 384-dimensional vector. Similar meaning gives you similar vectors. This is the semantic foundation.
>
> Step two: **UMAP** reduces those 384 dimensions to 10 for clustering, and 2 for the scatter plot you're about to see.
>
> Step three: **HDBSCAN** — this is the key choice over k-means. It finds dense groups *automatically*, no preset count, and it marks outliers as noise instead of forcing them into a wrong group.
>
> Step four: **c-TF-IDF** names every cluster with its most distinctive keywords.
>
> And from the same embeddings we build a **cosine similarity graph** that powers the dependency tree.
>
> The whole thing is **deterministic and runs fully offline**. No API keys. No data leaves your network."

*Transition:*
> "Let me show you this live."

---

---

# SLIDE 4 — LIVE DEMO

## 🖥️ What to show
**Switch to the running app.** Navigate through:
1. Upload page
2. Overview / cluster cards
3. Scatter plot
4. Similarity Graph
5. Dependency Tree
6. Export page

---

## 🎤 What to say — beat by beat

### 4a · Upload

*Navigate to the Upload page, drag in `data/aerospace_requirements.csv`*

> "I'm uploading our aerospace requirements dataset — 64 requirements spanning 8 subsystems: navigation, power, propulsion, structures, avionics, and more.
>
> You can see the clustering parameters here — min cluster size, min samples, similarity threshold. These are tunable, but the defaults work well.
>
> I'll hit **Run Clustering Pipeline** and you'll see a live progress bar as it goes through each stage."

*Click run. Watch the progress bar.*

> "Preprocessing, embedding, reduction, clustering, labeling, graph — done.
> **Under 5 seconds for 64 requirements.** We've benchmarked this up to 50,000 — I'll come back to those numbers."

---

### 4b · Overview

*Navigate to Overview*

> "Here's the dashboard. We got 8 clusters — which matches the 8 subsystems in the dataset, exactly as designed.
>
> Each card shows the cluster label — these were generated automatically by c-TF-IDF — the number of requirements, and a coherence score.
>
> Notice the noise count. HDBSCAN correctly flagged the outlier requirements that don't cleanly belong to any group. **That honesty is important in engineering contexts — you don't want your tool lying to you and forcing a wrong assignment.**"

---

### 4c · Scatter Plot

*Navigate to Scatter*

> "This is the 2D UMAP projection. Every dot is a requirement, colored by cluster.
>
> Look at how cleanly separated these groups are. That's the embedding quality — semantically similar requirements end up physically close in this space.
>
> The gray dots are noise — requirements that sit in low-density regions, between clusters.
>
> This renders with WebGL, so it stays smooth at tens of thousands of requirements."

---

### 4d · Similarity Graph

*Navigate to Graph*

> "Here's the similarity graph. Requirements are nodes; edges connect semantically similar ones.
>
> This isn't a pretty visualization for the slide deck. **This graph is the computational artifact** — it's what we use to infer dependencies. Tightly connected nodes share technical concerns. Bridges between clusters are cross-cutting requirements."

---

### 4e · Dependency Tree

*Navigate to Dependency Tree, click Generate Dependencies*

> "This is the DP5 deliverable — the dependency tree.
>
> We infer four types of edges from the requirement text itself:
> - **Hierarchical** — parent/child functional relationships
> - **Sequential** — pre-conditions, things that must happen first
> - **Data** — producer/consumer links
> - **Reference** — explicit cross-references
>
> And for every edge, we generate a rationale — a written explanation of *why* these requirements are related.
>
> You can switch between 2D and 3D views. Click any node to see its full rationale.
>
> **This is what the problem statement asked for: a categorized list, a dependency diagram, and a rationale document. All three, automatically generated.**"

---

### 4f · Export

*Navigate to Export, briefly show formats*

> "And when you're done, you export. Five formats:
> - **PDF** — a formatted report with metrics, charts, and per-cluster requirements
> - **ReqIF 1.2** — the OMG standard, imports directly into DOORS
> - **SysML/UML XMI** — for model-based systems engineering tools
> - **Jama bundle** — imports directly into Jama Connect
> - **CSV** — for anything else
>
> Every format includes the dependency links when a dependency tree exists."

*Switch back to slides.*

---

---

# SLIDE 5 — Intelligence Features (5 Phases)

## 🖥️ What to show
A timeline or phase progression:

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
Core        LLM      Refine   Human    Active
Pipeline  Enrichment  +Audit  in Loop  Learning
```

---

## 🎤 What to say

> "The live demo showed Phase 1 — the core pipeline. Let me walk you through what's on top of it.
>
> **Phase 2 — LLM Enrichment.**
> Before embedding, we can optionally expand each requirement using a language model — adding inferred intent, assumed components, technical context.
> Then we re-cluster on three different embedding modes: base only, enriched only, or a hybrid blend.
> The Enrichment page shows you a side-by-side comparison with ablation metrics so you can see whether the LLM expansion actually improved cluster quality — not just assume it did.
>
> **Phase 3 — ClusterLLM Refinement.**
> After clustering, we run an automated quality pass. Silhouette analysis finds clusters that are too similar to each other — merge candidates. Bimodality analysis finds clusters that are internally inconsistent — split candidates.
> Every suggestion comes with a coherence score and a rationale. You accept or reject each one, and every decision goes into an audit log.
>
> **Phase 4 — Human in the Loop.**
> Your analysts can manually reassign any requirement to a different cluster. Those corrections become must-link and cannot-link constraints. There's a review queue where a lead engineer approves or rejects each correction before it takes effect.
>
> **Phase 5 — Active Learning.**
> The constraints from Phase 4 feed into a constrained re-cluster. The system also surfaces the *least confident* assignments — the requirements it's most unsure about — and asks your analysts to review those first. Quality is tracked across iterations so you can see the clustering improving over time.
>
> **The key point: every phase is optional.** You can stop at Phase 1 and get a great result. Or you can go all the way to Phase 5 for a fully human-supervised, iteratively refined output. The system meets you where you are."

---

---

# SLIDE 6 — Architecture

## 🖥️ What to show
The architecture diagram:

```
React + Vite + Tailwind + Plotly (WebGL)
          ↓
FastAPI routes
Services: enrichment · refinement · feedback · dependency · active-learning · export
Core ML: embeddings · reduction · clustering · labeling · graph · dependency_tree
LLM: mock (offline) / local Ollama / OpenAI-compatible
          ↓
PostgreSQL / SQLite (fallback)    Redis (embedding cache)
```

---

## 🎤 What to say

> "Three-tier architecture, each independently testable.
>
> The **frontend** is React with Vite and Tailwind. Interactive visualizations use Plotly with WebGL rendering — that's what makes the scatter plot smooth at scale.
>
> The **backend** is FastAPI on Python. Clean separation: API routes → service layer → core ML modules → LLM services. Clustering runs as an async background job — the API returns immediately and you poll for progress.
>
> The **data layer** is PostgreSQL in production with SQLite as a zero-setup local fallback. Embedding vectors are stored as numpy arrays on disk, not in the database rows. Redis caches per-text embeddings — so if you add 50 new requirements to a 500-requirement dataset, it only re-embeds those 50.
>
> **LLM is offline-first.** The default provider is a deterministic mock — no model, no API key, no network. You can switch to a local Ollama model, or any OpenAI-compatible endpoint like Groq. Any LLM failure falls back to deterministic output — **the app never breaks because an LLM was unavailable**.
>
> One command to run the full stack: `docker compose up`. That brings up Postgres, Redis, backend, and frontend behind nginx."

---

---

# SLIDE 7 — Performance & Scale

## 🖥️ What to show
The performance table:

| Requirements | CPU (best → worst) | GPU (best → worst) |
|--:|--:|--:|
| 500 | 4 → 17 s | 4 → 16 s |
| 1,000 | 8 → 20 s | 7 → 19 s |
| 5,000 | 23 → 43 s | 15 → 35 s |
| 10,000 | 41 → 71 s | 26 → 55 s |
| 50,000 | 146 → 188 s | 65 → 108 s |

---

## 🎤 What to say

> "We didn't just build a proof of concept. We benchmarked it.
>
> These are real measured numbers on a 12-core CPU laptop with an RTX 2050 — consumer hardware, not a datacenter.
>
> **500 requirements: 4 seconds on CPU.** That's the use case the problem statement described.
>
> **50,000 requirements: under 3 minutes on CPU. Under 2 minutes on GPU.**
>
> And the pipeline is size-adaptive. For small datasets it uses a fixed random seed — fully reproducible. For large datasets it automatically switches to a parallel, approximate path that stays near-linear instead of exploding quadratically.
>
> The similarity graph uses approximate nearest-neighbor search — O(N log N) — so there's no quadratic blow-up as you scale.
>
> GPU gives roughly **7-8× speedup on embeddings** — from ~440 to ~3,650 requirements per second. On a datacenter GPU you'd expect 20-40×. The embedding cache means re-runs are nearly instantaneous.
>
> **Our target is 500,000 requirements in ~10 seconds on GPU hardware. The architecture is already designed for it.**"

---

---

# SLIDE 8 — Engineering Quality

## 🖥️ What to show
- **140+ automated tests** — green in CI
- GitHub Actions CI badge
- `uv run pytest -q` output (all passing)
- `.github/workflows/ci.yml` file exists

---

## 🎤 What to say

> "One more thing before I wrap up.
>
> We take engineering quality seriously. The project has **140+ automated tests** — unit tests, integration tests, API tests for every endpoint, and service-level tests.
>
> All tests run **offline and deterministically** — the embedding model is mocked in `conftest.py`, so CI doesn't need to download anything. Real UMAP and HDBSCAN run when installed.
>
> GitHub Actions runs the full backend test suite and the frontend production build on every push and pull request.
>
> This isn't a demo that works in one specific order. It's a tested, deployable system."

---

---

# SLIDE 9 — What We Built (Recap)

## 🖥️ What to show
Clean bullet list:

✅ **Phase 1** — Core pipeline: SBERT → UMAP → HDBSCAN → c-TF-IDF → similarity graph  
✅ **Phase 2** — LLM enrichment with mode comparison and ablation  
✅ **Phase 3** — Automated merge/split suggestions with audit log  
✅ **Phase 4** — Human-in-the-loop corrections with constraint generation  
✅ **Phase 5** — Active learning, constrained re-clustering, quality tracking  
✅ **DP5 deliverable** — Dependency tree + rationale document  
✅ **5 export formats** — PDF, ReqIF, SysML XMI, Jama, CSV  
✅ **Production-ready** — Docker Compose, PostgreSQL, Redis, GPU support, CI  
✅ **Fully offline** — No API keys required, no data leaves your network  

---

## 🎤 What to say

> "So let me bring it all together.
>
> We implemented all five phases of the pipeline — from the core ML clustering all the way through active learning.
>
> We delivered everything the DP5 problem statement asked for: a categorized list, a dependency diagram, and a rationale document — fully automated.
>
> We built five export formats so the output drops directly into DOORS, Jama, or SysML tools with no manual reformatting.
>
> The system is production-ready: Docker Compose, PostgreSQL, Redis, optional GPU acceleration, and CI.
>
> And it runs **completely offline**. Honeywell's requirements stay on Honeywell's hardware. No API keys, no cloud calls, no data leaving the network.
>
> **We're not showing you a prototype. We're showing you a system that's ready to be deployed.**"

---

---

# SLIDE 10 — Close + Q&A

## 🖥️ What to show
Final slide: **"ReqCluster"**  
Tagline: *"8 hours → under a minute. Repeatable, explainable, offline."*  
GitHub / contact info if applicable.

---

## 🎤 What to say

> "Requirements engineering is one of the most human-intensive parts of systems development. It doesn't have to be.
>
> ReqCluster turns an 8-hour analyst task into something you run before your morning coffee.
> The results are explainable — every cluster has a rationale. Every dependency has a reason.
> The process is repeatable — run it again tomorrow and get the same answer.
> And it's supervised — your analysts stay in the loop at every phase.
>
> **Thank you. I'm happy to take questions.**"

---

---

# 🛡️ Q&A Prep — Likely Questions

### "How does this compare to just asking ChatGPT?"
> "Three key differences. First, **determinism** — ChatGPT gives you a different answer every time you ask. Our clustering pipeline is deterministic and reproducible. Second, **scale** — ChatGPT's context window limits you to maybe 50-100 requirements in one shot. We've benchmarked to 50,000. Third, **data privacy** — this runs fully offline. Honeywell's requirements never leave the network. ChatGPT can't offer any of those three."

### "What if the clusters are wrong?"
> "That's exactly why we built Phases 3, 4, and 5. Phase 3 gives you automated suggestions to merge or split. Phase 4 lets analysts manually correct any assignment. Phase 5 feeds those corrections back as hard constraints into a re-cluster. You can iterate until you're happy. And every decision is recorded in an audit log."

### "Does it work on real Honeywell requirements?"
> "We couldn't test on internal IP, but we validated on three types of data: our aerospace sample dataset with known subsystem structure, PROMISE NFR which is 625 labelled real-world requirements, and a 600-requirement UAV flight management system dataset. The clustering consistently recovers the ground-truth domain structure."

### "What about requirements that span multiple domains — cross-cutting concerns?"
> "HDBSCAN marks those as noise — cluster -1. That's actually the honest answer: a requirement that sits between multiple domains belongs to none of them cleanly. The similarity graph shows these cross-cutting requirements as bridge nodes with edges to multiple clusters. You can see them explicitly in the scatter plot as the gray dots between groups."

### "How does the dependency tree work without explicit trace links?"
> "We infer dependencies from the requirement text using four heuristic patterns: hierarchical (parent/child function), sequential (pre-conditions and triggering events), data (producer/consumer via variable names), and reference (explicit 'shall comply with' style cross-references). It's not perfect — it's a heuristic — but it surfaces the high-confidence links and presents rationale for each, so analysts can verify rather than discover from scratch."

### "Can it integrate with our existing DOORS environment?"
> "Yes — the ReqIF 1.2 export is specifically designed for DOORS import. It carries cluster assignments, dependency links, and requirement metadata in the standard schema. The Jama export is a structured JSON bundle that Jama's importer accepts directly."

### "How much does the LLM cost to run?"
> "Zero, by default. The mock provider is deterministic and has no model behind it — it's there so the pipeline never breaks. If you want real LLM rationales, you use a local Ollama model — a 7B model like Qwen2.5 runs on 6GB of VRAM, no cloud costs. Or you connect any OpenAI-compatible endpoint. The cost is entirely your choice."

---

---

# 🗒️ Demo Prep Checklist

Before you present, verify:

- [ ] Backend is running on `:8000`
- [ ] Frontend is running on `:5173`
- [ ] `data/aerospace_requirements.csv` is ready to upload
- [ ] Browser is on the Upload page, file already loaded if possible
- [ ] Clustering parameters are at defaults
- [ ] Previous sessions are cleared for a clean demo (or use an existing completed session to skip wait time)
- [ ] Dependency tree page has a completed session pre-loaded if generation takes too long live
- [ ] Export page shows a completed session

---

*Good luck tomorrow. You built something real — just explain it clearly and let the product speak for itself.*
