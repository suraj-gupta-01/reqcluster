# ReqCluster Scalability & Performance Roadmap

> **Detailed analysis:** See `/memories/repo/scalability-analysis.md` for full technical specifications, code examples, and cost breakdown.

---

## Current State vs. Goals

```
Current:         10K requirements  5 users  3 min clustering   2-3 GB RAM/session
Target (Phase 5): 500K+ requirements 100 users 10 sec clustering 200 MB RAM/session
```

---

## 7 CRITICAL BOTTLENECKS

### 🔴 **CRITICAL (Block Scale > 20K req)**
1. **SQLite Single-Writer Lock** → PostgreSQL migration
2. **Missing Database Indexes** → Add 6 composite indexes
3. **SBERT Embedding Speed** → Increase batch size + caching
4. **In-Memory Embeddings** → Use memory-mapped files

### 🟠 **HIGH (Block Scale > 100K req)**
5. **UMAP Refitting** → Implement streaming transform
6. **LLM Provider Serialization** → Async with connection pooling
7. **N+1 Query Pattern** → Eager loading with joinedload()

### 🟡 **MEDIUM (Degraded Performance > 50K req)**
- Graph truncation strategy
- Request coalescing for polling
- Constraint conflict detection optimization

---

## 5-PHASE EXECUTION PLAN

| Phase | Duration | Effort | Key Changes | Expected Gain |
|-------|----------|--------|-------------|---------------|
| **1: Foundation** | 2 wks | 🟠 Medium | Indexes, PostgreSQL, connection pooling | 5-10x query speed |
| **2: Compute** | 2 wks | 🟠 Medium | Embedding cache (Redis), batch optimization | 3-4x pipeline speed |
| **3: Concurrency** | 2 wks | 🔴 High | Process pool, async LLM, request coalescing | 4-8x throughput |
| **4: Memory** | 1 wk | 🟠 Medium | Memmap, pagination, UMAP serialization | 10x memory reduction |
| **5: Operations** | 1-2 wks | 🟡 Low | Kubernetes, Prometheus, load testing | Production-ready |

**Total:** 8-10 weeks for full deployment

---

## QUICK IMPLEMENTATION CHECKLIST

### Week 1: Database Foundation
- [ ] Add 6 composite indexes to requirements/constraints/feedback tables
- [ ] Replace N+1 queries with SQLAlchemy `joinedload()`
- [ ] Set up PostgreSQL container in docker-compose.yml
- [ ] Configure connection pooling (QueuePool, pool_size=20)
- [ ] Test migration on staging with 50K sample data

### Week 2-3: Caching & Compute
- [ ] Deploy Redis container for embedding cache
- [ ] Implement content-versioned embedding cache with 30-day TTL
- [ ] Increase SBERT batch_size from 64 → 512
- [ ] Implement UMAP incremental transform (use stored model)
- [ ] Add streaming CSV upload with chunksize=10K

### Week 3-4: Concurrency
- [ ] Replace `run_in_threadpool` with `ProcessPoolExecutor`
- [ ] Refactor LLM provider to async with semaphore-limited concurrency
- [ ] Add 1-second TTL cache to `/progress/{session_id}` endpoint
- [ ] Implement pagination for `/requirements` and `/graph` endpoints

### Week 4-5: Memory & Monitoring
- [ ] Use `np.memmap()` for embedding storage instead of RAM
- [ ] Serialize fitted UMAP models to disk per session
- [ ] Implement incremental constraint conflict detection (Union-Find)
- [ ] Add Prometheus metrics + Grafana dashboard
- [ ] Set up OpenTelemetry distributed tracing

### Week 5-6: Deployment
- [ ] Create Kubernetes deployment manifests (3-node cluster)
- [ ] Build load test suite with 100+ concurrent virtual users
- [ ] Document capacity planning procedure
- [ ] Validate 500K+ requirement handling on staging
- [ ] Production rollout with blue-green deployment

---

## Performance Targets by Phase

```
Metric                | Current | Phase 1 | Phase 3 | Phase 5
──────────────────────┼─────────┼─────────┼─────────┼─────────
Max requirements      | 10K     | 50K     | 100K    | 500K+
Clustering time (10K) | 3 min   | 2 min   | 30 sec  | 10 sec
Concurrent users      | 5       | 15      | 50      | 100+
List endpoint latency | 500ms   | 100ms   | 20ms    | 10ms
Memory/session        | 2.5 GB  | 1.5 GB  | 500 MB  | 200 MB
Database IOPS         | 100     | 500     | 2K      | 10K+
```

---

## Infrastructure Additions

### Docker Compose Additions (Phase 1-2)
```yaml
postgres:
  image: postgres:15-alpine
  ports: ["5432:5432"]
  environment:
    POSTGRES_DB: reqcluster
    POSTGRES_PASSWORD: change_me

redis:
  image: redis:7-alpine
  ports: ["6379:6379"]
  volumes:
    - redis_data:/data
```

### Kubernetes Resources (Phase 5)
- **3-node cluster** (AWS EKS, GCP GKE, or on-prem)
- **PostgreSQL RDS** (managed, multi-AZ for HA)
- **Redis ElastiCache** (managed cluster, 3-node)
- **Prometheus + Grafana** (helm charts)
- **Load balancer** (auto-scale 1-10 backend replicas)

### Cost Estimate
| Component | Monthly |
|-----------|---------|
| PostgreSQL RDS (db.t3.medium) | $80 |
| Redis ElastiCache (cache.t3.small) | $30 |
| Kubernetes cluster (3 t3.medium nodes) | $300 |
| Monitoring (Prometheus + Grafana) | $100 |
| **Total** | **~$500-600/month** |

---

## Success Criteria

✅ **Phase 1 Complete:** 50K requirements, 15 concurrent users, <100ms API latency  
✅ **Phase 3 Complete:** 100K requirements, 50 concurrent users, clustering in <30 sec  
✅ **Phase 5 Complete:** 500K requirements, 100 concurrent users, <50ms p99 latency, production-grade monitoring  

---

## Risk Mitigation

| Risk | Mitigation | Timeline |
|------|-----------|----------|
| Database migration breaks existing data | Test migration on copy; dual-write pattern | Week 1-2 |
| Memory spikes during constraint ops | Add memory watchdog; kill process if >90% | Week 4 |
| LLM API rate limiting | Exponential backoff + queue; 10 req/sec max | Week 3 |
| Stale embedding cache | Version cache keys; auto-invalidate every 30d | Week 2 |
| PostgreSQL index bloat | Schedule VACUUM ANALYZE weekly; monitor size | Week 5 |

---

## File Changes Summary

### Created/Modified Files
- **backend/models/database.py** — Add indexes, migrate to PostgreSQL
- **backend/core/embedding_cache.py** — NEW: Redis-backed cache
- **backend/core/embeddings.py** — Increase batch size
- **backend/core/reduction.py** — Incremental UMAP transform
- **backend/core/feedback_bridge.py** — Union-Find incremental validation
- **backend/core/graph.py** — Adaptive truncation strategy
- **backend/api/routes.py** — Pagination, request coalescing, eager loading
- **backend/llm_services/providers.py** — Async + semaphore limiting
- **backend/main.py** — ProcessPoolExecutor, monitoring setup
- **docker-compose.yml** — Add PostgreSQL, Redis
- **k8s/deployment.yaml** — NEW: Kubernetes manifests
- **tools/load_test_reqcluster.py** — NEW: Locust load test suite
- **monitoring/metrics.py** — NEW: Prometheus metrics

---

## Next Steps

1. **Immediate (This Week):**
   - Review [detailed analysis](/memories/repo/scalability-analysis.md)
   - Set up staging environment with PostgreSQL
   - Audit current database schema for missing indexes

2. **Short Term (2-4 Weeks):**
   - Execute Phase 1 & 2 changes
   - Load test with 50K requirements
   - Benchmark before/after metrics

3. **Medium Term (6-8 Weeks):**
   - Complete Phases 3-4
   - Validate 500K+ handling
   - Prepare Kubernetes infrastructure

4. **Long Term:**
   - Production deployment with monitoring
   - Capacity planning dashboard
   - Regular load testing (monthly)

---

**Last Updated:** 2026-06-06  
**Status:** Analysis Complete → Ready for Phase 1 Implementation  
**Owner:** ReqCluster Engineering Team
