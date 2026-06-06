# ReqCluster Deployment & Operations Roadmap

> **Detailed analysis:** See `/memories/repo/deployment-analysis.md` for full technical specifications, architecture diagrams, and implementation details.

---

## Current State vs. Goals

```
Current:         Docker Compose local-only   No secrets mgmt   SQLite   No monitoring
Target (Phase 8): Multi-cloud Kubernetes     Encrypted secrets  PostgreSQL  Full observability
                  3 environments (dev/staging/prod)  99.9% uptime   Auto-scaling   24hr RPO
```

---

## 12 DEPLOYMENT BOTTLENECKS

### 🔴 **CRITICAL (Blocks Production)**
1. **No Multi-Environment Support** — dev/staging/prod use identical config
2. **SQLite Database** — Non-networked, no concurrent writes, ephemeral
3. **Plaintext Secrets** — API keys/passwords in .env files
4. **Single-Host Architecture** — No horizontal scaling, no HA failover
5. **No TLS/HTTPS** — All traffic unencrypted
6. **No Persistent Storage** — embeddings/data lost on container restart

### 🟠 **HIGH (Degrades Operations)**
7. **Incomplete CI/CD Pipeline** — Tests run, but no automated deployment
8. **No Monitoring/Logging** — Blind to production issues until user complaints
9. **No Load Balancer** — Cannot distribute traffic across multiple backends
10. **No Backup/Recovery** — No disaster recovery plan or automated backups

### 🟡 **MEDIUM (Increases Operational Burden)**
11. **No Health Checks** — Kubernetes can't auto-recover failed pods
12. **Large Docker Images** — Backend 2GB (includes build tools); slow deploys

---

## 8-PHASE DEPLOYMENT ROADMAP

| Phase | Duration | Effort | Key Changes | Go-Live |
|-------|----------|--------|-------------|---------|
| **1: Containerization** | 1 wk | 🟡 Low | Multi-stage builds, registry | Week 1 |
| **2: Secrets Mgmt** | 1 wk | 🟠 Medium | Remove plaintext secrets | Week 2 |
| **3: Database & Persistence** | 1 wk | 🟠 Medium | PostgreSQL, migrations, volumes | Week 3 |
| **4: Kubernetes** | 2 wks | 🔴 High | K8s manifests, deployments, auto-scaling | Week 5 |
| **5: TLS/Ingress** | 1 wk | 🟠 Medium | HTTPS, certificates, public DNS | Week 6 |
| **6: Monitoring & Logging** | 2 wks | 🟠 Medium | Prometheus, alerts, ELK, dashboards | Week 8 |
| **7: CI/CD Pipeline** | 1 wk | 🟠 Medium | Automated builds, deployments, GitOps | Week 9 |
| **8: Backup & DR** | 1 wk | 🟡 Low | Backup automation, recovery procedures | Week 10 |

**Total:** 10 weeks → Production-ready

---

## QUICK IMPLEMENTATION CHECKLIST

### Week 1: Containerization & Registry
- [ ] Convert Dockerfile.backend to multi-stage build (2GB → 600MB)
- [ ] Set up container registry (Docker Hub / ECR / GCR)
- [ ] Add .dockerignore to reduce image size
- [ ] Configure GitHub Actions to build & push images on every commit
- [ ] Test local builds work with pushed images

### Week 2: Secrets & Configuration
- [ ] Remove plaintext secrets from docker-compose.yml
- [ ] Set up Kubernetes Secrets (or AWS Secrets Manager)
- [ ] Create ConfigMaps for dev/staging/prod environments
- [ ] Implement secret rotation policy (90-day keys)
- [ ] Document secret access procedures

### Week 3: Database & Persistence
- [ ] Migrate from SQLite → PostgreSQL (docker-compose first)
- [ ] Set up Alembic for database migrations
- [ ] Create backup CronJob (daily PostgreSQL dumps)
- [ ] Configure persistent volumes for embeddings
- [ ] Test backup/restore procedure

### Week 4-5: Kubernetes Deployment
- [ ] Create k8s/backend-deployment.yaml (3 replicas, auto-scale to 10)
- [ ] Create k8s/frontend-deployment.yaml (2 replicas)
- [ ] Set up Services for internal routing
- [ ] Create StatefulSet for PostgreSQL (or use RDS)
- [ ] Test rolling updates (zero-downtime)
- [ ] Validate pod auto-recovery on failure

### Week 6: TLS & Ingress
- [ ] Install Cert-Manager for automatic certificate renewal
- [ ] Install Nginx Ingress Controller
- [ ] Set up Let's Encrypt issuer (staging + prod)
- [ ] Create Ingress resource with TLS
- [ ] Configure DNS (Route 53 / Cloud DNS)
- [ ] Validate HTTPS works; test auto-renewal

### Week 7-8: Monitoring & Logging
- [ ] Install Prometheus + Grafana (helm charts)
- [ ] Create alerting rules (high error rate, pod crashes, high latency)
- [ ] Install Fluent Bit daemonset for log collection
- [ ] Set up Elasticsearch + Kibana (or use managed service)
- [ ] Create Grafana dashboards for key metrics
- [ ] Test alert firing (PagerDuty integration)

### Week 9: CI/CD Pipeline
- [ ] Update GitHub Actions: test → build → push images
- [ ] Create k8s/deploy.yml with image updates
- [ ] Set up ArgoCD for GitOps (optional but recommended)
- [ ] Test automated deployments to staging
- [ ] Implement canary rollout strategy (10% → 50% → 100%)
- [ ] Test automatic rollback on failure

### Week 10: Backup & Disaster Recovery
- [ ] Create CronJob for daily database backups
- [ ] Upload backups to S3 (cross-region)
- [ ] Create CronJob for embedding cache backup (weekly)
- [ ] Document recovery procedure
- [ ] Run monthly disaster recovery drill
- [ ] Set RTO (1 hour) and RPO (24 hours) targets

---

## ENVIRONMENT SPECIFICATIONS

### Development (Local Docker Compose)
```yaml
- Backend: 1 instance, 512MB RAM
- Frontend: 1 instance, 128MB RAM
- PostgreSQL: 1 container (non-persistent)
- Redis: 1 container
- No monitoring
- Health checks every 15s
```

### Staging (Kubernetes, 2 nodes)
```yaml
- Backend: 2 replicas, auto-scale to 5
- Frontend: 1 replica
- PostgreSQL: Managed (daily backups)
- Redis: 3-node cluster
- Prometheus + Grafana (7-day retention)
- Let's Encrypt staging certificates
```

### Production (Kubernetes, 5+ nodes)
```yaml
- Backend: 5 replicas, auto-scale to 15
- Frontend: 2 replicas
- PostgreSQL: RDS (multi-AZ, automated failover)
- Redis: ElastiCache (3-node cluster)
- Full monitoring + alerts (30-day retention)
- ELK stack (Elasticsearch + Kibana)
- Let's Encrypt production certificates
- Daily backups → S3 (cross-region)
- On-call alerting (PagerDuty)
```

---

## Infrastructure Cost Estimate

| Component | Staging | Production | Notes |
|-----------|---------|-----------|-------|
| Kubernetes cluster | $60/mo | $600/mo | 2 nodes vs 5 nodes |
| PostgreSQL RDS | $35/mo | $500/mo | db.t3.small vs db.t3.large |
| Redis ElastiCache | $20/mo | $150/mo | single vs 3-node cluster |
| Load Balancer | — | $20/mo | Production only |
| Monitoring | $5/mo | $200/mo | Prometheus vs Datadog |
| Backup storage (S3) | $5/mo | $30/mo | Cold storage |
| DNS/TLS | — | $20/mo | Let's Encrypt free; Route 53 paid |
| **TOTAL** | **~$125/mo** | **~$1,520/mo** | Production enterprise-grade |

---

## Deployment Architecture

```
Clients (HTTPS)
     ↓
Load Balancer (TLS termination)
     ↓
Kubernetes Cluster (3-5 nodes)
  ├─ Backend Pods (3-15 replicas, auto-scaling)
  ├─ Frontend Pods (2 replicas)
  ├─ PostgreSQL StatefulSet / RDS
  ├─ Redis Cluster
  ├─ Prometheus + Grafana (monitoring)
  ├─ Fluent Bit → Elasticsearch (logging)
  └─ ArgoCD (GitOps, auto-deployment)
     ↓
Persistent Volumes (EBS/PD/Azure Disk)
  ├─ PostgreSQL data (100GB+)
  ├─ Embeddings cache (500GB+)
  └─ Backup snapshots (1TB+)
```

---

## Success Criteria by Phase

### ✅ Phase 1-2 (Week 1-2)
- Images in container registry
- No plaintext secrets in git
- CI pipeline builds on every commit

### ✅ Phase 3 (Week 3)
- PostgreSQL running (staging + local)
- Database migrations working
- Backup automation operational

### ✅ Phase 4-5 (Week 4-6)
- Kubernetes cluster deployed
- 3-5 replicas auto-scaling
- HTTPS working
- Zero-downtime deployments

### ✅ Phase 6-8 (Week 7-10)
- Full monitoring + alerting
- Disaster recovery tested
- **Production-ready**
- Support 100+ concurrent users
- 99.9% uptime SLA

---

## Deployment Strategies

### Rolling Updates (Default)
```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1        # Launch 1 new pod before killing old
    maxUnavailable: 0  # No downtime
```
**Result:** Smooth zero-downtime updates; gradual traffic shift.

### Canary (Safer)
```yaml
steps:
- setWeight: 10%  # Route 10% to new version
- pause: 5m       # Monitor for errors
- setWeight: 50%
- pause: 5m
- setWeight: 100%
```
**Result:** Detect issues early; automatic rollback on error.

### Blue-Green (Fastest Rollback)
```yaml
# Run 2 complete deployments; switch traffic instantly
- Production (Blue): v1.0.0 (active)
- Staging (Green): v1.0.1 (ready)
# Switch: traffic → Green
# Rollback: traffic → Blue
```
**Result:** Instant rollback; full environment available for testing.

---

## Monitoring & Alerting

### Key Metrics to Monitor
- **Application:**
  - Request latency (p50/p95/p99)
  - Error rate (5xx errors)
  - Active clustering jobs
  - API throughput (req/sec)

- **Infrastructure:**
  - CPU/Memory usage per pod
  - Disk I/O (database reads/writes)
  - Network I/O (ingress/egress)
  - Database connection pool usage

- **Business:**
  - Active users (sessions)
  - Clustering jobs completed
  - Requirement records processed
  - Feature usage (enrichment, suggestions)

### Alert Thresholds
```
🔴 CRITICAL:
  - Error rate > 1% → Page on-call
  - Latency p99 > 10 sec → Page on-call
  - Pod crash loop → Page on-call
  - Database connection pool > 95% → Page on-call

🟠 HIGH:
  - Error rate > 0.1% → Send Slack
  - Latency p95 > 5 sec → Send Slack
  - Memory usage > 85% → Send Slack
  - Disk usage > 80% → Send Slack
```

---

## Disaster Recovery

### Backup Strategy
- **Daily:** PostgreSQL full backup → S3 (cross-region)
- **Weekly:** Embedding cache backup → S3
- **RPO (Recovery Point Objective):** 24 hours
- **RTO (Recovery Time Objective):** 1 hour

### Recovery Procedure
```bash
# 1. Identify latest backup
aws s3 ls s3://backups/reqcluster/ | sort | tail -1

# 2. Restore database
gunzip < reqcluster-20260606.sql.gz | \
  psql -h postgres -U postgres -d reqcluster

# 3. Restore embeddings
aws s3 sync s3://backups/embeddings/ /embeddings/

# 4. Restart pods
kubectl rollout restart deployment/reqcluster-backend

# 5. Verify
kubectl exec pod/reqcluster-backend-xyz -- \
  curl http://localhost:8000/health
```
**Time:** ~45 minutes total

---

## Pre-Launch Checklist

### Pre-Production (Staging)
- [ ] Load test: 50 concurrent users, 10K requirements
- [ ] Test database backup/restore cycle
- [ ] Test pod auto-recovery (kill pod, verify restart)
- [ ] Test rolling update (verify zero downtime)
- [ ] Monitor 24 hours (check all alerts fire correctly)
- [ ] Document runbooks for on-call team
- [ ] Conduct disaster recovery drill

### Production Launch
- [ ] Set up production Kubernetes cluster
- [ ] Configure load balancer + DNS
- [ ] Deploy with blue-green strategy
- [ ] Monitor closely for 24 hours
- [ ] Set up on-call rotation (PagerDuty)
- [ ] Brief team on runbooks + escalation

### Post-Launch (First Month)
- [ ] Daily monitoring review
- [ ] Weekly capacity planning meeting
- [ ] Biweekly disaster recovery drill
- [ ] Monthly cost optimization review
- [ ] Security audit (container scanning, OWASP)

---

## Key Files & Locations

### New Kubernetes Manifests
```
k8s/
├── backend-deployment.yaml      # Backend service definition
├── frontend-deployment.yaml     # Frontend service definition
├── service.yaml                 # Internal routing
├── ingress.yaml                 # Public routing + TLS
├── secrets.yaml                 # Encrypted secrets
├── configmap-dev.yaml           # Dev environment config
├── configmap-prod.yaml          # Production config
├── postgres-statefulset.yaml    # PostgreSQL (optional)
├── cronjob-backup.yaml          # Daily DB backup
├── cronjob-embedding-backup.yaml # Weekly embedding backup
├── prometheus-rules.yaml        # Alerting rules
├── hpa.yaml                     # Horizontal Pod Autoscaler
└── pvc.yaml                     # Persistent volumes
```

### Modified Files
```
Dockerfile.backend              # Multi-stage build
Dockerfile.frontend             # (Already optimized)
docker-compose.yml              # Add PostgreSQL, Redis
.github/workflows/ci.yml        # Add build/push/deploy steps
.github/workflows/deploy.yml    # Automated deployments
backend/main.py                 # Add metrics endpoint
backend/models/database.py      # PostgreSQL URL
backend/alembic/                # Database migrations
backend/monitoring/metrics.py   # Prometheus metrics
```

---

## Next Steps

1. **This Week:**
   - Review [detailed deployment analysis](/memories/repo/deployment-analysis.md)
   - Set up local PostgreSQL via docker-compose
   - Create container registry account

2. **Next 2 Weeks:**
   - Complete Phase 1-2 (containerization, secrets)
   - Test multi-stage backend image build
   - Implement secret management

3. **Weeks 3-5:**
   - Complete Phase 3-4 (database, Kubernetes)
   - Deploy staging cluster
   - Load test staging

4. **Weeks 6-10:**
   - Complete Phase 5-8 (TLS, monitoring, CI/CD, backup)
   - Production deployment
   - Go live!

---

**Last Updated:** 2026-06-06  
**Status:** Analysis Complete → Ready for Phase 1 Implementation  
**Target Go-Live:** Week 10  
**Success Metric:** Production deployment supporting 100+ concurrent users, 99.9% uptime SLA
