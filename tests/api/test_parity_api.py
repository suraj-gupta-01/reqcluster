import numpy as np

from reqcluster.ml import refine

from .test_sessions_api import TOPICS


def _upload(client, n=30, extra=True):
    rows = ["id,text,module,section"] + [
        f"R{i},The {TOPICS[i % 3]} subsystem shall report code {i}"
        + (f" as defined in R{i - 1}" if i else "") + f",M{i % 3},S{i}" for i in range(n)]
    r = client.post("/api/sessions", files={"file": ("r.csv", "\n".join(rows).encode())})
    return r.json()["session"]["id"]


def test_attributes_and_csv_export(client):
    sid = _upload(client)
    client.post(f"/api/sessions/{sid}/cluster")
    reqs = client.get(f"/api/sessions/{sid}/requirements").json()
    assert reqs[0]["attributes"] == {"module": "M0", "section": "S0"}
    csv_text = client.get(f"/api/sessions/{sid}/export/csv").content.decode("utf-8-sig")
    assert csv_text.splitlines()[0].endswith(",module,section")


def test_reports_in_metrics(client):
    sid = _upload(client)
    client.post(f"/api/sessions/{sid}/enrich", json={})
    client.post(f"/api/sessions/{sid}/cluster",
                json={"enable_embedding_comparison": True, "run_ablation": True})
    m = client.get(f"/api/sessions/{sid}/metrics").json()
    assert m["embedding_comparison"] and m["ablation_report"]["base"]
    sid2 = _upload(client)
    client.post(f"/api/sessions/{sid2}/cluster")
    m = client.get(f"/api/sessions/{sid2}/metrics").json()
    assert m["embedding_comparison"] is None and m["ablation_report"] is None


def test_enrichment_summary_and_quality(client):
    sid = _upload(client, 6)
    assert client.get(f"/api/sessions/{sid}/enrichment/summary").status_code == 409
    client.post(f"/api/sessions/{sid}/enrich", json={})
    assert "length_ratio" in client.get(f"/api/sessions/{sid}/enrichment").json()[0]["quality"]
    body = client.get(f"/api/sessions/{sid}/enrichment/summary").json()
    assert body["quality_report"]["total"] == 6 and body["domain_vocabulary"]


def test_dependency_rationale_and_coherence(client):
    sid = _upload(client)
    client.post(f"/api/sessions/{sid}/cluster")
    client.post(f"/api/sessions/{sid}/dependencies", json={"sim_threshold": 0.0})
    body = client.get(f"/api/sessions/{sid}/dependencies").json()
    rat = body["rationale"]
    n_clusters = len(client.get(f"/api/sessions/{sid}/clusters").json())
    assert len(rat["grouping"]) == n_clusters
    assert len(rat["dependencies"]) == len(body["edges"]) > 0
    coh = client.get(f"/api/sessions/{sid}/coherence").json()
    assert coh and {"cluster_id", "coherence_score", "assessment"} <= set(coh[0])


def test_score_all_clusters_tight_beats_scattered():
    rng = np.random.default_rng(0)
    tight = np.array([1.0, 0, 0, 0]) + rng.normal(0, 0.01, (10, 4))
    loose = rng.normal(0, 1, (10, 4))
    X = np.vstack([tight, loose])
    out = refine.score_all_clusters(X, np.array([0] * 10 + [1] * 10), {0: ["a"]})
    assert out[0]["coherence_score"] > out[1]["coherence_score"] and out[0]["top_keywords"] == ["a"]


def test_metrics_record_refinement_analysis(client):
    # the UI said "No suggestions yet" after an analysis that found nothing to improve
    sid = _upload(client)
    client.post(f"/api/sessions/{sid}/cluster")
    assert client.get(f"/api/sessions/{sid}/metrics").json()["refinement_report"] is None
    client.post(f"/api/sessions/{sid}/suggestions", json={"top_n_merges": 0, "top_n_splits": 0})
    rep = client.get(f"/api/sessions/{sid}/metrics").json()["refinement_report"]
    assert rep["n_merges"] == 0 and rep["n_splits"] == 0 and rep["edit_seq"] == 0
