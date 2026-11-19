from .test_feedback_api import setup


def test_failed_rerun_keeps_revision(client):
    sid, _, _ = setup(client)
    u = f"/api/sessions/{sid}"
    jid = client.post(f"{u}/cluster").json()["id"]
    # Simulate the re-run failing after revision 1 exists.
    from reqcluster.db import session_scope
    from reqcluster.models import JobRow
    with session_scope() as s:
        s.get(JobRow, jid).status = "failed"
    body = client.get(u).json()
    assert body["status"] == "error" and body["revision"] >= 1
    assert body["last_job"]["status"] == "failed"
