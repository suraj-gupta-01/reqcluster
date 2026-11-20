import pytest
from sqlalchemy import select

from reqcluster import db
from reqcluster.models import AssignmentRow, JobRow, RequirementRow, RunRow, SessionRow


def _populate(s):
    s.add(SessionRow(id=1, name="a"))
    s.flush()
    s.add(RequirementRow(id=1, session_id=1, ordinal=0, req_id="R1", text="t"))
    s.add(RunRow(id=1, session_id=1, revision=1, kind="cluster"))
    s.flush()
    s.add(AssignmentRow(run_id=1, requirement_id=1, cluster_id=-1))
    s.add(JobRow(id="j" * 32, session_id=1, kind="cluster", status="queued"))


def test_cascade_delete():
    db.init_db()
    with db.session_scope() as s:
        _populate(s)
    with db.session_scope() as s:
        s.delete(s.get(SessionRow, 1))
    with db.session_scope() as s:
        for m in (RequirementRow, RunRow, AssignmentRow, JobRow):
            assert s.scalars(select(m)).all() == []


def test_rollback_on_error():
    db.init_db()
    with pytest.raises(RuntimeError), db.session_scope() as s:
        s.add(SessionRow(id=1, name="x"))
        s.flush()
        raise RuntimeError
    with db.session_scope() as s:
        assert s.get(SessionRow, 1) is None


def test_init_db_refuses_a_database_from_an_older_schema(tmp_path):
    # an older build's DB lacked new columns: create_all never adds them, so every query 500'd
    import sqlite3

    import pytest

    from reqcluster import db

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, session_id INTEGER)")  # missing columns
    con.commit()
    con.close()
    db.configure(f"sqlite:///{path.as_posix()}")
    with pytest.raises(RuntimeError, match="runs.*(max_cluster_id|revision)"):
        db.init_db()
