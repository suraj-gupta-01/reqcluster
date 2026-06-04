from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "reqcluster.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

DATABASE_URL = f"sqlite:///{os.path.abspath(DB_PATH)}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="uploaded")  # uploaded, processing, done, error
    total_requirements = Column(Integer, default=0)
    total_clusters = Column(Integer, default=0)
    noise_count = Column(Integer, default=0)


class Requirement(Base):
    __tablename__ = "requirements"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    req_id = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    module = Column(String, nullable=True)
    section = Column(String, nullable=True)
    cluster_id = Column(Integer, nullable=True)
    membership_prob = Column(Float, nullable=True)
    umap_x = Column(Float, nullable=True)
    umap_y = Column(Float, nullable=True)
    is_noise = Column(Boolean, default=False)


class Cluster(Base):
    __tablename__ = "clusters"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    cluster_id = Column(Integer, nullable=False)
    label = Column(String, nullable=False)
    keywords = Column(JSON, nullable=True)
    size = Column(Integer, default=0)


class Graph(Base):
    __tablename__ = "graphs"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, unique=True)
    nodes = Column(JSON, nullable=False)
    edges = Column(JSON, nullable=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)


def reset_stale_sessions():
    """Mark any session left in 'processing' (e.g. from a crash/restart) as
    'error' so it isn't permanently blocked by the in-progress guard."""
    db = SessionLocal()
    try:
        stale = db.query(Session).filter(Session.status == "processing").all()
        for s in stale:
            s.status = "error"
        if stale:
            db.commit()
    finally:
        db.close()
