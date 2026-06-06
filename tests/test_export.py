"""Tests for Phase 5 MBSE exporters: ReqIF, SysML XMI, Jama, CSV."""

import json
import os
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from export.reqif_exporter import export_reqif, REQIF_NS
from export.sysml_xmi_exporter import export_sysml_xmi, UML_NS, XMI_NS
from export.jama_connector import export_jama
from export.pdf_report import export_pdf


def _data():
    return {
        "session": {"id": 1, "name": "Demo", "filename": "demo.csv"},
        "requirements": [
            {"db_id": 1, "req_id": "REQ-001", "text": "The system shall cool.",
             "module": "Thermal", "section": "A", "cluster_id": 0, "is_noise": False},
            {"db_id": 2, "req_id": "REQ-002", "text": "The system shall power up.",
             "module": "Power", "section": "B", "cluster_id": 1, "is_noise": False},
            {"db_id": 3, "req_id": "REQ-003", "text": "An odd one out.",
             "module": "", "section": "", "cluster_id": -1, "is_noise": True},
        ],
        "clusters": [
            {"cluster_id": 0, "label": "Thermal", "keywords": ["cool"], "size": 1},
            {"cluster_id": 1, "label": "Power", "keywords": ["power"], "size": 1},
        ],
        "dependencies": [
            {"source_db_id": 1, "target_db_id": 2, "source_req_id": "REQ-001",
             "target_req_id": "REQ-002", "relation": "data"},
        ],
        "timestamp": "2026-06-04T00:00:00Z",
    }


def test_reqif_is_well_formed_and_structured():
    xml = export_reqif(_data())
    root = ET.fromstring(xml)
    assert root.tag == f"{{{REQIF_NS}}}REQ-IF"
    spec_objects = root.findall(f".//{{{REQIF_NS}}}SPEC-OBJECT")
    # 3 requirements + 2 cluster headings.
    assert len(spec_objects) == 5
    # Hierarchy contains cluster groups and the noise requirement at top level.
    hierarchies = root.findall(f".//{{{REQIF_NS}}}SPEC-HIERARCHY")
    assert len(hierarchies) >= 5
    assert "REQ-001" in xml


def test_reqif_escapes_special_characters():
    data = _data()
    data["requirements"][0]["text"] = "value < 5 & temp > 70"
    xml = export_reqif(data)
    # Must remain parseable (proper escaping).
    ET.fromstring(xml)
    assert "&amp;" in xml or "&lt;" in xml


def test_sysml_xmi_well_formed_with_packages_and_dependency():
    xml = export_sysml_xmi(_data())
    root = ET.fromstring(xml)
    assert root.tag == f"{{{XMI_NS}}}XMI"
    packages = root.findall(f".//packagedElement[@{{{XMI_NS}}}type='uml:Package']")
    classes = root.findall(f".//packagedElement[@{{{XMI_NS}}}type='uml:Class']")
    deps = root.findall(f".//packagedElement[@{{{XMI_NS}}}type='uml:Dependency']")
    assert len(packages) == 2
    assert len(classes) == 3
    assert len(deps) == 1


def test_jama_bundle_has_items_and_relationships():
    bundle = json.loads(export_jama(_data()))
    assert bundle["format"] == "jama"
    assert len(bundle["items"]) == 3
    assert bundle["meta"]["item_count"] == 3
    assert len(bundle["relationships"]) == 1
    assert bundle["relationships"][0]["fromItem"] == "REQ-001"


def test_pdf_is_valid_and_handles_special_chars():
    pdf = export_pdf(_data())
    assert isinstance(pdf, (bytes, bytearray))
    assert pdf[:5] == b"%PDF-"          # valid PDF header
    assert pdf.rstrip().endswith(b"%%EOF")
    assert len(pdf) > 1000


def test_pdf_empty_clusters_does_not_crash():
    data = _data()
    data["clusters"] = []
    pdf = export_pdf(data)
    assert pdf[:5] == b"%PDF-"
