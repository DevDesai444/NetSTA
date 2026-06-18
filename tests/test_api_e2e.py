"""End-to-end FastAPI smoke tests via TestClient."""

from fastapi.testclient import TestClient

from netsta.api import app


client = TestClient(app)


def test_health_reports_status():
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "checkpoint_present" in data
    assert "lora_active" in data


def test_topologies_lists_analog():
    r = client.get("/api/topologies")
    assert r.status_code == 200
    data = r.json()
    assert "two_stage_opamp" in data["analog"]


def test_diagnose_digital_returns_report():
    r = client.post("/api/diagnose", json={"kind": "digital", "gates": 20, "seed": 7})
    assert r.status_code == 200
    data = r.json()
    assert "graph" in data
    assert "report" in data
    assert data["report"]["backend"] in ("deterministic", "autogen")
    assert isinstance(data["report"]["bottlenecks"], list)
    # graph has node + edge views
    assert "nodes" in data["graph"]
    assert "edges" in data["graph"]


def test_diagnose_analog_two_stage_opamp():
    r = client.post("/api/diagnose",
                    json={"kind": "analog", "topology": "two_stage_opamp", "seed": 7})
    assert r.status_code == 200
    data = r.json()
    assert data["circuit_name"] == "two_stage_opamp"


def test_diagnose_rejects_unknown_topology():
    r = client.post("/api/diagnose",
                    json={"kind": "analog", "topology": "not_a_real_topology"})
    assert r.status_code == 400


def test_diagnose_nl_requires_query():
    r = client.post("/api/diagnose", json={"kind": "nl", "query": ""})
    assert r.status_code == 400
