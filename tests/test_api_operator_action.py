import json

from fastapi.testclient import TestClient

from garuda_edge.api.server import app


def test_allowed_operator_action_is_logged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    response = TestClient(app).post("/operator-action", json={"event_id": "E1", "action": "request_recheck"})
    assert response.status_code == 200
    logged = [json.loads(line) for line in (tmp_path / "data/outputs/operator_actions.jsonl").read_text().splitlines()]
    assert logged[0]["action"] == "request_recheck"


def test_unsupported_or_unsafe_action_is_rejected_and_not_logged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    for action in ("engage_target", "launch_strike", "made_up_action"):
        response = client.post("/operator-action", json={"event_id": "E1", "action": action})
        assert response.status_code == 422
        assert "request_recheck" in response.json()["detail"]["allowed_actions"]
    assert not (tmp_path / "data/outputs/operator_actions.jsonl").exists()
