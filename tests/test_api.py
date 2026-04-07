"""FastAPI endpoint tests for the EmailTriage server.

Uses FastAPI TestClient to exercise /reset, /step, /state, /health.
Requirements: 8.4, 8.5, 8.6
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from email_triage.server import app

client = TestClient(app)


def _get_first_email_id() -> str:
    """Helper: reset and return the current email id from the observation."""
    resp = client.post("/reset", json={})
    assert resp.status_code == 200
    return resp.json()["current_email"]["id"]


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------

def test_reset_no_body_returns_observation():
    resp = client.post("/reset", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "current_email" in data
    assert "inbox_summary" in data
    assert "step" in data
    assert data["step"] == 0


def test_reset_with_task_and_seed():
    resp = client.post("/reset", json={"task": "medium", "seed": 7})
    assert resp.status_code == 200
    data = resp.json()
    assert data["inbox_summary"]["total"] == 20


def test_reset_easy_task_inbox_size():
    resp = client.post("/reset", json={"task": "easy", "seed": 1})
    assert resp.status_code == 200
    assert resp.json()["inbox_summary"]["total"] == 10


def test_reset_hard_task_inbox_size():
    resp = client.post("/reset", json={"task": "hard", "seed": 1})
    assert resp.status_code == 200
    assert resp.json()["inbox_summary"]["total"] == 30


def test_reset_same_seed_same_email():
    r1 = client.post("/reset", json={"task": "easy", "seed": 99})
    r2 = client.post("/reset", json={"task": "easy", "seed": 99})
    assert r1.json()["current_email"]["id"] == r2.json()["current_email"]["id"]


# ---------------------------------------------------------------------------
# /step
# ---------------------------------------------------------------------------

def test_step_valid_action_returns_step_response():
    email_id = _get_first_email_id()
    resp = client.post("/step", json={
        "action_type": "skip",
        "target_email_id": email_id,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "observation" in data
    assert "reward" in data
    assert "done" in data
    assert "info" in data


def test_step_reward_value_in_range():
    email_id = _get_first_email_id()
    resp = client.post("/step", json={
        "action_type": "skip",
        "target_email_id": email_id,
    })
    assert resp.status_code == 200
    reward_value = resp.json()["reward"]["value"]
    assert 0.0 <= reward_value <= 1.0


def test_step_done_is_bool():
    email_id = _get_first_email_id()
    resp = client.post("/step", json={
        "action_type": "skip",
        "target_email_id": email_id,
    })
    assert isinstance(resp.json()["done"], bool)


def test_step_invalid_action_type_returns_422():
    resp = client.post("/step", json={
        "action_type": "not_a_real_action",
        "target_email_id": "email-001",
    })
    assert resp.status_code == 422


def test_step_wrong_email_id_returns_error_in_info():
    _get_first_email_id()  # ensure env is reset
    resp = client.post("/step", json={
        "action_type": "skip",
        "target_email_id": "nonexistent-id-xyz",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data["info"]
    assert data["reward"]["value"] == 0.0
    assert data["done"] is False


def test_step_categorize_missing_category_returns_error():
    email_id = _get_first_email_id()
    resp = client.post("/step", json={
        "action_type": "categorize",
        "target_email_id": email_id,
        # category intentionally omitted
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data["info"]
    assert data["reward"]["value"] == 0.0


# ---------------------------------------------------------------------------
# /state
# ---------------------------------------------------------------------------

def test_state_returns_dict_with_expected_keys():
    client.post("/reset", json={})
    resp = client.get("/state")
    assert resp.status_code == 200
    data = resp.json()
    for key in ("task_name", "seed", "step_count", "max_steps", "current_index",
                "inbox_size", "inbox", "actions_taken", "episode_actions"):
        assert key in data, f"Missing key: {key}"


def test_state_step_count_increments():
    client.post("/reset", json={"task": "easy", "seed": 1})
    state_before = client.get("/state").json()
    email_id = state_before["inbox"][0]["id"]
    client.post("/step", json={"action_type": "skip", "target_email_id": email_id})
    state_after = client.get("/state").json()
    assert state_after["step_count"] == state_before["step_count"] + 1
