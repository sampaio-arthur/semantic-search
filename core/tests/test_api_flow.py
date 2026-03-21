from __future__ import annotations

import importlib
import os
import pytest
from fastapi.testclient import TestClient


def _boot_app():
    test_database_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not test_database_url:
        pytest.skip("TEST_DATABASE_URL or DATABASE_URL must be configured for PostgreSQL integration test.")
    os.environ["DATABASE_URL"] = test_database_url
    os.environ["JWT_SECRET"] = "test-secret"
    os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "60"

    import infrastructure.config as config_module
    import infrastructure.db.session as session_module

    config_module.get_settings.cache_clear()
    session_module._engine = None
    session_module._SessionLocal = None

    import infrastructure.api.fastapi_app as app_module

    importlib.reload(app_module)
    return app_module.app


def _auth_headers(client: TestClient) -> dict[str, str]:
    register = client.post("/auth/register", json={"email": "user@example.com", "password": "12345678"})
    assert register.status_code == 201, register.text

    login = client.post("/auth/login", data={"username": "user@example.com", "password": "12345678"})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_minimal_required_flow():
    app = _boot_app()
    client = TestClient(app)
    headers = _auth_headers(client)

    chat = client.post("/api/chats", json={"title": "Teste"}, headers=headers)
    assert chat.status_code == 201, chat.text
    chat_id = chat.json()["id"]

    index_resp = client.post("/api/index", json={"dataset_id": "beir/trec-covid"}, headers=headers)
    assert index_resp.status_code == 200, index_resp.text

    search_resp = client.post(
        "/api/search",
        json={"dataset_id": "beir/trec-covid", "query": "trade tariffs imports exports", "mode": "compare", "chat_id": chat_id},
        headers=headers,
    )
    assert search_resp.status_code == 200, search_resp.text
    payload = search_resp.json()
    assert payload.get("chat_persisted") is True
    assert payload["comparison"]["classical"]["results"]
    assert payload["comparison"]["quantum"]["results"]
    assert payload["comparison"]["statistical"]["results"]

    chat_detail = client.get(f"/api/chats/{chat_id}", headers=headers)
    assert chat_detail.status_code == 200, chat_detail.text
    assert any(msg["role"] == "assistant" for msg in chat_detail.json()["messages"])

    first_doc_id = payload["comparison"]["classical"]["results"][0]["doc_id"]
    gt_resp = client.post(
        "/api/ground-truth",
        json={
            "dataset_id": "beir/trec-covid",
            "query_id": "q-trade",
            "query_text": "trade tariffs imports exports",
            "relevant_doc_ids": [first_doc_id],
        },
        headers=headers,
    )
    assert gt_resp.status_code == 200, gt_resp.text

    eval_resp = client.post("/api/evaluate", json={"dataset_id": "beir/trec-covid", "pipeline": "compare", "k": 25}, headers=headers)
    assert eval_resp.status_code == 200, eval_resp.text
    eval_payload = eval_resp.json()
    assert len(eval_payload["pipelines"]) == 3
