"""Tests for certus.proxy.gateway (requires the `proxy` extra: FastAPI + httpx)."""

from __future__ import annotations

import threading
import time

import pytest
from pydantic import BaseModel

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from certus.core.policy import PolicyEngine  # noqa: E402
from certus.proxy.approval import (  # noqa: E402
    ApprovalManager,
    PendingApprovalStore,
    WebhookApprovalCallback,
)
from certus.proxy.gateway import create_app  # noqa: E402
from certus.proxy.middleware import CertusGuard  # noqa: E402

POLICY = {
    "version": 1,
    "default_action": "deny",
    "allowlist": ["get_weather"],
    "critical_tools": [
        {
            "id": "critical-delete",
            "tools": ["delete_file"],
            "risk_level": "high",
            "requires_approval": True,
            "reason": "Deletion requires approval.",
        }
    ],
}


class WeatherArgs(BaseModel):
    city: str


class DeleteFileArgs(BaseModel):
    path: str


def test_healthz():
    guard = CertusGuard(policy_engine=PolicyEngine.from_dict(POLICY))
    client = TestClient(create_app(guard))

    response = client.get("/v1/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_executes_registered_handler():
    guard = CertusGuard(policy_engine=PolicyEngine.from_dict(POLICY))
    guard.register_tool("get_weather", WeatherArgs, lambda city: f"sunny in {city}")
    client = TestClient(create_app(guard))

    response = client.post(
        "/v1/tool-calls", json={"name": "get_weather", "arguments": {"city": "Izmir"}}
    )

    body = response.json()
    assert response.status_code == 200
    assert body == {
        "status": "executed",
        "ok": True,
        "result": "sunny in Izmir",
        "risk_level": None,
        "requires_approval": None,
        "approved": None,
        "reason": None,
    }


def test_decision_only_mode_for_unregistered_tool():
    guard = CertusGuard(policy_engine=PolicyEngine.from_dict(POLICY))
    guard.schema_validator.register_schema("get_weather", WeatherArgs)
    client = TestClient(create_app(guard))

    response = client.post(
        "/v1/tool-calls", json={"name": "get_weather", "arguments": {"city": "Izmir"}}
    )

    body = response.json()
    assert body["status"] == "decided"
    assert body["ok"] is True
    assert body["requires_approval"] is False


def test_approval_decision_endpoint_unblocks_pending_call():
    store = PendingApprovalStore()
    callback = WebhookApprovalCallback(store=store, notify=lambda request: None, timeout=5.0)
    guard = CertusGuard(
        policy_engine=PolicyEngine.from_dict(POLICY),
        approval_manager=ApprovalManager(callback=callback),
    )
    guard.register_tool("delete_file", DeleteFileArgs, lambda path: f"deleted {path}")
    client = TestClient(create_app(guard, approval_store=store))

    results: dict[str, object] = {}

    def call_delete_file():
        results["response"] = client.post(
            "/v1/tool-calls", json={"name": "delete_file", "arguments": {"path": "a.txt"}}
        )

    thread = threading.Thread(target=call_delete_file)
    thread.start()
    time.sleep(0.1)  # let the tool-call request register itself in the store first

    decision_response = client.post(
        "/v1/approvals/pending-lookup/decision", json={"approved": True}
    )
    assert decision_response.status_code == 404  # wrong id: nothing pending under this name

    # Resolve every currently-pending request id directly via the store, since
    # the real id is generated server-side and not returned until the call finishes.
    with store._lock:  # test-only introspection of pending ids
        pending_ids = list(store._events.keys())
    assert len(pending_ids) == 1
    ok_response = client.post(
        f"/v1/approvals/{pending_ids[0]}/decision",
        json={"approved": True, "approver": "ops-team"},
    )
    assert ok_response.status_code == 200

    thread.join(timeout=5)
    tool_call_response = results["response"]
    body = tool_call_response.json()  # type: ignore[attr-defined]
    assert body["status"] == "executed"
    assert body["result"] == "deleted a.txt"


def test_approval_decision_endpoint_404_without_store_configured():
    guard = CertusGuard(policy_engine=PolicyEngine.from_dict(POLICY))
    client = TestClient(create_app(guard))  # no approval_store passed

    response = client.post("/v1/approvals/some-id/decision", json={"approved": True})

    assert response.status_code == 404
