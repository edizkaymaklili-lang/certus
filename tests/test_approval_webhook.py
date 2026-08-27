"""Tests for the webhook-based approval channel (certus.proxy.approval).

`slack_webhook_notifier`/`generic_json_webhook_notifier` are exercised against
a real local HTTP server (not a mocked `urlopen`) so the test proves an actual
HTTP POST is delivered, not just that a function was called.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from certus.core.models import PolicyDecision, RiskLevel, ToolCall
from certus.proxy.approval import (
    ApprovalManager,
    ApprovalRequest,
    ApprovalResponse,
    PendingApprovalStore,
    WebhookApprovalCallback,
    generic_json_webhook_notifier,
    slack_webhook_notifier,
)


class _CapturingHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that records the last POST body it received."""

    received: list[bytes] = []

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming convention
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _CapturingHandler.received.append(body)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args: object) -> None:  # silence test output
        pass


@pytest.fixture
def local_webhook_server():
    _CapturingHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def make_approval_request() -> ApprovalRequest:
    return ApprovalRequest(
        tool_call=ToolCall(name="delete_file", arguments={"path": "a.txt"}),
        decision=PolicyDecision(
            allowed=True,
            requires_approval=True,
            risk_level=RiskLevel.HIGH,
            rule_id="critical-delete",
            reason="Deletion requires approval.",
        ),
    )


def test_slack_webhook_notifier_delivers_real_http_post(local_webhook_server):
    notify = slack_webhook_notifier(local_webhook_server)
    request = make_approval_request()

    notify(request)

    assert len(_CapturingHandler.received) == 1
    payload = json.loads(_CapturingHandler.received[0])
    assert "delete_file" in payload["text"]
    assert request.request_id in payload["text"]


def test_generic_json_webhook_notifier_delivers_full_request(local_webhook_server):
    notify = generic_json_webhook_notifier(local_webhook_server)
    request = make_approval_request()

    notify(request)

    payload = json.loads(_CapturingHandler.received[0])
    assert payload["request_id"] == request.request_id
    assert payload["tool_call"]["name"] == "delete_file"


def test_webhook_callback_returns_response_once_resolved():
    store = PendingApprovalStore()
    delivered = threading.Event()

    def notify(request: ApprovalRequest) -> None:
        delivered.set()

        def resolve_later():
            time.sleep(0.05)
            store.resolve(
                request.request_id,
                ApprovalResponse(request_id=request.request_id, approved=True, approver="ops-team"),
            )

        threading.Thread(target=resolve_later, daemon=True).start()

    callback = WebhookApprovalCallback(store=store, notify=notify, timeout=2.0)
    manager = ApprovalManager(callback=callback)

    response = manager.request_approval(
        ToolCall(name="delete_file", arguments={"path": "a.txt"}),
        PolicyDecision(
            allowed=True,
            requires_approval=True,
            risk_level=RiskLevel.HIGH,
            rule_id="critical-delete",
            reason="Deletion requires approval.",
        ),
    )

    assert delivered.is_set()
    assert response.approved
    assert response.approver == "ops-team"


def test_webhook_callback_fails_closed_on_timeout():
    store = PendingApprovalStore()
    callback = WebhookApprovalCallback(store=store, notify=lambda request: None, timeout=0.05)

    response = callback(make_approval_request())

    assert not response.approved
    assert "Timed out" in (response.reason or "")


def test_webhook_callback_fails_closed_when_notify_raises():
    store = PendingApprovalStore()

    def broken_notify(request: ApprovalRequest) -> None:
        raise ConnectionError("webhook unreachable")

    callback = WebhookApprovalCallback(store=store, notify=broken_notify, timeout=1.0)

    response = callback(make_approval_request())

    assert not response.approved
    assert "Failed to deliver" in (response.reason or "")
