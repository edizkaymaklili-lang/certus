"""Tests for examples/slack_interactivity_handler.py (requires the `proxy` extra).

Loaded by file path via importlib since `examples/` is a scripts folder,
not an installed package. Signature verification is tested against a real
HMAC-SHA256 computation (Slack's actual algorithm), not a mocked check.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

_MODULE_PATH = Path(__file__).parent.parent / "examples" / "slack_interactivity_handler.py"
_spec = importlib.util.spec_from_file_location("slack_interactivity_handler", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
handler_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(handler_module)

from fastapi.testclient import TestClient  # noqa: E402

SIGNING_SECRET = "test-signing-secret"


def sign(body: bytes, timestamp: str, secret: str = SIGNING_SECRET) -> str:
    basestring = f"v0:{timestamp}:{body.decode()}"
    return "v0=" + hmac.new(secret.encode(), basestring.encode(), hashlib.sha256).hexdigest()


def test_verify_signature_accepts_a_correctly_signed_request():
    body = b'{"hello": "world"}'
    timestamp = str(int(time.time()))
    signature = sign(body, timestamp)

    assert handler_module.verify_slack_signature(
        body, timestamp, signature, signing_secret=SIGNING_SECRET
    )


def test_verify_signature_rejects_wrong_secret():
    body = b'{"hello": "world"}'
    timestamp = str(int(time.time()))
    signature = sign(body, timestamp, secret="wrong-secret")

    assert not handler_module.verify_slack_signature(
        body, timestamp, signature, signing_secret=SIGNING_SECRET
    )


def test_verify_signature_rejects_tampered_body():
    timestamp = str(int(time.time()))
    signature = sign(b'{"hello": "world"}', timestamp)

    assert not handler_module.verify_slack_signature(
        b'{"hello": "tampered"}', timestamp, signature, signing_secret=SIGNING_SECRET
    )


def test_verify_signature_rejects_stale_timestamp():
    body = b'{"hello": "world"}'
    stale_timestamp = str(int(time.time()) - 60 * 60)  # 1 hour old
    signature = sign(body, stale_timestamp)

    assert not handler_module.verify_slack_signature(
        body, stale_timestamp, signature, signing_secret=SIGNING_SECRET
    )


def test_verify_signature_rejects_missing_secret():
    body = b'{"hello": "world"}'
    timestamp = str(int(time.time()))
    signature = sign(body, timestamp)

    assert not handler_module.verify_slack_signature(
        body, timestamp, signature, signing_secret=""
    )


def test_endpoint_rejects_unsigned_request():
    client = TestClient(handler_module.app)

    response = client.post(
        "/slack/interactions",
        data={"payload": json.dumps({"type": "block_actions", "actions": []})},
    )

    assert response.status_code == 401


class _CapturingHandler(BaseHTTPRequestHandler):
    received: list[bytes] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        _CapturingHandler.received.append(self.rfile.read(length))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def fake_gateway():
    _CapturingHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_endpoint_forwards_approve_click_to_gateway(fake_gateway, monkeypatch):
    monkeypatch.setenv("CERTUS_GATEWAY_URL", fake_gateway)
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SIGNING_SECRET)
    client = TestClient(handler_module.app)
    slack_payload = {
        "type": "block_actions",
        "user": {"username": "ops-alice"},
        "actions": [{"action_id": "certus_approve", "value": "req-123"}],
    }
    # Build the exact raw request body ourselves (rather than letting the
    # test client encode a dict) so the bytes we sign are byte-for-byte the
    # same ones verify_slack_signature will see server-side — signature
    # verification is exact-match, so any encoding mismatch here would
    # produce a false negative unrelated to the logic under test.
    from urllib.parse import urlencode

    encoded_body = urlencode({"payload": json.dumps(slack_payload)}).encode()
    timestamp = str(int(time.time()))
    signature = sign(encoded_body, timestamp)

    response = client.post(
        "/slack/interactions",
        content=encoded_body,
        headers={
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "recorded", "request_id": "req-123", "approved": True}
    assert len(_CapturingHandler.received) == 1
    forwarded = json.loads(_CapturingHandler.received[0])
    assert forwarded == {
        "approved": True,
        "approver": "ops-alice",
        "reason": "Decided via Slack by @ops-alice.",
    }
