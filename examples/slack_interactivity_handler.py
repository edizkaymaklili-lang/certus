"""Slack interactivity handler: the receiving half of the Slack approval flow.

`certus.proxy.approval.slack_webhook_notifier` only covers sending a
notification *to* Slack when a call needs approval. This script is the
other half — a small standalone service that receives the click when
someone presses "Approve"/"Deny" on that Slack message, and turns it into
a call to the Certus gateway's
``POST /v1/approvals/{request_id}/decision`` endpoint, which unblocks the
`WebhookApprovalCallback` that's been waiting on it.

Setup (once):

1. Create a Slack app at https://api.slack.com/apps with an Incoming
   Webhook and Interactivity enabled, pointing the Interactivity Request
   URL at wherever this script is deployed (e.g.
   ``https://your-domain.example/slack/interactions``).
2. Set environment variables:
   - ``SLACK_SIGNING_SECRET`` — from the Slack app's "Basic Information" page.
   - ``CERTUS_GATEWAY_URL`` — base URL of your Certus gateway
     (default: ``http://localhost:8000``).
3. Run it:  ``pip install "certus-ai[proxy]" && uvicorn examples.slack_interactivity_handler:app``

Security note: every request is verified against Slack's HMAC-SHA256
request signature (https://api.slack.com/authentication/verifying-requests)
before anything in its payload is trusted — an unsigned or replayed
request is rejected with 401, not silently ignored.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.request
from typing import Any
from urllib.parse import parse_qs

try:
    from fastapi import FastAPI, HTTPException, Request
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        'This example requires the proxy extra: pip install "certus-ai[proxy]"'
    ) from exc

MAX_REQUEST_AGE_SECONDS = 60 * 5  # reject anything older, to block replay attacks

app = FastAPI(title="Certus Slack Interactivity Handler")


def verify_slack_signature(
    raw_body: bytes, timestamp: str, signature: str, *, signing_secret: str | None = None
) -> bool:
    """Verify a request actually came from Slack, per Slack's documented algorithm.

    https://api.slack.com/authentication/verifying-requests

    Args:
        signing_secret: Defaults to the ``SLACK_SIGNING_SECRET`` environment
            variable, read at call time (not import time, so tests can set
            it per-case without reloading the module).
    """
    if signing_secret is not None:
        secret = signing_secret
    else:
        secret = os.environ.get("SLACK_SIGNING_SECRET", "")
    if not secret or not timestamp or not signature:
        return False
    try:
        request_age = abs(time.time() - float(timestamp))
    except ValueError:
        return False
    if request_age > MAX_REQUEST_AGE_SECONDS:
        return False  # too old: likely a replayed request

    basestring = f"v0:{timestamp}:{raw_body.decode('utf-8')}"
    computed = "v0=" + hmac.new(
        secret.encode("utf-8"), basestring.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(computed, signature)


def post_decision(
    request_id: str,
    *,
    approved: bool,
    approver: str,
    reason: str | None,
    gateway_url: str | None = None,
) -> None:
    """Deliver the human's decision to the Certus gateway.

    Args:
        gateway_url: Defaults to the ``CERTUS_GATEWAY_URL`` environment
            variable (or ``http://localhost:8000``), read at call time.
    """
    base_url = gateway_url if gateway_url is not None else os.environ.get(
        "CERTUS_GATEWAY_URL", "http://localhost:8000"
    )
    body = json.dumps({"approved": approved, "approver": approver, "reason": reason}).encode()
    url = f"{base_url}/v1/approvals/{request_id}/decision"
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
        resp.read()


@app.post("/slack/interactions")
async def handle_interaction(request: Request) -> dict[str, Any]:
    # Deliberately NOT using FastAPI's `Form(...)` parameter binding here:
    # it parses the body (consuming the request stream) before this
    # function's own code runs, which would leave nothing for
    # `request.body()` to read for signature verification below. Reading
    # the raw body ourselves first, then parsing it, keeps the exact bytes
    # Slack signed available for verify_slack_signature.
    raw_body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")
    if not verify_slack_signature(raw_body, timestamp, signature):
        raise HTTPException(status_code=401, detail="Invalid Slack request signature.")

    form_fields = parse_qs(raw_body.decode("utf-8"))
    payload_values = form_fields.get("payload")
    if not payload_values:
        raise HTTPException(status_code=400, detail="Missing 'payload' form field.")
    data = json.loads(payload_values[0])
    if data.get("type") != "block_actions":
        return {"status": "ignored"}

    actions = data.get("actions", [])
    if not actions:
        return {"status": "ignored"}

    action = actions[0]
    action_id = action.get("action_id")
    request_id = action.get("value")
    approver = data.get("user", {}).get("username", "unknown-slack-user")

    if action_id not in ("certus_approve", "certus_deny") or not request_id:
        return {"status": "ignored"}

    post_decision(
        request_id,
        approved=(action_id == "certus_approve"),
        approver=approver,
        reason=f"Decided via Slack by @{approver}.",
    )
    return {
        "status": "recorded",
        "request_id": request_id,
        "approved": action_id == "certus_approve",
    }


def slack_webhook_notifier_with_buttons(webhook_url: str, *, timeout: float = 5.0):
    """Build a WebhookNotifier that posts a Slack message with clickable Approve/Deny buttons.

    Drop-in alternative to `certus.proxy.approval.slack_webhook_notifier` for
    setups that also run this interactivity handler — the plain-text
    notifier has no buttons, since Certus's core `approval.py` intentionally
    stays Slack-Block-Kit-agnostic; this richer version lives here in the
    example instead.

    Example:
        >>> from certus.proxy.approval import PendingApprovalStore, WebhookApprovalCallback
        >>> store = PendingApprovalStore()
        >>> callback = WebhookApprovalCallback(
        ...     store=store,
        ...     notify=slack_webhook_notifier_with_buttons("https://hooks.slack.com/services/..."),
        ...     timeout=300,
        ... )
    """
    from certus.proxy.approval import ApprovalRequest  # local import: keeps this an optional extra

    def notify(request: ApprovalRequest) -> None:
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":rotating_light: *Certus approval required*\n"
                        f"*Tool:* `{request.tool_call.name}`\n"
                        f"*Arguments:* `{request.tool_call.arguments}`\n"
                        f"*Risk level:* {request.decision.risk_level.value}\n"
                        f"*Reason:* {request.decision.reason}"
                    ),
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "action_id": "certus_approve",
                        "value": request.request_id,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Deny"},
                        "style": "danger",
                        "action_id": "certus_deny",
                        "value": request.request_id,
                    },
                ],
            },
        ]
        body = json.dumps({"blocks": blocks}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            resp.read()

    return notify
