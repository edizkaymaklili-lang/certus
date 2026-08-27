"""Human-in-the-loop approval for high-risk tool calls.

The approval mechanism is a thin, pluggable interface. Certus ships three
concrete channels out of the box:

* :func:`cli_approval_callback` — blocking terminal prompt, for local dev.
* :func:`auto_deny_callback` — fail-closed default when nothing else is wired.
* :class:`WebhookApprovalCallback` — stages the request, notifies an
  out-of-band channel (Slack via :func:`slack_webhook_notifier`, or any
  other webhook), and blocks until that channel posts a decision back
  (typically via the gateway's ``POST /v1/approvals/{request_id}/decision``
  endpoint — see :mod:`certus.proxy.gateway`).

Wire your own callback for anything else (email, a ticketing system) —
:class:`ApprovalManager` only needs a ``Callable[[ApprovalRequest], ApprovalResponse]``.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from certus.core.exceptions import ApprovalDeniedError
from certus.core.models import PolicyDecision, ToolCall


class ApprovalRequest(BaseModel):
    """A pending request for a human to approve or deny a tool call."""

    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tool_call: ToolCall
    decision: PolicyDecision
    created_at: float = Field(default_factory=time.time)


class ApprovalResponse(BaseModel):
    """The outcome of a human review of an :class:`ApprovalRequest`."""

    request_id: str
    approved: bool
    approver: str | None = None
    reason: str | None = None
    resolved_at: float = Field(default_factory=time.time)


ApprovalCallback = Callable[[ApprovalRequest], ApprovalResponse]
AsyncApprovalCallback = Callable[[ApprovalRequest], Awaitable[ApprovalResponse]]


def cli_approval_callback(request: ApprovalRequest) -> ApprovalResponse:
    """Blocking CLI prompt asking an operator to approve/deny in a terminal.

    Intended for local development and demos, not production services.
    """
    print("\n[Certus] Approval required")
    print(f"  tool        : {request.tool_call.name}")
    print(f"  arguments   : {request.tool_call.arguments}")
    print(f"  risk level  : {request.decision.risk_level.value}")
    print(f"  reason      : {request.decision.reason}")
    answer = input("  Approve? [y/N]: ").strip().lower()
    return ApprovalResponse(
        request_id=request.request_id,
        approved=answer in ("y", "yes"),
        approver="cli-operator",
    )


def auto_deny_callback(request: ApprovalRequest) -> ApprovalResponse:
    """Fail-closed default: denies every request that reaches it.

    Useful as a safety net in non-interactive environments (CI, batch jobs)
    where no human is available to approve anything.
    """
    return ApprovalResponse(
        request_id=request.request_id,
        approved=False,
        approver=None,
        reason="No approval channel configured; failing closed.",
    )


class ApprovalManager:
    """Routes :class:`ApprovalRequest` objects to a pluggable callback.

    Example:
        >>> manager = ApprovalManager(callback=cli_approval_callback)
        >>> response = manager.request_approval(tool_call, decision)
        >>> if not response.approved:
        ...     raise ApprovalDeniedError(tool_call.name, response.approver, response.reason)
    """

    def __init__(
        self,
        callback: ApprovalCallback | None = None,
        async_callback: AsyncApprovalCallback | None = None,
    ) -> None:
        self.callback = callback or auto_deny_callback
        self.async_callback = async_callback
        self._history: list[tuple[ApprovalRequest, ApprovalResponse]] = []

    def request_approval(self, tool_call: ToolCall, decision: PolicyDecision) -> ApprovalResponse:
        """Synchronously request approval and return the resulting response."""
        request = ApprovalRequest(tool_call=tool_call, decision=decision)
        response = self.callback(request)
        self._history.append((request, response))
        return response

    async def request_approval_async(
        self, tool_call: ToolCall, decision: PolicyDecision
    ) -> ApprovalResponse:
        """Asynchronously request approval, if an async callback was configured.

        Falls back to running the synchronous callback if no async callback
        was provided.
        """
        request = ApprovalRequest(tool_call=tool_call, decision=decision)
        response = (
            await self.async_callback(request) if self.async_callback else self.callback(request)
        )
        self._history.append((request, response))
        return response

    def require_approval_or_raise(
        self, tool_call: ToolCall, decision: PolicyDecision
    ) -> ApprovalResponse:
        """Request approval and raise :class:`ApprovalDeniedError` if denied."""
        response = self.request_approval(tool_call, decision)
        if not response.approved:
            raise ApprovalDeniedError(tool_call.name, response.approver, response.reason)
        return response

    def history(self) -> list[dict[str, Any]]:
        """Return the full approval history for this manager instance, as dicts."""
        return [
            {"request": req.model_dump(), "response": resp.model_dump()}
            for req, resp in self._history
        ]


class PendingApprovalStore:
    """Thread-safe bridge between a blocking :class:`ApprovalCallback` and an
    out-of-band decision channel (Slack, a webhook, a ticketing system).

    :class:`WebhookApprovalCallback` calls :meth:`register`, fires off a
    notification, then blocks on :meth:`wait`. Whatever receives the human's
    decision out-of-band — typically an HTTP handler, such as the gateway's
    ``POST /v1/approvals/{request_id}/decision`` route — calls
    :meth:`resolve` from a different thread to unblock it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}
        self._responses: dict[str, ApprovalResponse] = {}

    def register(self, request_id: str) -> None:
        """Mark ``request_id`` as pending, before notifying the approval channel."""
        with self._lock:
            self._events[request_id] = threading.Event()

    def resolve(self, request_id: str, response: ApprovalResponse) -> bool:
        """Record a decision for ``request_id`` and wake up anyone waiting on it.

        Returns:
            True if ``request_id`` was pending and is now resolved; False if
            it was never registered or already timed out and was discarded.
        """
        with self._lock:
            event = self._events.get(request_id)
            if event is None:
                return False
            self._responses[request_id] = response
        event.set()
        return True

    def wait(self, request_id: str, timeout: float) -> ApprovalResponse | None:
        """Block up to ``timeout`` seconds for a decision on ``request_id``.

        Returns:
            The :class:`ApprovalResponse` if one arrived in time, else None.
        """
        with self._lock:
            event = self._events.get(request_id)
        if event is None or not event.wait(timeout):
            return None
        with self._lock:
            return self._responses.get(request_id)

    def discard(self, request_id: str) -> None:
        """Drop bookkeeping for ``request_id`` (e.g. after a timeout)."""
        with self._lock:
            self._events.pop(request_id, None)
            self._responses.pop(request_id, None)


WebhookNotifier = Callable[[ApprovalRequest], None]


def slack_webhook_notifier(webhook_url: str, *, timeout: float = 5.0) -> WebhookNotifier:
    """Build a :data:`WebhookNotifier` that posts to a Slack Incoming Webhook.

    Args:
        webhook_url: A Slack "Incoming Webhook" URL
            (``https://hooks.slack.com/services/...``).
        timeout: HTTP request timeout in seconds.

    Returns:
        A callable suitable for :class:`WebhookApprovalCallback`'s ``notify`` argument.
    """

    def notify(request: ApprovalRequest) -> None:
        text = (
            f":rotating_light: *Certus approval required*\n"
            f"*Tool:* `{request.tool_call.name}`\n"
            f"*Arguments:* `{request.tool_call.arguments}`\n"
            f"*Risk level:* {request.decision.risk_level.value}\n"
            f"*Reason:* {request.decision.reason}\n"
            f"*Request ID:* `{request.request_id}`"
        )
        body = json.dumps({"text": text}).encode("utf-8")
        http_request = urllib.request.Request(
            webhook_url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(http_request, timeout=timeout) as resp:  # noqa: S310
            resp.read()

    return notify


def generic_json_webhook_notifier(webhook_url: str, *, timeout: float = 5.0) -> WebhookNotifier:
    """Build a :data:`WebhookNotifier` that POSTs the raw request as JSON.

    Use this for ticketing systems / custom internal services instead of
    Slack's specific message format; the receiving endpoint gets
    ``{"request_id", "tool_call", "decision"}`` and is expected to call the
    Certus gateway's decision endpoint once a human has responded.
    """

    def notify(request: ApprovalRequest) -> None:
        body = request.model_dump_json().encode("utf-8")
        http_request = urllib.request.Request(
            webhook_url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(http_request, timeout=timeout) as resp:  # noqa: S310
            resp.read()

    return notify


class WebhookApprovalCallback:
    """Blocking :data:`ApprovalCallback` backed by a real, out-of-band channel.

    On each call: the request is registered in ``store``, ``notify`` is
    fired (e.g. a Slack message via :func:`slack_webhook_notifier`), and the
    calling thread blocks for up to ``timeout`` seconds for someone to call
    ``store.resolve(request_id, response)`` — typically triggered by
    ``POST /v1/approvals/{request_id}/decision`` on the Certus gateway
    (see :func:`certus.proxy.gateway.create_app`).

    Fails closed: if the notification can't be delivered, or no decision
    arrives before ``timeout``, the call is denied rather than left hanging
    or silently allowed.

    Example:
        >>> store = PendingApprovalStore()
        >>> callback = WebhookApprovalCallback(
        ...     store=store,
        ...     notify=slack_webhook_notifier("https://hooks.slack.com/services/..."),
        ...     timeout=300,
        ... )
        >>> manager = ApprovalManager(callback=callback)
    """

    def __init__(
        self,
        store: PendingApprovalStore,
        notify: WebhookNotifier,
        *,
        timeout: float = 300.0,
    ) -> None:
        self.store = store
        self.notify = notify
        self.timeout = timeout

    def __call__(self, request: ApprovalRequest) -> ApprovalResponse:
        self.store.register(request.request_id)
        try:
            self.notify(request)
        except Exception as exc:
            self.store.discard(request.request_id)
            return ApprovalResponse(
                request_id=request.request_id,
                approved=False,
                reason=f"Failed to deliver approval notification: {exc}",
            )

        response = self.store.wait(request.request_id, self.timeout)
        self.store.discard(request.request_id)
        if response is None:
            return ApprovalResponse(
                request_id=request.request_id,
                approved=False,
                reason=f"Timed out after {self.timeout}s waiting for a decision; failing closed.",
            )
        return response
