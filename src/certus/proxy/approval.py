"""Human-in-the-loop approval for high-risk tool calls.

The approval mechanism is intentionally a thin, pluggable interface: Certus
ships a synchronous CLI prompt for local development and a fully scriptable
in-memory queue for building a web/webhook-based approver on top, but does
not hardcode any specific notification channel (Slack, email, a ticketing
system) — wire your own callback and pass it to :class:`ApprovalManager`.
"""

from __future__ import annotations

import time
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
