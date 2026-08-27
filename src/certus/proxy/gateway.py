"""Optional FastAPI Gateway/Proxy exposing :class:`CertusGuard` over HTTP.

This is the "sits in front of the agent" deployment mode: an agent runtime
(or the LLM provider's tool-calling loop) POSTs a proposed tool call here
instead of executing it directly. Certus validates, evaluates policy, and
(if configured) drives human approval, then either executes a locally
registered handler or hands back a bare decision for the caller's own
executor to act on.

Requires the ``proxy`` extra::

    pip install "certus-ai[proxy]"
    uvicorn certus.proxy.gateway:app --reload
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "certus.proxy.gateway requires the 'proxy' extra. "
        'Install it with: pip install "certus-ai[proxy]"'
    ) from exc

from certus.core.exceptions import CertusError
from certus.core.models import ToolCall
from certus.proxy.middleware import CertusGuard


class ToolCallRequest(BaseModel):
    """Request body for ``POST /v1/tool-calls``."""

    name: str
    arguments: dict[str, Any] = {}
    agent_id: str | None = None
    request_id: str | None = None


class ToolCallResponse(BaseModel):
    """Response body for ``POST /v1/tool-calls``."""

    status: str  # "executed" | "decided"
    ok: bool
    result: Any | None = None
    risk_level: str | None = None
    requires_approval: bool | None = None
    approved: bool | None = None
    reason: str | None = None


def create_app(guard: CertusGuard) -> FastAPI:
    """Build a FastAPI app that proxies tool calls through ``guard``.

    Args:
        guard: A configured :class:`~certus.proxy.middleware.CertusGuard`.
            Register handlers on it beforehand if this gateway should fully
            execute calls; leave tools unregistered to run in
            decision-only mode (the caller executes the call itself once
            ``ok`` is True).
    """
    app = FastAPI(
        title="Certus Guardrail Gateway",
        description="Deterministic policy/schema enforcement proxy for AI agent tool calls.",
        version="0.1.0",
    )

    @app.post("/v1/tool-calls", response_model=ToolCallResponse)
    def submit_tool_call(payload: ToolCallRequest) -> ToolCallResponse:
        tool_call = ToolCall(
            name=payload.name,
            arguments=payload.arguments,
            agent_id=payload.agent_id,
            request_id=payload.request_id,
        )
        if guard.has_handler(tool_call.name):
            try:
                result = guard.intercept(tool_call)
            except CertusError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            return ToolCallResponse(status="executed", ok=True, result=result)

        decision = guard.evaluate(tool_call)
        return ToolCallResponse(
            status="decided",
            ok=decision.ok,
            risk_level=decision.decision.risk_level.value if decision.decision else None,
            requires_approval=decision.decision.requires_approval if decision.decision else None,
            approved=decision.approved,
            reason=decision.reason,
        )

    @app.get("/v1/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


# Convenience module-level app for `uvicorn certus.proxy.gateway:app`, using an
# unconfigured, fail-closed guard. Real deployments should build their own
# app via create_app(guard) with a properly configured CertusGuard instead.
app = create_app(CertusGuard())
