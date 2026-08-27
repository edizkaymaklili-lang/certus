"""Execution Interceptor: the wrapper that sits between an agent and its tools.

:class:`CertusGuard` is the single orchestration point that chains together
the deterministic core (:mod:`certus.core.schema_validator`,
:mod:`certus.core.policy`) with the human-in-the-loop layer
(:mod:`certus.proxy.approval`) and an append-only audit journal, then
finally invokes the real tool handler.

Pipeline for every intercepted call::

    ToolCall
        -> schema validation      (fail-closed: unknown tool -> rejected)
        -> policy evaluation      (deny outright / allow / allow+approval)
        -> human approval         (only if the policy decision requires it)
        -> real handler execution
        -> audit record persisted (always, including on failure)

Every stage before "real handler execution" is pure Python + regex/schema
matching — no network call, no model inference — which is what keeps the
guard's own overhead in the low-single-digit milliseconds for typical
payloads. The audit write is the one unavoidable I/O; disable it
(``audit_enabled=False``) or point it at a fast local disk if you are
operating under a hard latency budget.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from certus.core.exceptions import (
    ApprovalDeniedError,
    ApprovalRequiredError,
    PolicyViolationError,
    SchemaValidationError,
    UnknownToolError,
)
from certus.core.models import AuditRecord, PolicyDecision, ToolCall, ValidationResult
from certus.core.policy import PolicyEngine
from certus.core.schema_validator import SchemaSource, SchemaValidator
from certus.proxy.approval import ApprovalManager

F = TypeVar("F", bound=Callable[..., Any])


class GuardDecision(BaseModel):
    """Outcome of :meth:`CertusGuard.evaluate` — a verdict without execution.

    Attributes:
        ok: True only if the call is fully cleared to run (valid schema,
            policy-allowed, and either no approval was required or it was
            granted). A gateway should forward the call to its real
            executor if and only if ``ok`` is True.
    """

    tool_call: ToolCall
    validation: ValidationResult
    decision: PolicyDecision | None = None
    approved: bool | None = None
    approver: str | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        if not self.validation.valid:
            return False
        if self.decision is None or not self.decision.allowed:
            return False
        if self.decision.requires_approval:
            return bool(self.approved)
        return True


class AuditJournal:
    """Append-only JSON-lines log of every call the guard has intercepted."""

    def __init__(self, path: str | Path = ".certus/audit-journal.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: AuditRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")

    def read_all(self) -> list[AuditRecord]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [AuditRecord.model_validate_json(line) for line in fh if line.strip()]


class CertusGuard:
    """Central interceptor tying schema validation, policy, and approval together.

    Args:
        policy_engine: Deterministic allow/deny/approval engine. Defaults to
            a fail-closed engine with an empty policy (denies everything
            not explicitly allowlisted) if not provided.
        schema_validator: Registry of per-tool argument schemas. Created
            automatically if not provided; populated via :meth:`register_tool`
            or the :meth:`protect` decorator.
        approval_manager: Handles human-in-the-loop approval for calls the
            policy flags with ``requires_approval=True``. If ``None``, any
            call requiring approval raises :class:`ApprovalRequiredError`
            immediately (fail-closed).
        audit_enabled: Whether to persist an :class:`AuditRecord` for every
            intercepted call. Defaults to True.
        journal: Custom :class:`AuditJournal` instance (e.g. pointed at a
            different path); a default one is created if omitted.
    """

    def __init__(
        self,
        policy_engine: PolicyEngine | None = None,
        schema_validator: SchemaValidator | None = None,
        approval_manager: ApprovalManager | None = None,
        audit_enabled: bool = True,
        journal: AuditJournal | None = None,
    ) -> None:
        self.policy_engine = policy_engine or PolicyEngine()
        self.schema_validator = schema_validator or SchemaValidator()
        self.approval_manager = approval_manager
        self.audit_enabled = audit_enabled
        self.journal = journal or AuditJournal()
        self._handlers: dict[str, Callable[..., Any]] = {}

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register_tool(self, name: str, schema: SchemaSource, handler: Callable[..., Any]) -> None:
        """Register a callable ``handler`` as tool ``name``, guarded by ``schema``."""
        self.schema_validator.register_schema(name, schema)
        self._handlers[name] = handler

    def protect(
        self, schema: SchemaSource, *, name: str | None = None
    ) -> Callable[[F], F]:
        """Decorator that registers a function as a guarded tool.

        This is the 3-line integration path::

            guard = CertusGuard(policy_engine=PolicyEngine.from_file("policy.yaml"))

            @guard.protect(schema=DeleteFileArgs)
            def delete_file(path: str) -> str:
                ...

        Calling the decorated function now transparently goes through
        :meth:`intercept`: schema validation, policy evaluation, and (if
        required) human approval all happen before the original function
        body ever runs.

        Args:
            schema: JSON Schema dict or Pydantic model describing accepted
                keyword arguments.
            name: Tool name to register under; defaults to the function's
                ``__name__``.
        """

        def decorator(func: F) -> F:
            tool_name = name or func.__name__
            signature = inspect.signature(func)
            self.register_tool(tool_name, schema, func)

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                agent_id = kwargs.pop("_certus_agent_id", None)
                request_id = kwargs.pop("_certus_request_id", None)
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                tool_call = ToolCall(
                    name=tool_name,
                    arguments=dict(bound.arguments),
                    agent_id=agent_id,
                    request_id=request_id,
                )
                return self.intercept(tool_call)

            return wrapper  # type: ignore[return-value]

        return decorator

    def has_handler(self, tool_name: str) -> bool:
        """Return True if a real handler is registered for ``tool_name``.

        Used by gateway/proxy deployments to decide between fully executing
        a call (:meth:`intercept`) and returning a bare decision for a
        remote executor to act on (:meth:`evaluate`).
        """
        return tool_name in self._handlers

    # ------------------------------------------------------------------ #
    # Decision-only pipeline (no handler execution)
    # ------------------------------------------------------------------ #

    def evaluate(self, tool_call: ToolCall) -> GuardDecision:
        """Run schema validation, policy evaluation, and approval — but do not execute.

        This is the mode a pure Gateway/Proxy deployment uses: Certus sits
        in front of a separate tool-execution service and only needs to
        hand back a verdict (allowed / denied / approved / approval-pending)
        plus the normalized arguments, never invoking a local handler.

        Unlike :meth:`intercept`, this never raises on denial or pending
        approval — the outcome is always encoded in the returned
        :class:`GuardDecision` so a caller (e.g. an HTTP handler) can map it
        to the appropriate response without a try/except.
        """
        validation = self.schema_validator.validate(tool_call, strict=False)
        decision: PolicyDecision | None = None
        approved: bool | None = None
        approver: str | None = None
        reason: str | None = None

        if not validation.valid:
            reason = "; ".join(validation.errors) or "Schema validation failed."
        else:
            decision = self.policy_engine.evaluate(tool_call)
            if not decision.allowed:
                reason = decision.reason
            elif decision.requires_approval:
                if self.approval_manager is None:
                    reason = "Approval required but no approval manager is configured."
                else:
                    response = self.approval_manager.request_approval(tool_call, decision)
                    approved = response.approved
                    approver = response.approver
                    reason = response.reason

        if self.audit_enabled:
            self.journal.write(
                AuditRecord(
                    tool_call=tool_call,
                    validation=validation,
                    decision=decision,
                    approved=approved,
                    approver=approver,
                    executed=False,
                    error=(
                        reason
                        if not validation.valid or (decision and not decision.allowed)
                        else None
                    ),
                )
            )

        return GuardDecision(
            tool_call=tool_call,
            validation=validation,
            decision=decision,
            approved=approved,
            approver=approver,
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    # Interception pipeline
    # ------------------------------------------------------------------ #

    def intercept(self, tool_call: ToolCall) -> Any:
        """Run ``tool_call`` through the full guard pipeline and execute it.

        Raises:
            UnknownToolError: No schema/handler registered for the tool.
            SchemaValidationError: Arguments fail the registered schema.
            PolicyViolationError: The policy engine denies the call outright.
            ApprovalRequiredError: The call needs approval but no
                :class:`~certus.proxy.approval.ApprovalManager` is configured.
            ApprovalDeniedError: A configured approval manager rejected the call.

        Returns:
            Whatever the underlying handler returns.
        """
        started_at = time.perf_counter()
        validation: ValidationResult | None = None
        decision: PolicyDecision | None = None
        approved: bool | None = None
        approver: str | None = None
        error: str | None = None
        executed = False
        result: Any = None

        try:
            validation = self.schema_validator.validate(tool_call, strict=True)
            if not validation.valid:
                raise SchemaValidationError(tool_call.name, validation.errors)

            decision = self.policy_engine.evaluate(tool_call)
            if not decision.allowed:
                raise PolicyViolationError(tool_call.name, decision.rule_id, decision.reason)

            if decision.requires_approval:
                if self.approval_manager is None:
                    raise ApprovalRequiredError(
                        tool_call.name, decision.risk_level.value, tool_call.arguments
                    )
                response = self.approval_manager.request_approval(tool_call, decision)
                approved, approver = response.approved, response.approver
                if not approved:
                    raise ApprovalDeniedError(tool_call.name, approver, response.reason)

            handler = self._handlers.get(tool_call.name)
            if handler is None:
                raise UnknownToolError(tool_call.name)

            call_args = validation.normalized_arguments or tool_call.arguments
            result = handler(**call_args)
            executed = True
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            if self.audit_enabled:
                self.journal.write(
                    AuditRecord(
                        tool_call=tool_call,
                        validation=validation
                        or ValidationResult(
                            valid=False, tool_name=tool_call.name, errors=["not reached"]
                        ),
                        decision=decision,
                        approved=approved,
                        approver=approver,
                        executed=executed,
                        error=error,
                    )
                )
            _record_latency(tool_call.name, elapsed_ms)


_LATENCY_LOG_ENABLED = False


def _record_latency(tool_name: str, elapsed_ms: float) -> None:
    """Optional lightweight stderr latency trace; off by default.

    Flip :data:`_LATENCY_LOG_ENABLED` to True (or monkeypatch this function)
    to verify the guard is meeting the <20ms overhead budget in your
    environment; kept out of the audit journal to avoid bloating it.
    """
    if _LATENCY_LOG_ENABLED:  # pragma: no cover - diagnostic aid only
        print(json.dumps({"tool": tool_name, "guard_overhead_ms": round(elapsed_ms, 3)}))
