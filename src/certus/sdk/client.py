"""The `certus-ai` developer-facing SDK entrypoint.

This module exists to make the common case a 3-line integration::

    from certus import Certus

    guard = Certus(policy_path="policy.yaml")

    @guard.protect(schema=DeleteFileArgs)
    def delete_file(path: str) -> str:
        ...

Everything :class:`Certus` does is delegate to
:class:`~certus.proxy.middleware.CertusGuard`; this class only trims the
constructor boilerplate (loading a policy file, wiring an approval
callback, picking an audit path) down to keyword arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from certus.config import default_policy_path
from certus.core.models import ToolCall
from certus.core.policy import PolicyConfig, PolicyEngine
from certus.core.schema_validator import SchemaSource, SchemaValidator
from certus.proxy.approval import ApprovalCallback, ApprovalManager
from certus.proxy.middleware import AuditJournal, CertusGuard, GuardDecision


class Certus:
    """Top-level SDK client wrapping schema validation, policy, and approval.

    Args:
        policy_path: Path to a YAML/JSON policy file. If omitted and
            ``policy`` is also omitted, Certus loads its packaged
            fail-closed default policy.
        policy: An already-constructed :class:`~certus.core.policy.PolicyConfig`,
            for callers that build policy programmatically instead of from
            a file. Takes precedence over ``policy_path`` if both are given.
        approval_callback: Synchronous callback invoked when a call is
            flagged ``requires_approval``. Defaults to auto-deny (fail-closed)
            if not provided. See :mod:`certus.proxy.approval` for built-ins
            such as :func:`~certus.proxy.approval.cli_approval_callback`.
        audit_path: Where to persist the append-only audit journal. Defaults
            to ``.certus/audit-journal.jsonl`` in the current working directory.
        audit_enabled: Set False to disable audit persistence entirely
            (e.g. in latency-sensitive hot paths or ephemeral test runs).
    """

    def __init__(
        self,
        policy_path: str | Path | None = None,
        policy: PolicyConfig | None = None,
        approval_callback: ApprovalCallback | None = None,
        audit_path: str | Path | None = None,
        audit_enabled: bool = True,
    ) -> None:
        if policy is not None:
            engine = PolicyEngine(policy)
        elif policy_path is not None:
            engine = PolicyEngine.from_file(policy_path)
        else:
            engine = PolicyEngine.from_file(default_policy_path())

        approval_manager = (
            ApprovalManager(callback=approval_callback) if approval_callback else None
        )
        journal = AuditJournal(audit_path) if audit_path else None

        self._guard = CertusGuard(
            policy_engine=engine,
            schema_validator=SchemaValidator(),
            approval_manager=approval_manager,
            audit_enabled=audit_enabled,
            journal=journal,
        )

    @property
    def guard(self) -> CertusGuard:
        """The underlying :class:`~certus.proxy.middleware.CertusGuard`, for advanced use."""
        return self._guard

    def protect(self, schema: SchemaSource, *, name: str | None = None) -> Callable[..., Any]:
        """Decorator: register a function as a guarded tool. See module docstring."""
        return self._guard.protect(schema, name=name)

    def register_tool(self, name: str, schema: SchemaSource, handler: Callable[..., Any]) -> None:
        """Register ``handler`` as tool ``name``, guarded by ``schema``."""
        self._guard.register_tool(name, schema, handler)

    def call(self, name: str, arguments: dict[str, Any], *, agent_id: str | None = None) -> Any:
        """Build a :class:`~certus.core.models.ToolCall` and run it through the guard.

        Use this to intercept calls originating from an LLM's tool-use
        response (e.g. an Anthropic/OpenAI tool-call block) without writing
        a decorator for every tool.
        """
        tool_call = ToolCall(name=name, arguments=arguments, agent_id=agent_id)
        return self._guard.intercept(tool_call)

    def evaluate(
        self, name: str, arguments: dict[str, Any], *, agent_id: str | None = None
    ) -> GuardDecision:
        """Like :meth:`call`, but only returns a decision without executing anything."""
        tool_call = ToolCall(name=name, arguments=arguments, agent_id=agent_id)
        return self._guard.evaluate(tool_call)
