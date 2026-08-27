"""Exception hierarchy for the Certus guardrail engine.

All exceptions are deterministic and carry enough structured context
(tool name, offending field, rule id) to be logged, audited, or surfaced
to a human approver without additional parsing.
"""

from __future__ import annotations

from typing import Any


class CertusError(Exception):
    """Base class for every exception raised by Certus."""


class SchemaValidationError(CertusError):
    """Raised when a tool call's arguments fail JSON Schema / Pydantic validation.

    Attributes:
        tool_name: Name of the tool call that failed validation.
        errors: List of human-readable validation error messages.
    """

    def __init__(self, tool_name: str, errors: list[str]) -> None:
        self.tool_name = tool_name
        self.errors = errors
        message = f"Schema validation failed for tool '{tool_name}': {'; '.join(errors)}"
        super().__init__(message)


class UnknownToolError(CertusError):
    """Raised when a tool call references a tool with no registered schema.

    Certus defaults to fail-closed: an unrecognized tool is never assumed safe.
    """

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(
            f"No schema registered for tool '{tool_name}'. "
            "Fail-closed policy: unregistered tools are rejected by default."
        )


class PolicyViolationError(CertusError):
    """Raised when a tool call is denied outright by the policy engine.

    Attributes:
        tool_name: Name of the denied tool call.
        rule_id: Identifier of the policy rule that triggered the denial.
        reason: Human-readable explanation of the denial.
    """

    def __init__(self, tool_name: str, rule_id: str, reason: str) -> None:
        self.tool_name = tool_name
        self.rule_id = rule_id
        self.reason = reason
        super().__init__(f"Policy '{rule_id}' denied tool '{tool_name}': {reason}")


class ApprovalRequiredError(CertusError):
    """Raised (or used as a signal) when a tool call needs human approval.

    This is not necessarily a failure: callers may catch it and route the
    pending call through an :class:`~certus.proxy.approval.ApprovalManager`.
    """

    def __init__(self, tool_name: str, risk_level: str, arguments: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.risk_level = risk_level
        self.arguments = arguments
        super().__init__(
            f"Tool '{tool_name}' requires human approval (risk level: {risk_level})."
        )


class ApprovalDeniedError(CertusError):
    """Raised when a human approver explicitly rejects a pending tool call."""

    def __init__(self, tool_name: str, approver: str | None, reason: str | None) -> None:
        self.tool_name = tool_name
        self.approver = approver
        self.reason = reason
        approver_part = f" by '{approver}'" if approver else ""
        reason_part = f": {reason}" if reason else ""
        super().__init__(f"Tool '{tool_name}' was denied approval{approver_part}{reason_part}")


class SandboxExecutionError(CertusError):
    """Raised when a sandboxed simulation of a critical action fails."""

    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Sandbox execution failed for tool '{tool_name}': {reason}")
