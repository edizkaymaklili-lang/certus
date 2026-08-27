"""Core data models shared across the Certus guardrail pipeline.

These are the only structures that flow between the schema validator,
policy engine, approval manager, and sandbox executor. Keeping them as
strict Pydantic models (rather than free-form dicts) is what makes the
rest of the pipeline deterministic and easy to unit test.
"""

from __future__ import annotations

import re
import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(StrEnum):
    """Deterministic risk classification for a tool call.

    Ordering matters: members are declared from least to most severe so
    callers can compare them with :meth:`RiskLevel.at_least`.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def _rank(self) -> int:
        return list(RiskLevel).index(self)

    def at_least(self, other: RiskLevel) -> bool:
        """Return True if this risk level is >= ``other``."""
        return self._rank >= other._rank


class ToolCall(BaseModel):
    """A single tool/function call an agent is attempting to execute.

    This is the canonical unit of work Certus intercepts. It is intentionally
    transport-agnostic: it can represent an OpenAI/Anthropic-style tool call,
    an internal RPC, or a shell command wrapped as a tool.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Tool/function name, e.g. 'delete_file'.")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Keyword arguments.")
    agent_id: str | None = Field(default=None, description="Identifier of the calling agent.")
    request_id: str | None = Field(default=None, description="Correlation id for audit logs.")
    issued_at: float = Field(default_factory=time.time, description="Unix timestamp.")


class ValidationResult(BaseModel):
    """Outcome of running a :class:`ToolCall` through the schema validator."""

    valid: bool
    tool_name: str
    errors: list[str] = Field(default_factory=list)
    normalized_arguments: dict[str, Any] | None = Field(
        default=None,
        description="Arguments coerced/normalized by the schema (e.g. Pydantic defaults applied).",
    )


class PolicyDecision(BaseModel):
    """Outcome of running a :class:`ToolCall` through the policy engine.

    ``allowed`` and ``requires_approval`` are independent axes: a call can be
    allowed outright, allowed-with-approval, or denied outright. A denied
    call always has ``allowed=False`` and ``requires_approval=False``.
    """

    allowed: bool
    requires_approval: bool
    risk_level: RiskLevel
    rule_id: str
    reason: str
    matched_pattern: str | None = None


class AuditRecord(BaseModel):
    """Immutable record of one intercepted call, written to the audit journal."""

    tool_call: ToolCall
    validation: ValidationResult
    decision: PolicyDecision | None = None
    approved: bool | None = None
    approver: str | None = None
    executed: bool = False
    sandboxed: bool = False
    rolled_back: bool = False
    error: str | None = None
    recorded_at: float = Field(default_factory=time.time)


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")


def is_valid_tool_name(name: str) -> bool:
    """Deterministic guard against malformed/injected tool names.

    Rejects empty strings, whitespace, and anything containing characters
    outside a conservative identifier charset. This runs before schema
    lookup so a crafted tool name can never be used to probe the registry.
    """
    return bool(name) and bool(_IDENTIFIER_RE.match(name))
