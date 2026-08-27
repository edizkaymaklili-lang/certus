"""Deterministic, rule-based policy engine (allowlist / denylist / regex).

Certus never asks an LLM whether a tool call "looks safe" — every decision
here is a plain rule match: exact tool-name membership or a compiled regex
against argument values. Given the same policy file and the same tool call,
:meth:`PolicyEngine.evaluate` always returns the same :class:`PolicyDecision`.

Evaluation order (first match wins within each stage, stages run in order):

1. Tool name well-formedness.
2. ``denylist_tools``   — exact tool name match -> deny outright.
3. ``denylist_patterns`` — regex match against argument values -> deny outright.
4. ``critical_tools``   — exact tool name match -> allow, but flagged with a
   risk level and (usually) ``requires_approval=True``.
5. ``allowlist``        — if non-empty, any tool not already classified and
   not present here is rejected by ``default_action``.
6. Fallback to ``default_action`` / ``default_risk``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from certus.core.models import PolicyDecision, RiskLevel, ToolCall, is_valid_tool_name

DefaultAction = Literal["allow", "deny", "require_approval"]


class DenylistToolRule(BaseModel):
    """Blocks a tool call outright by exact name match."""

    id: str
    tools: list[str]
    reason: str


class DenylistPatternRule(BaseModel):
    """Blocks a tool call whose argument values match a regex pattern.

    Attributes:
        fields: Argument field names (dotted paths for nested values) to
            scan. Use ``"*"`` to scan every string-valued field.
        pattern: A Python regular expression (case sensitivity is the
            author's responsibility via inline flags, e.g. ``(?i)``).
    """

    id: str
    fields: list[str] = Field(default_factory=lambda: ["*"])
    pattern: str
    reason: str

    @field_validator("pattern")
    @classmethod
    def _pattern_compiles(cls, v: str) -> str:
        re.compile(v)  # raises re.error at config-load time, not at request time
        return v

    @property
    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern)


class CriticalToolRule(BaseModel):
    """Marks a tool as requiring elevated handling (risk level / approval)."""

    id: str
    tools: list[str]
    risk_level: RiskLevel = RiskLevel.HIGH
    requires_approval: bool = True
    reason: str


class PolicyConfig(BaseModel):
    """Fully parsed policy document, typically loaded from YAML or JSON."""

    version: int = 1
    default_action: DefaultAction = "deny"
    default_risk: RiskLevel = RiskLevel.LOW
    allowlist: list[str] = Field(default_factory=list)
    denylist_tools: list[DenylistToolRule] = Field(default_factory=list)
    denylist_patterns: list[DenylistPatternRule] = Field(default_factory=list)
    critical_tools: list[CriticalToolRule] = Field(default_factory=list)


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Recursively flatten a nested argument structure into (path, str) pairs.

    Only leaf values are stringified for pattern scanning; container
    structure is preserved in the dotted path (e.g. ``"headers.Authorization"``).
    """
    out: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for k, v in value.items():
            out.extend(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.extend(_flatten(v, f"{prefix}[{i}]"))
    else:
        out.append((prefix, str(value)))
    return out


class PolicyEngine:
    """Loads a :class:`PolicyConfig` and evaluates :class:`ToolCall` objects against it."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    @classmethod
    def from_file(cls, path: str | Path) -> PolicyEngine:
        """Load a policy from a YAML or JSON file on disk.

        Args:
            path: Path to a ``.yaml``, ``.yml``, or ``.json`` policy file.

        Returns:
            A ready-to-use :class:`PolicyEngine`.
        """
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        return cls(PolicyConfig.model_validate(data))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyEngine:
        """Load a policy from an already-parsed dict (e.g. from an API payload)."""
        return cls(PolicyConfig.model_validate(data))

    def evaluate(self, tool_call: ToolCall) -> PolicyDecision:
        """Evaluate a tool call and return a single, deterministic decision.

        This method never raises for a "normal" denial — it always returns a
        :class:`PolicyDecision`. Callers that want an exception on denial
        should use :meth:`evaluate_or_raise`.
        """
        if not is_valid_tool_name(tool_call.name):
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                risk_level=RiskLevel.CRITICAL,
                rule_id="malformed-tool-name",
                reason=f"Malformed tool name: {tool_call.name!r}",
            )

        # Stage 1: exact-name denylist.
        for tool_rule in self.config.denylist_tools:
            if tool_call.name in tool_rule.tools:
                return PolicyDecision(
                    allowed=False,
                    requires_approval=False,
                    risk_level=RiskLevel.CRITICAL,
                    rule_id=tool_rule.id,
                    reason=tool_rule.reason,
                )

        # Stage 2: regex denylist over argument values.
        flattened = _flatten(tool_call.arguments)
        for pattern_rule in self.config.denylist_patterns:
            scan_all = "*" in pattern_rule.fields
            for field_path, str_value in flattened:
                field_name = field_path.split(".")[-1].split("[")[0]
                in_scope = field_path in pattern_rule.fields or field_name in pattern_rule.fields
                if not (scan_all or in_scope):
                    continue
                match = pattern_rule.compiled.search(str_value)
                if match:
                    return PolicyDecision(
                        allowed=False,
                        requires_approval=False,
                        risk_level=RiskLevel.CRITICAL,
                        rule_id=pattern_rule.id,
                        reason=pattern_rule.reason,
                        matched_pattern=match.group(0),
                    )

        # Stage 3: critical tools -> allowed, but flagged.
        for critical_rule in self.config.critical_tools:
            if tool_call.name in critical_rule.tools:
                return PolicyDecision(
                    allowed=True,
                    requires_approval=critical_rule.requires_approval,
                    risk_level=critical_rule.risk_level,
                    rule_id=critical_rule.id,
                    reason=critical_rule.reason,
                )

        # Stage 4: allowlist, if configured, is authoritative for anything left.
        if self.config.allowlist:
            if tool_call.name in self.config.allowlist:
                return PolicyDecision(
                    allowed=True,
                    requires_approval=False,
                    risk_level=self.config.default_risk,
                    rule_id="allowlist",
                    reason="Tool is explicitly allowlisted.",
                )
            return self._default_decision(
                tool_call.name, reason="Tool is not present in the allowlist."
            )

        # Stage 5: no allowlist configured -> fall back to default_action.
        return self._default_decision(
            tool_call.name, reason="No allowlist, denylist, or critical rule matched."
        )

    def evaluate_or_raise(self, tool_call: ToolCall) -> PolicyDecision:
        """Like :meth:`evaluate`, but raises :class:`PolicyViolationError` on denial."""
        from certus.core.exceptions import PolicyViolationError

        decision = self.evaluate(tool_call)
        if not decision.allowed:
            raise PolicyViolationError(tool_call.name, decision.rule_id, decision.reason)
        return decision

    def _default_decision(self, tool_name: str, *, reason: str) -> PolicyDecision:
        action = self.config.default_action
        if action == "allow":
            return PolicyDecision(
                allowed=True,
                requires_approval=False,
                risk_level=self.config.default_risk,
                rule_id="default-allow",
                reason=reason,
            )
        if action == "require_approval":
            return PolicyDecision(
                allowed=True,
                requires_approval=True,
                risk_level=self.config.default_risk,
                rule_id="default-require-approval",
                reason=reason,
            )
        return PolicyDecision(
            allowed=False,
            requires_approval=False,
            risk_level=self.config.default_risk,
            rule_id="default-deny",
            reason=reason,
        )
