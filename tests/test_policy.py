"""Tests for certus.core.policy."""

from __future__ import annotations

from certus.core.models import RiskLevel, ToolCall
from certus.core.policy import PolicyEngine

BASE_CONFIG = {
    "version": 1,
    "default_action": "deny",
    "default_risk": "low",
    "allowlist": ["read_file", "search_web"],
    "denylist_tools": [
        {"id": "deny-shell", "tools": ["exec_shell"], "reason": "No raw shell execution."}
    ],
    "denylist_patterns": [
        {
            "id": "deny-secrets",
            "fields": ["*"],
            "pattern": r"(?i)\.env\b|id_rsa",
            "reason": "Credential access blocked.",
        }
    ],
    "critical_tools": [
        {
            "id": "critical-delete",
            "tools": ["delete_file"],
            "risk_level": "high",
            "requires_approval": True,
            "reason": "Deletion requires approval.",
        }
    ],
}


def make_engine() -> PolicyEngine:
    return PolicyEngine.from_dict(BASE_CONFIG)


def test_allowlisted_tool_is_allowed():
    engine = make_engine()
    decision = engine.evaluate(ToolCall(name="read_file", arguments={"path": "a.txt"}))
    assert decision.allowed
    assert not decision.requires_approval


def test_non_allowlisted_tool_defaults_to_deny():
    engine = make_engine()
    decision = engine.evaluate(ToolCall(name="unknown_tool", arguments={}))
    assert not decision.allowed
    assert decision.rule_id == "default-deny"


def test_denylisted_tool_name_is_denied():
    engine = make_engine()
    decision = engine.evaluate(ToolCall(name="exec_shell", arguments={"command": "ls"}))
    assert not decision.allowed
    assert decision.rule_id == "deny-shell"


def test_denylist_pattern_blocks_regardless_of_tool():
    engine = make_engine()
    decision = engine.evaluate(ToolCall(name="read_file", arguments={"path": "/root/.env"}))
    assert not decision.allowed
    assert decision.rule_id == "deny-secrets"
    assert ".env" in (decision.matched_pattern or "")


def test_denylist_pattern_scans_nested_arguments():
    engine = make_engine()
    decision = engine.evaluate(
        ToolCall(name="read_file", arguments={"options": {"paths": ["ok.txt", "~/.ssh/id_rsa"]}})
    )
    assert not decision.allowed
    assert decision.rule_id == "deny-secrets"


def test_critical_tool_requires_approval_but_is_allowed():
    engine = make_engine()
    decision = engine.evaluate(ToolCall(name="delete_file", arguments={"path": "a.txt"}))
    assert decision.allowed
    assert decision.requires_approval
    assert decision.risk_level == RiskLevel.HIGH


def test_denylist_pattern_wins_over_critical_tool():
    engine = make_engine()
    decision = engine.evaluate(ToolCall(name="delete_file", arguments={"path": "/root/.env"}))
    assert not decision.allowed
    assert decision.rule_id == "deny-secrets"


def test_malformed_tool_name_is_denied():
    engine = make_engine()
    decision = engine.evaluate(ToolCall(name="../etc/passwd", arguments={}))
    assert not decision.allowed
    assert decision.risk_level == RiskLevel.CRITICAL


def test_default_action_allow_permits_unclassified_tools():
    config = {**BASE_CONFIG, "allowlist": [], "default_action": "allow"}
    engine = PolicyEngine.from_dict(config)
    decision = engine.evaluate(ToolCall(name="get_weather", arguments={"city": "Istanbul"}))
    assert decision.allowed
    assert not decision.requires_approval


def test_default_action_require_approval():
    config = {**BASE_CONFIG, "allowlist": [], "default_action": "require_approval"}
    engine = PolicyEngine.from_dict(config)
    decision = engine.evaluate(ToolCall(name="get_weather", arguments={"city": "Istanbul"}))
    assert decision.allowed
    assert decision.requires_approval
