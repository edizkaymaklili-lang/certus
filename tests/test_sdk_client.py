"""Tests for the top-level `Certus` SDK client (certus.sdk.client)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from certus import Certus
from certus.core.exceptions import ApprovalRequiredError, PolicyViolationError
from certus.core.policy import PolicyConfig
from certus.proxy.approval import ApprovalResponse

POLICY = PolicyConfig.model_validate(
    {
        "version": 1,
        "default_action": "deny",
        "allowlist": ["get_weather"],
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
)


class WeatherArgs(BaseModel):
    city: str


class DeleteFileArgs(BaseModel):
    path: str


def test_default_constructor_loads_packaged_policy(tmp_path):
    guard = Certus(audit_path=tmp_path / "audit.jsonl")

    # The packaged default policy is fail-closed with no allowlist entry
    # for a made-up tool, so this must be denied rather than raise on
    # construction.
    decision = guard.evaluate("some_never_registered_tool", {})
    assert not decision.ok


def test_constructor_accepts_policy_object_directly(tmp_path):
    guard = Certus(policy=POLICY, audit_path=tmp_path / "audit.jsonl")
    guard.guard.schema_validator.register_schema("get_weather", WeatherArgs)

    decision = guard.evaluate("get_weather", {"city": "Ankara"})

    assert decision.ok


def test_constructor_accepts_policy_path(tmp_path):
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(
        "version: 1\ndefault_action: allow\ndefault_risk: low\n", encoding="utf-8"
    )

    guard = Certus(policy_path=policy_file, audit_path=tmp_path / "audit.jsonl")
    guard.guard.schema_validator.register_schema("get_weather", WeatherArgs)

    decision = guard.evaluate("get_weather", {"city": "Izmir"})

    assert decision.ok


def test_protect_decorator_executes_allowed_call(tmp_path):
    guard = Certus(policy=POLICY, audit_path=tmp_path / "audit.jsonl")

    @guard.protect(schema=WeatherArgs)
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    assert get_weather(city="Istanbul") == "sunny in Istanbul"


def test_register_tool_and_call(tmp_path):
    guard = Certus(policy=POLICY, audit_path=tmp_path / "audit.jsonl")
    guard.register_tool("get_weather", WeatherArgs, lambda city: f"sunny in {city}")

    result = guard.call("get_weather", {"city": "Bursa"})

    assert result == "sunny in Bursa"


def test_call_raises_on_policy_denial(tmp_path):
    guard = Certus(policy=POLICY, audit_path=tmp_path / "audit.jsonl")
    guard.register_tool("exec_shell", WeatherArgs, lambda city: "never reached")

    with pytest.raises(PolicyViolationError):
        guard.call("exec_shell", {"city": "n/a"})


def test_approval_callback_wiring_lets_approved_call_through(tmp_path):
    guard = Certus(
        policy=POLICY,
        approval_callback=lambda req: ApprovalResponse(
            request_id=req.request_id, approved=True, approver="tester"
        ),
        audit_path=tmp_path / "audit.jsonl",
    )
    guard.register_tool("delete_file", DeleteFileArgs, lambda path: f"deleted {path}")

    assert guard.call("delete_file", {"path": "a.txt"}) == "deleted a.txt"


def test_no_approval_callback_raises_approval_required(tmp_path):
    guard = Certus(policy=POLICY, audit_path=tmp_path / "audit.jsonl")
    guard.register_tool("delete_file", DeleteFileArgs, lambda path: "never reached")

    with pytest.raises(ApprovalRequiredError):
        guard.call("delete_file", {"path": "a.txt"})


def test_evaluate_does_not_execute_handler(tmp_path):
    guard = Certus(policy=POLICY, audit_path=tmp_path / "audit.jsonl")
    calls = []
    guard.register_tool("get_weather", WeatherArgs, lambda city: calls.append(city))

    decision = guard.evaluate("get_weather", {"city": "Adana"})

    assert decision.ok
    assert calls == []


def test_custom_audit_path_is_used(tmp_path):
    audit_path = tmp_path / "custom" / "journal.jsonl"
    guard = Certus(policy=POLICY, audit_path=audit_path)
    guard.register_tool("get_weather", WeatherArgs, lambda city: f"sunny in {city}")

    guard.call("get_weather", {"city": "Trabzon"})

    assert audit_path.exists()
    assert guard.guard.journal.path == audit_path


def test_audit_disabled_writes_no_journal_file(tmp_path):
    audit_path = tmp_path / "should-not-be-created" / "journal.jsonl"
    guard = Certus(policy=POLICY, audit_path=audit_path, audit_enabled=False)
    guard.register_tool("get_weather", WeatherArgs, lambda city: f"sunny in {city}")

    guard.call("get_weather", {"city": "Konya"})

    assert not audit_path.exists()


def test_guard_property_exposes_underlying_certus_guard(tmp_path):
    guard = Certus(policy=POLICY, audit_path=tmp_path / "audit.jsonl")

    from certus.proxy.middleware import CertusGuard

    assert isinstance(guard.guard, CertusGuard)
