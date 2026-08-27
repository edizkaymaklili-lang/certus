"""Tests for certus.proxy.middleware.CertusGuard (the execution interceptor)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from certus.core.exceptions import (
    ApprovalDeniedError,
    ApprovalRequiredError,
    PolicyViolationError,
    SchemaValidationError,
)
from certus.core.models import ToolCall
from certus.core.policy import PolicyEngine
from certus.proxy.approval import ApprovalManager, ApprovalResponse
from certus.proxy.middleware import AuditJournal, CertusGuard

POLICY = {
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


class WeatherArgs(BaseModel):
    city: str


class DeleteFileArgs(BaseModel):
    path: str


def make_guard(tmp_path, approval_manager=None) -> CertusGuard:
    return CertusGuard(
        policy_engine=PolicyEngine.from_dict(POLICY),
        approval_manager=approval_manager,
        journal=AuditJournal(tmp_path / "audit.jsonl"),
    )


def test_protect_decorator_executes_allowed_call(tmp_path):
    guard = make_guard(tmp_path)

    @guard.protect(schema=WeatherArgs)
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    assert get_weather(city="Istanbul") == "sunny in Istanbul"
    records = guard.journal.read_all()
    assert len(records) == 1
    assert records[0].executed


def test_schema_violation_blocks_execution(tmp_path):
    guard = make_guard(tmp_path)

    @guard.protect(schema=WeatherArgs)
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    with pytest.raises(SchemaValidationError):
        get_weather(city=123)  # wrong type, fails pydantic coercion... see note below


def test_policy_denial_blocks_execution(tmp_path):
    guard = make_guard(tmp_path)
    calls = []

    @guard.protect(schema=WeatherArgs, name="not_allowlisted_tool")
    def handler(city: str) -> str:
        calls.append(city)
        return "never reached"

    with pytest.raises(PolicyViolationError):
        handler(city="Ankara")
    assert calls == []


def test_approval_required_without_manager_raises(tmp_path):
    guard = make_guard(tmp_path)

    @guard.protect(schema=DeleteFileArgs)
    def delete_file(path: str) -> str:
        return f"deleted {path}"

    with pytest.raises(ApprovalRequiredError):
        delete_file(path="a.txt")


def test_approval_granted_executes_call(tmp_path):
    manager = ApprovalManager(
        callback=lambda req: ApprovalResponse(
            request_id=req.request_id, approved=True, approver="tester"
        )
    )
    guard = make_guard(tmp_path, approval_manager=manager)

    @guard.protect(schema=DeleteFileArgs)
    def delete_file(path: str) -> str:
        return f"deleted {path}"

    assert delete_file(path="a.txt") == "deleted a.txt"


def test_approval_denied_blocks_execution(tmp_path):
    manager = ApprovalManager(
        callback=lambda req: ApprovalResponse(
            request_id=req.request_id, approved=False, reason="nope"
        )
    )
    guard = make_guard(tmp_path, approval_manager=manager)
    calls = []

    @guard.protect(schema=DeleteFileArgs)
    def delete_file(path: str) -> str:
        calls.append(path)
        return "never reached"

    with pytest.raises(ApprovalDeniedError):
        delete_file(path="a.txt")
    assert calls == []


def test_evaluate_decision_only_mode_does_not_execute(tmp_path):
    guard = make_guard(tmp_path)
    guard.schema_validator.register_schema("get_weather", WeatherArgs)

    decision = guard.evaluate(ToolCall(name="get_weather", arguments={"city": "Izmir"}))

    assert decision.ok
    assert decision.decision is not None and decision.decision.allowed


def test_evaluate_reports_unregistered_tool_as_not_ok(tmp_path):
    guard = make_guard(tmp_path)

    decision = guard.evaluate(ToolCall(name="get_weather", arguments={"city": "Izmir"}))

    assert not decision.ok
    assert not decision.validation.valid
