"""Tests for certus.core.schema_validator."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from certus.core.exceptions import UnknownToolError
from certus.core.models import ToolCall
from certus.core.schema_validator import SchemaValidator


class DeleteFileArgs(BaseModel):
    path: str
    force: bool = False


def test_pydantic_schema_accepts_valid_call():
    validator = SchemaValidator()
    validator.register_schema("delete_file", DeleteFileArgs)

    result = validator.validate(ToolCall(name="delete_file", arguments={"path": "a.txt"}))

    assert result.valid
    assert result.normalized_arguments == {"path": "a.txt", "force": False}


def test_pydantic_schema_rejects_missing_required_field():
    validator = SchemaValidator()
    validator.register_schema("delete_file", DeleteFileArgs)

    result = validator.validate(ToolCall(name="delete_file", arguments={}))

    assert not result.valid
    assert any("path" in err for err in result.errors)


def test_jsonschema_accepts_valid_call():
    validator = SchemaValidator()
    validator.register_schema(
        "search_web",
        {
            "type": "object",
            "properties": {"query": {"type": "string", "maxLength": 200}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )

    result = validator.validate(ToolCall(name="search_web", arguments={"query": "certus ai"}))

    assert result.valid


def test_jsonschema_rejects_additional_properties():
    validator = SchemaValidator()
    validator.register_schema(
        "search_web",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )

    result = validator.validate(
        ToolCall(name="search_web", arguments={"query": "x", "evil_field": "rm -rf /"})
    )

    assert not result.valid


def test_unregistered_tool_is_fail_closed_by_default():
    validator = SchemaValidator()

    with pytest.raises(UnknownToolError):
        validator.validate(ToolCall(name="never_registered", arguments={}))


def test_unregistered_tool_non_strict_returns_invalid_result():
    validator = SchemaValidator()

    result = validator.validate(ToolCall(name="never_registered", arguments={}), strict=False)

    assert not result.valid


def test_malformed_tool_name_is_rejected():
    validator = SchemaValidator()
    validator.register_schema("delete_file", DeleteFileArgs)

    result = validator.validate(
        ToolCall(name="delete_file; rm -rf /", arguments={}), strict=False
    )

    assert not result.valid
