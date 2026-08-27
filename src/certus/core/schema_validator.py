"""Deterministic schema validation for agent tool calls.

This module is the first gate a :class:`~certus.core.models.ToolCall` passes
through. It never uses an LLM or any probabilistic scoring: a call either
conforms to a registered JSON Schema (or Pydantic model) or it does not.

Design notes
------------
* Fail-closed: a tool with no registered schema is rejected, never assumed
  safe. This forces every capability exposed to an agent to be explicitly
  declared, which is the single biggest lever against indirect prompt
  injection ("the agent was tricked into calling a tool it was never meant
  to expose arbitrary arguments to").
* Two schema sources are supported side by side:
    1. Raw JSON Schema (``dict``) validated via the ``jsonschema`` library.
    2. A Pydantic ``BaseModel`` subclass, validated via ``model_validate``.
  Pydantic models additionally produce *normalized* arguments (defaults
  applied, types coerced), which downstream code can rely on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jsonschema
from pydantic import BaseModel, ValidationError

from certus.core.exceptions import SchemaValidationError, UnknownToolError
from certus.core.models import ToolCall, ValidationResult, is_valid_tool_name

SchemaSource = dict[str, Any] | type[BaseModel]


@dataclass(frozen=True)
class _RegisteredSchema:
    tool_name: str
    source: SchemaSource

    @property
    def kind(self) -> str:
        return "pydantic" if isinstance(self.source, type) else "jsonschema"


class SchemaValidator:
    """Registry + validation engine mapping tool names to argument schemas.

    Example:
        >>> from pydantic import BaseModel
        >>> class DeleteFileArgs(BaseModel):
        ...     path: str
        >>> validator = SchemaValidator()
        >>> validator.register_schema("delete_file", DeleteFileArgs)
        >>> result = validator.validate(ToolCall(name="delete_file", arguments={"path": "a.txt"}))
        >>> result.valid
        True
    """

    def __init__(self) -> None:
        self._schemas: dict[str, _RegisteredSchema] = {}

    def register_schema(self, tool_name: str, schema: SchemaSource) -> None:
        """Register the argument schema for a tool.

        Args:
            tool_name: Exact name the agent will use to invoke the tool.
            schema: Either a JSON Schema dict or a Pydantic ``BaseModel``
                subclass describing the tool's keyword arguments.

        Raises:
            ValueError: If ``tool_name`` is empty or ``schema`` is neither
                a dict nor a Pydantic model class.
        """
        if not is_valid_tool_name(tool_name):
            raise ValueError(f"Invalid tool name for schema registration: {tool_name!r}")
        is_pydantic_model = isinstance(schema, type) and issubclass(schema, BaseModel)
        if not (isinstance(schema, dict) or is_pydantic_model):
            raise ValueError(
                "schema must be a JSON Schema dict or a pydantic.BaseModel subclass, "
                f"got {type(schema)!r}"
            )
        self._schemas[tool_name] = _RegisteredSchema(tool_name=tool_name, source=schema)

    def unregister_schema(self, tool_name: str) -> None:
        """Remove a previously registered schema, if present."""
        self._schemas.pop(tool_name, None)

    def is_registered(self, tool_name: str) -> bool:
        """Return True if ``tool_name`` has a schema registered."""
        return tool_name in self._schemas

    def registered_tools(self) -> list[str]:
        """Return the sorted list of currently registered tool names."""
        return sorted(self._schemas)

    def validate(self, tool_call: ToolCall, *, strict: bool = True) -> ValidationResult:
        """Validate a tool call's arguments against its registered schema.

        Args:
            tool_call: The call to validate.
            strict: If True (default) and no schema is registered for
                ``tool_call.name``, raise :class:`UnknownToolError` instead
                of returning an "invalid" result. This enforces fail-closed
                behavior at the type level for callers that forget to check.

        Returns:
            A :class:`ValidationResult` describing success/failure. On
            success for a Pydantic-backed schema, ``normalized_arguments``
            contains the coerced, default-applied argument dict.

        Raises:
            UnknownToolError: If ``strict`` is True and no schema is
                registered for the tool.
        """
        if not is_valid_tool_name(tool_call.name):
            return ValidationResult(
                valid=False,
                tool_name=tool_call.name,
                errors=[f"Malformed tool name: {tool_call.name!r}"],
            )

        registered = self._schemas.get(tool_call.name)
        if registered is None:
            if strict:
                raise UnknownToolError(tool_call.name)
            return ValidationResult(
                valid=False,
                tool_name=tool_call.name,
                errors=[f"No schema registered for tool '{tool_call.name}'"],
            )

        if registered.kind == "pydantic":
            return self._validate_pydantic(tool_call, registered.source)  # type: ignore[arg-type]
        return self._validate_jsonschema(tool_call, registered.source)  # type: ignore[arg-type]

    def validate_or_raise(self, tool_call: ToolCall, *, strict: bool = True) -> ValidationResult:
        """Like :meth:`validate`, but raises :class:`SchemaValidationError` on failure."""
        result = self.validate(tool_call, strict=strict)
        if not result.valid:
            raise SchemaValidationError(tool_call.name, result.errors)
        return result

    @staticmethod
    def _validate_pydantic(tool_call: ToolCall, model: type[BaseModel]) -> ValidationResult:
        try:
            instance = model.model_validate(tool_call.arguments)
        except ValidationError as exc:
            errors = [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
            return ValidationResult(valid=False, tool_name=tool_call.name, errors=errors)
        return ValidationResult(
            valid=True,
            tool_name=tool_call.name,
            normalized_arguments=instance.model_dump(),
        )

    @staticmethod
    def _validate_jsonschema(tool_call: ToolCall, schema: dict[str, Any]) -> ValidationResult:
        validator_cls = jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)
        validator = validator_cls(schema)
        errors = sorted(validator.iter_errors(tool_call.arguments), key=lambda e: list(e.path))
        if errors:
            messages = [
                f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors
            ]
            return ValidationResult(valid=False, tool_name=tool_call.name, errors=messages)
        return ValidationResult(
            valid=True,
            tool_name=tool_call.name,
            normalized_arguments=dict(tool_call.arguments),
        )
