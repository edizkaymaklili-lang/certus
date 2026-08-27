"""Certus: a deterministic guardrail layer for autonomous AI agents.

Quickstart::

    from certus import Certus
    from pydantic import BaseModel

    class DeleteFileArgs(BaseModel):
        path: str

    guard = Certus()  # loads the packaged fail-closed default policy

    @guard.protect(schema=DeleteFileArgs)
    def delete_file(path: str) -> str:
        ...
        return f"deleted {path}"

See ``certus.core`` for the schema/policy engine (offline, no dependencies
beyond pydantic/jsonschema/PyYAML), ``certus.proxy`` for the execution
interceptor, sandboxing, and approval workflow, and ``certus.edge`` for the
Colab-to-edge-device model packaging pipeline.
"""

from certus.core.exceptions import (
    ApprovalDeniedError,
    ApprovalRequiredError,
    CertusError,
    PolicyViolationError,
    SandboxExecutionError,
    SchemaValidationError,
    UnknownToolError,
)
from certus.core.models import AuditRecord, PolicyDecision, RiskLevel, ToolCall, ValidationResult
from certus.core.policy import PolicyConfig, PolicyEngine
from certus.core.schema_validator import SchemaValidator
from certus.sdk.client import Certus

__version__ = "0.1.0"

__all__ = [
    "ApprovalDeniedError",
    "ApprovalRequiredError",
    "AuditRecord",
    "Certus",
    "CertusError",
    "PolicyConfig",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyViolationError",
    "RiskLevel",
    "SandboxExecutionError",
    "SchemaValidationError",
    "SchemaValidator",
    "ToolCall",
    "UnknownToolError",
    "ValidationResult",
    "__version__",
]
