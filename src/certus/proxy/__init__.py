"""Execution interception: middleware, sandboxing, and human approval.

``certus.proxy.gateway`` is imported lazily by design (it requires the
``proxy`` extra) — it is intentionally NOT re-exported here so that
``import certus`` never pulls in FastAPI.
"""

from certus.proxy.approval import ApprovalManager, ApprovalRequest, ApprovalResponse
from certus.proxy.middleware import AuditJournal, CertusGuard, GuardDecision
from certus.proxy.sandbox import DbTransaction, FileSandbox, SandboxJournal, StagedFileOperation

__all__ = [
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalResponse",
    "AuditJournal",
    "CertusGuard",
    "DbTransaction",
    "FileSandbox",
    "GuardDecision",
    "SandboxJournal",
    "StagedFileOperation",
]
