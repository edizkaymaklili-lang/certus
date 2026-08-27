"""Sandbox and rollback primitives for critical, hard-to-reverse actions.

Two concrete sandboxes are provided:

* :class:`FileSandbox` — stages file deletes/writes into a quarantine
  directory instead of touching the real path immediately. The staged
  operation is only made permanent on an explicit :meth:`StagedFileOperation.commit`
  call (typically triggered by an approved :class:`~certus.proxy.approval.ApprovalManager`
  decision); otherwise it can be reverted with :meth:`StagedFileOperation.rollback`,
  and an unresolved staged operation is safe to simply discard.
* :class:`DbTransaction` — wraps any DBAPI2-style connection (an object
  exposing ``commit()``/``rollback()``) in a context manager that rolls
  back by default, so a write is only durable if the code explicitly calls
  :meth:`DbTransaction.commit`.

Every staged/committed/rolled-back operation is appended to an on-disk,
append-only JSON-lines journal so an operator can audit or manually replay
history even if the process crashes mid-flight.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from certus.core.exceptions import SandboxExecutionError


class _DbApiConnection(Protocol):
    def commit(self) -> None: ...
    def rollback(self) -> None: ...


class SandboxJournal:
    """Append-only JSON-lines audit journal for sandboxed operations."""

    def __init__(self, path: str | Path = ".certus/sandbox-journal.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: dict[str, Any]) -> None:
        event = {"recorded_at": time.time(), **event}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


@dataclass
class StagedFileOperation:
    """A file delete/write staged in quarantine, pending commit or rollback."""

    operation_id: str
    kind: str  # "delete" | "write"
    target_path: Path
    quarantine_path: Path | None
    new_content: bytes | None
    _sandbox: FileSandbox = field(repr=False)
    resolved: bool = False

    def commit(self) -> None:
        """Make the staged operation permanent.

        For a ``delete``, the quarantined copy is purged for good. For a
        ``write``, the staged content is flushed to ``target_path``.
        """
        if self.resolved:
            raise SandboxExecutionError(
                self.kind, f"Operation {self.operation_id} already resolved."
            )
        try:
            if self.kind == "delete":
                if self.quarantine_path and self.quarantine_path.exists():
                    self.quarantine_path.unlink()
            elif self.kind == "write":
                self.target_path.parent.mkdir(parents=True, exist_ok=True)
                self.target_path.write_bytes(self.new_content or b"")
            self.resolved = True
            self._sandbox.journal.record(
                {"event": "commit", "operation_id": self.operation_id, "kind": self.kind,
                 "target_path": str(self.target_path)}
            )
        except OSError as exc:
            raise SandboxExecutionError(self.kind, f"Commit failed: {exc}") from exc

    def rollback(self) -> None:
        """Discard the staged operation, restoring the original state."""
        if self.resolved:
            raise SandboxExecutionError(
                self.kind, f"Operation {self.operation_id} already resolved."
            )
        try:
            if self.kind == "delete" and self.quarantine_path and self.quarantine_path.exists():
                self.target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(self.quarantine_path), str(self.target_path))
            # "write": nothing to restore on disk since target was never touched.
            self.resolved = True
            self._sandbox.journal.record(
                {"event": "rollback", "operation_id": self.operation_id, "kind": self.kind,
                 "target_path": str(self.target_path)}
            )
        except OSError as exc:
            raise SandboxExecutionError(self.kind, f"Rollback failed: {exc}") from exc


class FileSandbox:
    """Stages destructive/mutating file operations behind a commit/rollback gate.

    Example:
        >>> sandbox = FileSandbox(quarantine_dir=".certus/quarantine")
        >>> op = sandbox.stage_delete("reports/q3.csv")
        >>> # ... route through ApprovalManager ...
        >>> op.commit()   # or op.rollback()
    """

    def __init__(
        self,
        quarantine_dir: str | Path = ".certus/quarantine",
        journal: SandboxJournal | None = None,
    ) -> None:
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.journal = journal or SandboxJournal()

    def stage_delete(self, path: str | Path) -> StagedFileOperation:
        """Move ``path`` into quarantine instead of deleting it immediately."""
        target = Path(path)
        operation_id = uuid.uuid4().hex
        quarantine_path: Path | None = None
        if target.exists():
            quarantine_path = self.quarantine_dir / f"{operation_id}__{target.name}"
            shutil.move(str(target), str(quarantine_path))
        op = StagedFileOperation(
            operation_id=operation_id,
            kind="delete",
            target_path=target,
            quarantine_path=quarantine_path,
            new_content=None,
            _sandbox=self,
        )
        self.journal.record(
            {
                "event": "stage",
                "operation_id": operation_id,
                "kind": "delete",
                "target_path": str(target),
            }
        )
        return op

    def stage_write(self, path: str | Path, content: bytes | str) -> StagedFileOperation:
        """Stage new ``content`` for ``path`` without touching the real file yet."""
        target = Path(path)
        operation_id = uuid.uuid4().hex
        payload = content.encode("utf-8") if isinstance(content, str) else content
        op = StagedFileOperation(
            operation_id=operation_id,
            kind="write",
            target_path=target,
            quarantine_path=None,
            new_content=payload,
            _sandbox=self,
        )
        self.journal.record(
            {
                "event": "stage",
                "operation_id": operation_id,
                "kind": "write",
                "target_path": str(target),
            }
        )
        return op


class DbTransaction:
    """Rollback-by-default wrapper around a DBAPI2-style connection.

    The wrapped connection's transaction is rolled back unless
    :meth:`commit` is called explicitly before the ``with`` block exits —
    including when the block raises.

    Example:
        >>> with DbTransaction(conn) as tx:
        ...     conn.cursor().execute("UPDATE accounts SET balance = balance - 1 WHERE id = 1")
        ...     if approved:
        ...         tx.commit()
    """

    def __init__(self, connection: _DbApiConnection, journal: SandboxJournal | None = None) -> None:
        self._conn = connection
        self._committed = False
        self.journal = journal or SandboxJournal()

    def __enter__(self) -> DbTransaction:
        return self

    def commit(self) -> None:
        self._conn.commit()
        self._committed = True
        self.journal.record({"event": "db-commit"})

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self._committed:
            self._conn.rollback()
            self.journal.record(
                {"event": "db-rollback", "reason": str(exc) if exc else "not committed"}
            )
        # Returning None (falsy) never suppresses an exception raised inside the block.
