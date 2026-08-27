"""Tests for certus.proxy.sandbox (staged file ops and DB transactions)."""

from __future__ import annotations

from certus.proxy.sandbox import DbTransaction, FileSandbox, SandboxJournal


class FakeDbConnection:
    """Minimal DBAPI2-shaped fake for testing DbTransaction."""

    def __init__(self):
        self.value = 0
        self.committed = False
        self.rolled_back = False

    def apply(self, delta: int) -> None:
        self.value += delta

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True
        self.value = 0


def test_stage_delete_moves_file_to_quarantine(tmp_path):
    target = tmp_path / "report.csv"
    target.write_text("important data")
    sandbox = FileSandbox(
        quarantine_dir=tmp_path / "quarantine", journal=SandboxJournal(tmp_path / "journal.jsonl")
    )

    op = sandbox.stage_delete(target)

    assert not target.exists()
    assert op.quarantine_path is not None and op.quarantine_path.exists()


def test_rollback_restores_deleted_file(tmp_path):
    target = tmp_path / "report.csv"
    target.write_text("important data")
    sandbox = FileSandbox(quarantine_dir=tmp_path / "quarantine")

    op = sandbox.stage_delete(target)
    op.rollback()

    assert target.exists()
    assert target.read_text() == "important data"


def test_commit_permanently_removes_file(tmp_path):
    target = tmp_path / "report.csv"
    target.write_text("important data")
    sandbox = FileSandbox(quarantine_dir=tmp_path / "quarantine")

    op = sandbox.stage_delete(target)
    op.commit()

    assert not target.exists()
    assert op.quarantine_path is not None and not op.quarantine_path.exists()


def test_stage_write_does_not_touch_disk_until_commit(tmp_path):
    target = tmp_path / "new_file.txt"
    sandbox = FileSandbox(quarantine_dir=tmp_path / "quarantine")

    op = sandbox.stage_write(target, "hello world")
    assert not target.exists()

    op.commit()
    assert target.read_text() == "hello world"


def test_db_transaction_rolls_back_by_default(tmp_path):
    conn = FakeDbConnection()
    journal = SandboxJournal(tmp_path / "db-journal.jsonl")

    with DbTransaction(conn, journal=journal):
        conn.apply(-100)
        # No commit() call -> should roll back on exit.

    assert conn.rolled_back
    assert not conn.committed
    assert conn.value == 0


def test_db_transaction_commits_when_explicitly_requested(tmp_path):
    conn = FakeDbConnection()

    with DbTransaction(conn) as tx:
        conn.apply(-100)
        tx.commit()

    assert conn.committed
    assert not conn.rolled_back
    assert conn.value == -100


def test_db_transaction_rolls_back_on_exception(tmp_path):
    conn = FakeDbConnection()

    try:
        with DbTransaction(conn):
            conn.apply(-100)
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert conn.rolled_back
    assert not conn.committed
