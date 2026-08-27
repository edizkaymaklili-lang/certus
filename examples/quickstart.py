"""Certus quickstart: guard a destructive tool in three lines.

Run with:

    pip install -e .
    python examples/quickstart.py
"""

from __future__ import annotations

from pydantic import BaseModel

from certus import Certus
from certus.proxy.approval import ApprovalResponse


class DeleteFileArgs(BaseModel):
    """Argument schema for the `delete_file` tool below."""

    path: str


# --- The 3-line integration -------------------------------------------------
guard = Certus(approval_callback=lambda req: ApprovalResponse(
    request_id=req.request_id, approved=True, approver="demo-auto-approver"
))


@guard.protect(schema=DeleteFileArgs)
def delete_file(path: str) -> str:
    """Pretend to delete a file. In a real integration this touches disk/API."""
    return f"deleted: {path}"


# -----------------------------------------------------------------------------


def main() -> None:
    # `delete_file` is in the packaged default policy's `critical_tools` list,
    # so this call is schema-validated, policy-classified as HIGH risk, routed
    # through the (auto-approving, for this demo) approval callback, and only
    # then actually executed.
    result = delete_file(path="reports/q3-draft.csv")
    print(f"Guarded call result: {result}")

    # A call that matches a denylist pattern is rejected outright, with no
    # approval step at all — denylist rules always win over approval flows.
    try:
        delete_file(path="/etc/passwd")
    except Exception as exc:  # certus.CertusError subclasses in practice
        print(f"Blocked as expected: {exc}")

    # Decision-only mode: evaluate a call without executing anything, useful
    # when Certus is deployed as a pure gateway in front of a remote executor.
    # This second client never registered a schema for `send_email`, so it
    # fails closed: an unrecognized tool is never assumed safe.
    gateway_guard = Certus()
    email_args = {"to": "finance@example.com", "body": "wire $10k"}
    decision = gateway_guard.evaluate("send_email", email_args)
    print(f"Gateway-style decision: ok={decision.ok}, reason={decision.reason}")


if __name__ == "__main__":
    main()
