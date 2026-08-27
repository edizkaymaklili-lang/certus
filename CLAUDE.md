# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Certus is two things sharing one package:

1. **Guardrail & Proxy** (`certus.core`, `certus.proxy`, `certus.sdk`) — a deterministic
   interception layer for autonomous AI agent tool calls: schema validation → policy
   evaluation → human approval → sandboxed/real execution → audit log. No stage before
   the real handler call uses an LLM or any probabilistic scoring — every decision is an
   exact-match or regex rule, by design (see "Design invariants" below).
2. **Claude-Colab edge bridge** (`certus.edge`) — a file-based contract (`config.json` /
   `metrics.json`) plus a packaging pipeline that lets an orchestrating agent drive a
   Colab training loop and produce INT8-quantized ONNX/TFLite artifacts for edge hardware.

## Commands

```bash
# Environment setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + pytest/mypy/ruff
pip install -e ".[proxy]"        # + FastAPI gateway (needed for certus.proxy.gateway)

# The `edge` extra (torch/onnx/onnxruntime/tensorflow/onnx2tf) is heavy and its
# wheel availability can lag behind the newest CPython — use a dedicated
# venv pinned to a well-supported interpreter (3.11) rather than mixing it
# into the main dev venv:
python3.11 -m venv .venv-edge && source .venv-edge/bin/activate
pip install -e ".[edge,dev]"

# Test
pytest                                              # full suite (edge integration tests auto-skip without the `edge` extra)
pytest tests/test_policy.py                         # one file
pytest tests/test_policy.py::test_allowlisted_tool_is_allowed -v   # one test
pytest -k "approval"                                # by keyword
pytest tests/test_edge_packager_integration.py -v   # real ONNX/TFLite conversion (.venv-edge only)

# Lint / type-check (both run in CI, both must be clean before committing)
ruff check .          # add --fix for auto-fixable issues
mypy                  # strict mode, config in pyproject.toml [tool.mypy]

# Run the examples
python examples/quickstart.py                                  # guardrail SDK, no extras needed
python examples/colab_training_template.py --config examples/config.json --output-dir artifacts/run-001  # needs the `edge` extra
uvicorn certus.proxy.gateway:app --reload                       # needs the `proxy` extra
```

`certus.core` has zero optional dependencies (pydantic, jsonschema, PyYAML only) — never
add an import at module scope in `certus/core/**` or `certus/proxy/{middleware,sandbox,approval}.py`
that isn't in the base dependency list. `certus.proxy.gateway` (FastAPI) and
`certus.edge.quantize` (torch/onnx/onnxruntime/tensorflow) import their heavy dependencies
lazily — inside functions/methods, or behind a top-of-module `try/except ImportError` that
raises a message pointing at the right extra — for exactly this reason.

## Architecture

### The interception pipeline (the core mental model)

Every tool call flows through the same ordered pipeline, implemented once in
`CertusGuard` ([src/certus/proxy/middleware.py](src/certus/proxy/middleware.py)) and
reused by both the decorator API and the gateway:

```
ToolCall -> SchemaValidator.validate -> PolicyEngine.evaluate -> ApprovalManager (if required) -> handler -> AuditJournal
```

- `SchemaValidator` ([core/schema_validator.py](src/certus/core/schema_validator.py)) is
  **fail-closed**: a tool with no registered schema raises `UnknownToolError` rather than
  being treated as valid. This is enforced at the type level, not by convention.
- `PolicyEngine` ([core/policy.py](src/certus/core/policy.py)) evaluates stages in a fixed
  order — denylist-by-name, denylist-by-regex, critical-tools, allowlist, default action —
  and the **first match wins**; denylist rules always beat a `critical_tools` "allowed with
  approval" classification for the same tool (see `test_denylist_pattern_wins_over_critical_tool`).
- `CertusGuard` exposes two execution modes that share the same validation/policy/approval
  logic but diverge after that:
  - `intercept(tool_call)` — requires a registered handler, executes it, raises on any
    denial/rejection. This is what `@guard.protect(schema=...)` wires up.
  - `evaluate(tool_call)` — never raises; returns a `GuardDecision` with an `.ok` property.
    Used by `certus.proxy.gateway` when no local handler is registered, so a remote
    executor can act on the verdict itself (decision-only / pure-gateway deployment mode).

### Sandbox and approval are separate, composable concerns

`FileSandbox` / `DbTransaction` ([proxy/sandbox.py](src/certus/proxy/sandbox.py)) implement
commit/rollback for destructive operations (quarantine directory for files, rollback-by-default
transaction wrapper for DB connections) but are **not wired into the guard pipeline
automatically** — a tool handler decides whether to use them internally. `ApprovalManager`
([proxy/approval.py](src/certus/proxy/approval.py)) is the pluggable human-in-the-loop gate;
it defaults to `auto_deny_callback` (fail-closed) if no channel is configured, so a
`critical_tools` policy match with no approval manager wired raises `ApprovalRequiredError`
rather than silently proceeding.

`WebhookApprovalCallback` is the real out-of-band channel (Slack or any webhook): it
registers the pending request in a `PendingApprovalStore`, fires a `notify` callable
(e.g. `slack_webhook_notifier(url)`), and *blocks the calling thread* up to `timeout`
seconds waiting for something to call `store.resolve(request_id, response)` — normally
the gateway's `POST /v1/approvals/{request_id}/decision` route (pass the same `store` to
`create_app(guard, approval_store=store)`). It fails closed on both a notify error and a
timeout. Because FastAPI runs sync `def` route handlers in a thread pool, blocking inside
`submit_tool_call` this way does not deadlock the gateway — see `test_gateway.py::test_approval_decision_endpoint_unblocks_pending_call`
for the concurrency pattern (call the tool in a background thread, resolve the pending
request from the test's main thread, join).

### Edge module has an independent data contract

`certus.edge.colab_bridge` defines the `TrainingConfig`/`TrainingMetrics` Pydantic models
that are the *only* coupling between an orchestrating agent and the Colab training script —
there is no shared runtime, just JSON files. `suggest_next_config` is a deterministic
decision tree (overfit / plateau / failed-run detection), mirroring the same
"no black-box decisions" principle as the guardrail side; it's meant to be called and
overridden, not treated as authoritative.

`EdgePackager.to_tflite_int8_from_onnx` converts ONNX straight to a fully INT8-quantized
TFLite model via `onnx2tf`, calibrated on a real (or realistic) representative dataset saved
to a temporary `.npy` file — `onnx2tf`'s calibration API takes a *file path*, not an
in-memory array. Two non-obvious gotchas discovered by actually running this end to end
(both now fixed, but worth knowing if this code changes):
- `_require()` must use `importlib.import_module`, not the `__import__` builtin directly —
  `__import__("onnxruntime.quantization")` returns the top-level `onnxruntime` package, not
  the `quantization` submodule.
- `EdgePackager.to_onnx` defaults to `dynamo=False` (the legacy TorchScript-based ONNX
  exporter). PyTorch's newer `torch.export`-based exporter (default in `torch.onnx.export`
  since 2.9) has been observed to emit `ReduceMean`/`Reshape` graphs for pooling layers that
  fail ONNX shape inference inside `quantize_onnx_int8`/`to_tflite_int8_from_onnx` with
  `InferenceError: Inferred shape and existing shape differ`. Only flip `dynamo=True` after
  verifying the specific model converts cleanly end to end.

### Design invariants to preserve when extending this codebase

- **Fail-closed everywhere**: an unregistered tool, an unmatched policy, or a missing
  approval channel must resolve to "denied", never "allowed by omission."
- **No probabilistic gating in `certus.core` or `certus.proxy`**: policy/schema decisions
  are exact-match or compiled-regex only. If a feature needs an LLM call to decide
  allow/deny, it belongs in a layer clearly separated from this guarantee, not inside
  `PolicyEngine`/`SchemaValidator`.
- **Low overhead**: every stage before the real handler call is pure Python (no network,
  no model inference). The one deliberate I/O cost is the audit journal write; it's the
  documented escape hatch (`audit_enabled=False`) for latency-sensitive deployments.

## Project history

- **Phase 1 (complete)** — built per the original 4-step plan: project scaffold
  (`pyproject.toml`, src-layout), the schema/policy validator engine, the execution
  interceptor + sandbox + approval + FastAPI gateway, and the Colab-Claude bridge with a
  runnable training template. Verified with 38 passing pytest cases, clean `ruff`, and
  clean strict `mypy` — commit `0ded709`.
- Added `LICENSE` (Apache-2.0) and `.github/workflows/ci.yml` (ruff + mypy + pytest on
  3.11/3.12, plus a gateway import smoke test) — commit `0c7689e`.
- Added `CLAUDE.md` — commit `3af919a`.
- Pushed to GitHub as a **public** repo: https://github.com/edizkaymaklili-lang/certus
  (decision: monetization plans — open-core, hosted gateway, enterprise support — all
  depend on adoption, which a private repo can't build). Added `.github/FUNDING.yml`
  wiring the Sponsor button to GitHub Sponsors (goes live once that application, a manual
  step for the repo owner, is approved).
- **Gaps closed** (all three verified with real, non-mocked runs, not just unit tests):
  1. Real human approval channel: `WebhookApprovalCallback` + `PendingApprovalStore` +
     the gateway's `POST /v1/approvals/{request_id}/decision` endpoint, tested against an
     actual local HTTP server (`tests/test_approval_webhook.py`) and a real threaded
     request/resolve/unblock cycle through the FastAPI gateway (`tests/test_gateway.py`).
  2. Real dataset: `examples/colab_training_template.py` now trains on
     `sklearn.datasets.load_digits` (real handwritten digits, bundled — no network
     download) instead of random tensors.
  3. Real ONNX→TFLite conversion: `EdgePackager.to_tflite_int8_from_onnx` via `onnx2tf`,
     producing a genuinely INT8-quantized model (verified with the `ai_edge_litert`
     interpreter — int8 input tensor). Getting this working end to end surfaced and fixed
     two real bugs — see the two gotchas under "Edge module" above (`_require`'s
     `__import__` vs `importlib.import_module`, and the dynamo exporter's shape-inference
     failure) — plus the full `edge` extra dependency list in `pyproject.toml` (onnx2tf
     pulls in `onnx-graphsurgeon`, `sng4onnx`, `onnxsim`, `ai-edge-litert`, `psutil`,
     `tf-keras`, `onnxscript`, none of which it declares as hard requirements itself).
  A dedicated CI job (`edge-extra-test`) now installs the `edge` extra and runs the full
  training template on every push/PR, so none of this can silently regress. That job
  installs the **CPU-only torch wheel** (`--index-url https://download.pytorch.org/whl/cpu`)
  before the `edge` extra — the default Linux wheel pulls in the full CUDA toolkit and
  crashed the GitHub Actions runner with `Fatal Python error: Floating point exception`
  (no GPU/driver present); this only surfaced in CI (Linux), not in local testing (macOS).

## Roadmap

**Near-term engineering:**
- GitHub Sponsors: the repo side (`FUNDING.yml`) is ready; the actual application at
  github.com/sponsors is a manual, off-platform step for the repo owner.
- Consider adding a Slack *interactivity* handler example (a small server that turns a
  Slack button click into a call to the gateway's approval-decision endpoint) — right now
  `slack_webhook_notifier` only covers the outbound notification half.

**Product direction (from monetization discussion, not yet started):**
Certus is positioned as an open-core B2B infra project — the plan is to keep
`certus.core`/`certus.proxy`/`certus.sdk` permissively licensed to drive adoption, and
build revenue on top rather than by gating the core:
1. **Open-core enterprise module**: centralized policy management across many
   agents/tools, SIEM export for the audit journal, SSO/RBAC, compliance reporting
   (SOC 2, EU AI Act) — a paid layer on top of the free `CertusGuard`.
2. **Hosted gateway ("Certus Cloud")**: run `certus.proxy.gateway` as a managed service,
   billed per tool-call volume or seat, instead of requiring self-hosting.
3. **Edge pipeline as a service**: hosted build pipeline around `EdgePackager`
   (config in → quantized artifact out), billed per build-minute or per device deployment.
4. **Enterprise support/SLA**: paid support and custom policy authoring on top of the free
   OSS core — the classic model, lowest engineering lift, always available as a fallback.

None of the product-direction items have corresponding code yet; when work starts on one,
it should land as an additively-licensed layer (e.g. a separate package or a clearly
gated module) rather than changing the license or behavior of the existing `certus.core`.
