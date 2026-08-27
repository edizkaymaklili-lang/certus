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
pip install -e ".[edge]"         # + torch/onnx/onnxruntime/tensorflow

# Test
pytest                                              # full suite
pytest tests/test_policy.py                         # one file
pytest tests/test_policy.py::test_allowlisted_tool_is_allowed -v   # one test
pytest -k "approval"                                # by keyword

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

### Edge module has an independent data contract

`certus.edge.colab_bridge` defines the `TrainingConfig`/`TrainingMetrics` Pydantic models
that are the *only* coupling between an orchestrating agent and the Colab training script —
there is no shared runtime, just JSON files. `suggest_next_config` is a deterministic
decision tree (overfit / plateau / failed-run detection), mirroring the same
"no black-box decisions" principle as the guardrail side; it's meant to be called and
overridden, not treated as authoritative.

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
  clean strict `mypy` — see commit `0ded709`.
- Added `LICENSE` (Apache-2.0, matching `pyproject.toml`'s declared classifier) and
  `.github/workflows/ci.yml` (ruff + mypy + pytest on 3.11/3.12, plus a gateway import
  smoke test) — commit `0c7689e`.
- Local git repo initialized; not yet pushed to GitHub. `gh` CLI installed via Homebrew;
  device-login flow was started but the repo's public/private visibility and the actual
  `gh repo create` + push are still pending a decision from the project owner.

## Roadmap

**Near-term engineering (natural continuation of Phase 1):**
- Wire a real approval channel beyond the CLI/auto-deny built-ins (Slack, a webhook, a
  ticketing system) as a concrete `ApprovalCallback` implementation.
- Swap the Colab template's synthetic tensors for a real dataset (e.g. `torchvision.MNIST`)
  and add the ONNX→TFLite conversion step (`onnx2tf` or equivalent) currently left as a
  documented gap in `examples/colab_training_template.py`.
- Push the repo to GitHub once visibility (public/private) is decided; CI is already
  configured to run on push/PR.

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
