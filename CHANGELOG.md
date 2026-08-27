# Changelog

All notable changes to Certus are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project intends to adhere to [Semantic Versioning](https://semver.org/)
once it reaches 1.0.0. Before 1.0.0, minor versions may include breaking changes.

## [Unreleased]

## [0.1.0] — 2026-08-27

Initial release: the guardrail SDK/proxy and the Claude-Colab edge pipeline,
end to end and verified with real (non-mocked) runs.

### Added

- **Core (`certus.core`)**: `SchemaValidator` (fail-closed JSON Schema /
  Pydantic argument validation), `PolicyEngine` (deterministic
  allowlist/denylist/regex rule engine), and the shared `ToolCall` /
  `PolicyDecision` / `ValidationResult` / `AuditRecord` models. Zero
  optional dependencies beyond `pydantic`, `jsonschema`, and `PyYAML`.
- **Proxy (`certus.proxy`)**: `CertusGuard`, the execution interceptor
  tying schema validation, policy evaluation, and approval together, with
  a `@guard.protect(schema=...)` decorator for 3-line integration.
  `FileSandbox` and `DbTransaction` for commit/rollback around destructive
  operations. `ApprovalManager` with a CLI prompt, fail-closed auto-deny,
  and a real `WebhookApprovalCallback` (Slack or any webhook) backed by
  `PendingApprovalStore`. An optional FastAPI gateway
  (`certus.proxy.gateway`) exposing `POST /v1/tool-calls` and
  `POST /v1/approvals/{request_id}/decision`.
- **SDK (`certus.sdk`)**: the `Certus` client — the top-level, 3-line
  developer-facing entrypoint wrapping schema/policy/approval.
- **Edge (`certus.edge`)**: `TrainingConfig`/`TrainingMetrics` (the
  Claude-Colab `config.json`/`metrics.json` contract), the deterministic
  `suggest_next_config` hyperparameter heuristic, and `EdgePackager`
  (PyTorch → ONNX → ONNX Runtime INT8 → TFLite INT8 via `onnx2tf`).
- A packaged, restrictive `default_policy.yaml` (denylists for shell/SQL
  injection, credential access, SSRF, and prompt-injection markers;
  approval-gated critical tools for file/DB/payment/comms operations).
- Runnable examples: `examples/quickstart.py` (guardrail SDK) and
  `examples/colab_training_template.py` (real `sklearn.datasets.load_digits`
  training run producing real INT8 ONNX/TFLite artifacts and a manifest).
- Apache-2.0 `LICENSE`, GitHub Actions CI (lint, strict mypy, tests across
  Python 3.11/3.12, a FastAPI gateway smoke test, and a dedicated `edge`
  extra integration test job).

### Known limitations

- `EdgePackager.to_tflite_int8_from_onnx` is verified working on macOS but
  crashes the process (`Fatal Python error: Floating point exception`)
  inside TensorFlow Lite's native calibrator on Linux
  (`tensorflow==2.21.0`, reproduced on GitHub Actions' `ubuntu-latest`
  runner). Not yet root-caused upstream — see the "Known platform
  limitation" note in [CLAUDE.md](CLAUDE.md). CI works around it; a real
  Linux user requesting `int8_tflite` may hit the same crash.
- Not yet published to PyPI — install via `pip install -e .` from a clone
  until a release is published (see `.github/workflows/publish.yml`).
