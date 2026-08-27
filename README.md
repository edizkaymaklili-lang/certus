# Certus

**A deterministic guardrail layer for autonomous AI agents, plus an edge-AI MLOps pipeline for shipping quantized models to constrained hardware.**

Certus is two things under one roof:

1. **Guardrail & Proxy** — a lightweight, self-hostable Python SDK and optional FastAPI gateway that sits between an autonomous agent (LLM tool-use, Anthropic/OpenAI function calling, or any custom agent runtime) and the real world. It validates every tool call against a strict schema, evaluates it against deterministic allow/deny policies, routes high-risk actions through human approval, and can sandbox destructive operations behind a commit/rollback gate.
2. **Claude-Colab Bridge** — a file-based contract (`config.json` / `metrics.json`) plus a packaging pipeline that lets an orchestrating agent (e.g. Claude) drive a Google Colab training loop and automatically produce INT8-quantized ONNX/TFLite artifacts for Raspberry Pi / mobile / embedded deployment.

No part of the guardrail's decision-making is probabilistic. A tool call is either schema-valid or it isn't; it either matches a policy rule or it doesn't. That determinism is the point — it's what makes the layer auditable and testable like any other piece of infrastructure.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # core SDK: pydantic, jsonschema, PyYAML only
pip install -e ".[proxy]"        # + FastAPI gateway
pip install -e ".[edge]"         # + torch/onnx/onnxruntime/tensorflow for the edge pipeline
pip install -e ".[dev]"          # + pytest/mypy/ruff for contributing
```

## Quickstart (3 lines)

```python
from certus import Certus
from pydantic import BaseModel

class DeleteFileArgs(BaseModel):
    path: str

guard = Certus()  # loads the packaged fail-closed default policy

@guard.protect(schema=DeleteFileArgs)
def delete_file(path: str) -> str:
    ...  # your real implementation
```

Calling `delete_file(path=...)` now transparently runs through:

```
ToolCall -> schema validation -> policy evaluation -> human approval (if required) -> your function -> audit log
```

`delete_file` matches the packaged default policy's `critical_tools` list, so it requires approval before it runs; wire a real approval channel (Slack, a generic webhook, a CLI prompt) via `approval_callback=` — see [Human approval channels](#human-approval-channels) below. See [`examples/quickstart.py`](examples/quickstart.py) for a runnable end-to-end demo, including a denylisted call being blocked outright and decision-only "gateway mode" usage.

## Project layout

```
src/certus/
  core/            Deterministic engine — no network, no LLM calls.
    models.py        ToolCall, PolicyDecision, ValidationResult, AuditRecord, RiskLevel
    schema_validator.py   JSON Schema / Pydantic argument validation (fail-closed)
    policy.py             Allowlist / denylist / regex rule engine
    exceptions.py         Typed, structured exception hierarchy
  proxy/           Execution interception layer.
    middleware.py    CertusGuard — the interceptor pipeline + @guard.protect decorator
    gateway.py       Optional FastAPI HTTP gateway (requires the `proxy` extra)
    sandbox.py       FileSandbox (quarantine + rollback) and DbTransaction (rollback-by-default)
    approval.py      CLI/auto-deny + WebhookApprovalCallback (Slack or any webhook) approval channels
  sdk/
    client.py        `Certus` — the top-level developer-facing client
  config/
    default_policy.yaml   Packaged, restrictive starting policy
  edge/            Edge-AI MLOps pipeline (Colab <-> orchestrator bridge).
    colab_bridge.py  TrainingConfig / TrainingMetrics contract + deterministic hyperparameter heuristic
    quantize.py      EdgePackager — PyTorch -> ONNX -> INT8 ONNX / INT8 TFLite (direct, via onnx2tf)

examples/
  quickstart.py                 Guardrail SDK end-to-end demo
  config.json                   Sample TrainingConfig
  colab_training_template.py    Real dataset -> trained model -> INT8 ONNX/TFLite, end to end

tests/            pytest suite covering every module above, plus a real
                  torch/onnx/onnx2tf integration test (tests/test_edge_packager_integration.py)
                  that's skipped unless the `edge` extra is installed
```

## Policy configuration

Policies are plain YAML (or JSON), evaluated top-to-bottom, first match wins per stage — see [`src/certus/config/default_policy.yaml`](src/certus/config/default_policy.yaml) for the full packaged default and inline comments on evaluation order:

```yaml
default_action: deny        # fail-closed: anything unclassified is denied
allowlist: [read_file, search_web]

denylist_tools:
  - id: deny-raw-shell
    tools: [exec_shell, run_command]
    reason: "Raw shell execution is not permitted."

denylist_patterns:
  - id: deny-secret-access
    fields: ["*"]
    pattern: '(?i)(\.env\b|/etc/passwd|id_rsa)'
    reason: "Attempted access to credentials or secrets."

critical_tools:
  - id: critical-payment
    tools: [charge_payment, transfer_funds]
    risk_level: critical
    requires_approval: true
    reason: "Financial transactions require human approval."
```

Load it with `Certus(policy_path="policy.yaml")` or `PolicyEngine.from_file(...)` for lower-level use.

## Sandbox & rollback

Destructive actions never have to hit real state immediately:

```python
from certus.proxy.sandbox import FileSandbox, DbTransaction

sandbox = FileSandbox(quarantine_dir=".certus/quarantine")
op = sandbox.stage_delete("reports/q3.csv")   # file is moved into quarantine, not deleted
# ... route through your ApprovalManager ...
op.commit()      # or op.rollback() to restore it exactly as it was

with DbTransaction(db_connection) as tx:
    cursor.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
    if approved:
        tx.commit()   # rolls back automatically if you never call commit()
```

## Human approval channels

Beyond the CLI prompt and fail-closed auto-deny built-ins, `WebhookApprovalCallback` bridges a real out-of-band channel — Slack, a ticketing system, or any webhook — into the guard pipeline. It stages the pending request, fires a notification, and blocks (with a timeout, failing closed if nothing responds) until a decision arrives:

```python
from certus import Certus
from certus.proxy.approval import PendingApprovalStore, WebhookApprovalCallback, slack_webhook_notifier

store = PendingApprovalStore()
guard = Certus(approval_callback=WebhookApprovalCallback(
    store=store,
    notify=slack_webhook_notifier("https://hooks.slack.com/services/..."),
    timeout=300,  # seconds to wait for a human before failing closed
))
```

Whatever receives the Slack message (an interactivity handler, or a person `curl`-ing directly) delivers the decision back via the gateway:

```bash
curl -X POST localhost:8000/v1/approvals/<request_id>/decision \
  -H 'content-type: application/json' \
  -d '{"approved": true, "approver": "ops-team"}'
```

Pass the same `store` to `create_app(guard, approval_store=store)` to expose that endpoint — see [`tests/test_gateway.py`](tests/test_gateway.py) for a full working example of a call being staged, approved out-of-band, and then completing.

## Gateway / Proxy mode

Deploy Certus in front of a separate tool-execution service instead of embedding it in-process:

```bash
pip install "certus-ai[proxy]"
uvicorn certus.proxy.gateway:app --reload
```

```bash
curl -X POST localhost:8000/v1/tool-calls \
  -H 'content-type: application/json' \
  -d '{"name": "delete_file", "arguments": {"path": "a.txt"}}'
```

Build your own `FastAPI` app with `create_app(guard)` for a properly configured `CertusGuard` (registered handlers, a real approval callback, a custom policy) rather than using the bare module-level `app`.

## Claude-Colab edge pipeline

```python
from certus.edge.colab_bridge import load_config, load_metrics, suggest_next_config

config = load_config("config.json")
metrics = load_metrics("metrics.json")
next_config = suggest_next_config(config, metrics, next_run_id="run-002")
```

`suggest_next_config` is a deterministic rule set (overfit detection, plateau detection, failed-run retry) an orchestrating agent can call, inspect, and override — never a hidden decision the pipeline makes on its own.

Run [`examples/colab_training_template.py`](examples/colab_training_template.py) end-to-end to see the full loop for real — no mocks, no synthetic placeholder data:

```bash
pip install "certus-ai[edge]"
python examples/colab_training_template.py --config examples/config.json --output-dir artifacts/run-001
```

It trains on the real, bundled `sklearn.datasets.load_digits` dataset (1,797 real handwritten-digit images — no network download required, unlike `torchvision.datasets.MNIST`), then produces `metrics.json`, a `model.onnx`, an ONNX Runtime-quantized `model.int8.onnx`, and a genuinely INT8-quantized `model.int8.tflite` (int8 input tensor, converted directly from ONNX via [onnx2tf](https://github.com/PINTO0309/onnx2tf), calibrated on real held-out validation images), plus a `manifest.json` describing every artifact.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The `edge` extra's heavy dependencies (torch, tensorflow, onnx2tf) are best installed in their own environment, since PyTorch/TensorFlow wheel availability can lag behind the newest CPython release:

```bash
python3.11 -m venv .venv-edge && source .venv-edge/bin/activate
pip install -e ".[edge,dev]"
pytest tests/test_edge_packager_integration.py -v   # real ONNX/TFLite conversion, not mocked
```

> **Known platform limitation:** the TFLite conversion (`to_tflite_int8_from_onnx`, and the `int8_tflite` target format) is verified working on macOS but crashes the process (`Fatal Python error: Floating point exception`) inside TensorFlow Lite's native calibrator on the GitHub Actions `ubuntu-latest` runner with `tensorflow==2.21.0`, independent of model or data. Not yet root-caused upstream. CI works around it — see `edge-extra-test` in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and the note in [CLAUDE.md](CLAUDE.md).

## Design principles

- **Zero black-box dependencies.** Security decisions are deterministic rule matches (exact names, compiled regex, JSON Schema), never model scores.
- **Fail-closed.** An unregistered tool or an unclassified call is rejected by default, not assumed safe.
- **Low latency.** Every stage before the real handler call is pure Python; no network round-trip, no model inference. Disable the audit journal (`audit_enabled=False`) if you're operating under a hard sub-20ms budget.
- **Modular & offline-capable.** The core engine (`certus.core`) depends only on `pydantic`, `jsonschema`, and `PyYAML` — it runs fully self-hosted, with no calls to any Certus-operated service.

## License

Apache-2.0
