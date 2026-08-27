# Contributing to Certus

Thanks for considering a contribution. This is a young, actively-developed
project — expect the process below to be lightweight.

## Development setup

Certus splits into a lightweight core and two heavy optional extras
(`proxy`, `edge`). Use two virtual environments so the `edge` extra's
multi-gigabyte ML dependencies never leak into everyday development:

```bash
# Main dev environment: core + proxy + dev tooling
python -m venv .venv && source .venv/bin/activate
pip install -e ".[proxy,dev]"

# Separate environment for the edge pipeline (pinned to 3.11: PyTorch/
# TensorFlow wheel availability lags behind the newest CPython release)
python3.11 -m venv .venv-edge && source .venv-edge/bin/activate
pip install -e ".[edge,dev]"
```

## Running checks

All three must be clean before opening a PR — they all run in CI:

```bash
ruff check .          # lint; add --fix for auto-fixable issues
mypy                  # strict mode, config in pyproject.toml [tool.mypy]
pytest                # full suite (edge-only tests auto-skip without .venv-edge)
```

Edge-specific tests (real ONNX/TFLite conversion, not mocked) only run
with the `edge` extra installed:

```bash
source .venv-edge/bin/activate
pytest tests/test_edge_packager_integration.py -v
```

See the **Known platform limitation** note in [CLAUDE.md](CLAUDE.md)
before debugging a TFLite-conversion crash on Linux — it's a known,
unresolved upstream issue, not necessarily your code.

## Design invariants (please preserve these)

These are the properties that make Certus's guardrail layer trustworthy;
changes that weaken them need a strong justification in the PR description:

- **Fail-closed everywhere.** An unregistered tool, an unmatched policy,
  or a missing approval channel must resolve to "denied," never "allowed
  by omission."
- **No probabilistic gating in `certus.core` or `certus.proxy`.**
  Policy/schema decisions are exact-match or compiled-regex only — never
  an LLM call or a model score. If a feature needs probabilistic judgment,
  it belongs in a layer clearly separated from this guarantee.
- **Low overhead.** Every stage before the real handler call in
  `CertusGuard.intercept`/`evaluate` is pure Python — no network call, no
  model inference. The audit journal write is the one deliberate I/O cost;
  it has a documented escape hatch (`audit_enabled=False`).
- **`certus.core` stays dependency-light.** Only `pydantic`, `jsonschema`,
  and `PyYAML` — no import in `certus/core/**` (or the always-loaded parts
  of `certus/proxy/`) that isn't in the base dependency list. Heavy or
  optional dependencies (FastAPI, torch, tensorflow, onnx2tf) are imported
  lazily, inside functions or behind a `try/except ImportError` pointing
  at the right extra.

## Commit / PR conventions

- Keep commits focused; a commit message should explain *why*, not just
  restate the diff.
- Run the full check suite (`ruff`, `mypy`, `pytest`) locally before
  pushing — CI runs the same checks and will block merge otherwise.
- For a behavior change, update the relevant test file in `tests/` in the
  same PR — this project does not accept untested behavior changes to
  `certus.core` or `certus.proxy`.
- Update [CHANGELOG.md](CHANGELOG.md)'s `[Unreleased]` section for any
  user-visible change.

## Reporting bugs / security issues

Regular bugs: open a GitHub issue with a minimal reproduction.

Security vulnerabilities: **do not** open a public issue — see
[SECURITY.md](SECURITY.md).
