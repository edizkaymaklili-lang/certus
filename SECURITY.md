# Security Policy

Certus is a security-focused project (a guardrail layer for autonomous AI
agent tool calls), so we take vulnerability reports seriously and will
prioritize them over regular bug reports.

## Supported versions

Certus is pre-1.0 (`0.x`). Only the latest released version — and `main`
— receive security fixes; there is no long-term-support branch yet.

| Version | Supported |
| ------- | --------- |
| latest `0.x` release / `main` | ✅ |
| older `0.x` releases | ❌ |

## Reporting a vulnerability

**Do not open a public GitHub issue for a security vulnerability.**

Instead, use GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/edizkaymaklili-lang/certus/security) of this repository.
2. Click **"Report a vulnerability"**.
3. Describe the issue: affected file(s)/function(s), a minimal
   reproduction if possible, and the potential impact (e.g. "a crafted
   tool-call argument bypasses the `denylist_patterns` regex stage").

This opens a private advisory visible only to the maintainers until a fix
is ready, so the issue isn't disclosed before it's patched.

## Scope

Certus's core security guarantee is that `certus.core` (schema validation)
and `certus.proxy` (policy evaluation, approval, sandboxing) make
**deterministic, fail-closed** decisions. In-scope reports include, but
aren't limited to:

- A tool call that should be denied by the packaged `default_policy.yaml`
  (or by a documented deterministic rule) but is allowed instead.
- A way to bypass `SchemaValidator`'s fail-closed behavior for an
  unregistered tool.
- A `FileSandbox`/`DbTransaction` operation that becomes irreversible
  despite never being committed.
- A `WebhookApprovalCallback`/gateway approval-decision flow that can be
  triggered or spoofed without the intended out-of-band authorization.

Denial-of-service reports (e.g. a regex that's slow on adversarial input)
are welcome but lower priority than a correctness bypass of an allow/deny
decision.

## Disclosure

We aim to acknowledge a report within a few days and to publish a fix
(with credit to the reporter, unless they prefer to stay anonymous) once
one is available. As a young project without a dedicated security team,
timelines are best-effort, not contractual.
