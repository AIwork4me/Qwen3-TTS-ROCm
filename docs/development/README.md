# Development history

Internal working documents kept for provenance and audit. Ordinary users do
not need anything here; start at the [README](../../README.md) instead.

| File | What it is |
|---|---|
| `2026-08-27-design-spec.md` | Approved v0.1.0 design spec — scope arbiter for "is a change in scope?" (referenced from [CONTRIBUTING.md](../../CONTRIBUTING.md)) |
| `2026-08-27-implementation-plan.md` | Step-by-step implementation plan that produced v0.1.0 |
| `2026-08-27-final-acceptance.md` | v0.1.0 acceptance record: CPU/GPU suite runs, packaging, Docker, browser E2E matrix, hardening passes |
| `gpu-ci-runbook.md` | GPU CI runbook (2026-09-21): how to activate and maintain the self-hosted `gfx1151` runner behind `gpu-nightly.yml` — **STATUS: BLOCKED ON RUNNER INFRASTRUCTURE**, workflow validated locally, never executed on a runner |

New process documents for future releases go here rather than the repository
root, so the public tree stays focused on user-facing documentation and
validation evidence (`evidence/`).
