# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | yes       |
| 0.1.x   | yes       |

## Reporting a vulnerability

Please **do not open a public GitHub issue** for anything you believe is a
security problem.

**Contact channel:** GitHub Private Security Advisories — this is the only
reporting channel; there is no security-contact email for this project. On
the repository page: *Security → Advisories → New draft security advisory*
("Report a vulnerability"). This keeps details embargoed and lets maintainers
collaborate and publish a fix and advisory properly.

What to include, if possible:

- affected component (`env.py`, `models.py`, `loader.py`, demo backend/UI,
  `scripts/*`);
- reproduction steps or a minimal proof of concept;
- impact assessment and any suggested mitigation.

## What to expect

- Acknowledgement within 7 days.
- Assessment and fix target within 90 days for accepted reports.
- You will be credited in the release notes unless you prefer otherwise.

## Scope notes

In scope:

- this repository's own code: environment diagnostics, model registry,
  downloader, loader, demo backend/UI validation, packaging scripts;
- supply-chain issues in how this project pins or fetches dependencies and
  model weights (e.g. unexpected sources or tampered downloads).

Out of scope:

- vulnerabilities inside the upstream `qwen-tts` package, PyTorch, ROCm or
  model weights themselves — report those to Alibaba's Qwen team via their
  official channels;
- the demo server's plaintext-HTTP microphone limitation when run without a
  reverse proxy: it is documented behavior for local-only use, not a defect;
- missing GPU/driver hardening on the operator's machine.

## Hardening posture of this project

The downloader only talks to the two official registries (Hugging Face /
ModelScope) using pinned repository ids; model weights are never stored in
this repository or shipped in wheels; environment probing is read-only and
designed never to raise; uploaded audio files are validated server-side by
the demo backend before use.
