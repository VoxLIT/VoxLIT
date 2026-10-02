# Security Policy

## Supported Versions

Security fixes are applied to the latest release of VoxLIT on the `main` branch.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0 (LIT for Voice) | :x: |

## Reporting a Vulnerability

We take the security of VoxLIT seriously. If you believe you have found a vulnerability, please follow these steps:

1. **Do not disclose publicly.** Do not open a public issue, pull request or discussion about the vulnerability until it has been fixed.

2. **Report it privately.** Use GitHub's private reporting (**Security → Report a vulnerability**) on this repository, or contact one of the maintainers listed under [Authors](README.md#authors) directly.

3. **Provide details.** Please include:
   - A description of the vulnerability
   - The affected component (Frontend, Backend API, a specific task workbench, or the Modal deployment)
   - Steps to reproduce the issue
   - The potential impact
   - Any suggested fix (optional)

4. **Response.** We aim to acknowledge reports within 72 hours. VoxLIT is maintained by a small academic team, so fixes are made on a best-effort basis, prioritised by severity.

5. **Disclosure.** We will work with you to validate the issue, agree on a fix, and credit you once it is released (if you wish).

## Scope

In scope:

- The FastAPI backend (`Backend/app/`), including file upload, inference and analysis endpoints
- The React frontend (`Frontend/`)
- The deployment configuration (`deploy/`)

Out of scope:

- Vulnerabilities in third-party models, datasets or dependencies themselves (please report those upstream; we will update our pinned versions)
- Attacks that require access to the Modal account, the Hugging Face token or the host machine

## Security Best Practices

When developing or deploying VoxLIT:

1. **Secrets.** Never commit tokens or credentials. Keep the Hugging Face token in `.env` locally (already ignored by git) and in the `voxlit-hf` Modal Secret in production.

2. **Model checkpoints.** Load checkpoints only from trusted, pinned sources. Prefer `torch.load(..., weights_only=True)`; a checkpoint loaded with full pickling can execute arbitrary code.

3. **Uploaded audio.** Treat uploaded files as untrusted input. Validate file type and size, and do not keep user uploads longer than needed.

4. **Data protection.** Voice recordings are personal data. Only process audio you have consent to use, and follow the licence terms of each reference dataset (VoxCeleb, AMI, RAVDESS, Common Voice, ASVspoof).

5. **API exposure.** Restrict CORS origins (`ALLOWED_ORIGINS`) to the frontend's domain, serve everything over HTTPS, and add authentication and rate limiting before exposing compute-heavy endpoints publicly.

6. **Dependencies.** Keep `Frontend/package.json` and `Backend/requirements.txt` up to date, and review `npm audit` and `pip-audit` output periodically.

Thank you for helping to keep VoxLIT secure!
