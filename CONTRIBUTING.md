# Contributing to VoxLIT

Thank you for your interest in contributing to VoxLIT! This document explains how to set up the project, how work is organised across the task workbenches, and how changes reach `main`.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Development Workflow](#development-workflow)
  - [Task Ownership](#task-ownership)
  - [Branching Strategy](#branching-strategy)
  - [Making Changes](#making-changes)
  - [Testing](#testing)
  - [Code Style and Linting](#code-style-and-linting)
  - [What Not to Commit](#what-not-to-commit)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behaviour to the project maintainers listed in the [README](README.md#authors).

## Getting Started

### Prerequisites

- **Frontend**: Node.js 18 or higher, and npm
- **Backend**: Python 3.11
- **Redis**: Docker Desktop
- **Gated models**: a Hugging Face account and access token (needed for pyannote and other gated checkpoints)

### Setting Up the Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VoxLIT/VoxLIT.git
   cd VoxLIT
   ```

2. **Set up the Frontend**:
   ```bash
   cd Frontend
   npm install
   ```

3. **Set up the Backend**:
   ```bash
   cd Backend
   python3.11 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Put your Hugging Face token in a `.env` file. It is ignored by git; never commit it.

4. **Start Redis** (Docker Desktop must be running):
   ```bash
   cd Backend
   docker compose up -d
   ```

5. **Run the development servers**:

   Frontend:
   ```bash
   cd Frontend
   npm run dev
   ```

   Backend (in a second terminal):
   ```bash
   cd Backend
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   uvicorn app.main:app --reload
   ```

## Development Workflow

### Task Ownership

VoxLIT is split into task workbenches, each owned by one team member. Ownership is recorded in [CODEOWNERS](CODEOWNERS).

- **Your task area**: `Frontend/src/features/<task>/` and `Backend/app/tasks/<task>/`. Work here freely.
- **Other members' task areas**: do not edit them. If a shared component does not fit your task, create a task-specific version inside your own feature folder instead of changing someone else's.
- **Shared code** (`Frontend/src/components/`, `Frontend/src/tasks/`, `Backend/app/api/`, `Backend/app/services/`, `Backend/app/tasks/registry.py`, `Backend/app/main.py`): changes need review from the whole team.

See [STRUCTURE.md](STRUCTURE.md) for how the task registry works and how to add a new task, model or dataset.

### Branching Strategy

- `main` is the stable branch and is what gets deployed.
- Each member works on their own branch (for example `Chanupa`, `nethsith`, `hatheem`). Short-lived branches such as `feature/short-description` or `bugfix/short-description` are also fine.
- Keep your branch up to date with `main` before opening a pull request, either with GitHub's **Update branch** button or locally:
  ```bash
  git checkout <your-branch>
  git pull origin main
  ```

### Making Changes

1. Switch to your branch and bring it up to date:
   ```bash
   git checkout <your-branch>
   git pull origin main
   ```

2. Stage only the files you changed, and commit with a descriptive message:
   ```bash
   git add <files>
   git commit -m "Describe what changed and why"
   ```

3. Push your branch:
   ```bash
   git push origin <your-branch>
   ```

### Testing

- **Frontend** (Vitest):
  ```bash
  cd Frontend
  npm test
  ```
  Run a single task's tests with a path filter, for example `npx vitest run src/features/deepfake`.

- **Backend** (pytest):
  ```bash
  cd Backend
  pytest
  ```

- **Production build** (catches type and import errors that the dev server tolerates):
  ```bash
  cd Frontend
  npm run build
  ```

### Code Style and Linting

- **Frontend**: TypeScript with ESLint:
  ```bash
  cd Frontend
  npm run lint
  ```
- **Backend**: follow PEP 8. Type hints are encouraged.
- **Interpretability methods**: every attribution or analysis shown in the interface should state its method and known limits, and per-clip views must not show ground-truth labels (see [Research Practice](README.md#research-practice)).

### What Not to Commit

- Secrets: `.env`, tokens, Modal credentials
- Model weights, datasets and uploaded audio (`Backend/pretrained_models/`, `Backend/data/`, `Backend/uploads/`)
- Build output and caches: `Frontend/dist/`, `node_modules/`, `__pycache__/`, `.venv/`
- OS files such as `.DS_Store` (already in `.gitignore`)

If a dataset or model is needed, add a download or preparation script under `Backend/scripts/` instead.

## Pull Request Process

1. Make sure your branch is up to date with `main`, and that tests, lint and the frontend build pass.
2. Open a pull request into `main` with a clear title and a description of what changed and how you tested it.
3. Reference related issues with "Fixes #issue_number".
4. Changes to shared code need approval from the other members. Changes confined to your own task area can be merged after a quick check.
5. Address review comments, then merge.

## Documentation

- Update [README.md](README.md) when you add a task, model, dataset or user-facing feature.
- Update [STRUCTURE.md](STRUCTURE.md) when the task registry or project layout changes.
- Record notable changes in [CHANGELOG.md](CHANGELOG.md).
- Comment non-obvious logic, especially model loading, preprocessing and metric calculations.

## Issue Reporting

When reporting an issue, please include:

- A clear and descriptive title
- The affected task workbench and model
- Steps to reproduce the problem
- Expected and actual behaviour
- Screenshots or backend logs if applicable
- Environment details (OS, browser, Python and Node versions)

For security vulnerabilities, follow the [Security Policy](SECURITY.md) instead of opening a public issue.

## Feature Requests

We welcome feature requests! Please describe:

- The feature and the task workbench it belongs to
- The motivation and research use case
- Any implementation ideas
- Mockups or examples if applicable

Thank you for contributing to VoxLIT!
