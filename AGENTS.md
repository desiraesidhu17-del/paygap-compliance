# How to work on this repo (for Codex & humans)

## Stack
- Python 3.11, Flask
- Data: pandas
- PDF: Playwright (Chromium)

## Setup
- `pip install -r requirements.txt`
- `playwright install chromium`  # required for PDF export smoke tests

## Commands
- Tests: `pytest -q`
- Lint (if installed): `ruff check .`
- Type check (if installed): `mypy .`

## Constraints
- Keep business logic in paygap_compliance/services/*
- Routes live in paygap_compliance/routes/*
- Don’t edit storage/public/reports/* (generated)
- Aim for small PRs with tests

## Env
- No secrets in repo. Use env vars.

## Definition of done
- Tests pass (`pytest -q`)
- New code covered by tests
- Accessible, print-safe HTML for reports
