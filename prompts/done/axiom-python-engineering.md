# Refresh: axiom-python-engineering

**Verdict:** HIGH / M effort. Tooling guidance frozen Q3-Q4 2023.

## Context

- Pack path: `/home/john/skillpacks/plugins/axiom-python-engineering/`
- Full review: `/tmp/skillpack-refresh-review/axiom-python-engineering.md`
- Purpose: Python engineering excellence — typing, async, testing, scaffolding, profiling, delinting.

## Why refresh

The pack has excellent router structure and timeless type/async/testing content, but tooling guidance is visibly written before the modern Python tooling shift:

- **Zero `uv` references.** uv is now the dominant Python installer/resolver.
- **Stale ruff/mypy pre-commit revs.**
- **`actions/setup-python@v4`** — current is `v5`.
- **Scaffold/skill version mismatch** — examples target Python 3.11 in some places, 3.12 in others.
- **No Python 3.13** support (free-threaded build, PEP 703 awareness).
- **No `ty` / `pyrefly`** — newer type-checkers worth at least mentioning as alternatives to mypy / pyright.
- **Internal contradiction** — one place recommends isort, another configures ruff for import sorting.

## Scope — DO

1. **Scaffolding skill.** Default to `uv` (`uv init`, `uv add`, `uv sync`, `uv tool`). Keep pip/poetry/pdm as alternatives with one-line "use this if" guidance.
2. **CI sheet.** Bump `actions/setup-python@v5`, `actions/checkout@v4`+, `actions/upload-artifact@v4`. Add `astral-sh/setup-uv` action.
3. **Linting/formatting.** Make ruff the single source of truth for both lint and format. Remove standalone isort recommendation. Bump pre-commit revs to current. Mention `ruff format` replacing black.
4. **Type-checking.** Default mypy + add `ty` (Astral) and `pyrefly` (Meta) as emerging options with status note (preview / production-ready).
5. **Python version.** Default 3.12 across all examples. Mention 3.13 (free-threaded build, JIT) with caveats.
6. **Scaffold project layout.** Verify `pyproject.toml` matches modern uv-managed layout.
7. **Async/typing/testing sheets.** Likely keep as-is (reviewer says timeless), but spot-check for `typing_extensions` patterns superseded by stdlib in 3.12+.

## Scope — DO NOT

- Do not remove pip/poetry/pdm guidance — many shops still use them.
- Do not change router skill structure.
- Do not refactor the timeless content (typing, async, testing).

## Acceptance criteria

1. `uv` is the default scaffolding path; pip/poetry/pdm are documented alternatives.
2. All GitHub Actions versions current (`@v5` for setup-python, `@v4`+ for checkout/upload-artifact).
3. Single Python default version across all skill files; 3.13 mentioned with status.
4. `ruff` is the single linter/formatter; no remaining `isort` standalone advice.
5. `ty` and `pyrefly` mentioned in type-checking skill.
6. `plugin.json` version bumped (minor).

## Process

1. Read `/tmp/skillpack-refresh-review/axiom-python-engineering.md`.
2. Read every SKILL.md.
3. Sweep for staleness markers (pip vs uv, Python version, action versions).
4. Edit. Verify each code example with `uv` actually works (`uv init` → `uv add` → `uv run`).
5. Bump version.

## Constraints

- Every CLI invocation must be verified to work with current tool versions.
- No fabrication of `uv` flags — verify against `uv --help`.
- Don't recommend `ty` as production-ready until it actually is — describe as preview.
