---
description: Modern Python 3.12+ engineering with uv-first tooling - types (mypy/pyright/ty/pyrefly), testing, async, scientific computing, ML workflows, Textual TUI. Routes to 10 specialist reference sheets, 4 commands, 3 agents.
---

# Python Engineering Routing

**Different Python problems require different specialists. Match symptoms to the right specialist sheet; do not guess at solutions. Diagnose before optimizing, set up tooling before delinting, profile before performance tuning. For non-Python languages, algorithm theory, deployment infrastructure, or database design, use a different pack.**

Use the `using-python-engineering` skill from the `axiom-python-engineering` plugin to route to the right specialist sheet. Content authority lives in `plugins/axiom-python-engineering/skills/using-python-engineering/SKILL.md` - this wrapper is a thin pointer.

## When to Use

- Working with Python and unsure which specialist to load
- User mentions Python, type hints, mypy, pytest, async, pandas, numpy, Textual, TUI
- Implementing Python projects, optimizing performance, or scaffolding tooling
- Setting up Python tooling, fixing lint warnings, or resolving mypy errors
- Debugging Python code or profiling performance
- Building terminal user interfaces with Textual

**Don't use** for: non-Python languages, algorithm theory, deployment infrastructure (use `/axiom-devops-engineering`), or database design (use `/embedded-database` or `/web-backend`).

## Sheets

- **modern-syntax-and-types** - type hints, mypy/pyright/ty/pyrefly, Python 3.10-3.12 features, generics, protocols
- **resolving-mypy-errors** - systematic mypy error resolution, `type: ignore` discipline, typing legacy code
- **project-structure-and-tooling** - pyproject.toml, uv, ruff, pre-commit, dependency management, packaging, src vs flat layout
- **systematic-delinting** - process for fixing lint warnings without disabling or over-refactoring
- **testing-and-quality** - pytest patterns, fixtures, mocking, coverage, property-based testing
- **async-patterns-and-concurrency** - async/await, asyncio, TaskGroup, structured concurrency, threading
- **scientific-computing-foundations** - NumPy/pandas, vectorization, memory efficiency, large datasets
- **ml-engineering-workflows** - MLflow, experiment tracking, reproducibility, monitoring, model lifecycle
- **debugging-and-profiling** - pdb/debugpy, cProfile, memory_profiler, optimization strategies
- **textual-tui-development** - Textual framework, reactive lifecycle, compose/mount discipline, CSS layout, async event loop

## Commands

- `/axiom-python-engineering:delint` - systematically fix lint warnings using category-by-category approach
- `/axiom-python-engineering:typecheck` - run mypy type checking with systematic error resolution
- `/axiom-python-engineering:profile` - profile Python code to find actual bottlenecks before optimizing
- `/axiom-python-engineering:create-project-scaffold` - scaffold a new Python project with modern tooling (uv, ruff, mypy, pytest, pre-commit)

## Agents

- `python-code-reviewer` - reviews Python code for correctness, idiom, and anti-patterns; hands off security to `/security-architect` (SME protocol with Confidence/Risk/Information Gaps/Caveats)
- `refactoring-architect` - restructures Python code; refuses feature work and lint-only sweeps (SME protocol)
- `delinting-specialist` - autonomous executor of the systematic-delinting workflow (haiku model, non-SME)

## Cross-references

- Pytest organization, test architecture, flaky-test diagnosis → `/quality-engineering`
- Security review of Python code → `/security-architect`
- CI/CD pipeline for Python projects → `/axiom-devops-engineering`
- SQLite/DuckDB drivers (`sqlite3`, `aiosqlite`, `duckdb`) → `/embedded-database`
- PyTorch-specific engineering → `/pytorch-engineering`
- LLM-API engineering in Python → `/llm-specialist`
