---
description: Scaffold a new Python project with modern tooling (uv, ruff, mypy, pytest, pre-commit)
allowed-tools: ["Read", "Write", "Bash", "Skill"]
argument-hint: "<project-name> [--minimal|--ml] [--pip-only]"
---

# Create Project Scaffold

Scaffold a new Python project with modern tooling configured correctly from the
start. Default workflow is uv-managed (Astral's uv handles venv, deps,
lockfile, Python install, and build/publish in one tool). Pass `--pip-only` if
the target environment cannot install uv (locked-down corporate setups).

## Process (uv path — default)

1. **Parse arguments**
   - Project name (required, becomes both the directory and the package name).
   - `--minimal`: skip pre-commit and mypy strict; pytest only.
   - `--ml`: include MLflow / hydra-core / pydantic in deps; add `mlruns/` and
     `outputs/` to `.gitignore`. See `ml-engineering-workflows.md`.
   - `--pip-only`: skip uv, fall back to plain `pip` + `pip-tools`.

2. **Initialise the project** (uv path)
   ```bash
   uv init <project-name> --lib --python 3.12
   cd <project-name>
   ```
   This creates:
   ```
   <project-name>/
   ├── .git/
   ├── .gitignore
   ├── .python-version
   ├── pyproject.toml
   ├── README.md
   └── src/
       └── <project_name>/
           └── __init__.py
   ```

3. **Add dev dependencies** as a PEP 735 group:
   ```bash
   uv add --dev pytest pytest-cov mypy ruff pre-commit
   # If --ml:
   uv add mlflow hydra-core pydantic
   ```

4. **Author additional config** in `pyproject.toml` (ruff, mypy, pytest tables —
   see template below).

5. **Author `.pre-commit-config.yaml`** (see template below) and install hooks:
   ```bash
   uvx pre-commit install
   ```

6. **Create `tests/` and `py.typed`:**
   ```bash
   mkdir tests && touch tests/__init__.py
   touch src/<project_name>/py.typed
   ```

7. **Lock and sync:**
   ```bash
   uv sync
   ```

8. **Verify** by running the empty test suite and the linters:
   ```bash
   uv run pytest
   uv run ruff check .
   uv run mypy src/
   ```

## Process (pip-only path — fallback when uv unavailable)

1. `mkdir <project-name> && cd <project-name> && git init`
2. `mkdir -p src/<project_name> tests && touch src/<project_name>/__init__.py src/<project_name>/py.typed tests/__init__.py`
3. Author `pyproject.toml`, `.pre-commit-config.yaml`, `.gitignore`.
4. `python -m venv .venv && source .venv/bin/activate`
5. `pip install -e ".[dev]"`
6. `pre-commit install`

## pyproject.toml template (extend after `uv init`)

```toml
# uv init writes [project] and [build-system]; append the rest.

[tool.ruff]
target-version = "py312"
line-length = 140

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM", "RUF"]
ignore = ["E501"]   # Handled by formatter

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101"]

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = ["--strict-markers", "--strict-config"]
```

For the `--pip-only` path, also add (uv handles this via `[dependency-groups]`):

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-cov>=5.0",
    "mypy>=1.13",
    "ruff>=0.8",
    "pre-commit>=4.0",
]
```

## .pre-commit-config.yaml template

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.12
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.20.2
    hooks:
      - id: mypy

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
      - id: check-merge-conflict
```

## Load detailed guidance

For comprehensive project structure patterns:

```
Load skill: axiom-python-engineering:using-python-engineering
Then read: project-structure-and-tooling.md
```

## Example session (uv path)

```
User: /python-engineering:create-project-scaffold my-api

Claude:
1. Running `uv init my-api --lib --python 3.12`...
   Created my-api/{pyproject.toml, .python-version, src/my_api/__init__.py, README.md, .gitignore, .git}

2. Adding dev dependencies via `uv add --dev pytest pytest-cov mypy ruff pre-commit`...

3. Extending pyproject.toml with [tool.ruff], [tool.mypy], [tool.pytest.ini_options]...

4. Writing .pre-commit-config.yaml...

5. Installing pre-commit hooks via `uvx pre-commit install`...

6. Running `uv sync` to lock and provision the venv...

7. Verification:
   - uv run pytest        → no tests collected (expected)
   - uv run ruff check .  → All checks passed
   - uv run mypy src/     → Success: no issues found

Project ready. Next steps:
   cd my-api
   uv add <runtime deps you need>
   uv run pytest -v
```

## Why uv by default

- One tool covers venv, dependency resolution, lockfile, Python interpreter
  install, tool install (replaces pipx), and build/publish (replaces
  build/twine).
- 10-100× faster than pip/Poetry on cold and warm caches.
- Native PEP 735 dev/group dependencies — no abusing
  `[project.optional-dependencies]` for tooling.
- `uvx` runs one-shot tools without polluting the project venv.
- Falls back gracefully: `uv pip install` is a drop-in for pip if you only want
  the speedup and not the new workflow.

If uv cannot be installed in the target environment, use `--pip-only` to fall
back to a plain `pip + pip-tools + venv` workflow.
