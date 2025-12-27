---
description: Scaffold a new Python project with modern tooling (ruff, mypy, pytest, pre-commit)
allowed-tools: ["Read", "Write", "Bash", "Skill"]
argument-hint: "<project-name> [--minimal|--ml]"
---

# New Project Command

Create a new Python project with modern tooling configured correctly from the start.

## Process

1. **Parse arguments**
   - Project name (required)
   - `--minimal`: Basic structure only
   - `--ml`: Include ML-specific tooling (MLflow, hydra patterns)

2. **Create directory structure**
   ```
   project-name/
   ├── src/
   │   └── project_name/
   │       └── __init__.py
   ├── tests/
   │   └── __init__.py
   ├── pyproject.toml
   ├── .pre-commit-config.yaml
   ├── .gitignore
   └── README.md
   ```

3. **Configure pyproject.toml**
   - Project metadata
   - ruff configuration (linting + formatting)
   - mypy configuration (strict mode)
   - pytest configuration

4. **Configure pre-commit hooks**
   - ruff (lint + format)
   - mypy
   - trailing whitespace, end-of-file fixes

5. **Initialize git and pre-commit**
   ```bash
   cd project-name && git init && pre-commit install
   ```

## Standard pyproject.toml Template

```toml
[project]
name = "project-name"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "mypy>=1.8",
    "ruff>=0.3",
    "pre-commit>=3.6",
]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

## Load Detailed Guidance

For comprehensive project structure patterns:
```
Load skill: axiom-python-engineering:using-python-engineering
Then read: project-structure-and-tooling.md
```

## Example Session

```
User: /python-engineering:new-project my-api

Claude:
1. Creating project structure for my-api...
   ├── src/my_api/__init__.py
   ├── tests/__init__.py
   ├── pyproject.toml (ruff, mypy, pytest configured)
   ├── .pre-commit-config.yaml
   ├── .gitignore
   └── README.md

2. Initializing git repository...

3. Installing pre-commit hooks...

4. Project ready! Next steps:
   - cd my-api
   - python -m venv .venv && source .venv/bin/activate
   - pip install -e ".[dev]"
   - pre-commit run --all-files
```
