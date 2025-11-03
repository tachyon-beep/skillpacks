---
name: project-structure-and-tooling
description: Modern Python project structure, pyproject.toml, ruff, mypy, pre-commit hooks, dependency management, packaging
---

# Project Structure and Tooling

## Overview

**Core Principle:** Project setup is infrastructure. Good infrastructure is invisible when working, painful when missing. Set it up once, benefit forever.

Modern Python projects use `pyproject.toml` for all configuration, `ruff` for linting and formatting, `mypy` for type checking, and `pre-commit` for automated quality gates. The choice between src layout and flat layout determines import patterns and package discoverability.

This skill covers SETUP of tooling. For FIXING lint warnings systematically, see `systematic-delinting`.

## When to Use

**Use this skill when:**
- Starting a new Python project
- "How should I structure my project?"
- Setting up pyproject.toml
- Configuring ruff, mypy, or pre-commit
- "What dependency manager should I use?"
- Packaging Python projects for distribution

**Don't use when:**
- Fixing existing lint warnings (use systematic-delinting)
- Writing type hints (use modern-syntax-and-types)
- Setting up tests (use testing-and-quality)

**Symptoms triggering this skill:**
- "New Python project setup"
- "Configure ruff/black/mypy"
- "src layout vs flat layout"
- "Poetry vs pip-tools"
- "Package my project"

---

## Project Layout Decisions

### Src Layout vs Flat Layout

**Decision tree:**
```
Distributing as package? → src layout
Testing import behavior? → src layout
Simple script/app? → flat layout
Learning project? → flat layout
Production library? → src layout
```

### Flat Layout

```
my_project/
├── pyproject.toml
├── README.md
├── my_package/
│   ├── __init__.py
│   ├── module1.py
│   └── module2.py
└── tests/
    ├── __init__.py
    ├── test_module1.py
    └── test_module2.py
```

**Pros:**
- Simpler structure
- Easier to understand for beginners
- Fewer directories

**Cons:**
- Can accidentally import from source instead of installed package
- Harder to test actual install behavior
- Package and project root mixed

**Use when:**
- Simple applications
- Learning projects
- Not distributing as package

### Src Layout (Recommended for Libraries)

```
my_project/
├── pyproject.toml
├── README.md
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── module1.py
│       └── module2.py
└── tests/
    ├── __init__.py
    ├── test_module1.py
    └── test_module2.py
```

**Pros:**
- Forces testing against installed package
- Clear separation: src/ is package, tests/ is tests
- Prevents accidental imports from source
- Industry standard for libraries

**Cons:**
- One extra directory level
- Slightly more complex

**Use when:**
- Creating a library
- Distributing on PyPI
- Want production-quality setup

**Why this matters**: Src layout forces you to install your package in editable mode (`pip install -e .`), ensuring tests run against the installed package, not loose Python files. Catches import issues early.

---

## pyproject.toml Fundamentals

### Basic Structure

**File:** `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-package"
version = "0.1.0"
description = "A short description"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
readme = "README.md"
requires-python = ">=3.12"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "requests>=2.31.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "mypy>=1.5.0",
    "ruff>=0.1.0",
]

[project.urls]
Homepage = "https://github.com/username/my-package"
Documentation = "https://my-package.readthedocs.io"
Repository = "https://github.com/username/my-package"

[tool.ruff]
target-version = "py312"
line-length = 140

[tool.mypy]
python_version = "3.12"
strict = true
```

**Why this matters**: Single file for all configuration. No setup.py, setup.cfg, or scattered config files. Modern standard (PEP 621).

### Build System Selection

**hatchling (recommended for most projects):**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**setuptools (traditional, still common):**
```toml
[build-system]
requires = ["setuptools>=68.0.0", "wheel"]
build-backend = "setuptools.build_meta"
```

**poetry (if using Poetry for dependencies):**
```toml
[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

**Decision tree:**
```
Using Poetry for deps? → poetry-core
Need advanced features? → setuptools
Simple project? → hatchling
```

**Why hatchling?**
- Modern, fast, minimal configuration
- Good defaults
- Works with standard tools
- No legacy baggage

### Version Management

**Static version:**
```toml
[project]
version = "0.1.0"
```

**Dynamic version from file:**
```toml
[project]
dynamic = ["version"]

[tool.hatch.version]
path = "src/my_package/__init__.py"
```

**File:** `src/my_package/__init__.py`
```python
__version__ = "0.1.0"
```

**Dynamic version from git tag:**
```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]

[tool.hatch.version]
source = "vcs"
```

**Recommendation**: Start with static version. Add dynamic versioning when you need it.

---

## Ruff Configuration

### Core Configuration

**File:** `pyproject.toml`

```toml
[tool.ruff]
target-version = "py312"
line-length = 140  # Note: 140, not default 88

# Exclude patterns
exclude = [
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "*.egg-info",
]

[tool.ruff.lint]
# Enable rule sets
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
    "RUF", # ruff-specific
]

# Ignore specific rules
ignore = [
    "E501",  # Line too long (handled by formatter)
]

# Per-file ignores
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = [
    "S101",  # Allow assert in tests
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"
```

**Why line-length = 140?**
- Modern screens are wide
- Default 88 is too restrictive for complex type hints
- 140 balances readability and fitting multiple windows
- Industry trend toward 100-140

**Rule set breakdown:**

| Set | Purpose | Example Rules |
|-----|---------|---------------|
| E/W | PEP 8 style | Whitespace, indentation |
| F | Logical errors | Undefined names, unused imports |
| I | Import sorting | isort compatibility |
| N | Naming | PEP 8 naming conventions |
| UP | Python upgrades | Use Python 3.10+ features |
| B | Bug detection | Likely bugs (mutable defaults) |
| C4 | Comprehensions | Better list/dict comprehensions |
| SIM | Simplification | Simplify complex code |
| RUF | Ruff-specific | Ruff's custom checks |

### Import Sorting (isort compatibility)

```toml
[tool.ruff.lint.isort]
known-first-party = ["my_package"]
known-third-party = ["numpy", "pandas"]
section-order = ["future", "standard-library", "third-party", "first-party", "local-folder"]
lines-after-imports = 2
```

**Expected import order:**
```python
# Future imports
from __future__ import annotations

# Standard library
import json
import sys
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import requests

# First-party
from my_package import utils
from my_package.core import Engine


def my_function():
    ...
```

**Why this matters**: Consistent import ordering improves readability and prevents merge conflicts.

### Advanced Configuration

```toml
[tool.ruff.lint.flake8-bugbear]
# Extend immutable calls (prevent mutation)
extend-immutable-calls = ["fastapi.Depends", "fastapi.Query"]

[tool.ruff.lint.flake8-quotes]
docstring-quotes = "double"
inline-quotes = "double"

[tool.ruff.lint.mccabe]
max-complexity = 10

[tool.ruff.lint.pydocstyle]
convention = "google"  # or "numpy", "pep257"
```

**Complexity limit explanation:**
- Complexity < 10: Good
- 10-15: Acceptable, monitor
- 15+: Refactor

---

## Type Checking with mypy

### Strict Configuration

**File:** `pyproject.toml`

```toml
[tool.mypy]
python_version = "3.12"
strict = true

# Strict mode includes:
# - warn_return_any
# - warn_unused_configs
# - disallow_untyped_defs
# - disallow_any_generics
# - disallow_subclassing_any
# - disallow_untyped_calls
# - disallow_untyped_decorators
# - disallow_incomplete_defs
# - check_untyped_defs
# - warn_redundant_casts
# - warn_unused_ignores
# - warn_no_return
# - warn_unreachable
# - strict_equality

# Exclude patterns
exclude = [
    "^build/",
    "^dist/",
]

# Per-module overrides
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false  # Tests can be less strict

[[tool.mypy.overrides]]
module = "third_party.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "untyped_library"
ignore_missing_imports = true
```

### Incremental Adoption

**Start lenient, get stricter:**

```toml
# Phase 1: Basic type checking
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true

# Phase 2: Add more checks
check_untyped_defs = true
warn_redundant_casts = true
warn_unused_ignores = true

# Phase 3: Require types
disallow_untyped_defs = true
disallow_incomplete_defs = true

# Phase 4: Full strict
strict = true
```

**Per-module migration:**

```toml
[tool.mypy]
python_version = "3.12"
# Default: lenient

[[tool.mypy.overrides]]
module = "my_package.new_module"
strict = true  # New code is strict

[[tool.mypy.overrides]]
module = "my_package.legacy"
ignore_errors = true  # TODO: Fix legacy code
```

**Why this matters**: Incremental adoption prevents overwhelming backlog of type errors. Strict mode for new code, lenient for legacy.

---

## Dependency Management

### pip-tools

**Recommended for most projects. Simple, standard, no lock-in.**

**Setup:**
```bash
pip install pip-tools
```

**File:** `requirements.in` (high-level dependencies)
```
requests>=2.31.0
pydantic>=2.0.0
```

**Generate locked requirements:**
```bash
pip-compile requirements.in
# Creates requirements.txt with exact versions
```

**File:** `requirements.txt` (auto-generated)
```
certifi==2023.7.22
    # via requests
charset-normalizer==3.2.0
    # via requests
idna==3.4
    # via requests
pydantic==2.3.0
    # via -r requirements.in
pydantic-core==2.6.3
    # via pydantic
requests==2.31.0
    # via -r requirements.in
urllib3==2.0.4
    # via requests
```

**Development dependencies:**

**File:** `requirements-dev.in`
```
-c requirements.txt  # Constrain to production versions
pytest>=7.4.0
mypy>=1.5.0
ruff>=0.1.0
```

**Compile:**
```bash
pip-compile requirements-dev.in
```

**Sync environment:**
```bash
pip-sync requirements.txt requirements-dev.txt
```

**Why pip-tools?**
- Uses standard requirements.txt format
- No proprietary lock file
- Simple mental model
- Works everywhere
- No lock-in

### Poetry

**Better for libraries, more features, heavier.**

**Setup:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**File:** `pyproject.toml`
```toml
[tool.poetry]
name = "my-package"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.12"
requests = "^2.31.0"
pydantic = "^2.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
mypy = "^1.5.0"
ruff = "^0.1.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

**Commands:**
```bash
poetry install          # Install dependencies
poetry add requests     # Add dependency
poetry add --group dev pytest  # Add dev dependency
poetry update           # Update dependencies
poetry lock             # Update lock file
poetry build            # Build package
poetry publish          # Publish to PyPI
```

**Why Poetry?**
- Manages dependencies AND build system
- Better dependency resolution
- Built-in virtual environment management
- Integrated publishing

**Why NOT Poetry?**
- Heavier tool
- Proprietary lock format
- Slower than pip-tools
- Lock-in to Poetry workflow

### Comparison Decision Tree

```
Publishing to PyPI? → Poetry (integrated workflow)
Simple project? → pip-tools (minimal)
Need reproducible builds? → Either (both lock)
Team unfamiliar with tools? → pip-tools (simpler)
Complex dependency constraints? → Poetry (better resolver)
CI/CD integration? → pip-tools (faster)
```

---

## Pre-commit Hooks

### Setup

**Install:**
```bash
pip install pre-commit
```

**File:** `.pre-commit-config.yaml`

```yaml
repos:
  # Ruff for linting and formatting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      # Run linter
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      # Run formatter
      - id: ruff-format

  # mypy for type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]

  # Standard pre-commit hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=1000]
      - id: check-merge-conflict
      - id: check-case-conflict

  # Python-specific
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        name: isort (python)
```

**Install hooks:**
```bash
pre-commit install
```

**Run manually:**
```bash
pre-commit run --all-files
```

**Update hooks:**
```bash
pre-commit autoupdate
```

### Hook Selection Strategy

**Essential hooks (always use):**
- `ruff` - Linting and formatting
- `trailing-whitespace` - Clean files
- `end-of-file-fixer` - Proper file endings
- `check-yaml` - YAML syntax
- `check-merge-conflict` - Prevent merge markers

**Recommended hooks:**
- `mypy` - Type checking
- `check-toml` - pyproject.toml syntax
- `check-added-large-files` - Prevent large files

**Optional hooks:**
- `pytest` - Run tests (slow!)
- `bandit` - Security checks
- `interrogate` - Docstring coverage

**Why NOT include slow hooks:**
```yaml
# ❌ WRONG: Tests in pre-commit (too slow)
- repo: local
  hooks:
    - id: pytest
      name: pytest
      entry: pytest
      language: system
      pass_filenames: false
```

**Why this matters**: Pre-commit hooks run on EVERY commit. Keep them fast (<5 seconds total). Run tests in CI, not pre-commit.

### Skipping Hooks

**Skip all hooks (use sparingly):**
```bash
git commit --no-verify -m "Quick fix"
```

**Skip specific hook:**
```bash
SKIP=mypy git commit -m "WIP: type errors to fix"
```

**When to skip:**
- WIP commits on feature branch (will fix before PR)
- Emergency hotfixes (fix hooks after)
- Known false positives (fix hook config instead)

**When NOT to skip:**
- Merging to main
- Creating PR
- "Too lazy to fix" ← Never valid reason

---

## Formatting and Linting Workflow

### Ruff as Formatter and Linter

**Ruff replaces: black, isort, flake8, pyupgrade, and more.**

**Format code:**
```bash
ruff format .
```

**Check linting:**
```bash
ruff check .
```

**Fix auto-fixable issues:**
```bash
ruff check --fix .
```

**Show what would fix without changing:**
```bash
ruff check --fix --diff .
```

### IDE Integration

**VS Code** (`.vscode/settings.json`):
```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "none",
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "charliermarsh.ruff",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": true,
      "source.organizeImports": true
    },
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

**PyCharm:**
- Install Ruff plugin
- Settings → Tools → Ruff → Enable
- Settings → Tools → Actions on Save → Ruff format

**Why this matters**: Format on save prevents formatting commits. Linting in IDE catches issues before commit.

---

## Packaging and Distribution

### Minimal Package

**File structure:**
```
my_package/
├── pyproject.toml
├── README.md
├── LICENSE
└── src/
    └── my_package/
        ├── __init__.py
        └── main.py
```

**File:** `pyproject.toml`
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-package"
version = "0.1.0"
description = "A short description"
readme = "README.md"
requires-python = ">=3.12"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
classifiers = [
    "Programming Language :: Python :: 3.12",
]
dependencies = []

[project.urls]
Homepage = "https://github.com/username/my-package"
```

**Build:**
```bash
pip install build
python -m build
```

**Creates:**
```
dist/
├── my_package-0.1.0-py3-none-any.whl
└── my_package-0.1.0.tar.gz
```

### Publishing to PyPI

**Test on TestPyPI first:**

```bash
pip install twine

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test install
pip install --index-url https://test.pypi.org/simple/ my-package
```

**Publish to real PyPI:**

```bash
twine upload dist/*
```

**Better: Use GitHub Actions**

**File:** `.github/workflows/publish.yml`
```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.12"

      - name: Install build
        run: pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

**Why this matters**: Automated publishing on GitHub release. Consistent process, no manual uploads.

### Entry Points

**Console scripts:**

```toml
[project.scripts]
my-cli = "my_package.cli:main"
my-tool = "my_package.tools:run"
```

**Creates command-line tools:**
```bash
pip install my-package
my-cli --help  # Runs my_package.cli:main()
```

**File:** `src/my_package/cli.py`
```python
def main() -> None:
    print("Hello from my-cli!")

if __name__ == "__main__":
    main()
```

---

## Complete Example: Production Project

### Project Structure

```
awesome_project/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── publish.yml
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   └── awesome_project/
│       ├── __init__.py
│       ├── core.py
│       ├── utils.py
│       └── py.typed
└── tests/
    ├── __init__.py
    ├── test_core.py
    └── test_utils.py
```

### pyproject.toml (Complete)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "awesome-project"
version = "0.1.0"
description = "An awesome Python project"
readme = "README.md"
requires-python = ">=3.12"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3 :: Only",
    "Typing :: Typed",
]
dependencies = [
    "requests>=2.31.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "mypy>=1.5.0",
    "ruff>=0.1.0",
    "pre-commit>=3.5.0",
    "types-requests>=2.31.0",
]

[project.urls]
Homepage = "https://github.com/username/awesome-project"
Documentation = "https://awesome-project.readthedocs.io"
Repository = "https://github.com/username/awesome-project"
Issues = "https://github.com/username/awesome-project/issues"

[project.scripts]
awesome = "awesome_project.cli:main"

# Ruff configuration
[tool.ruff]
target-version = "py312"
line-length = 140

exclude = [
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "*.egg-info",
]

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
    "RUF", # ruff-specific
]

ignore = [
    "E501",  # Line too long (handled by formatter)
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = [
    "S101",  # Allow assert in tests
]

[tool.ruff.lint.isort]
known-first-party = ["awesome_project"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

# mypy configuration
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

# pytest configuration
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=awesome_project",
    "--cov-report=term-missing",
]

# Coverage configuration
[tool.coverage.run]
source = ["src"]
omit = ["tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

### .pre-commit-config.yaml (Complete)

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, pydantic]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
      - id: check-merge-conflict
```

### .gitignore (Complete)

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Ruff
.ruff_cache/

# OS
.DS_Store
Thumbs.db
```

### CI Workflow

**File:** `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run ruff (lint)
        run: ruff check .

      - name: Run ruff (format check)
        run: ruff format --check .

      - name: Run mypy
        run: mypy src/

      - name: Run pytest
        run: pytest --cov --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        if: matrix.os == 'ubuntu-latest'
```

---

## Anti-Patterns

### Scattered Configuration Files

```
# ❌ WRONG: Configuration in multiple files
setup.py
setup.cfg
requirements.txt
requirements-dev.txt
.flake8
mypy.ini
pytest.ini
.isort.cfg
```

```toml
# ✅ CORRECT: Single pyproject.toml
# All configuration in one place
[tool.ruff]
...

[tool.mypy]
...

[tool.pytest.ini_options]
...
```

**Why this matters**: Single source of truth. Easier to maintain, version control, and share.

### Not Using Src Layout for Libraries

```
# ❌ WRONG: Flat layout for distributed package
my_package/
├── my_package/
│   └── __init__.py
└── tests/
```

**Problem**: Tests might pass locally but fail when installed:
```bash
# Works locally (imports from source)
pytest  # PASS

# Fails when installed (package not installed correctly)
pip install .
python -c "import my_package"  # ImportError
```

```
# ✅ CORRECT: Src layout forces proper install
my_package/
├── src/
│   └── my_package/
│       └── __init__.py
└── tests/
```

**Why this matters**: Src layout catches packaging issues early by forcing editable install.

### Too Many Dependencies

```toml
# ❌ WRONG: Kitchen sink approach
dependencies = [
    "requests",
    "httpx",       # Both requests and httpx?
    "urllib3",     # Already included with requests
    "pandas",
    "polars",      # Both pandas and polars?
    "numpy",       # Included with pandas
    # ... 50 more
]
```

```toml
# ✅ CORRECT: Minimal direct dependencies
dependencies = [
    "requests>=2.31.0",  # Only what YOU directly use
    "pydantic>=2.0.0",
]

# Transitive deps (requests → urllib3) handled automatically
```

**Why this matters**: More dependencies = more conflict risk, slower installs, larger attack surface.

### Ignoring Lock Files

```bash
# ❌ WRONG: Install from requirements.in
pip install -r requirements.in
```

**Problem**: Gets different versions each time, breaks reproducibility.

```bash
# ✅ CORRECT: Install from locked requirements
pip install -r requirements.txt
```

**Why this matters**: Locked dependencies ensure reproducible builds and deployments.

### Pre-commit Hooks Too Slow

```yaml
# ❌ WRONG: Run full test suite on every commit
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest tests/
        language: system
        pass_filenames: false
```

**Problem**: 5-minute test suite blocks every commit. Developers will skip hooks.

```yaml
# ✅ CORRECT: Fast checks only
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
      - id: ruff-format
```

**Why this matters**: Pre-commit must be fast (<5s total). Run tests in CI, not pre-commit.

---

## Decision Trees

### Choosing Project Layout

```
├─ Distributing as package?
│  ├─ Yes → src layout
│  └─ No
│     ├─ Complex project? → src layout (future-proof)
│     └─ Simple script? → flat layout
```

### Choosing Dependency Manager

```
├─ Publishing to PyPI?
│  ├─ Yes → Poetry (integrated workflow)
│  └─ No
│     ├─ Need simple workflow? → pip-tools
│     ├─ Complex constraints? → Poetry
│     └─ Existing requirements.txt? → pip-tools
```

### Choosing Build Backend

```
├─ Using Poetry? → poetry-core
├─ Need setuptools features? → setuptools
└─ Simple project? → hatchling
```

### Line Length Configuration

```
├─ Team preference for 88? → 88
├─ Complex type hints? → 120-140
├─ Modern screens? → 120-140
└─ No strong opinion? → 120
```

---

## Common Workflows

### New Project from Scratch

```bash
# 1. Create structure
mkdir my_project
cd my_project
git init

# 2. Create directory structure
mkdir -p src/my_project tests

# 3. Create pyproject.toml (see example above)
# 4. Create .pre-commit-config.yaml (see example above)
# 5. Create .gitignore (see example above)

# 6. Initialize package
cat > src/my_project/__init__.py << 'EOF'
"""My awesome project."""
__version__ = "0.1.0"
EOF

# 7. Create py.typed marker for type checking
touch src/my_project/py.typed

# 8. Install in editable mode
pip install -e ".[dev]"

# 9. Install pre-commit hooks
pre-commit install

# 10. First commit
git add .
git commit -m "feat: Initial project structure"
```

### Adding Ruff to Existing Project

```bash
# 1. Install ruff
pip install ruff

# 2. Add to pyproject.toml
cat >> pyproject.toml << 'EOF'
[tool.ruff]
target-version = "py312"
line-length = 140

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM", "RUF"]
ignore = ["E501"]
EOF

# 3. Check what would change
ruff check --diff .

# 4. Apply fixes
ruff check --fix .

# 5. Format code
ruff format .

# 6. Add to pre-commit
cat >> .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
EOF

pre-commit install
```

### Migrating from Black/Flake8 to Ruff

```bash
# 1. Install ruff
pip install ruff

# 2. Remove old tools
pip uninstall black flake8 isort pyupgrade

# 3. Convert black config to ruff
# Old .flake8:
# [flake8]
# max-line-length = 88
# ignore = E203, W503

# New pyproject.toml:
[tool.ruff]
line-length = 88

[tool.ruff.lint]
ignore = ["E203", "W503"]

# 4. Remove old config files
rm .flake8 .isort.cfg

# 5. Update pre-commit
# Replace black, isort, flake8 hooks with ruff

# 6. Reformat everything
ruff format .
```

---

## Integration with Other Skills

**Before using this skill:**
- No prerequisites (start here for new projects)

**After using this skill:**
- Fix lint warnings → See `systematic-delinting`
- Add type hints → See `modern-syntax-and-types`
- Setup testing → See `testing-and-quality`
- Add CI/CD → (Future skill)

**Cross-references:**
- Type checking setup → `modern-syntax-and-types` for type hint patterns
- Delinting process → `systematic-delinting` for fixing warnings
- Testing setup → `testing-and-quality` for pytest configuration

---

## Quick Reference

### Essential Commands

```bash
# Project setup
pip install -e ".[dev]"       # Editable install with dev deps
pre-commit install            # Install git hooks

# Daily workflow
ruff check .                  # Lint
ruff check --fix .           # Lint and auto-fix
ruff format .                # Format
mypy src/                    # Type check
pytest                       # Run tests

# Pre-commit
pre-commit run --all-files   # Run all hooks manually
pre-commit autoupdate        # Update hook versions

# Dependency management (pip-tools)
pip-compile requirements.in              # Lock dependencies
pip-compile requirements-dev.in          # Lock dev dependencies
pip-sync requirements.txt requirements-dev.txt  # Sync environment

# Building and publishing
python -m build              # Build package
twine upload dist/*          # Upload to PyPI
```

### Configuration Checklist

**Minimum viable pyproject.toml:**
- [x] `[build-system]` - hatchling or setuptools
- [x] `[project]` - name, version, dependencies
- [x] `[tool.ruff]` - target-version, line-length
- [x] `[tool.mypy]` - python_version, strict

**Production-ready additions:**
- [x] `[project.optional-dependencies]` - dev dependencies
- [x] `[project.scripts]` - console scripts
- [x] `[tool.ruff.lint]` - rule selection
- [x] `[tool.pytest.ini_options]` - test configuration
- [x] `.pre-commit-config.yaml` - automated checks
- [x] `.gitignore` - ignore build artifacts
- [x] `src/package/py.typed` - typed package marker

### Ruff Rule Sets Quick Reference

| Code | Name | Purpose |
|------|------|---------|
| E/W | pycodestyle | PEP 8 style |
| F | Pyflakes | Logical errors |
| I | isort | Import ordering |
| N | pep8-naming | Naming conventions |
| UP | pyupgrade | Modern syntax |
| B | flake8-bugbear | Bug detection |
| C4 | flake8-comprehensions | Better comprehensions |
| SIM | flake8-simplify | Code simplification |
| RUF | Ruff | Ruff-specific |

**Enable progressively:**
1. Start: `["E", "W", "F"]` - Core errors
2. Add: `["I", "N", "UP"]` - Style and modernization
3. Add: `["B", "C4", "SIM"]` - Quality improvements
4. Add: `["RUF"]` - Ruff-specific checks

---

## Why This Matters: Real-World Impact

**Good tooling setup prevents:**
- ❌ "Works on my machine" - Locked dependencies ensure consistency
- ❌ Import errors in production - Src layout catches packaging issues
- ❌ Style arguments in PRs - Automated formatting ends debates
- ❌ Type errors in production - mypy catches before deploy
- ❌ Breaking dependencies - Lock files ensure reproducibility
- ❌ Manual quality checks - Pre-commit automates enforcement

**Good tooling setup enables:**
- ✅ Fast onboarding - `pip install -e ".[dev]"` gets developers running
- ✅ Consistent code style - Ruff format ensures uniformity
- ✅ Early bug detection - Type checking and linting catch issues
- ✅ Confident refactoring - Types and tests enable safe changes
- ✅ Automated publishing - CI/CD handles releases
- ✅ Professional polish - Well-configured projects attract contributors

**Time investment:**
- Initial setup: 1-2 hours
- Saved per month: 10+ hours (no style debates, fewer bugs, faster onboarding)
- ROI: Positive after first month, compounds over project lifetime
