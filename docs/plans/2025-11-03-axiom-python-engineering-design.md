# axiom-python-engineering Design

**Date**: 2025-11-03
**Status**: Design validated, ready for implementation
**Faction**: Axiom - Creators of Technological Marvels
**Philosophy**: Innovation through systematic process, quality through testing, accessibility through clear methodology

---

## Overview

The **axiom-python-engineering** plugin provides comprehensive Python 3.12+ engineering guidance following symptom-based routing patterns established in the skillpacks marketplace. This is the first Axiom faction plugin, embodying Python's core philosophy: explicit over implicit, readability counts, one obvious way to do it.

**Target users**: Engineers working with modern Python seeking production-ready patterns, from general-purpose engineering to AI/ML workflows with monitoring.

**Scope**: General Python engineering fundamentals (primary) + scientific/ML Python patterns (secondary) + ML monitoring (as relates to ML workflows).

---

## Architecture

### Plugin Structure

```
plugins/axiom-python-engineering/
├── .claude-plugin/
│   └── plugin.json
└── skills/
    ├── using-python-engineering/SKILL.md          (router, ~400-500 lines)
    ├── modern-syntax-and-types/SKILL.md           (~1,500-1,800 lines)
    ├── project-structure-and-tooling/SKILL.md     (~1,400-1,600 lines)
    ├── systematic-delinting/SKILL.md              (~1,200-1,400 lines)
    ├── testing-and-quality/SKILL.md               (~1,600-1,800 lines)
    ├── async-patterns-and-concurrency/SKILL.md    (~1,500-1,700 lines)
    ├── scientific-computing-foundations/SKILL.md  (~1,400-1,600 lines)
    ├── ml-engineering-workflows/SKILL.md          (~1,300-1,500 lines)
    └── debugging-and-profiling/SKILL.md           (~1,500-1,700 lines)
```

**Total**: 9 skills, ~12,200-14,100 lines of production-ready content

### Design Principles

1. **Symptom-based routing** - Router skill directs based on user problems, not abstract categories
2. **Python 3.12+ focus** - Modern features while noting version requirements (3.10, 3.11, 3.12)
3. **Runnable examples** - All code examples are production-ready and executable
4. **Anti-pattern awareness** - Explicitly show wrong approaches alongside correct ones
5. **Systematic methodology** - Process-driven (Axiom faction alignment)
6. **No refactoring** - Fix issues minimally, don't restructure (especially in delinting)

### Faction Alignment: Axiom

**Axiom Philosophy**: "The world is altruism and innovation" - Make technology available to all through science and systematic process.

**How Python embodies Axiom values**:
- **Accessibility**: Python's readability makes code accessible to all skill levels
- **Systematic process**: Clear patterns, explicit over implicit
- **Quality through testing**: pytest, type checking, linting
- **Innovation**: Modern features (match statements, async, type system evolution)
- **One obvious way**: Zen of Python aligns with structured methodology

---

## Skill Breakdown

### 1. using-python-engineering (Router)

**Purpose**: Symptom-based navigation to specialist skills

**Target length**: 400-500 lines

**Pattern**: Follows `yzmir-pytorch-engineering` router model

**Key routing table**:

| Symptom | Route To | Why |
|---------|----------|-----|
| Type errors, mypy/pyright issues | modern-syntax-and-types | Type system specialist |
| Project setup, structure questions | project-structure-and-tooling | Setup and configuration |
| "Too many lint warnings" | systematic-delinting | Process for fixing warnings |
| Test failures, flaky tests | testing-and-quality | Test architecture specialist |
| Async/await not working | async-patterns-and-concurrency | Async patterns expert |
| "Code is slow" | debugging-and-profiling | ALWAYS profile first |
| Array/DataFrame operations | scientific-computing-foundations | NumPy/pandas specialist |
| ML experiment tracking | ml-engineering-workflows | ML lifecycle expert |

**Cross-cutting scenarios**:
- "Slow pandas code" → debugging-and-profiling (profile) THEN scientific-computing-foundations (optimize)
- "New ML project setup" → project-structure-and-tooling (structure) THEN ml-engineering-workflows (ML specifics)
- "Legacy code with no types" → project-structure-and-tooling (setup mypy) THEN modern-syntax-and-types (add types)

**Key sections**:
- Routing by symptom
- Ambiguous queries (ask clarifying question)
- Common routing mistakes
- Red flags checklist (self-check before answering)
- Diagnosis-first principle

---

### 2. modern-syntax-and-types

**Purpose**: Type system mastery and modern Python 3.10-3.12 syntax

**Target length**: 1,500-1,800 lines

**Core sections**:
1. Type hints fundamentals - Basic annotations, Optional, Union, type aliases
2. Advanced typing - Generics, Protocol, TypedDict, ParamSpec, TypeVar bounds
3. Python 3.10 features - Match statements, structural pattern matching, `|` union syntax
4. Python 3.11 features - Exception groups, task groups, improved error messages
5. Python 3.12 features - PEP 695 type parameter syntax, @override decorator
6. Static analysis setup - mypy strict mode, pyright, dealing with untyped libraries
7. Common type errors - How to diagnose and fix mypy/pyright errors
8. Anti-patterns - Over-typing, type: ignore abuse, incompatible generics

**Key patterns**:
```python
# ✅ Python 3.12+ generic syntax (PEP 695)
class Container[T]:
    def __init__(self, value: T) -> None:
        self.value = value

# ✅ Union types with | (Python 3.10+)
def process(value: str | int) -> dict | None:
    ...

# ✅ Structural pattern matching (Python 3.10+)
match status:
    case 200:
        return "OK"
    case 404:
        return "Not Found"
    case _:
        return "Unknown"
```

**Anti-patterns emphasized**:
- Using old-style Union when `|` available
- type: ignore without explanation
- Not using strict mode in new projects
- Over-engineering with complex generics

---

### 3. project-structure-and-tooling

**Purpose**: Modern Python project setup, configuration, and tooling

**Target length**: 1,400-1,600 lines

**Core sections**:
1. Project layouts - src layout vs flat layout, when to use each
2. pyproject.toml - Modern build system, dependency specification, all tool configs
3. Dependency management - pip-tools, poetry comparison, lock files
4. Code formatting - ruff format (replaces black), configuration
5. Linting setup - ruff check (replaces flake8/pylint), rule selection
6. Pre-commit hooks - Setup, hook selection, local vs CI
7. Import ordering - ruff's isort compatibility
8. Packaging and distribution - Building wheels, PyPI, versioning

**Key configuration example**:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "myproject"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy>=1.26", "pandas>=2.0"]

[tool.ruff]
target-version = "py312"
line-length = 140  # Note: 140, not default 88

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "A", "C4", "PT"]
ignore = ["E501"]  # Handled by formatter

[tool.mypy]
python_version = "3.12"
strict = true
```

**Decision trees**:
- When to use src layout vs flat layout
- Poetry vs pip-tools selection criteria
- When to enable mypy strict mode

**Note**: This skill covers SETUP of linting. The systematic-delinting skill covers FIXING warnings.

---

### 4. systematic-delinting

**Purpose**: Process for fixing lint warnings without refactoring

**Target length**: 1,200-1,400 lines

**Core principle**: Fix warnings systematically, never disable them. Delinting ≠ refactoring.

**Core sections**:
1. Delinting philosophy - Why fix vs disable, technical debt, team standards
2. Ruff configuration - Rule sets by project maturity, line-length = 140
3. Pylint configuration - When to use pylint vs ruff, severity levels
4. Triage methodology - Categorize by effort, count by rule, baseline
5. Systematic workflow - File-by-file vs rule-by-rule, tracking progress
6. Common rule fixes - How to fix specific rules (E501, F401, N806) without refactoring
7. CI integration - Ratcheting (don't get worse), pre-commit for new code only
8. Team adoption - Legacy project introduction, gradual strictness

**Systematic workflow example**:
```bash
# Step 1: Baseline
$ ruff check . --statistics
Found 1247 errors:
- F401 [unused-import]: 423
- E501 [line-too-long]: 312
- N806 [non-lowercase-variable]: 189

# Step 2: Pick highest-volume auto-fixable rule
$ ruff check . --select F401 --fix

# Step 3: Review and commit
$ git diff
$ git commit -m "fix: Remove unused imports (F401)"

# Step 4: Repeat for next rule
```

**Key anti-pattern**:
```python
# ❌ WRONG: Disabling warnings
# ruff: noqa: F401
import unused_module  # "We might need this later"

# ✅ CORRECT: Just remove it
# (nothing - the import is gone)
```

**Line length 140 enforcement**:
```toml
[tool.ruff]
line-length = 140  # Don't ignore E501, set proper length

[tool.ruff.lint]
ignore = []  # Don't ignore E501!
```

**Router symptoms**:
- "Fix lint warnings" → systematic-delinting
- "Too many lint errors" → systematic-delinting
- "Legacy code needs linting" → systematic-delinting

**Distinction from project-structure-and-tooling**:
- project-structure-and-tooling: Initial setup, configuration
- systematic-delinting: Process for fixing existing warnings

---

### 5. testing-and-quality

**Purpose**: pytest mastery, test architecture, quality gates

**Target length**: 1,600-1,800 lines

**Core sections**:
1. pytest fundamentals - Test discovery, assertions, organization
2. Fixtures - Scope, factories, conftest.py, composition
3. Parametrization - pytest.mark.parametrize, indirect parametrization
4. Mocking - pytest-mock, when to mock, anti-patterns (don't mock what you don't own)
5. Coverage - pytest-cov, branch coverage, targets, gap analysis
6. Property-based testing - Hypothesis basics, strategies, edge cases
7. Test architecture - Unit/integration/e2e, test pyramids, fast vs slow
8. Flaky tests - Diagnosis, causes (time, randomness, ordering), fixes
9. CI integration - GitHub Actions, parallel tests, caching

**Key patterns**:
```python
# ✅ Test behavior, not implementation
def test_processing_behavior():
    obj = MyClass()
    result = obj.process()
    assert result.status == "completed"
    assert len(result.items) == 5

# ✅ Appropriate fixture scope
@pytest.fixture
def database():
    db = Database()
    db.connect()
    yield db
    db.rollback()  # Isolate each test
    db.disconnect()
```

**Anti-patterns**:
- Testing implementation details (private methods, internal state)
- Fixtures with too broad scope (session/module when function appropriate)
- Mocking external libraries you don't control
- Ignoring flaky tests instead of fixing root cause

**Emphasis**: Testing async code, testing exceptions, testing file I/O, testing random behavior

---

### 6. async-patterns-and-concurrency

**Purpose**: Modern async/await patterns, asyncio, concurrency

**Target length**: 1,500-1,700 lines

**Core sections**:
1. Async fundamentals - Event loop, async/await syntax, when to use
2. asyncio patterns - gather, create_task, TaskGroup (3.11+), timeouts
3. Async context managers - async with, __aenter__/__aexit__
4. Async iterators - async for, async generators, __aiter__/__anext__
5. Common pitfalls - Blocking event loop, forgetting await, cancellation
6. Structured concurrency - TaskGroup benefits (3.11+), exception handling
7. concurrent.futures - ThreadPoolExecutor, ProcessPoolExecutor, thread vs process choice
8. Async libraries - aiohttp, httpx, async database drivers
9. Debugging async - asyncio debug mode, detecting blocking calls

**Key pattern - Python 3.11+ TaskGroup**:
```python
# ✅ Structured concurrency (Python 3.11+)
async def main():
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch(url)) for url in urls]
    # Automatic cleanup, exception propagation
    results = [task.result() for task in tasks]
```

**Critical anti-pattern**:
```python
# ❌ Blocking the event loop
async def process():
    result = requests.get(url)  # BLOCKS entire event loop!
    return result.json()

# ✅ Use async libraries
async def process():
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

**Decision framework**: When to use async vs threading vs multiprocessing

---

### 7. scientific-computing-foundations

**Purpose**: NumPy/pandas patterns, vectorization, performance

**Target length**: 1,400-1,600 lines

**Core sections**:
1. NumPy fundamentals - Array creation, indexing, broadcasting, dtypes
2. Vectorization - Replacing loops, ufuncs, performance patterns
3. Memory efficiency - Views vs copies, in-place operations, memory layout
4. Pandas patterns - DataFrame operations, groupby, method chaining
5. Performance anti-patterns - Iterating DataFrames, chained indexing, fragmentation
6. Data pipelines - Method chaining, pipe(), assign()
7. Large datasets - Chunking, Dask basics, memory-mapped arrays
8. Type hints for arrays - numpy.typing, DataFrame annotations

**Critical anti-pattern**:
```python
# ❌ WRONG: Iterating DataFrame (100-500x slower!)
total = 0
for idx, row in df.iterrows():
    total += row['value'] * row['weight']

# ✅ CORRECT: Vectorized
total = (df['value'] * df['weight']).sum()
```

**Performance patterns**:
```python
# ✅ Avoid chained indexing (SettingWithCopyWarning)
df.loc[df['age'] > 30, 'salary'] *= 1.1

# ✅ Pre-allocate DataFrames
df = pd.DataFrame({
    'a': range(1000),
    'b': [i**2 for i in range(1000)]
})
```

**Profiling section**: %timeit, memory_profiler, identifying bottlenecks

---

### 8. ml-engineering-workflows

**Purpose**: ML experiment tracking, reproducibility, monitoring

**Target length**: 1,300-1,500 lines

**Core sections**:
1. Experiment tracking - MLflow, Weights & Biases, logging patterns
2. Data pipeline architecture - Reproducible splits, versioning, validation
3. Model lifecycle - Training, checkpointing, versioning, model registry
4. Hydra configuration - Hyperparameter management, experiment configs
5. Metrics and logging - Structured logging, metric computation, tracking
6. Reproducibility - Random seeds, determinism, environment pinning
7. Production monitoring - Metric collection, alerting, drift detection basics
8. ML project structure - Cookiecutter patterns, research vs production

**Reproducibility pattern**:
```python
# ✅ Reproducible split
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# ✅ Tracked experiment
import mlflow

@hydra.main(config_path="conf", config_name="config")
def train_model(cfg):
    with mlflow.start_run():
        mlflow.log_params(cfg)
        model = build_model(cfg)
        # ... training ...
        mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.pytorch.log_model(model, "model")
```

**Monitoring emphasis**: Collecting metrics for production alerting, drift detection

---

### 9. debugging-and-profiling

**Purpose**: Diagnosis methodology, profiling, optimization

**Target length**: 1,500-1,700 lines

**Core sections**:
1. Debugging fundamentals - pdb basics, breakpoints, post-mortem
2. Modern debugging - debugpy, VS Code integration, conditional breakpoints
3. CPU profiling - cProfile, snakeviz, line_profiler, hotspots
4. Memory profiling - memory_profiler, tracemalloc, memray, leak detection
5. Profiling async - yappi, async-specific patterns
6. Performance optimization - Caching, lru_cache, memoization
7. Diagnosis methodology - **Profile before optimizing**, benchmarking
8. Common bottlenecks - I/O vs CPU bound, GIL implications, C extensions

**Core principle**: **ALWAYS profile before optimizing**

**Diagnosis workflow**:
```python
# Step 1: Profile first
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
result = process_data(items)
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 time consumers

# Step 2: Optimize the actual bottleneck
@lru_cache(maxsize=1000)
def complex_transform(item):
    # Now cached
    ...
```

**Diagnosis flowchart**: Slow? → Profile CPU → If not CPU → Profile I/O → If not I/O → Check algorithm

---

## Cross-Skill Integration

### Router Patterns

**Simple routing** (single skill):
- "Type errors" → modern-syntax-and-types
- "How to structure project?" → project-structure-and-tooling
- "Too many lint warnings" → systematic-delinting

**Sequential routing** (multiple skills in order):
- "Setup new ML project" → project-structure-and-tooling THEN ml-engineering-workflows
- "Legacy code needs types" → project-structure-and-tooling (mypy setup) THEN modern-syntax-and-types (add types)
- "Configure linting and fix warnings" → project-structure-and-tooling (setup) THEN systematic-delinting (fix)

**Diagnostic routing** (diagnose first):
- "Code is slow" → debugging-and-profiling (profile) THEN appropriate optimization skill
- "Pandas too slow" → debugging-and-profiling (profile) THEN scientific-computing-foundations (vectorize)
- "Async performance issue" → debugging-and-profiling (async profiling) THEN async-patterns-and-concurrency (fix)

### Skill Dependencies

```
using-python-engineering (router)
    ├─→ modern-syntax-and-types
    ├─→ project-structure-and-tooling ──→ systematic-delinting
    ├─→ testing-and-quality
    ├─→ async-patterns-and-concurrency
    ├─→ debugging-and-profiling ──→ scientific-computing-foundations
    │                          └──→ async-patterns-and-concurrency
    ├─→ scientific-computing-foundations
    └─→ ml-engineering-workflows
```

---

## Plugin Metadata

**File**: `plugins/axiom-python-engineering/.claude-plugin/plugin.json`

```json
{
  "name": "axiom-python-engineering",
  "version": "1.0.0",
  "description": "Modern Python 3.12+ engineering: types, testing, async, scientific computing, ML workflows",
  "category": "development"
}
```

**Marketplace entry** (to be added to `.claude-plugin/marketplace.json`):
```json
{
  "name": "axiom-python-engineering",
  "source": "./plugins/axiom-python-engineering"
}
```

---

## Quality Standards

### Per-Skill Requirements

1. **YAML front matter** - name, description (for skill discovery)
2. **Production-ready examples** - All code runnable, tested patterns
3. **Anti-patterns** - Show wrong approach alongside correct
4. **Why it matters** - Explain underlying issues, not just fixes
5. **Version awareness** - Note when features introduced (3.10, 3.11, 3.12)
6. **Target length adherence** - Stay within specified ranges
7. **Markdown quality** - Code blocks with language tags, clear headers

### Example Format Template

```python
# ❌ WRONG: [Brief explanation]
bad_code_example()

# ✅ CORRECT: [Brief explanation]
good_code_example()

# Why this matters: [Explanation of underlying issue]
```

### Testing Approach

Skills will be validated using the RED-GREEN-REFACTOR pattern for process documentation:
1. **RED**: Identify scenarios where Claude would give wrong/incomplete Python advice
2. **GREEN**: Write/refine skills to guide Claude to correct answers
3. **REFACTOR**: Iterate based on testing with subagents

---

## Implementation Notes

### Priorities

**Phase 1 - Core Python** (implement first):
1. using-python-engineering (router)
2. modern-syntax-and-types
3. project-structure-and-tooling
4. systematic-delinting

**Phase 2 - Quality & Performance**:
5. testing-and-quality
6. debugging-and-profiling
7. async-patterns-and-concurrency

**Phase 3 - AI/ML**:
8. scientific-computing-foundations
9. ml-engineering-workflows

### Key Design Decisions

1. **Line length 140** - Explicitly configured in systematic-delinting, not ignored
2. **Delinting ≠ refactoring** - Minimal changes to fix warnings
3. **Ruff over black/flake8** - Modern unified tooling
4. **Python 3.12+ focus** - While noting feature versions
5. **Symptom-based routing** - Matches established pytorch pattern
6. **Profile before optimize** - Core principle in debugging-and-profiling

### Future Considerations

- **Additional skills**: May add packaging-and-distribution (separate from project-structure)
- **Web frameworks**: Could add fastapi-patterns or django-patterns as separate skills
- **Data engineering**: Could expand scientific-computing into separate ETL/pipeline skill

---

## Success Criteria

**Design validated when**:
1. ✅ Faction alignment clear (Axiom values)
2. ✅ Router symptom coverage comprehensive
3. ✅ Skill boundaries well-defined
4. ✅ No overlaps or gaps in coverage
5. ✅ Consistent with repository patterns
6. ✅ Target lengths realistic (based on pytorch pack analysis)

**Implementation successful when**:
1. All 9 skills written to target lengths
2. Examples tested and runnable
3. Router comprehensive with all symptoms
4. Validated via RED-GREEN-REFACTOR testing
5. Plugin registered in marketplace.json
6. README updated with axiom-python-engineering

---

## Next Steps

1. Write design document ✅ (this document)
2. Set up git worktree for implementation
3. Create implementation plan with task breakdown
4. Implement skills in phases (Core → Quality → AI/ML)
5. Test with RED-GREEN-REFACTOR approach
6. Register plugin in marketplace
7. Update repository README
