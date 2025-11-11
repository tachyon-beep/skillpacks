## Python Engineering Plugin

**Location:** `/home/john/skillpacks/plugins/axiom-python-engineering/`

**Responsibility:** Provides comprehensive Python 3.12+ engineering guidance including type systems, testing, async patterns, scientific computing, and ML workflows.

**Key Components:**
- `using-python-engineering/SKILL.md` - Router skill that directs to 9 specialized Python skills based on symptoms (399 lines)
- `resolving-mypy-errors/SKILL.md` - Systematic mypy error resolution process (1,136 lines)
- `testing-and-quality/SKILL.md` - Pytest patterns, fixtures, mocking, and property-based testing (1,867 lines)
- `project-structure-and-tooling/SKILL.md` - Project setup, pyproject.toml, ruff, pre-commit configuration (1,612 lines)
- `systematic-delinting/SKILL.md` - Process for fixing lint warnings without disabling rules (1,524 lines)

**Dependencies:**
- Inbound: None observed (standalone plugin)
- Outbound: References to external Python tools (mypy, ruff, pytest, MLflow, NumPy, pandas) as implementation targets, not code dependencies

**Patterns Observed:**
- Router pattern: `using-python-engineering` catalogs all 9 skills and routes by symptom
- Process-driven: Skills emphasize PROCESS over syntax (e.g., "resolving" vs "learning types")
- Cross-skill separation: Clear boundaries (setup vs fixing, syntax vs resolution, diagnosis vs optimization)
- Skill completeness: All 10 skills fully implemented with line counts ranging from 399-1,867 lines

**Concerns:**
- None observed

**Confidence:** High - Router skill provides complete catalog of all 9 specialist skills, verified by sampling 3 representative skills (resolving-mypy-errors, systematic-delinting, using-python-engineering), all skills confirmed as complete implementations with substantial content

---
