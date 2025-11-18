
# Assessing Code Quality

## Purpose

Analyze code quality indicators beyond architecture to identify maintainability issues, code smells, and technical debt - produces quality scorecard with actionable improvement recommendations.

## When to Use

- Coordinator delegates quality assessment after subsystem catalog completion
- Task specifies analyzing code quality in addition to architecture
- Need to identify refactoring priorities beyond structural concerns
- Output feeds into architect handover reports or improvement planning

## Core Principle: Evidence-Based Quality Assessment

**Good quality analysis identifies specific, actionable issues. Poor quality analysis makes vague claims about "bad code."**

Your goal: Provide evidence-based quality metrics with concrete examples and remediation guidance.

## Quality Analysis Dimensions

### 1. Code Complexity

**What to assess:**
- Function/method length (lines of code)
- Cyclomatic complexity (decision points)
- Nesting depth (indentation levels)
- Parameter count

**Evidence to collect:**
- Longest functions with line counts
- Functions with highest decision complexity
- Deeply nested structures (> 4 levels)
- Functions with >5 parameters

**Thresholds (guidelines, not rules):**
- Functions > 50 lines: Flag for review
- Cyclomatic complexity > 10: Consider refactoring
- Nesting > 4 levels: Simplification candidate
- Parameters > 5: Consider parameter object

**Example documentation:**
```markdown
### Complexity Concerns

**High complexity functions:**
- `src/api/order_processing.py:process_order()` - 127 lines, complexity ~15
  - 8 nested if statements
  - Handles validation, pricing, inventory, shipping in single function
  - **Recommendation:** Extract validation, pricing, inventory, shipping into separate functions

- `src/utils/data_transform.py:transform_dataset()` - 89 lines, 7 parameters
  - **Recommendation:** Create DatasetConfig object to replace parameter list
```

### 2. Code Duplication

**What to assess:**
- Repeated code blocks (copy-paste patterns)
- Similar functions with slight variations
- Duplicated logic across subsystems

**Evidence to collect:**
- Quote repeated code blocks (5+ lines)
- List functions with similar structure
- Note duplication percentage (if tool available)

**Analysis approach:**
1. Read representative files from each subsystem
2. Look for similar patterns, function structures
3. Note copy-paste indicators (similar variable names, comment duplication)
4. Assess if duplication is deliberate or accidental

**Example documentation:**
```markdown
### Duplication Concerns

**Copy-paste pattern in validation:**
- `src/api/users.py:validate_user()` (lines 45-67)
- `src/api/orders.py:validate_order()` (lines 89-111)
- `src/api/products.py:validate_product()` (lines 23-45)

All three functions follow identical structure:
1. Check required fields
2. Validate format with regex
3. Check database constraints
4. Return validation result

**Recommendation:** Extract common validation framework to `src/utils/validation.py`
```

### 3. Code Smells

**Common smells to identify:**

**Long parameter lists:**
- Functions with >5 parameters
- Recommendation: Parameter object or builder pattern

**God objects/functions:**
- Classes with >10 methods
- Functions doing multiple unrelated things
- Recommendation: Single Responsibility Principle refactoring

**Magic numbers:**
- Hardcoded values without named constants
- Recommendation: Extract to configuration or named constants

**Dead code:**
- Commented-out code blocks
- Unused functions/classes (no references found)
- Recommendation: Remove or document why kept

**Shotgun surgery indicators:**
- Single feature change requires edits in 5+ files
- Indicates high coupling
- Recommendation: Improve encapsulation

**Example documentation:**
```markdown
### Code Smell Observations

**Magic numbers:**
- `src/services/cache.py`: Hardcoded timeout values (300, 3600, 86400)
  - **Recommendation:** Extract to CacheConfig with named durations

**Dead code:**
- `src/legacy/` directory contains 15 files, no imports found in active code
  - Last modified: 2023-06-15
  - **Recommendation:** Archive or remove if truly unused

**Shotgun surgery:**
- Adding new payment method requires changes in:
  - `src/api/payment.py`
  - `src/models/transaction.py`
  - `src/utils/validators.py`
  - `src/services/notification.py`
  - `config/payment_providers.json`
  - **Recommendation:** Introduce payment provider abstraction layer
```

### 4. Maintainability Indicators

**What to assess:**
- Documentation coverage (docstrings, comments)
- Test coverage (if test files visible)
- Error handling patterns
- Logging consistency

**Evidence to collect:**
- Percentage of functions with docstrings
- Test file presence per module
- Error handling approaches (try/except, error codes, etc.)
- Logging statements (presence, consistency)

**Example documentation:**
```markdown
### Maintainability Assessment

**Documentation:**
- 12/45 functions (27%) have docstrings
- Public API modules better documented than internal utilities
- **Recommendation:** Add docstrings to all public functions, focus on "why" not "what"

**Error handling inconsistency:**
- `src/api/` uses exception raising
- `src/services/` uses error code returns
- `src/utils/` mixes both approaches
- **Recommendation:** Standardize on exceptions with custom exception hierarchy

**Logging:**
- Inconsistent log levels (some files use DEBUG for errors)
- No structured logging (difficult to parse)
- **Recommendation:** Adopt structured logging library, establish level conventions
```

### 5. Dependency Quality

**What to assess:**
- Coupling between subsystems
- Circular dependencies
- External dependency management

**Evidence from subsystem catalog:**
- Review "Dependencies - Outbound" sections
- Count dependencies per subsystem
- Identify bidirectional dependencies (A→B and B→A)

**Analysis approach:**
1. Use subsystem catalog dependency data
2. Count inbound/outbound dependencies per subsystem
3. Identify highly coupled subsystems (>5 dependencies)
4. Note circular dependency patterns

**Example documentation:**
```markdown
### Dependency Concerns

**High coupling:**
- `API Gateway` subsystem: 8 outbound dependencies (most in system)
  - Depends on: Auth, User, Product, Order, Payment, Notification, Logging, Cache
  - **Observation:** Acts as orchestrator, coupling may be appropriate
  - **Recommendation:** Monitor for API Gateway becoming bloated

**Circular dependencies:**
- `User Service` ↔ `Notification Service`
  - User triggers notifications, Notification updates user preferences
  - **Recommendation:** Introduce event bus to break circular dependency

**Dependency concentration:**
- 6/10 subsystems depend on `Database Layer`
- Database Layer has no abstraction (direct SQL queries)
- **Recommendation:** Consider repository pattern to isolate database logic
```

## Output Contract

Write findings to workspace as `05-quality-assessment.md`:

```markdown
# Code Quality Assessment

**Analysis Date:** YYYY-MM-DD
**Scope:** [Subsystems analyzed]
**Methodology:** Static code review, pattern analysis

## Quality Scorecard

| Dimension | Rating | Severity | Evidence Count |
|-----------|--------|----------|----------------|
| Complexity | Medium | 3 High, 5 Medium | 8 functions flagged |
| Duplication | High | 2 Critical, 4 Medium | 6 patterns identified |
| Code Smells | Medium | 0 Critical, 7 Medium | 7 smells documented |
| Maintainability | Medium-Low | 1 Critical, 3 Medium | 4 concerns noted |
| Dependencies | Low | 1 Medium | 2 concerns noted |

**Overall Rating:** Medium - Several actionable improvements identified, no critical blockers

## Detailed Findings

### 1. Complexity Concerns

[List from analysis above]

### 2. Duplication Concerns

[List from analysis above]

### 3. Code Smell Observations

[List from analysis above]

### 4. Maintainability Assessment

[List from analysis above]

### 5. Dependency Concerns

[List from analysis above]

## Prioritized Recommendations

### Critical (Address Immediately)
1. [Issue with highest impact]

### High (Next Sprint)
2. [Important issues]
3. [Important issues]

### Medium (Next Quarter)
4. [Moderate issues]
5. [Moderate issues]

### Low (Backlog)
6. [Nice-to-have improvements]

## Methodology Notes

**Analysis approach:**
- Sampled [N] representative files across [M] subsystems
- Focused on [specific areas of concern]
- Did NOT use automated tools (manual review only)

**Limitations:**
- Sample-based (not exhaustive)
- No runtime analysis (static review only)
- Test coverage estimates based on file presence
- No quantitative complexity metrics (manual assessment)

**For comprehensive analysis, consider:**
- Running static analysis tools (ruff, pylint, mypy for Python)
- Measuring actual test coverage
- Profiling runtime behavior
- Security-focused code review
```

## Severity Rating Guidelines

**Critical:**
- Blocks core functionality or deployment
- Security vulnerability present
- Data corruption risk
- Examples: SQL injection, hardcoded credentials, unhandled exceptions in critical path

**High:**
- Significant maintainability impact
- High effort to modify or extend
- Frequent source of bugs
- Examples: God objects, extreme duplication, shotgun surgery patterns

**Medium:**
- Moderate maintainability concern
- Refactoring beneficial but not urgent
- Examples: Long functions, missing documentation, inconsistent error handling

**Low:**
- Minor quality improvement
- Cosmetic or style issues
- Examples: Magic numbers, verbose naming, minor duplication

## Integration with Architect Handover

Quality assessment feeds directly into `creating-architect-handover.md`:

1. Quality scorecard provides severity ratings
2. Prioritized recommendations become architect's action items
3. Code smells inform refactoring strategy
4. Dependency concerns guide architectural improvements

The architect handover briefing will synthesize architecture + quality into comprehensive improvement plan.

## When to Skip Quality Assessment

**Optional scenarios:**
- User requested architecture-only analysis
- Extremely tight time constraints (< 2 hours total)
- Codebase is very small (< 1000 lines)
- Quality issues not relevant to stakeholder needs

**Document if skipped:**
```markdown
## Quality Assessment: SKIPPED

**Reason:** [Time constraints / Not requested / etc.]
**Recommendation:** Run focused quality review post-stakeholder presentation
```

## Systematic Analysis Checklist

```
[ ] Read subsystem catalog to understand structure
[ ] Sample 3-5 representative files per subsystem
[ ] Document complexity concerns (functions >50 lines, high nesting)
[ ] Identify duplication patterns (repeated code blocks)
[ ] Note code smells (god objects, magic numbers, dead code)
[ ] Assess maintainability (docs, tests, error handling)
[ ] Review dependencies from catalog (coupling, circular deps)
[ ] Rate severity for each finding (Critical/High/Medium/Low)
[ ] Prioritize recommendations by impact
[ ] Write to 05-quality-assessment.md following contract
[ ] Document methodology and limitations
```

## Success Criteria

**You succeeded when:**
- Quality assessment covers all 5 dimensions
- Each finding has concrete evidence (file paths, line numbers, examples)
- Severity ratings are justified
- Recommendations are specific and actionable
- Methodology and limitations documented
- Output written to 05-quality-assessment.md

**You failed when:**
- Vague claims without evidence ("code is messy")
- No severity ratings or priorities
- Recommendations are generic ("improve code quality")
- Missing methodology notes
- Skipped dimensions without documentation

## Common Mistakes

**❌ Analysis paralysis**
"Need to read every file" → Sample strategically, 20% coverage reveals patterns

**❌ Vague findings**
"Functions are too complex" → "process_order() is 127 lines with complexity ~15"

**❌ No prioritization**
Flat list of 50 issues → Prioritize by severity/impact, focus on Critical/High

**❌ Tool-dependent**
"Can't assess without linting tools" → Manual review reveals patterns, note as limitation

**❌ Perfectionism**
"Everything needs fixing" → Focus on high-impact issues, accept some technical debt

## Integration with Workflow

This briefing is typically invoked as:

1. **Coordinator** completes subsystem catalog (02-subsystem-catalog.md)
2. **Coordinator** (optionally) validates catalog
3. **Coordinator** writes task specification for quality assessment
4. **YOU** read subsystem catalog to understand structure
5. **YOU** perform systematic quality analysis (5 dimensions)
6. **YOU** write to 05-quality-assessment.md following contract
7. **Coordinator** proceeds to diagram generation or architect handover

**Your role:** Complement architectural analysis with code quality insights, providing evidence-based improvement recommendations.
