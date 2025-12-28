
# Incremental Analysis

## Purpose

Git-aware delta analysis for repeat users. Identify what changed since last architecture analysis, which subsystems are affected, and detect high-churn maintenance burden indicators.

## When to Use

- Prior analysis workspace exists (docs/arch-analysis-*)
- User requests "what changed" or "update analysis"
- Regular maintenance analysis cadence (weekly/monthly reviews)
- Detecting high-churn hotspots that indicate architectural problems
- User mentions: "incremental", "delta", "update", "refresh", "since last time"

## Core Principle: Delta Over Repeat

**Don't re-analyze unchanged code. Focus effort on what actually changed.**

Full analysis of a 50K LOC codebase takes hours. If only 5% changed, incremental analysis takes minutes and surfaces the important changes immediately.

## Prerequisite Check (MANDATORY)

Before incremental analysis, verify:

```bash
# 1. Find existing workspaces
find docs -name "arch-analysis-*" -type d | sort -r

# 2. Most recent workspace exists
WORKSPACE=$(find docs -name "arch-analysis-*" -type d | sort -r | head -1)
[ -z "$WORKSPACE" ] && echo "No prior workspace found"

# 3. Catalog exists and is valid
[ -f "$WORKSPACE/02-subsystem-catalog.md" ] || echo "Missing catalog"

# 4. Coordination log has commit reference
grep -q "commit\|Commit\|HEAD" "$WORKSPACE/00-coordination.md" || echo "No commit reference"
```

**If prerequisites missing:**

```
Cannot perform incremental analysis.

Reason: [No prior workspace / Missing catalog / No commit reference]

Options:
A) Run /analyze-codebase for full analysis
B) Specify prior workspace manually: /incremental [workspace-path]
```

## Process Steps

### Step 1: Establish Baseline

**Extract from prior analysis:**

```bash
# Get workspace path
WORKSPACE="docs/arch-analysis-YYYY-MM-DD-HHMM"

# Find baseline commit (from coordination log or catalog header)
BASELINE_COMMIT=$(grep -oE '[0-9a-f]{7,40}' "$WORKSPACE/00-coordination.md" | head -1)

# If no commit in coordination log, use file modification time
if [ -z "$BASELINE_COMMIT" ]; then
    BASELINE_DATE=$(stat -c %Y "$WORKSPACE/02-subsystem-catalog.md")
    BASELINE_COMMIT=$(git rev-list -1 --before="@$BASELINE_DATE" HEAD)
fi

echo "Baseline: $BASELINE_COMMIT"
echo "Current: $(git rev-parse HEAD)"
```

**Document baseline:**

```markdown
## Incremental Analysis Baseline

**Prior Workspace:** docs/arch-analysis-2025-01-15-1430/
**Baseline Commit:** abc1234 (2025-01-15)
**Current Commit:** def5678 (2025-01-28)
**Days Since Analysis:** 13
**Total Commits:** 47
```

### Step 2: Collect Git Delta

**Files changed since baseline:**

```bash
# List all changed files
git diff --name-only $BASELINE_COMMIT..HEAD

# Get change statistics
git diff --stat $BASELINE_COMMIT..HEAD

# Categorize by change type
git diff --name-status $BASELINE_COMMIT..HEAD
# A = Added, M = Modified, D = Deleted, R = Renamed
```

**Map changes to subsystems:**

For each changed file:
1. Determine which subsystem contains it (from prior catalog)
2. Classify change type: New file / Modified / Deleted / Moved
3. Flag if file is in NEW directory (potential new subsystem)

### Step 3: Classify Changes

**Change Classification Matrix:**

| Change Type | Example | Action Required |
|-------------|---------|-----------------|
| **New files in existing subsystem** | src/auth/oauth.py added | Update catalog entry, check patterns |
| **New directory (potential subsystem)** | src/payments/ created | Assess: new subsystem or expansion? |
| **Modified core files** | src/auth/handler.py changed | Check if responsibility changed |
| **Modified config/deps** | requirements.txt, package.json | Update dependency graph |
| **Deleted files** | src/legacy/ removed | Check if subsystem should be archived |
| **Renamed/moved** | src/old/ → src/new/ | Update location in catalog |
| **Test changes only** | tests/*.py | Update test infrastructure section |
| **Documentation only** | docs/*.md, README | Low priority for architecture |

**Document in change table:**

```markdown
## Change Summary

| Subsystem | Files Added | Modified | Deleted | Change Type |
|-----------|-------------|----------|---------|-------------|
| Auth | 2 | 3 | 0 | Expanded (new OAuth) |
| Payments | 8 | 0 | 0 | NEW SUBSYSTEM |
| Legacy | 0 | 0 | 12 | Removed |
| API Gateway | 0 | 5 | 0 | Modified |
| (Unknown) | 3 | 0 | 0 | Needs classification |
```

### Step 4: Detect High-Churn Hotspots

**High churn indicates maintenance burden or architectural problems.**

```bash
# Files changed frequently in last 3 months
git log --since="3 months ago" --pretty=format: --name-only | \
    sort | uniq -c | sort -rn | head -20

# Files with most commits
git log --since="3 months ago" --format='%H' -- [file] | wc -l
```

**Churn Thresholds:**

| Changes/Quarter | Classification | Implication |
|-----------------|----------------|-------------|
| 1-5 | Normal | Expected maintenance |
| 6-10 | Elevated | Watch for patterns |
| 11-20 | High Churn | Investigate root cause |
| 21+ | Hotspot | Likely architectural issue |

**Document hotspots:**

```markdown
## High-Churn Hotspots

| File | Changes (3mo) | Subsystem | Concern |
|------|---------------|-----------|---------|
| src/api/handler.py | 24 | API Gateway | Hotspot - possible god file |
| src/models/user.py | 18 | User Service | High churn - schema instability |
| config/settings.py | 15 | Config | High churn - environment drift |

**Recommendations:**
- API Gateway handler: Consider splitting into smaller modules
- User model: Stabilize schema before adding features
- Config: Audit environment differences
```

### Step 5: Dependency Impact Analysis

**For each changed subsystem, trace impact:**

```bash
# Find files that import changed module
grep -r "from changed_module import\|import changed_module" --include="*.py"
```

**Impact Categories:**

| Changed Subsystem | Dependents (Inbound) | Impact Level |
|-------------------|---------------------|--------------|
| Auth | API Gateway, User, Admin | HIGH - core service |
| Payments (NEW) | None yet | LOW - isolated |
| Logging | All subsystems | MEDIUM - cross-cutting |

**Document impact:**

```markdown
## Dependency Impact

### Auth Service (Modified)
**Inbound Dependencies:** API Gateway, User Service, Admin Panel

**Changes May Affect:**
- API Gateway: Auth middleware (gateway/middleware/auth.py)
- User Service: Login flow (user/views/auth.py)
- Admin Panel: Session handling (admin/session.py)

**Recommendation:** Verify auth interface compatibility after changes

### Payments (NEW Subsystem)
**Inbound Dependencies:** None (new)
**Outbound Dependencies:** Database, External Payment API

**Recommendation:** Add to subsystem catalog, establish dependency boundaries
```

### Step 6: Produce Incremental Report

**Synthesize findings into actionable report.**

## Output Contract (MANDATORY)

Write to `08-incremental-report.md` in current workspace:

```markdown
# Incremental Analysis Report

**Prior Analysis:** [workspace path]
**Baseline Commit:** [hash] ([date])
**Current Commit:** [hash] ([date])
**Analysis Date:** YYYY-MM-DD
**Time Range:** [X] days, [Y] commits

## Executive Summary

- **Files Changed:** [count]
- **Subsystems Affected:** [count]/[total]
- **New Subsystems:** [count]
- **Removed Subsystems:** [count]
- **High-Churn Hotspots:** [count]
- **Dependency Impact:** [HIGH/MEDIUM/LOW]

## Change Summary by Subsystem

| Subsystem | Status | Files Changed | Change Type | Impact |
|-----------|--------|---------------|-------------|--------|
| [Name] | Modified | 5 | Core logic changed | HIGH |
| [Name] | New | 8 | New subsystem | MEDIUM |
| [Name] | Removed | 12 | Archived | LOW |
| [Name] | Unchanged | 0 | - | - |

## Detailed Changes

### [Subsystem Name] (Modified)

**Files Changed:**
- `file1.py` - [A/M/D] - [brief description]
- `file2.py` - [A/M/D] - [brief description]

**Change Assessment:**
- Responsibility: [Unchanged / Expanded / Reduced / Changed]
- Patterns: [Consistent / New patterns introduced]
- Dependencies: [Unchanged / New deps / Removed deps]

**Catalog Update Required:** [Yes - specify what / No]

### [New Subsystem Name] (New)

**Location:** `src/new_subsystem/`
**Files:** [count]
**Detected Responsibility:** [inferred from code]

**Recommended Action:** Add to 02-subsystem-catalog.md with:
- Location, Responsibility, Key Components
- Dependencies (inbound/outbound)
- Initial confidence: Medium (new, not fully analyzed)

## High-Churn Hotspots

| Rank | File | Changes | Subsystem | Assessment |
|------|------|---------|-----------|------------|
| 1 | [path] | [count] | [subsystem] | [Concern] |
| 2 | [path] | [count] | [subsystem] | [Concern] |

**Root Cause Analysis:**
- [Hotspot 1]: [Why high churn? Schema changes? Bug fixes? Feature additions?]
- [Hotspot 2]: [Analysis]

## Dependency Impact Matrix

| Changed | Affects | Impact | Verification Needed |
|---------|---------|--------|---------------------|
| Auth | Gateway, User | HIGH | Interface compatibility |
| Config | All | MEDIUM | Environment drift check |

## Catalog Update Actions

### Updates Required:
1. **[Subsystem]:** Update [specific field] to reflect [change]
2. **[New Subsystem]:** Add new entry (template below)
3. **[Removed Subsystem]:** Archive entry, move to historical section

### Suggested Catalog Entry (New Subsystem):
```markdown
## [New Subsystem Name]

**Location:** `[path]`

**Responsibility:** [One sentence based on code review]

**Key Components:**
- `file1.py` - [description]
- `file2.py` - [description]

**Dependencies:**
- Inbound: [None yet / list]
- Outbound: [list observed]

**Patterns Observed:**
- [Observed patterns]

**Concerns:**
- None observed (newly added, limited history)

**Confidence:** Medium - Incremental analysis, recommend full review
```

## Recommendations

### Immediate Actions:
1. [Specific action for highest-impact change]
2. [Specific action for new subsystems]

### Catalog Maintenance:
- [ ] Add [X] new subsystem entries
- [ ] Update [Y] existing entries
- [ ] Archive [Z] removed subsystems

### Re-Analysis Triggers:
If any of these apply, recommend full re-analysis:
- [ ] >30% of files changed
- [ ] Core architecture patterns changed
- [ ] New subsystem count > 3
- [ ] Dependency graph significantly altered

## Limitations

- **Scope:** Only analyzed git diff, not semantic changes
- **Confidence:** File-level changes detected, logic changes require code review
- **Missing:** [What couldn't be determined from diff alone]

## Next Analysis Recommendation

Based on change velocity:
- **High velocity (>20 commits/week):** Weekly incremental analysis
- **Medium velocity (5-20 commits/week):** Bi-weekly incremental
- **Low velocity (<5 commits/week):** Monthly incremental
- **Major release:** Full re-analysis recommended

**Commit this workspace for future incremental baseline.**
```

## Common Rationalizations (STOP SIGNALS)

| Rationalization | Reality |
|-----------------|---------|
| "Lots of files changed, easier to do full analysis" | Incremental identifies WHAT changed. Full analysis is for HOW. |
| "Git diff is enough, don't need to check catalog" | Diff shows files; catalog shows subsystem impact. Both needed. |
| "Churn analysis is optional" | Churn hotspots reveal architectural problems. Don't skip. |
| "Small changes don't need dependency impact" | Small changes in core services have large impact. Always check. |
| "I'll just eyeball the diff" | Systematic classification catches what eyeballing misses. |
| "New subsystem can wait for next full analysis" | Document it now while context is fresh. |

## Anti-Patterns

**DON'T skip baseline verification:**
```
WRONG: "I'll just compare to recent commits"
RIGHT: Verify exact baseline commit from prior workspace coordination log
```

**DON'T ignore unclassified files:**
```
WRONG: "3 files in unknown location, probably not important"
RIGHT: Classify all files. Unknown files may indicate new subsystems or misorganization.
```

**DON'T report churn without analysis:**
```
WRONG: "handler.py changed 24 times (high churn)"
RIGHT: "handler.py changed 24 times. Analysis: 15 bug fixes, 6 features, 3 refactors.
        Root cause: Growing responsibilities suggest splitting module."
```

## Success Criteria

**You succeeded when:**
- Baseline commit identified from prior workspace
- All changed files classified by subsystem and change type
- New subsystems identified with recommended catalog entries
- High-churn hotspots detected with root cause analysis
- Dependency impact traced for changed subsystems
- Catalog update actions clearly specified
- Report follows output contract exactly
- Written to 08-incremental-report.md

**You failed when:**
- Baseline not verified (used arbitrary commit)
- Files left unclassified
- New directories not assessed as potential subsystems
- Churn analysis skipped
- Dependency impact not traced
- Report lacks actionable catalog updates
- Just produced git diff without analysis

## Integration with Workflow

Incremental analysis is invoked:
1. When prior workspace detected and user requests update
2. As periodic maintenance (weekly/monthly cadence)
3. Before major releases to assess change scope
4. When investigating why analysis feels "stale"

**Pipeline:** Detect Prior Workspace → Establish Baseline → Collect Delta → Classify → Churn Analysis → Impact Analysis → Report → Catalog Updates

## Git Best Practices for Incremental Analysis

**Record baseline in coordination log:**

```markdown
## Analysis Metadata
- Commit: abc1234
- Branch: main
- Date: 2025-01-15
```

**Tag analysis points (optional but recommended):**

```bash
git tag -a arch-analysis-2025-01-15 -m "Architecture analysis baseline"
```

**Use consistent branch:**

If analysis done on feature branch, note it. Incremental should compare same branch to avoid noise from merge commits.
