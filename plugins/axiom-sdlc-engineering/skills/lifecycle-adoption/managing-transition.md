---
parent-skill: lifecycle-adoption
reference-type: transition-strategy
load-when: Tool migrations, platform changes, maintaining compliance during transitions
---

# Managing the Transition Reference

**Parent Skill:** lifecycle-adoption
**When to Use:** Tool migrations (GitHub ↔ Azure DevOps), platform changes, parallel operations, preserving compliance

This reference provides guidance for managing transitions while maintaining CMMI compliance.

---

## Reference Sheet 7: Managing the Transition

### Purpose & Context

**What this achieves**: Preserve CMMI compliance during tool/process changes

**When to apply**:
- Migrating between platforms (GitHub ↔ Azure DevOps)
- Changing processes mid-project (waterfall → agile, gitflow → trunk-based)
- Team/org restructuring
- Tooling upgrades that might break existing practices

**Prerequisites**:
- Understanding of current CMMI compliance level
- New tool/process selected
- Migration timeline (2-4 weeks typical)

### CMMI Maturity Scaling

#### Level 2: Managed (Transition Management)

**Approach**: Preserve audit trail, minimal disruption

**Key Concerns**:
- Don't lose work product history
- Maintain traceability links
- Document the transition itself

**Acceptable Downtime**: 1-2 days (manual processes)

**Documentation**: Transition plan (1-page)

#### Level 3: Defined (Transition Management)

**Approach**: Parallel operation period, systematic migration

**Key Concerns** (beyond Level 2):
- Organizational templates migrated
- Training materials updated
- Process documentation reflects new tools
- Baselines preserved

**Acceptable Downtime**: 0 (parallel operation required)

**Documentation**: Detailed migration plan, rollback procedure

#### Level 4: Quantitatively Managed (Transition Management)

**Approach**: Metrics-driven validation of transition success

**Key Concerns** (beyond Level 3):
- Metrics collection continues uninterrupted
- Baselines remain valid (or recalculated if methodology changes)
- Statistical process control unaffected

**Acceptable Downtime**: 0 (metrics continuity critical)

**Documentation**: Impact analysis on process performance baselines

### Implementation Guidance

#### Quick Start Checklist

**Phase 1: Pre-Transition Planning** (1-2 weeks before)

- [ ] **Inventory current state**:
  - What work products exist? (requirements, ADRs, test plans)
  - What traceability links exist? (issue refs, commit associations)
  - What baselines exist? (metrics, process performance)

- [ ] **Map current → new**:
  - GitHub Issues → Azure DevOps Work Items (or vice versa)
  - GitHub Labels → Azure DevOps Tags
  - GitHub Projects → Azure DevOps Boards
  - Traceability links (issue #123 → work item 456)

- [ ] **Create migration plan**:
  - Timeline (parallel operation period, cutover date)
  - Responsibility (who migrates what)
  - Validation criteria (how to verify success)
  - Rollback procedure (if migration fails)

**Phase 2: Parallel Operation** (2-4 weeks)

**Strategy**: Both tools active, new work goes to new tool, old tool read-only

**Week 1-2**: New work in new tool only
- Create requirements in Azure DevOps (not GitHub)
- Reference old GitHub issues where relevant
- Update traceability: "Migrated from GitHub #123 → ADO 456"

**Week 3-4**: Selective migration of active work
- Migrate in-progress features (PRs not yet merged)
- Migrate recent ADRs (last 6 months)
- Leave old/completed work in GitHub (historical reference)

**Phase 3: Historical Data Migration** (1-2 weeks)

**Critical work products to migrate**:
- ✅ ADRs (all, especially recent)
- ✅ Requirements for active features
- ✅ Open bugs/issues
- ✅ Traceability matrix

**Nice-to-have (defer if time-constrained)**:
- ⏸ Closed bugs from >1 year ago
- ⏸ Deprecated features
- ⏸ Experimental/prototype documentation

**Migration tools**:
- **GitHub → Azure DevOps**: `gh-ado-migrator` (official MS tool)
- **Azure DevOps → GitHub**: `ado-to-github` (community tool)
- Manual export/import for small datasets

**Phase 4: Cutover & Validation** (1 day)

**Cutover**:
- Make old tool read-only (archive repository, disable new issues)
- Announce: "All new work in Azure DevOps as of Monday"
- Update onboarding docs with new tool links

**Validation checklist**:
- [ ] Can trace requirement → code → test in new tool?
- [ ] ADRs migrated and accessible?
- [ ] Metrics collection working in new tool?
- [ ] Team trained on new tool?
- [ ] Rollback procedure documented (in case of emergency)?
- [ ] **CRITICAL: Automated link checker passed** (see enforcement below)

**CRITICAL: Link Checker Enforcement**

**Problem**: Old tool links in code comments, documentation, and commit messages become broken after migration, destroying traceability.

**Required before cutover**:
- [ ] Run automated link checker to scan entire codebase for old tool references
- [ ] For GitHub→Azure DevOps: Search for `github.com/[org]/[repo]/issues/` patterns
- [ ] For Azure DevOps→GitHub: Search for `dev.azure.com/[org]/_workitems/` patterns
- [ ] Document ALL found links in migration inventory

**Link checker script example** (Python):
```python
# link_checker.py - Scan codebase for GitHub issue links
import os
import re

# Pattern to match GitHub issue links
GITHUB_ISSUE_PATTERN = r'github\.com/[\w-]+/[\w-]+/issues/\d+'

broken_links = []

for root, dirs, files in os.walk('.'):
    # Skip .git, node_modules, etc.
    dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', 'venv']]

    for file in files:
        # Only scan text files
        if file.endswith(('.py', '.js', '.md', '.ts', '.java', '.cpp')):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                matches = re.findall(GITHUB_ISSUE_PATTERN, content)
                if matches:
                    broken_links.append({
                        'file': filepath,
                        'links': matches
                    })

print(f"Found {sum(len(item['links']) for item in broken_links)} GitHub issue links")
for item in broken_links:
    print(f"{item['file']}: {len(item['links'])} links")
```

**Remediation options** (choose ONE for each found link):
1. **Add cross-reference**: Update comment with new work item ID
   - Old: `// See GitHub issue #123`
   - New: `// See GitHub issue #123 (migrated to ADO Work Item 789)`
2. **Create redirect**: Set up GitHub→ADO link mapping in docs
   - Maintain `LINK_MAP.md`: `GitHub #123 → ADO 789`
3. **Archive link**: If historical only, add "(archived)" note
   - `// Historical: GitHub issue #123 (archived, see ARCHIVE.md)`

**Week 4 cutover gate requirement**:
- Link checker shows 0 unaddressed links OR
- Every found link has remediation plan documented OR
- CTO approves risk of broken links (with documented justification)

**Red flags**:
- Link checker found 100+ links but "we'll fix them later" → Migration NOT ready for cutover
- Code comments reference specific issues but no migration mapping → Traceability will break

**Phase 5: Post-Cutover Cleanup** (1-2 weeks)

- Archive old tool (but don't delete - historical reference)
- Update all documentation links (point to new tool)
- Recalculate baselines if metrics methodology changed
- Retrospective: What went well, what didn't?

#### Templates & Examples

**Tool Migration Plan (GitHub → Azure DevOps)**:

```markdown
# Tool Migration Plan: GitHub → Azure DevOps

**Goal**: Migrate project "PaymentGateway" from GitHub to Azure DevOps while preserving CMMI Level 3 compliance

**Timeline**: 4 weeks (Feb 1-28, 2026)

## Current State

- **GitHub Repository**: github.com/acme/payment-gateway
- **Work Items**: 150 open issues, 500 closed
- **ADRs**: 25 decision records in docs/adr/
- **Traceability**: Issue refs in PR descriptions
- **Metrics**: GitHub Actions collecting coverage, cycle time

## Migration Phases

### Week 1: Setup & Parallel Start
- [ ] Create Azure DevOps project "PaymentGateway"
- [ ] Set up branch policies (match GitHub branch protection)
- [ ] Import Git repo (preserves commit history)
- [ ] Create work item templates (match GitHub issue templates)
- [ ] Train team (2-hour workshop on Azure DevOps)

### Week 2-3: Parallel Operation
- [ ] New features: Create work items in Azure DevOps
- [ ] New code: PRs in Azure DevOps (GitHub still accessible read-only)
- [ ] Migrate critical work products:
  - [ ] All 25 ADRs → Azure DevOps Wiki
  - [ ] 30 open issues (in-progress features) → Work items
  - [ ] Traceability matrix → Azure DevOps Queries

### Week 4: Cutover & Validation
- [ ] Feb 22: Announce cutover date (Feb 25)
- [ ] Feb 25: Make GitHub repo read-only
- [ ] Feb 25: Update all documentation links
- [ ] Feb 26-28: Validation and bug fixes

## Traceability Preservation

Old format (GitHub):
```
Issue #123: Add 2FA support
PR #456: Implements issue #123
```

New format (Azure DevOps):
```
Work Item 789: Add 2FA support
(Migrated from GitHub #123 on 2026-02-15)
PR 12: Linked to Work Item 789
```

## Rollback Procedure

If critical issues found within 1 week of cutover:
1. Re-enable GitHub for new work
2. Pause Azure DevOps migration
3. Root cause analysis
4. Decide: fix and retry, or abandon migration

## Success Criteria

- [ ] All active work items migrated
- [ ] Traceability functional (can trace requirement → code)
- [ ] Metrics collection restored
- [ ] Zero work lost
- [ ] Team confident in new tool (survey)
```

### Common Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| **Big Bang Migration** | Cutover on Friday, chaos Monday | Parallel operation for 2-4 weeks |
| **Migrate Everything** | 5 years of closed issues = wasted effort | Migrate active work + critical historical only |
| **No Rollback Plan** | "We'll fix forward" → panic if issues | Document rollback procedure before cutover |
| **Break Traceability** | Old links broken, compliance lost | Update links, preserve refs (e.g., "Migrated from GH #123") |
| **No Training** | Team confused, productivity drops | 2-hour workshop before cutover |

### Tool Integration

**GitHub → Azure DevOps**:
- Microsoft's `gh-ado-migrator` tool
- Preserves: Issue history, comments, labels → tags
- Manual: ADR migration (copy to Wiki)

**Azure DevOps → GitHub**:
- Community tool `ado-to-github` (less official support)
- Manual for complex migrations
- Alternative: Keep both active (Azure DevOps for work management, GitHub for code)

**Hybrid (Both Tools)**:
- Some organizations keep GitHub for code, Azure DevOps for work tracking
- Requires cross-tool linking automation

### Verification & Validation

**How to verify transition succeeded**:
- Spot check 10 random requirements: Can you trace to code and tests?
- Compare metrics before/after: Are baselines similar (accounting for growth)?
- Team survey: Confident in new tool?
- Audit dry-run: Would this pass compliance review?

**Common failure modes**:
- **Traceability links broken** → Update references, add "Migrated from X" notes
- **Team productivity drop** → More training, simpler workflows
- **Metrics discontinuity** → Recalculate baselines with new methodology

### Related Practices

- **Platform-specific migration**: See platform-integration skill
- **If team resists transition**: See Reference Sheet 8 (Change Management)
- **For metrics continuity**: See Reference Sheet 6 (Retrofitting Measurement)

---

