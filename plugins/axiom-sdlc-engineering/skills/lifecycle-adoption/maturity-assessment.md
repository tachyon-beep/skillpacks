---
parent-skill: lifecycle-adoption
reference-type: detailed-assessment-process
load-when: Starting adoption, gap analysis, level selection, audit preparation
---

# Maturity Assessment Reference

**Parent Skill:** lifecycle-adoption
**When to Use:** Starting CMMI adoption, determining maturity level fit, preparing for audit, measuring progress

This reference provides the complete maturity assessment process for evaluating your current CMMI maturity level and identifying gaps.

---

## Reference Sheet 1: Maturity Assessment

### Purpose & Context

**What this achieves**: Systematic gap analysis between current state and target CMMI level

**When to apply**:
- Before starting adoption (understand where you are)
- During pilot project (baseline current practices)
- After 6 months (measure progress)

**Prerequisites**:
- Target CMMI level identified (Level 2, 3, or 4)
- Willingness to honestly assess current state (no sandbagging)

### CMMI Maturity Scaling

#### Level 2: Managed (Assessment)

**Assessment Focus**: Do basic project management practices exist?

**Key Questions**:
- Are requirements documented before implementation?
- Is version control used consistently?
- Are work products reviewed before release?
- Are changes tracked and controlled?
- Is testing performed systematically?

**Assessment Method**: Simple checklist (yes/no)

**Time Required**: 2-4 hours interview + observation

#### Level 3: Defined (Assessment)

**Assessment Focus**: Are practices standardized across the organization?

**Key Questions** (in addition to Level 2):
- Do organizational process templates exist?
- Are processes tailored per project with justification?
- Are peer reviews formal with checklists?
- Are organizational baselines maintained?
- Is training provided for process execution?

**Assessment Method**: Process documentation review + practice observation + interview

**Time Required**: 1-2 days

#### Level 4: Quantitatively Managed (Assessment)

**Assessment Focus**: Is process performance measured statistically?

**Key Questions** (in addition to Level 3):
- Are quantitative process objectives set?
- Are statistical baselines established with control limits?
- Is process performance monitored with control charts?
- Are predictions made from quantitative models?
- Are out-of-control signals analyzed for root cause?

**Assessment Method**: Metrics analysis + statistical review + prediction model validation

**Time Required**: 3-5 days

### Implementation Guidance

#### Quick Start Checklist

**Step 1: Define Scope** (30 minutes)
- [ ] Identify target CMMI level (2, 3, or 4)
- [ ] Define project/organization boundary (single project vs. department vs. company)
- [ ] Select 1-2 projects as representative samples

**Step 2: Gather Evidence** (2-8 hours depending on level)
- [ ] Review existing documentation (requirements, design docs, test plans)
- [ ] Examine version control history (branching, commit messages, PR reviews)
- [ ] Check issue tracker (requirements tracking, traceability)
- [ ] Interview 3-5 team members (different roles: dev, QA, PM)
- [ ] Observe 1-2 ceremonies (sprint planning, design review, code review)

**Step 3: Score Current Practices** (1-2 hours)

Use this scoring rubric per process area:

| Score | Meaning | Evidence |
|-------|---------|----------|
| **0 - Not Performed** | Practice doesn't exist | No documentation, no mentions in interviews |
| **1 - Ad Hoc** | Practice performed inconsistently | Some evidence, but not systematic |
| **2 - Managed (L2)** | Practice performed per plan for projects | Documented plan, executed on projects |
| **3 - Defined (L3)** | Organizational standard exists, tailored per project | Template exists, tailoring documented |
| **4 - Quantitative (L4)** | Statistical process control applied | Metrics collected, control charts used |

**Example Scoring**:

| Process Area | Current Score | Evidence | Gap to Target (L3) |
|--------------|--------------|----------|-------------------|
| REQM | 1 (Ad Hoc) | Requirements in email threads, no tracking | 2 levels |
| CM | 2 (Managed) | Git used, no branching strategy | 1 level |
| VER | 1 (Ad Hoc) | Manual testing only, no reviews | 2 levels |
| DAR | 0 (Not Performed) | Decisions undocumented | 3 levels |

**Step 4: Prioritize Gaps** (1 hour)

Use this prioritization framework:

```
High Priority = High Risk × High Value × Low Effort
```

**Risk-Based Prioritization**:

| Gap | Risk if Not Addressed | Value if Addressed | Effort | Priority |
|-----|----------------------|-------------------|--------|----------|
| No CM workflow | High (lost work, conflicts) | High (prevents issues) | Low (1 day) | **CRITICAL** |
| No traceability | Medium (audit failure) | High (compliance) | Medium (2 weeks) | **HIGH** |
| No formal reviews | Medium (bugs escape) | Medium (quality) | Low (1 week) | **HIGH** |
| No metrics | Low (no improvement) | Medium (insight) | High (1 month) | **MEDIUM** |

**Step 5: Create Gap Closure Roadmap** (1-2 hours)

Map gaps to adoption phases:

- **Phase 1 (Quick Wins)**: Critical priority, low effort (CM workflow, PR templates)
- **Phase 2 (Foundations)**: High priority, medium effort (traceability, test coverage)
- **Phase 3 (Enhancements)**: Medium priority, high effort (metrics, baselines)
- **Phase 4 (Continuous Improvement)**: Long-term evolution

#### Templates & Examples

**Maturity Assessment Template**:

```markdown
# CMMI Maturity Assessment

**Project**: [Name]
**Date**: [YYYY-MM-DD]
**Assessor**: [Name]
**Target Level**: Level [2/3/4]

## Process Area Scores

| Process Area | Current | Target | Gap | Priority |
|--------------|---------|--------|-----|----------|
| REQM | [0-4] | [2/3/4] | [#] | [C/H/M/L] |
| RD | [0-4] | [2/3/4] | [#] | [C/H/M/L] |
| TS | [0-4] | [2/3/4] | [#] | [C/H/M/L] |
| CM | [0-4] | [2/3/4] | [#] | [C/H/M/L] |
| PI | [0-4] | [2/3/4] | [#] | [C/H/M/L] |
| VER | [0-4] | [2/3/4] | [#] | [C/H/M/L] |
| VAL | [0-4] | [2/3/4] | [#] | [C/H/M/L] |
| DAR | [0-4] | [2/3/4] | [#] | [C/H/M/L] |
| RSKM | [0-4] | [2/3/4] | [#] | [C/H/M/L] |
| MA | [0-4] | [2/3/4] | [#] | [C/H/M/L] |

## Top 5 Gaps (Priority Order)

1. **[Process Area]**: [Description] - [Effort estimate]
2. **[Process Area]**: [Description] - [Effort estimate]
3. **[Process Area]**: [Description] - [Effort estimate]
4. **[Process Area]**: [Description] - [Effort estimate]
5. **[Process Area]**: [Description] - [Effort estimate]

## Recommended Adoption Roadmap

- **Month 1**: [Quick wins]
- **Month 2**: [Foundations]
- **Month 3**: [Enhancements]
```

**Filled Example** (2-person startup, no process):

```markdown
# CMMI Maturity Assessment

**Project**: PaymentGateway API
**Date**: 2026-01-24
**Assessor**: John (Tech Lead)
**Target Level**: Level 2 (Managed)

## Process Area Scores

| Process Area | Current | Target | Gap | Priority |
|--------------|---------|--------|-----|----------|
| REQM | 0 | 2 | 2 | HIGH |
| CM | 1 (Git, no workflow) | 2 | 1 | CRITICAL |
| VER | 0 (manual only) | 2 | 2 | HIGH |
| DAR | 0 | 2 | 2 | MEDIUM |

## Top 5 Gaps

1. **CM Workflow**: No branching strategy, force pushes common - 1 day to implement
2. **REQM**: Requirements in Slack, not tracked - 2 days to set up GitHub Issues
3. **VER**: No code review, no automated tests - 1 week for basic CI + review policy
4. **DAR**: Decisions undocumented - 1 hour to create ADR template
5. **VAL**: No stakeholder acceptance process - 2 days to set up demo process

## Recommended Adoption Roadmap

- **Week 1**: CM workflow (branch protection, PR template), GitHub Issues for REQM
- **Week 2**: Basic CI (linting, unit tests), code review policy
- **Week 3**: ADR template, first 3 ADRs for existing decisions
- **Week 4**: Stakeholder demo process, acceptance criteria in issues
```

### Common Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| **Sandbagging Assessment** | Claiming worse state than reality to show "improvement" | Honest baseline enables realistic roadmap |
| **Optimism Bias** | "We're almost Level 3!" when barely Level 1 | Use evidence-based scoring, external validation |
| **Analysis Paralysis** | Spending 2 months on assessment | Time-box to 2-4 hours for Level 2, 1-2 days for Level 3 |
| **One-Size-Fits-All** | Using same assessment for 2-person and 50-person teams | Scale assessment depth to team size/risk |
| **Ignoring Assessment Results** | Conduct assessment but don't act on gaps | Turn gaps into actionable roadmap immediately |

### Tool Integration

**GitHub Approach**:
- **Evidence**: Review repository (branch protection rules, PR history, Issues, Actions)
- **Scoring**: Count PRs with reviews, calculate % issues with labels/milestones
- **Automation**: GitHub API to pull metrics (review rate, test coverage)

**Azure DevOps Approach**:
- **Evidence**: Review work item queries, branch policies, pipeline history
- **Scoring**: Analytics views for work item completion, pipeline success rate
- **Automation**: OData queries for process compliance metrics

**Tool-Agnostic Approach**:
- **Evidence**: Interview team, observe ceremonies, review any documentation
- **Scoring**: Manual checklist-based assessment
- **Automation**: N/A (use for very small teams or tool-less environments)

### Verification & Validation

**How to verify this assessment is accurate**:
- Cross-check with 2+ team members (do they agree on current state?)
- Compare with observable evidence (don't rely solely on claims)
- Pilot a practice (does "we do code review" match reality in 10 random PRs?)
- External validation (peer from another team reviews assessment)

**Common failure modes**:
- **Assessment shows gaps, team denies them** → Show evidence (check 10 PRs, find 8 with no review)
- **Assessment shows no gaps, audit finds violations** → Insufficient evidence gathering, biased sample
- **Roadmap created but not followed** → Lack of executive support, competing priorities (see Change Management sheet)

### Related Practices

- **Next step after assessment**: See Reference Sheet 2 (Incremental Adoption Roadmap)
- **If gaps include traceability**: See Reference Sheet 3 (Retrofitting Requirements)
- **If gaps include CM**: See Reference Sheet 4 (Retrofitting Configuration Management)
- **If team resists findings**: See Reference Sheet 8 (Change Management)

---

