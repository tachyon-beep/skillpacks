---
parent-skill: lifecycle-adoption
reference-type: phased-rollout-timeline
load-when: Planning incremental adoption, pilot projects, scaling CMMI practices
---

# Incremental Adoption Roadmap Reference

**Parent Skill:** lifecycle-adoption
**When to Use:** Planning phased CMMI rollout, pilot project strategy, scaling from one team to organization

This reference provides a week-by-week roadmap for incremental CMMI adoption.

---

## Reference Sheet 2: Incremental Adoption Roadmap

### Purpose & Context

**What this achieves**: Phased rollout plan for CMMI practices over 2-3 months

**When to apply**:
- After maturity assessment (you know the gaps)
- Before starting adoption (need structured plan)
- When pivoting from "big bang" to incremental approach

**Prerequisites**:
- Maturity assessment complete (know current and target state)
- Executive sponsorship secured (at least tentative approval)
- Team aware of adoption plan (no surprise mandates)

### CMMI Maturity Scaling

#### Level 2: Managed (Adoption Roadmap)

**Adoption Strategy**: Basic practices in 1 month

**Phases**:
1. **Week 1**: CM workflow (branching, PR reviews)
2. **Week 2**: REQM (issue tracking, basic traceability)
3. **Week 3**: VER (CI with basic tests, code review policy)
4. **Week 4**: Consolidation (team training, process refinement)

**Pilot Project**: Not required (can adopt organization-wide for Level 2)

**Rollout Approach**: All projects simultaneously (practices are lightweight)

#### Level 3: Defined (Adoption Roadmap)

**Adoption Strategy**: Pilot → Scale → Refine over 3 months

**Phases**:
1. **Month 1**: Pilot project with comprehensive practices
2. **Month 2**: Scale to 2-3 additional projects
3. **Month 3**: Organization-wide rollout

**Pilot Project**: Required (validate templates, identify issues)

**Rollout Approach**: Wave-based (pilot → early adopters → majority → laggards)

#### Level 4: Quantitatively Managed (Adoption Roadmap)

**Adoption Strategy**: Data infrastructure → Baselines → SPC over 6+ months

**Phases**:
1. **Month 1-2**: Establish metrics collection infrastructure
2. **Month 3-4**: Gather data, calculate initial baselines
3. **Month 5-6**: Implement control charts, train on SPC
4. **Month 7+**: Quantitative management, prediction models

**Pilot Project**: Required (prove feasibility of quantitative approach)

**Rollout Approach**: Sequential (must have data before SPC)

### Implementation Guidance

#### Quick Start Checklist

**Phase 0: Preparation** (1 week before adoption)
- [ ] Maturity assessment complete
- [ ] Target level selected and justified
- [ ] Executive sponsor identified
- [ ] Team informed (no surprise announcements)
- [ ] Tools selected (GitHub, Azure DevOps, etc.)

**Phase 1: Quick Wins** (Week 1-2)
- [ ] CM: Branch protection rules enabled
- [ ] CM: PR template created
- [ ] REQM: Issue templates set up
- [ ] VER: Basic CI pipeline (linting)
- [ ] Team Training: 1-hour workshop on new workflow

**Deliverables**: Branch protection active, first 5 PRs using template, 10 issues created with template

**Phase 2: Foundations** (Week 3-6)
- [ ] REQM: Traceability policy (PRs reference issues)
- [ ] CM: Branching workflow documented and enforced
- [ ] VER: Test coverage target set (e.g., 70%)
- [ ] TS: ADR template created, first 3 ADRs written
- [ ] Team Training: 2-hour workshop on traceability and ADRs

**Deliverables**: 90% of PRs reference issues, branching workflow adopted, 10 ADRs documenting major decisions

**Phase 3: Quality Practices** (Week 7-10)
- [ ] VER: Code review checklist created
- [ ] VER: Test coverage enforced in CI
- [ ] VAL: Stakeholder demo process established
- [ ] RSKM: Risk register template created
- [ ] Team Training: Peer review best practices

**Deliverables**: 100% of PRs reviewed against checklist, test coverage >70%, 3 stakeholder demos completed

**Phase 4: Measurement** (Week 11-12)
- [ ] MA: Metrics collection automated (velocity, defect rate, coverage)
- [ ] MA: Dashboard created (Grafana, GitHub Insights, Azure Analytics)
- [ ] Retrospective: Lessons learned from adoption
- [ ] Process refinement based on feedback

**Deliverables**: Metrics dashboard live, baselines calculated, process tailoring documented

#### Pilot Project Selection

**Criteria for pilot project**:

| Criterion | Why Important | Example |
|-----------|---------------|---------|
| **Medium complexity** | Not trivial (won't test practices), not critical (can tolerate issues) | 5-10K LOC, 2-4 developers, 3-6 month timeline |
| **Representative** | Practices must work for typical projects | Uses common tech stack, similar team structure |
| **Supportive team** | Early adopters willing to give feedback | Developers open to process experimentation |
| **Visible success** | Demonstrates value to rest of organization | Launches publicly, has metrics to show improvement |
| **Not time-critical** | Can tolerate learning curve slowdown | Not emergency project or tight external deadline |

**Bad pilot choices**:
- ❌ Mission-critical production system (too risky)
- ❌ Prototype with 1 developer (not representative)
- ❌ 2-year legacy monolith (too complex for first adoption)
- ❌ Project with skeptical team (will sabotage)

#### Wave-Based Rollout (Level 3)

**Wave 1: Pilot** (Month 1)
- 1 project, 3-5 developers
- Comprehensive practices implementation
- Daily feedback collection
- Process refinement in real-time

**Wave 2: Early Adopters** (Month 2)
- 2-3 projects, 10-15 developers
- Validated practices from pilot
- Weekly feedback sessions
- Minor process adjustments

**Wave 3: Majority** (Month 3)
- Remaining projects
- Standard templates and checklists
- Self-service onboarding docs
- Monthly process review

**Wave 4: Laggards** (Month 4+)
- Holdout projects (if any)
- Mandate enforcement (if necessary)
- Individual coaching
- Process exceptions granted case-by-case

#### Codebase Size Scaling

**Timeline adjustments based on existing codebase size**:

| Codebase Size | Level 2 Adoption | Level 3 Adoption | Key Challenges |
|---------------|------------------|------------------|----------------|
| **Small (1-10K LOC)** | 2-3 weeks | 1-2 months | Minimal retrofitting, quick wins dominate |
| **Medium (10-50K LOC)** | 1-2 months | 2-4 months | Moderate retrofitting, selective coverage acceptable |
| **Large (50-200K LOC)** | 2-4 months | 4-6 months | Extensive retrofitting, requires prioritization |
| **Very Large (200K+ LOC)** | 4-6 months | 6-12 months | Massive retrofitting effort, multi-team coordination |

**Effort scaling factors**:

**Retrofitting Requirements** (Reference Sheet 3):
- 1-10K LOC: 20-40 hours (1 week)
- 10-50K LOC: 80-160 hours (3-4 weeks)
- 50-200K LOC: 240-480 hours (6-12 weeks)
- 200K+ LOC: 600-1200 hours (15-30 weeks)

**Retrofitting CM** (Reference Sheet 4):
- 1-10K LOC: 1-2 days (cleanup minimal branches)
- 10-50K LOC: 1-2 weeks (moderate branch cleanup, migration)
- 50-200K LOC: 3-4 weeks (extensive history, many active branches)
- 200K+ LOC: 6-8 weeks (multi-repo coordination, complex history)

**Retrofitting Quality Practices** (Reference Sheet 5):
- 1-10K LOC: 1-2 weeks (write initial tests)
- 10-50K LOC: 4-6 weeks (selective test coverage)
- 50-200K LOC: 12-16 weeks (extensive test gap analysis)
- 200K+ LOC: 24-32 weeks (multi-team test strategy)

**Team size interaction**:

| Team Size | Small Codebase | Large Codebase | Coordination Overhead |
|-----------|----------------|----------------|----------------------|
| **2-4 developers** | 2-3 months | 6-9 months | Low (direct communication) |
| **5-10 developers** | 2-4 months | 4-8 months | Medium (need coordination meetings) |
| **10-20 developers** | 3-5 months | 6-12 months | High (formal coordination process) |
| **20+ developers** | 4-6 months | 9-18 months | Very high (multi-team orchestration) |

**Red flags**:
- Promising <1 month adoption for 50K+ LOC codebase → unrealistic
- Not adjusting timeline for codebase size → guaranteed failure
- Treating 10K LOC and 200K LOC identically → shows lack of understanding

**Calibration guideline**: Use 200 LOC per developer-day as baseline for analysis effort (understanding existing code, documenting requirements, writing tests). Adjust for complexity and quality of existing code.

#### Templates & Examples

**3-Month Incremental Adoption Roadmap (Level 3)**:

```markdown
# CMMI Level 3 Adoption Roadmap

**Organization**: Acme Software Inc.
**Target Level**: Level 3 (Defined)
**Timeline**: 3 months (Jan-Mar 2026)
**Pilot Project**: Mobile App Redesign

## Month 1: Pilot (Mobile App Team)

### Week 1-2: CM & REQM
- ✅ GitHub repository with branch protection
- ✅ GitFlow workflow documented
- ✅ Issue templates for features/bugs
- ✅ PR template with checklist
- **Training**: 2-hour Git workflow workshop
- **Success Metric**: 100% PRs follow template

### Week 3-4: TS & VER
- ✅ ADR template + first 5 ADRs
- ✅ Code review checklist
- ✅ CI pipeline with tests + coverage
- ✅ Test coverage target: 80%
- **Training**: 2-hour code review workshop
- **Success Metric**: All PRs reviewed, coverage >80%

## Month 2: Scale to Early Adopters

### Wave 2 Projects:
- Backend API Refactor (5 devs)
- Data Pipeline Modernization (4 devs)

### Rollout:
- Week 5: Onboarding sessions (1 hour per team)
- Week 6-8: Adoption with coach support
- **Deliverables**: Same as pilot (branch protection, templates, ADRs, CI)
- **Success Metric**: 90% compliance with standards

## Month 3: Organization-Wide Rollout

### Remaining Projects: All (20 developers across 5 projects)

### Rollout:
- Week 9: Self-service onboarding docs published
- Week 10: Team-by-team adoption
- Week 11: Compliance review
- Week 12: Retrospective and refinement

### Deliverables:
- All projects using standard templates
- 50+ ADRs documenting decisions
- Organizational metrics dashboard live
- Process tailoring guide published

## Success Criteria

- [ ] 100% of projects using CM workflow
- [ ] 95% of PRs reference issues (traceability)
- [ ] 90% test coverage across all projects
- [ ] 50+ ADRs documenting key decisions
- [ ] Metrics dashboard showing baselines
- [ ] <5 process exceptions granted
```

### Common Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| **Big Bang Adoption** | Overwhelming, no learning, high failure rate | Phased rollout over 3 months |
| **No Pilot** | Untested templates, organization-wide pain | Pilot on 1 project, refine, then scale |
| **Too Many Pilots** | Fragmented, no clear winner | 1-2 pilots max, standardize quickly |
| **Pilot Forever** | Never scale beyond initial project | Hard deadline for scaling (Month 2) |
| **Forced Rollout** | Team revolt, cargo cult compliance | Involve team in tailoring, demonstrate value |

### Tool Integration

**GitHub**:
- **Phase 1**: Enable branch protection, create templates (Settings → Branches, .github/)
- **Phase 2**: Enforce issue references in PRs (GitHub Actions check)
- **Phase 3**: Required status checks (CI must pass), required reviewers
- **Phase 4**: GitHub Insights for metrics

**Azure DevOps**:
- **Phase 1**: Branch policies, work item templates
- **Phase 2**: Linked work items required, PR templates
- **Phase 3**: Build validation policies, test coverage gates
- **Phase 4**: Analytics views, dashboards

### Verification & Validation

**How to verify adoption is on track**:
- Weekly spot checks (review 5 random PRs, do they follow template?)
- Metrics tracking (% PRs with issue refs, test coverage trend)
- Team feedback sessions (is process helping or hindering?)
- External audit (peer review from another team/organization)

**Common failure modes**:
- **Pilot succeeds, scaling fails** → Insufficient training materials, lack of coach support
- **Compliance drops after Month 1** → No enforcement, competing priorities, no demonstrated value
- **Teams create workarounds** → Process is too bureaucratic, needs tailoring
- **Executives pull support** → No visible ROI, need quick wins and metrics

### Related Practices

- **Before creating roadmap**: See Reference Sheet 1 (Maturity Assessment)
- **During rollout if team resists**: See Reference Sheet 8 (Change Management)
- **If traceability is a gap**: See Reference Sheet 3 (Retrofitting Requirements)
- **If CM is a gap**: See Reference Sheet 4 (Retrofitting Configuration Management)

---

