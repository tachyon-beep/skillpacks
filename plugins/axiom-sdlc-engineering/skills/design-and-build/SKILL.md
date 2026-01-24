---
name: design-and-build
description: Use when making architecture decisions, setting up CI/CD, managing technical debt, or choosing branching strategies - enforces ADR requirements and prevents resume-driven design
---

# Design and Build

## Overview

This skill implements the **Technical Solution (TS)**, **Product Integration (PI)**, and **Configuration Management (CM)** process areas from the CMMI-based SDLC prescription.

**Core principle**: Architecture decisions require documentation (ADRs). Emergency shortcuts require retrospective documentation. "Best practice" is never justification - requirements are.

**Reference**: See `docs/sdlc-prescription-cmmi-levels-2-4.md` Section 3.2 for complete policy and practice definitions.

---

## When to Use

Use this skill when:
- Making architecture or design decisions (technology choice, patterns, structure)
- Setting up build/integration systems (CI/CD, deployment pipelines)
- Managing technical debt (tracking, prioritization, paydown)
- Establishing configuration management (branching strategy, release process)
- Facing "should we use X?" questions where X is trendy technology
- Team experiencing git chaos, integration hell, or debt spiral

**Do NOT use for**:
- Implementation details within existing architecture → Use domain-specific skills (python-engineering, web-backend)
- Testing strategy → Use quality-assurance skill
- Requirements or specification → Use requirements-lifecycle skill

---

## Quick Reference

| Situation | Primary Reference Sheet | Key Decision |
|-----------|------------------------|--------------|
| "Should we use microservices?" | Architecture & Design | Requires ADR. Use decision framework: team size, domain complexity, ops maturity. |
| "Git workflow is chaos" | Configuration Management | Diagnose root cause first. GitFlow (L3 releases) vs GitHub Flow (continuous) vs Trunk (high maturity). Requires ADR. |
| "70% of time on bugs" | Technical Debt Management | CODE RED. Feature freeze, architectural audit, classify debt (architectural/tactical/unpayable). |
| "Setting up CI/CD" | Build & Integration | Requirements gathering first. Platform choice requires ADR. Start simple, add stages incrementally. |
| "Quick fix vs proper solution" | Level 2→3→4 Scaling | Level 3 requires retrospective ADR within 48 hours. Document as HOTFIX with paydown commitment. |

---

## Level-Based Governance

**CRITICAL**: This skill enforces governance based on project maturity level.

### Level Detection

Check for project level in:
1. `CLAUDE.md`: `CMMI Target Level: 3`
2. User message: "This is a Level 3 project..."
3. **Default if unspecified**: Level 3

### ADR Requirements by Level

| Level | ADR Required For | Exception Protocol |
|-------|------------------|-------------------|
| **Level 2** | Major architecture decisions (platform choice, deployment strategy) | Informal discussion OK, document decision in wiki/README |
| **Level 3** | ALL architectural decisions (tech stack, branching strategy, design patterns, CI/CD platform) | Emergency HOTFIX: Retrospective ADR within 48 hours mandatory |
| **Level 4** | Everything in L3 + quantitative justification with metrics | No exceptions - statistical baselines required |

### Emergency Exception Protocol (HOTFIX Pattern)

**When**: Production emergency, immediate fix needed, no time for full ADR process

**Level 3 Requirements**:
1. Fix the emergency (restore service)
2. Document the fix in issue tracker with "HOTFIX" label
3. Create retrospective ADR within **48 hours**:
   - Incident timeline
   - Why proper fix wasn't feasible
   - Technical debt introduced
   - Paydown commitment with date (max 2 weeks)
4. Track paydown as high-priority ticket

**Violation**: Skipping retrospective ADR = governance failure. See Enforcement section below.

**HOTFIX Frequency Limit**: >5 HOTFIXes per month = systemic problem requiring architectural audit, not process exception.

### What Counts as "Architectural Decision"? (Level 3)

**Architectural decisions** (ADR required):
- Technology platform choice (language, framework, database, cache, message queue)
- Branching strategy (GitFlow, GitHub Flow, trunk-based)
- CI/CD platform (GitHub Actions, Azure Pipelines, Jenkins)
- Deployment strategy (blue/green, canary, rolling)
- Design patterns with broad impact (event-driven, microservices, monolith)
- Module boundaries and interfaces (how system decomposes)
- Authentication/authorization approach (OAuth, JWT, session-based)
- Data storage strategy (SQL vs NoSQL, caching strategy, data partitioning)

**Implementation details** (no ADR required, track in code/PR):
- Specific library choice within chosen framework (e.g., logging library in Python)
- Variable/function naming conventions
- Code organization within module (file structure)
- Test framework choice (if it doesn't affect architecture)

**Borderline decisions** (when in doubt, write ADR):
- If change affects >3 modules or files → ADR
- If reversal would take >1 day → ADR
- If future developers would ask "why did they choose this?" → ADR

---

## Enforcement and Escalation

### Level 3 Enforcement Mechanisms

**Platform Enforcement** (automated gates):
- [ ] Branch protection: Main/master requires 2+ approvals
- [ ] CI gates: Build + tests must pass before merge
- [ ] ADR linking: PRs for architectural changes must reference ADR number (format: "Implements ADR-YYYY-MM-DD-NNN")

**Process Enforcement** (review gates):
- [ ] ADR review required before implementation begins
- [ ] HOTFIX retrospective ADR tracked in ticket system
- [ ] Automated reminder 24h after HOTFIX label applied (e.g., GitHub Action)

**Metrics Tracking Compliance**:
- % architectural changes with ADRs (target: 100%)
- HOTFIX retrospective ADR compliance (target: 100% within 48h)
- Average ADR review time (target: <24 hours)

**Violation Escalation Path**:
1. **First violation**: Team lead notified, retrospective scheduled within 7 days
2. **Second violation (within 30 days)**: Engineering manager notified, process audit required
3. **Systemic violations (>3 in 90 days)**: Escalate to governance committee, audit non-conformance report

**Consequences for Non-Compliance** (Level 3):
- ADR violations visible in team metrics dashboard
- Repeated violations block promotion/performance reviews
- Audit findings can fail SOC 2, ISO compliance

---

## Anti-Patterns and Red Flags

### Resume-Driven Design

**Detection**: User says "I've heard X is best practice" or "Everyone uses Y"

**Red Flags**:
- Technology mentioned before requirements
- Appeal to popularity ("everyone uses microservices")
- Buzzword bingo (serverless, Kubernetes, blockchain)
- Can't articulate WHAT PROBLEM they're solving

**Counter**:
1. "What requirements drive this choice?"
2. "What alternatives exist?"
3. "Why is current approach inadequate?"
4. If they can't answer → "You're doing resume-driven design. Requirements first, technology second."

**Forcing function**: Require ADR with alternatives analysis. If they can't justify with measurable requirements, ADR review will reject it.

### Architecture Astronaut

**Detection**: Over-engineered solution for simple problem

**Red Flags**:
- Microservices for CRUD app
- Service mesh for 2 services
- Event sourcing for simple data model
- "Future-proof" without concrete future requirements

**Counter**: "What's the SIMPLEST solution that meets requirements? Start there. Add complexity when demonstrated need exists, not hypothetically."

### Cowboy Coding

**Detection**: No reviews, no standards, "works on my machine"

**Red Flags**:
- Skipping pull requests
- Force pushing to main
- No CI/CD
- "I'll add tests later"

**Counter**: Enforce basic CM practices. Level 2 minimum: branch protection, required PR reviews, CI runs on PRs.

### Debt Spiral

**Detection**: Increasing % of time on bugs, velocity declining

**Red Flags**:
- >50% time on bugs = WARNING
- >60% time on bugs = CODE RED
- Every feature breaks something else
- Team morale declining

**Counter**: See Technical Debt Management reference sheet. CODE RED triggers feature freeze.

---

## Reference Sheets

The following reference sheets provide detailed guidance for specific domains. Load them on-demand when needed.

### 1. Architecture & Design

**When to use**: Making technology choices, selecting patterns, designing system structure

→ See [architecture-and-design.md](./architecture-and-design.md)

**Covers**:
- Architecture Decision Records (ADR) template and process
- Decision frameworks for technology selection
- Design patterns by use case
- C4 model for architecture documentation
- Design review checklists
- Anti-patterns: Resume-driven design, Architecture Astronaut, Big Design Up Front

### 2. Implementation Standards

**When to use**: Establishing coding standards, code review process, documentation requirements

→ See [implementation-standards.md](./implementation-standards.md)

**Covers**:
- Coding standards enforcement
- Code review checklists and best practices
- Naming conventions
- Documentation standards (inline comments, module docs, API docs)
- Level 2/3/4 rigor scaling

### 3. Configuration Management

**When to use**: Git chaos, branching strategy decisions, release management

→ See [configuration-management.md](./configuration-management.md)

**Covers**:
- Branching strategy decision framework (GitFlow vs GitHub Flow vs Trunk-based)
- Branch protection policies
- Merge strategies (squash, merge commit, rebase)
- Release management (tagging, changelogs, versioning)
- Migration from chaos to structure
- Level 2/3/4 control scaling

### 4. Build & Integration

**When to use**: Setting up CI/CD, build optimization, deployment pipelines

→ See [build-and-integration.md](./build-and-integration.md)

**Covers**:
- Requirements gathering for CI/CD
- Platform selection (GitHub Actions, Azure Pipelines, GitLab CI)
- Pipeline stages (build → test → integration → deploy)
- Build optimization (caching, parallelization)
- Deployment strategies (blue/green, canary, rolling)
- Rollback procedures

### 5. Technical Debt Management

**When to use**: Team spending >40% time on bugs, debt accumulating, velocity declining

→ See [technical-debt-management.md](./technical-debt-management.md)

**Covers**:
- Debt classification (architectural, tactical, unpayable)
- Crisis detection thresholds (>60% bug time = CODE RED)
- Debt tracking systems
- Paydown strategies (20% rule, debt sprints, feature freeze)
- Debt metrics (ratio, trend, hotspots)
- Recovery roadmaps for crisis situations

### 6. Level 2→3→4 Scaling

**When to use**: Understanding what rigor is appropriate for your project tier

→ See [level-scaling.md](./level-scaling.md)

**Covers**:
- Level 2 baseline practices
- Level 3 organizational standards
- Level 4 statistical control
- Escalation criteria (when to move up levels)
- De-escalation criteria (when level is overkill)
- ADR requirements by level

---

## Common Mistakes

| Mistake | Why It Fails | Better Approach |
|---------|--------------|-----------------|
| "Emergency exempts process" | Creates pattern where "urgent" = skip governance, accumulating undocumented debt | Use HOTFIX pattern: retrospective ADR within 48 hours, mandatory |
| "I'll document it later" | Later never comes, loses audit trail | Document NOW (ADR takes 15 min) or schedule retrospective (48 hours max) |
| "This is too simple for ADR" | Simple decisions have big impact, lose rationale for future | If it's truly simple, ADR takes 10 min. If it takes longer, it wasn't simple. |
| "Everyone uses X, so we should" | Resume-driven design, not requirements-driven | Require measurable justification. "Everyone" is not a requirement. |
| "20% debt allocation" when 70% bugs | Treats crisis as normal problem, ensures slow death | >60% bugs = CODE RED. Feature freeze, not incremental paydown. |
| "Pick branching strategy" without diagnosis | Treats symptom (conflicts) not cause (architecture? communication?) | Root cause analysis FIRST. Git strategy is symptom, not disease. |
| Generic CI/CD template without context | Wastes time on wrong solution | Requirements gathering FIRST: build characteristics, deployment context, risk profile |

---

## Integration with Other Skills

| When You're Doing | Also Use | For |
|-------------------|----------|-----|
| Designing Python architecture | `axiom-python-engineering/modern-python-patterns` | Python-specific patterns and idioms |
| Designing web API | `axiom-web-backend/api-design` | REST/GraphQL best practices |
| Making architecture decision | `governance-and-risk/decision-analysis` | Formal DAR process for critical choices |
| Setting up testing | `quality-assurance/verification-planning` | Test strategy and coverage |
| Choosing platforms | `platform-integration` | GitHub vs Azure DevOps specifics |

---

## Real-World Impact

**Without this skill**: Teams experience:
- Undocumented architecture decisions that haunt future developers
- Resume-driven design leading to over-engineered solutions
- Git chaos with daily conflicts and lost work
- Debt spirals consuming 70%+ of time
- Emergency shortcuts becoming permanent anti-patterns

**With this skill**: Teams achieve:
- Defensible audit trail through ADRs
- Technology choices driven by requirements, not hype
- Structured configuration management reducing conflicts by 80%+
- Early crisis detection preventing debt spirals
- Emergency protocols that maintain governance without blocking urgency

---

## Next Steps

1. **Determine project level**: Check CLAUDE.md or ask user for CMMI target level (default: Level 3)
2. **Identify situation**: Use Quick Reference table to find relevant reference sheet
3. **Load reference sheet**: Read detailed guidance for specific domain
4. **Enforce ADR requirements**: Level 3 requires ADR for architectural decisions - no exceptions without HOTFIX protocol
5. **Apply decision frameworks**: Use systematic evaluation, not gut feelings or hype
6. **Counter anti-patterns**: Watch for resume-driven design, debt spirals, cowboy coding
7. **Measure success**: Establish baseline, set targets, schedule retrospectives

**Remember**: "Best practice" is never justification. Requirements are. If you can't articulate the requirement, you can't justify the architecture.
