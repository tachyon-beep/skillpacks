# Design-and-Build - Rationalization Patterns from RED Phase

**Date**: 2026-01-24
**Purpose**: Document patterns from baseline testing to inform skill content

---

## Systematic Rationalization Patterns

### Pattern 1: "Emergency Exempts Process"
**Manifests as**: "For production emergencies, you can defer the ADR..."

**Why it's harmful**: Creates pattern where "urgent" becomes excuse to skip governance, accumulating debt and losing audit trail.

**What skill must counter**: Emergency protocols that REQUIRE documentation (retrospective ADR within 48 hours), not optional deferrals.

---

### Pattern 2: "Generic Best Practice Substitutes for Context"
**Manifests as**: "I recommend GitHub Flow because..." (without understanding their specific situation)

**Why it's harmful**: One-size-fits-all recommendations ignore project tier, team maturity, and actual pain points.

**What skill must counter**: Diagnostic frameworks that investigate root causes before recommending solutions.

---

### Pattern 3: "Incremental Response to Crisis"
**Manifests as**: "20% debt allocation" when team is at 70% bug time

**Why it's harmful**: Treats existential crisis as normal problem, ensuring gradual death instead of recovery.

**What skill must counter**: Crisis detection thresholds with explicit escalation (feature freeze when >60% bug time).

---

### Pattern 4: "Cookbook Solutions Without Requirements"
**Manifests as**: Provides YAML template without asking about build characteristics, deployment context, or risk profile

**Why it's harmful**: Generic templates fail to address actual constraints and waste time on wrong solutions.

**What skill must counter**: Requirements gathering frameworks and ADR enforcement for architectural choices.

---

### Pattern 5: "Missing Anti-Pattern Detection"
**Manifests as**: Treating "I've heard it's best practice" as sincere question, not resume-driven design red flag

**Why it's harmful**: Enables cargo-cult thinking and hype-driven architecture instead of requirements-first design.

**What skill must counter**: Explicit anti-pattern recognition with education on WHY patterns fail.

---

### Pattern 6: "No Governance Enforcement"
**Manifests as**: Never mentioning Level 3 requires ADRs, treating architectural decisions as informal

**Why it's harmful**: Undermines entire governance system, loses audit trail, enables shortcuts.

**What skill must counter**: Level-based governance with explicit ADR requirements and forcing functions.

---

### Pattern 7: "Lists Options Without Decision Support"
**Manifests as**: "Here are 3 branching strategies" without criteria, thresholds, or forcing functions

**Why it's harmful**: User still doesn't know HOW to choose, paralyzed by options or picks randomly.

**What skill must counter**: Decision frameworks with concrete thresholds and systematic evaluation methods.

---

### Pattern 8: "No Measurement or Success Criteria"
**Manifests as**: Proposes solutions without baselines, metrics, or retrospective schedules

**Why it's harmful**: Can't tell if solution works, wastes resources on ineffective approaches.

**What skill must counter**: Success metrics framework with baseline, targets, and monitoring plans.

---

## Required Skill Components (Derived from Patterns)

### 1. Level-Based Governance Framework
- Clear Level 1/2/3 definitions and requirements
- When ADRs are mandatory vs optional
- Exception protocols (HOT FIX pattern with retrospective ADR)
- Enforcement without bureaucracy

### 2. Decision Frameworks and Forcing Functions
- ADR templates that require alternatives analysis
- Decision trees with concrete thresholds
- Forcing functions: "justified ONLY if you can point to measurable requirements"
- Systematic evaluation criteria

### 3. Diagnostic Methodologies
- Root cause analysis before solutions
- Questions to ask, data to gather
- Pattern recognition (daily conflicts → architectural coupling, not just branching)
- Cultural/process diagnosis

### 4. Crisis Detection and Escalation
- Explicit thresholds: >60% bug time = CODE RED
- Emergency protocols (feature freeze, architectural audit)
- Stakeholder communication templates
- Escalation criteria

### 5. Technical Debt Classification
- Architectural debt (requires ADR, design)
- Code quality debt (refactor)
- Unpayable debt (rewrite)
- Metrics: complexity, coupling, change amplification, bug clustering

### 6. Anti-Pattern Catalog
- Resume-driven design detection and counter
- Git chaos diagnosis (not just "pick a strategy")
- Debt spiral recognition (70% bugs = crisis, not normal)
- Cowboy coding, architecture astronaut, etc.

### 7. Implementation Roadmaps
- Phased plans with gates and rollback triggers
- Migration strategies from chaos to structure
- Parallel tracks (feature freeze + recovery)
- Risk management integrated

### 8. Success Metrics Framework
- What to measure for each decision type
- How to baseline current state
- Target setting with retrospective schedules
- Leading indicators (complexity trends, not just bug counts)

---

## Pressure Scenarios Validated

All 5 scenarios combined multiple pressures:

**Scenario 1** (ADR for hotfix):
- Time pressure (production bug, needed today)
- Sunk cost (already spent 2 hours on hack)
- Authority ("Level 3 requires ADR" - will they comply?)

**Scenario 2** (Branching strategy):
- Frustration (morale low from daily conflicts)
- Complexity aversion ("Git is too complicated")
- Social proof ("everyone recommends X")

**Scenario 3** (Debt spiral):
- Exhaustion (team burning out at 70% bug time)
- Stakeholder pressure ("can't afford to stop features")
- Overwhelm (2 years of accumulated debt)

**Scenario 4** (CI/CD setup):
- Urgency (manual builds taking hours)
- Inexperience (Level 2 project, likely new to CI/CD)
- Blank slate (where to even start?)

**Scenario 5** (Microservices):
- Authority pressure ("everyone says microservices are best practice")
- Resume motivation (developer wants trendy tech on resume)
- Cargo-cult thinking ("I've heard" instead of requirements)

---

## Key Insight for GREEN Phase

The skill must:
1. **Detect context**: Level, team size, crisis indicators
2. **Enforce governance**: ADRs for architectural decisions, no exceptions without protocol
3. **Provide frameworks**: Decision trees, classification systems, diagnostic checklists
4. **Counter rationalizations**: Explicit anti-patterns and red flags
5. **Enable measurement**: Metrics, baselines, success criteria
6. **Guide implementation**: Roadmaps with phases, gates, rollback triggers

**Minimal viable content**: Address the 8 patterns above. Everything else is bonus.
