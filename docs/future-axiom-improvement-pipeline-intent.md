# Axiom/Improvement-Pipeline - Intent & High-Level Design

**Created**: 2025-11-13
**Status**: Intent Document (Not Yet Implemented)
**Faction**: Axiom (Creators of Marvels - Analysis, Strategy, Execution)
**Priority**: High Value - Creates Ecosystem Integration Story

---

## Core Insight

**"Legacy codebases need a pipeline: Document the mess → Assess critically → Manage improvements."**

The system-archaeologist plugin documents architecture but stops short of critical assessment and execution management. Users with "dogs breakfast" codebases need the complete pipeline:

1. **Archaeologist** - "Here's what exists" (neutral documentation)
2. **Architect** - "Here's what's wrong and how to fix it" (critical assessment)
3. **Project Manager** - "Here's the execution plan and progress tracking" (managed improvement)

---

## Why This Matters

### The "Legacy Codebase Problem"

Most real-world codebases are:
- ❌ Poorly architected (accumulated technical debt)
- ❌ Inconsistent patterns (3 ways to do authentication)
- ❌ Tightly coupled (can't change one thing without breaking 10 others)
- ❌ Undocumented architectural decisions (no one knows why it's this way)

**Current state**: System-archaeologist documents the mess but doesn't:
- Critique what's wrong
- Recommend improvements
- Track remediation efforts

**This creates the gap**: "I know what I have (archaeology), but what do I DO about it?"

### The Ecosystem Integration Value Proposition

**These skillpacks become the "glue" that makes the entire marketplace valuable together:**

| Existing Pack | Integration Point | Value Multiplier |
|---------------|-------------------|------------------|
| **ordis-security-architect** | Architect identifies security anti-patterns → Ordis provides threat modeling | Security issues become actionable work |
| **muna-technical-writer** | Architect identifies documentation debt → Muna structures docs | Documentation becomes architectural artifact |
| **axiom-python-engineering** | Architect detects Python-specific issues → Python pack provides solutions | Language-specific refactoring guided |
| **yzmir-ml-production** | Architect finds model deployment issues → Yzmir provides production patterns | ML architecture becomes production-ready |
| **bravos-simulation-tactics** | Architect sees game simulation issues → Bravos provides systemic solutions | Game architecture gets emergent design |
| **lyra-ux-designer** | Architect identifies UX architectural problems → Lyra provides user-centered fixes | UX becomes architectural concern |

**This is the killer feature**: The improvement pipeline creates demand for the entire marketplace by showing how packs work together on real problems.

---

## Faction Assignment: Axiom

**Why Axiom (Creators of Marvels)?**
- ✅ Systematic analysis and evidence-based assessment
- ✅ Tooling and infrastructure for managing work
- ✅ Builds on existing axiom-system-archaeologist
- ✅ Creates "marvel" of turning messes into maintainable systems
- ✅ Not security-specific (Ordis), not docs-focused (Muna), not algorithm-focused (Yzmir)

**Axiom's growing mission**:
- **axiom-system-archaeologist** - Document what exists (neutral archaeology)
- **axiom-system-architect** - Assess and strategize (critical analysis)
- **axiom-project-manager** - Execute and track (managed improvement)
- **axiom-python-engineering** - Language-specific engineering excellence

Axiom is becoming the "systems improvement" faction.

---

## Proposed Pack 1: axiom-system-architect (8 Skills)

### Purpose
Critical architectural assessment with actionable recommendations. Takes archaeologist's neutral documentation and provides strategic guidance.

### Meta-skill
**`axiom/system-architect/using-system-architect`** (Router)
- Reads archaeologist outputs
- Routes to appropriate assessment skill based on need
- Coordinates multi-skill analysis workflows

### Core Skills

#### 1. **assessing-architecture-quality**
**Purpose:** "I see what you're going for, but this whole thing sucks - here's why"

**Analyzes:**
- Architectural pattern adherence (MVC, microservices, event-driven)
- SOLID principle violations at system level
- Coupling/cohesion metrics from dependency analysis
- Layering violations (presentation → data without business logic)
- Missing abstractions (duplicate patterns not unified)
- Technology mismatches (wrong tool for the problem)

**Inputs:**
- `02-subsystem-catalog.md` (from archaeologist)
- `03-diagrams.md` (from archaeologist)
- Codebase structure

**Outputs:**
- `05-architecture-assessment.md` with severity ratings (Critical/High/Medium/Low)
- Evidence-based critique with line numbers and examples
- Pattern violations with architectural principles cited

**Integration Points:**
- **ordis-security-architect** - Security anti-patterns flagged for threat modeling
- **axiom-python-engineering** - Python-specific architectural issues flagged
- **yzmir-ml-production** - ML system architecture assessed against production patterns

---

#### 2. **identifying-technical-debt**
**Purpose:** Catalog and quantify the architectural mess

**Identifies:**
- **Structural debt** - God classes, tight coupling, circular dependencies
- **Duplication debt** - Same problem solved 3 different ways
- **Testing debt** - Untestable code, missing architectural boundaries
- **Configuration debt** - Hard-coded values, environment-specific hacks
- **Dead code** - Unused modules, zombie features, deprecated APIs
- **Migration debt** - Stuck on old versions, security vulnerabilities

**Inputs:**
- `02-subsystem-catalog.md` (dependencies, patterns observed)
- Codebase analysis (duplicates, coverage metrics)

**Outputs:**
- `06-technical-debt-catalog.md` with effort estimates
- Prioritized by impact (velocity drag, bug risk, security exposure)
- Organized by category (structural, duplication, testing, etc.)

**Integration Points:**
- **muna-technical-writer** - Documentation debt identified
- **ordis-security-architect** - Security debt flagged for remediation planning
- **axiom-python-engineering** - Python-specific debt (type annotations, async patterns)

---

#### 3. **analyzing-architectural-patterns**
**Purpose:** What patterns exist (intentional or accidental)?

**Detects:**
- **Actual patterns in use** - MVC, microservices, event-driven, layered
- **Pattern misapplication** - "They read about DDD once and got it wrong"
- **Inconsistent patterns** - 3 different auth implementations
- **Anti-patterns** - Big Ball of Mud, Distributed Monolith, God Object
- **Implicit architecture** - Accidental patterns that emerged
- **Pattern evolution** - How architecture changed over time (from catalog history)

**Inputs:**
- `02-subsystem-catalog.md` (patterns observed per subsystem)
- `03-diagrams.md` (structural patterns visible)

**Outputs:**
- `07-pattern-analysis.md` with pattern quality assessment
- Intentional vs accidental patterns
- Pattern consistency matrix (which subsystems use which patterns)
- Recommended patterns for unification

**Integration Points:**
- **bravos-systems-as-experience** - Game system patterns evaluated
- **yzmir-neural-architectures** - ML architecture patterns assessed
- **All packs** - Domain-specific patterns identified and evaluated

---

#### 4. **recommending-refactoring-strategies**
**Purpose:** Here's how to fix this mess (with practical approaches)

**Provides:**
- **Strangler Fig** - Gradual replacement without big-bang rewrite
- **Extract Service** - Subsystem boundaries with clear interfaces
- **Consolidate Duplicates** - Unify the 3 auth implementations
- **Introduce Abstraction** - Add layers where missing
- **Dependency Inversion** - Break tight coupling with interfaces
- **Data Migration Paths** - Untangle data models safely
- **Technology Updates** - Framework/library upgrade strategies

**Inputs:**
- `05-architecture-assessment.md` (what's wrong)
- `06-technical-debt-catalog.md` (what to fix)
- `07-pattern-analysis.md` (patterns to adopt)

**Outputs:**
- `08-refactoring-recommendations.md` with prioritized strategies
- Each strategy includes: goal, approach, risks, effort estimate
- Dependencies between strategies (must do X before Y)

**Integration Points:**
- **axiom-python-engineering** - Python-specific refactoring patterns
- **yzmir-ml-production** - ML system refactoring (model serving, pipelines)
- **ordis-security-architect** - Security-driven refactoring (auth, authorization)

---

#### 5. **prioritizing-improvements**
**Purpose:** What to fix first (triage the disaster)

**Ranks By:**
- **Business impact** - Which mess hurts users/revenue most?
- **Risk reduction** - Security, reliability, compliance exposure
- **Enablement value** - What unlocks future work (remove blockers)?
- **Effort vs benefit** - Quick wins vs long slogs
- **Dependency ordering** - Must fix X before Y is possible
- **Team capacity** - Realistic given current resources

**Inputs:**
- `05-architecture-assessment.md` (severity ratings)
- `06-technical-debt-catalog.md` (effort estimates)
- `08-refactoring-recommendations.md` (strategies)
- User input (business priorities, constraints)

**Outputs:**
- `09-improvement-roadmap.md` with phased approach
- Phase 1 (Quick wins), Phase 2 (Foundation), Phase 3 (Strategic)
- Each phase: goals, deliverables, success criteria, timeline
- Explicitly documents what's NOT being fixed (conscious tech debt)

**Integration Points:**
- **axiom-project-manager** - Roadmap becomes project input
- **All specialist packs** - Each phase references appropriate domain skills

---

#### 6. **documenting-architecture-decisions**
**Purpose:** Explain why current state exists and why changes matter

**Produces ADRs:**
- **Historical analysis** - "Why did they build it this way?" (context reconstruction)
- **Proposed decisions** - "Here's what we should do instead"
- **Trade-off analysis** - "This fix costs X, buys Y, risks Z"
- **Status tracking** - Proposed → Accepted → Implemented → Superseded
- **Alternatives considered** - What we didn't choose and why

**Inputs:**
- Codebase history (git log, commit messages)
- `02-subsystem-catalog.md` (concerns documented)
- `08-refactoring-recommendations.md` (proposed changes)

**Outputs:**
- `10-architecture-decisions/` directory
- ADR-0001.md, ADR-0002.md, etc. (numbered, dated)
- ADR index with status and supersession tracking

**Integration Points:**
- **muna-technical-writer** - ADRs become architecture documentation
- **ordis-security-architect** - Security ADRs reference compliance requirements
- **axiom-project-manager** - ADRs tracked as project decisions

---

#### 7. **estimating-refactoring-effort**
**Purpose:** Realistic cost/benefit for stakeholders

**Estimates:**
- **T-shirt sizing** - S/M/L/XL for each recommendation
- **Calendar time** - Realistic timelines given team capacity (not ideal person-weeks)
- **Resource requirements** - Skills needed, team size, external dependencies
- **Risk assessment** - What could go wrong during refactor? (regression, performance)
- **ROI calculation** - Time saved, bugs prevented, velocity gained
- **Incremental milestones** - Working software at each checkpoint

**Inputs:**
- `08-refactoring-recommendations.md` (strategies)
- `09-improvement-roadmap.md` (phases)
- Team capacity and skill assessment

**Outputs:**
- `11-effort-estimates.md` with stakeholder-ready breakdown
- Confidence levels (High/Medium/Low) per estimate
- Assumptions documented (team size, skill level, uninterrupted focus)

**Integration Points:**
- **axiom-project-manager** - Estimates feed into project planning
- **All specialist packs** - Domain expert time factored into estimates

---

#### 8. **generating-improvement-metrics**
**Purpose:** How to measure if improvements are working

**Defines:**
- **Baseline metrics** - Current state (build time, test coverage, bug rate, coupling metrics)
- **Target metrics** - Desired end state per improvement phase
- **Leading indicators** - Early signals of improvement (test coverage, code review time)
- **Lagging indicators** - Outcome measures (bug rate, deployment frequency, velocity)
- **Monitoring strategy** - How to track progress (automated vs manual, frequency)
- **Dashboard design** - Stakeholder-appropriate visualizations

**Inputs:**
- `05-architecture-assessment.md` (problems to measure)
- `09-improvement-roadmap.md` (phases with goals)
- Current metrics (if available)

**Outputs:**
- `12-improvement-metrics.md` with baseline and targets
- Metric collection strategy (tools, frequency, responsibility)
- Dashboard mockups for stakeholder reporting

**Integration Points:**
- **axiom-project-manager** - Metrics tracked as project KPIs
- **yzmir-ml-production** - ML-specific metrics (model performance, latency)
- **ordis-security-architect** - Security metrics (vulnerability count, audit findings)

---

### Workflow: System Architect Skills Together

```
archaeologist outputs → assessing-architecture-quality → 05-architecture-assessment.md
                     ↓
                     → identifying-technical-debt → 06-technical-debt-catalog.md
                     ↓
                     → analyzing-architectural-patterns → 07-pattern-analysis.md
                     ↓
       [05, 06, 07] → recommending-refactoring-strategies → 08-refactoring-recommendations.md
                     ↓
   [05, 06, 08, user input] → prioritizing-improvements → 09-improvement-roadmap.md
                     ↓
          [02, 08] → documenting-architecture-decisions → 10-architecture-decisions/
                     ↓
       [08, 09, team data] → estimating-refactoring-effort → 11-effort-estimates.md
                     ↓
          [05, 09, current metrics] → generating-improvement-metrics → 12-improvement-metrics.md

All outputs → axiom-project-manager for execution
```

---

## Proposed Pack 2: axiom-project-manager (8 Skills)

### Purpose
Manage any implementation effort - refactoring, features, migrations, etc. Architect's recommendations are just one project type it manages.

### Meta-skill
**`axiom/project-manager/using-project-manager`** (Router)
- Determines PM skill needed based on task
- Reads project state to route appropriately
- Coordinates multi-skill PM workflows

### Core Skills

#### 1. **initiating-projects**
**Purpose:** Convert any plan into trackable project structure

**Takes Inputs Like:**
- Architecture improvement roadmap (`09-improvement-roadmap.md`)
- Feature requirements document
- Security remediation list (from Ordis)
- Migration plan
- Infrastructure upgrade plan

**Creates:**
```
projects/[project-name]-YYYY-MM-DD/
├── 00-project-charter.md          # Scope, goals, constraints, success criteria
├── 01-work-breakdown.md           # Phases → Epics → Stories → Tasks
├── 02-resource-plan.md            # Team, budget, timeline, dependencies
├── 03-risk-register.md            # Risks, likelihood, impact, mitigation
└── tracking/
    └── (status updates added here)
```

**Outputs:**
- Structured project workspace with all planning artifacts
- Work broken down to sprint-sized tasks
- Dependencies identified and documented
- Risk register initialized

**Integration Points:**
- **axiom-system-architect** - Roadmap becomes project input
- **ordis-security-architect** - Security work becomes tracked project
- **muna-technical-writer** - Documentation initiatives become projects
- **Any pack** - Any systematic work becomes managed project

---

#### 2. **tracking-progress**
**Purpose:** Monitor execution against plan (what's done, what's blocked, what's at risk)

**Tracks:**
- **Task completion** - Planned vs actual (burndown)
- **Blockers** - What's stuck and why (technical, resource, external)
- **Velocity** - Story points or tasks per sprint
- **Quality metrics** - Test coverage, code review findings, bug counts
- **Timeline health** - Green/Yellow/Red status
- **Scope changes** - Additions/removals with rationale

**Produces:**
- `tracking/status-YYYY-MM-DD.md` with weekly updates
- Red/Yellow/Green indicators for each phase/epic
- Blocker list with owner and ETA
- Burndown charts (if applicable)
- Recommendations (stay course, adjust scope, add resources)

**Integration Points:**
- **axiom-system-architect** - Improvement metrics tracked here
- **All packs** - Any work tracked can use progress monitoring
- **Version control** - Reads commit history, PR status, etc.

---

#### 3. **managing-risks**
**Purpose:** Track and mitigate project risks (technical, resource, dependency)

**Maintains Risk Register:**
- **Technical risks** - "Refactor might break authentication"
- **Resource risks** - "Key developer on vacation 2 weeks"
- **Dependency risks** - "Blocked on vendor API update"
- **Scope risks** - "Client keeps adding requirements"
- **Quality risks** - "No tests, high regression risk"
- **Timeline risks** - "Slipping 3 weeks behind"

**For Each Risk:**
- Likelihood (Low/Medium/High)
- Impact (Low/Medium/High)
- Mitigation strategy
- Owner
- Status (Open/Mitigated/Realized/Closed)
- Trigger conditions (when does this become critical?)

**Updates:**
- `03-risk-register.md` with status changes
- Risk trending (are we reducing risk over time?)
- Realized risks become issues (tracked separately)

**Integration Points:**
- **ordis-security-architect** - Security risks identified and tracked
- **axiom-system-architect** - Refactoring risks from effort estimates
- **All packs** - Domain-specific risks (ML model performance, game balance, etc.)

---

#### 4. **planning-iterations**
**Purpose:** Sprint/iteration planning from backlog (what goes in next sprint?)

**Helps With:**
- **Capacity planning** - Team availability, vacation, realistic commitments
- **Prioritization** - Value vs effort, dependencies, critical path
- **Story sizing** - Breaking large tasks into sprint-sized chunks
- **Sprint goals** - Coherent theme, not random task grab-bag
- **Definition of done** - What "complete" means (coded, tested, reviewed, deployed?)
- **Risk mitigation** - Tackle risky items early

**Produces:**
- `tracking/sprint-NN-plan.md` with committed work
- Sprint goal statement
- Story list with points/estimates
- Capacity vs commitment comparison
- Known risks for sprint

**Integration Points:**
- **axiom-system-architect** - Roadmap phases feed into sprint planning
- **All packs** - Work from any domain gets sprint-planned

---

#### 5. **facilitating-retrospectives**
**Purpose:** Learn from what's happening (continuous improvement)

**Structured Retrospectives:**
- **What went well** - Double down on this
- **What went poorly** - Fix or mitigate this
- **Action items** - Specific, assigned, trackable (not vague "communicate better")
- **Metrics trends** - Velocity, quality, morale over time
- **Process improvements** - Update how we work
- **Blockers resolved** - What unblocked us?
- **Surprises** - What did we not anticipate?

**Produces:**
- `tracking/retro-YYYY-MM-DD.md` with findings
- Action items tracked to completion
- Process changes documented
- Velocity/quality trends visualized

**Integration Points:**
- **All packs** - Any work benefits from retrospectives
- **axiom-system-architect** - Architecture improvement effectiveness assessed

---

#### 6. **communicating-status**
**Purpose:** Generate stakeholder-appropriate reports (right detail for audience)

**Creates Different Views:**
- **Executive summary** - High-level, risks, decisions needed (1 page max)
- **Team status** - Detailed progress, blockers, next steps (working-level detail)
- **Client updates** - Feature progress, timeline, costs (business language)
- **Technical deep-dive** - For architects, leads (technical detail, metrics)
- **Audit trail** - For compliance, retrospectives (full history)

**Adapts To:**
- Audience technical level
- Update frequency (daily, weekly, monthly)
- Stakeholder concerns (cost, timeline, quality, scope)

**Produces:**
- `reports/stakeholder-update-YYYY-MM-DD.md`
- Appropriate visualizations (burndown, risk heat map, velocity)
- Recommendations and decision requests clearly stated

**Integration Points:**
- **muna-technical-writer** - Report formatting and clarity
- **ordis-security-architect** - Security status for compliance reporting
- **axiom-system-architect** - Architecture improvement progress

---

#### 7. **managing-dependencies**
**Purpose:** Track and coordinate cross-team/external dependencies (what's blocking what?)

**Manages:**
- **Internal dependencies** - "Backend API must complete before frontend"
- **External dependencies** - "Vendor contract renewal blocks this"
- **Technical dependencies** - "DB migration before schema changes"
- **Resource dependencies** - "Need designer available for this"
- **Critical path** - What's blocking everything else?
- **Dependency health** - Are dependencies on track? At risk?

**Tracks:**
- Dependency owner
- Status (Not Started/In Progress/Complete/Blocked)
- Impact if delayed (blocks what?)
- Alternative paths (can we work around this?)

**Updates:**
- `01-work-breakdown.md` with dependency status
- Dependency graph visualization
- Critical path highlighting
- Escalation for at-risk dependencies

**Integration Points:**
- **All packs** - Cross-domain dependencies (architecture + security + docs)
- **axiom-system-architect** - Refactoring dependency ordering

---

#### 8. **closing-projects**
**Purpose:** Proper project closure with lessons learned

**Performs:**
- **Deliverable verification** - All goals met? Acceptance criteria satisfied?
- **Documentation handoff** - Architecture, operations, maintenance docs complete
- **Lessons learned** - What worked, what didn't, what to repeat
- **Metrics analysis** - Did we hit targets? Velocity, quality, cost
- **Cleanup** - Archive workspace, close tracking, update organizational knowledge
- **Celebration** - Acknowledge team effort and success

**Produces:**
- `99-project-closure.md` with summary
- Lessons learned document
- Final metrics report vs targets
- Handoff documentation complete
- Organizational knowledge updated (wikis, runbooks, etc.)

**Integration Points:**
- **muna-technical-writer** - Closure docs formatted professionally
- **axiom-system-architect** - Architecture improvements verified
- **All packs** - Domain-specific deliverables verified

---

### Workflow: Project Manager Skills Together

```
Plan (from any source) → initiating-projects → Project workspace created
                                             ↓
                            planning-iterations → Sprint plans
                                             ↓
                               tracking-progress → Status updates
                                             ↓
                                 managing-risks → Risk register updated
                                             ↓
                            managing-dependencies → Dependency tracking
                                             ↓
                           communicating-status → Stakeholder reports
                                             ↓
                       facilitating-retrospectives → Lessons learned
                                             ↓
                                closing-projects → Project closure
```

---

## The Complete Improvement Pipeline

### End-to-End Workflow

```
1. USER: "Analyze my codebase and fix it"

2. ARCHAEOLOGIST:
   /system-archaeologist
   → Produces: 01-discovery-findings.md
              02-subsystem-catalog.md
              03-diagrams.md
              04-final-report.md

3. ARCHITECT:
   /system-architect
   → Reads archaeologist outputs
   → Produces: 05-architecture-assessment.md ("This is a mess")
              06-technical-debt-catalog.md ("14 critical issues")
              07-pattern-analysis.md ("Inconsistent patterns")
              08-refactoring-recommendations.md ("Here's how to fix")
              09-improvement-roadmap.md ("3 phases, 6 months")
              10-architecture-decisions/ (ADRs)
              11-effort-estimates.md ("Phase 1: 3 weeks, Phase 2: 8 weeks")
              12-improvement-metrics.md ("Track these KPIs")

4. PROJECT MANAGER:
   /project-manager
   → Reads 09-improvement-roadmap.md
   → Produces: projects/auth-refactor-2025-11/
              ├── 00-project-charter.md
              ├── 01-work-breakdown.md
              ├── 02-resource-plan.md
              ├── 03-risk-register.md
              └── tracking/
                  ├── sprint-01-plan.md
                  ├── status-2025-11-18.md
                  └── retro-2025-11-22.md

5. EXECUTION (with specialist packs):
   → Security issues → ordis-security-architect
   → Documentation → muna-technical-writer
   → Python code → axiom-python-engineering
   → ML systems → yzmir-ml-production
   → Game systems → bravos-systems-as-experience
   → UX issues → lyra-ux-designer

6. ONGOING TRACKING:
   PM tracks progress, architect validates improvements, archaeologist re-documents after changes
```

---

## Integration Matrix: The Ecosystem Value

**This is the killer feature** - showing how ALL packs work together through the improvement pipeline:

### Security Integration (Ordis)

| Pipeline Stage | Integration | Value Created |
|----------------|-------------|---------------|
| **Archaeologist** | Documents auth subsystems | Security attack surface visible |
| **Architect** | Identifies auth anti-patterns (05-assessment.md) | Security issues prioritized |
| **Architect** | Flags security debt (06-debt-catalog.md) | Quantifies security risk |
| **Architect** | Recommends auth consolidation (08-recommendations.md) | Security improvement strategy |
| **PM** | Creates security remediation project | Security work tracked |
| **Ordis** | `/security-architect` provides threat modeling | STRIDE applied to auth issues |
| **Ordis** | Designs secure auth service | Security controls specified |
| **PM** | Tracks implementation | Security improvements verified |
| **Archaeologist** | Re-documents after fix | New architecture validated |

**User sees**: "Claude found my security mess, told me how bad it was, created a fix plan, called in security experts, and tracked the fix."

---

### Documentation Integration (Muna)

| Pipeline Stage | Integration | Value Created |
|----------------|-------------|---------------|
| **Archaeologist** | Documents subsystems (02-catalog.md) | Raw architecture data |
| **Architect** | Identifies doc debt (06-debt-catalog.md) | Missing/outdated docs cataloged |
| **Architect** | Produces ADRs (10-architecture-decisions/) | Architecture decisions documented |
| **PM** | Creates documentation project | Doc work managed |
| **Muna** | `/technical-writer` structures docs | Professional documentation |
| **Muna** | Applies clarity & style guidelines | Docs readable |
| **Muna** | Creates diagrams from architect outputs | Visual documentation |
| **PM** | Tracks doc completion | Documentation verified |

**User sees**: "Claude found my doc gaps, turned architecture analysis into proper docs, and tracked completion."

---

### Python Engineering Integration (Axiom Python)

| Pipeline Stage | Integration | Value Created |
|----------------|-------------|---------------|
| **Archaeologist** | Identifies Python subsystems | Python codebase mapped |
| **Architect** | Flags Python anti-patterns (05-assessment.md) | Type hints missing, async misuse detected |
| **Architect** | Catalogs Python-specific debt (06-debt-catalog.md) | Python 3.8 → 3.12 upgrade needed |
| **PM** | Creates Python modernization project | Python work tracked |
| **Python Pack** | `/python-engineering` provides modern patterns | Type system, async, project structure |
| **Python Pack** | Systematic delinting | Code quality improved |
| **PM** | Tracks typing coverage, delinting progress | Quantified improvement |
| **Architect** | Validates Python architecture post-refactor | Quality gates passed |

**User sees**: "Claude found my Python mess, recommended modern patterns, and guided the upgrade."

---

### ML Production Integration (Yzmir)

| Pipeline Stage | Integration | Value Created |
|----------------|-------------|---------------|
| **Archaeologist** | Documents ML pipeline subsystems | Training, serving, monitoring mapped |
| **Architect** | Identifies ML anti-patterns (05-assessment.md) | Model serving bottlenecks, monitoring gaps |
| **Architect** | Flags ML technical debt (06-debt-catalog.md) | Hardcoded paths, no versioning, manual deploys |
| **PM** | Creates MLOps improvement project | ML work managed |
| **Yzmir ML-Prod** | `/ml-production` provides patterns | Model serving, quantization, monitoring |
| **Yzmir ML-Prod** | Implements MLOps practices | CI/CD for models, versioning, A/B testing |
| **PM** | Tracks deployment frequency, model metrics | MLOps KPIs monitored |
| **Architect** | Validates ML architecture improvements | Production-ready verified |

**User sees**: "Claude found my ML chaos, recommended production patterns, and tracked MLOps implementation."

---

### Game Development Integration (Bravos)

| Pipeline Stage | Integration | Value Created |
|----------------|-------------|---------------|
| **Archaeologist** | Documents game subsystems | Physics, AI, rendering mapped |
| **Architect** | Identifies game architecture issues (05-assessment.md) | Tight coupling between simulation and rendering |
| **Architect** | Flags game-specific debt (06-debt-catalog.md) | Fixed timestep issues, no determinism |
| **PM** | Creates simulation refactor project | Game work tracked |
| **Bravos Sim** | `/simulation-tactics` provides patterns | Fixed timestep, deterministic physics |
| **Bravos Systems** | `/systems-as-experience` for emergent design | Player-driven systems |
| **PM** | Tracks simulation stability, performance | Game quality metrics |
| **Architect** | Validates game architecture improvements | Emergent gameplay enabled |

**User sees**: "Claude found my game architecture mess, recommended systemic patterns, and tracked improvements."

---

### UX Architecture Integration (Lyra)

| Pipeline Stage | Integration | Value Created |
|----------------|-------------|---------------|
| **Archaeologist** | Documents UI/UX subsystems | Frontend components, design system |
| **Architect** | Identifies UX architectural issues (05-assessment.md) | Accessibility violations, inconsistent patterns |
| **Architect** | Flags UX technical debt (06-debt-catalog.md) | No design system, WCAG violations |
| **PM** | Creates design system project | UX work tracked |
| **Lyra** | `/ux-designer` provides patterns | Accessibility, design systems, interaction patterns |
| **Lyra** | Implements WCAG 2.1 AA compliance | Accessibility verified |
| **PM** | Tracks accessibility metrics, design consistency | UX quality measured |
| **Architect** | Validates UX architecture improvements | User experience architecturally sound |

**User sees**: "Claude found my UX mess, recommended design system and accessibility fixes, and tracked implementation."

---

## Cross-Pack Workflow Examples

### Example 1: "Fix My Legacy Auth System"

```
User: "My authentication is a mess - 3 different implementations, security issues, no docs"

1. Archaeologist:
   → Documents 3 auth subsystems (Session, JWT, OAuth - all different)
   → Maps dependencies (everything calls one of the three)
   → Produces diagrams showing the tangle

2. Architect:
   → CRITICAL: Pattern inconsistency (07-pattern-analysis.md)
   → HIGH: Security debt - no rate limiting, weak password hashing (06-debt-catalog.md)
   → Recommends: Extract unified auth service (08-recommendations.md)
   → Phase 1: Facade pattern to unify (3 weeks)
   → Phase 2: Extract service (6 weeks)
   → ADR-0001: Adopt OAuth2 + JWT standard

3. PM:
   → Creates auth-unification project
   → Sprint 1-2: Facade pattern
   → Sprint 3-5: Extract service
   → Sprint 6: Cutover

4. Ordis (Security):
   → Threat model for auth service (STRIDE)
   → Security controls spec (rate limiting, MFA, audit logging)
   → Compliance validation (GDPR, SOC2)

5. Muna (Docs):
   → Auth architecture documentation
   → Security model documentation
   → API documentation for new service

6. PM:
   → Tracks: Facade complete ✓, Service 60% done, Security review passed ✓
   → Risk: JWT library compatibility issue (mitigated with version pinning)
   → Status: Green, on track for 8-week completion

Result: "Claude found the mess, designed the fix, brought in security experts,
        documented everything, and tracked it to completion."
```

---

### Example 2: "My ML System Is Unreliable"

```
User: "My ML models work in notebooks but break in production"

1. Archaeologist:
   → Documents: Training notebooks, manual deployment scripts, no monitoring
   → Maps: Data pipeline → Training → Manual copy to server → Hope

2. Architect:
   → CRITICAL: No MLOps infrastructure (05-assessment.md)
   → HIGH: Manual deployment = high failure rate (06-debt-catalog.md)
   → Recommends: Implement MLOps pipeline (08-recommendations.md)
   → Phase 1: Model versioning (2 weeks)
   → Phase 2: Automated deployment (4 weeks)
   → Phase 3: Monitoring and A/B testing (4 weeks)
   → ADR-0002: Adopt MLflow for model registry

3. PM:
   → Creates ml-production project
   → Sprint 1-2: MLflow setup
   → Sprint 3-6: CI/CD pipeline
   → Sprint 7-10: Monitoring and experimentation

4. Yzmir ML-Production:
   → Model serving patterns (FastAPI + TorchServe)
   → Quantization for performance
   → Monitoring dashboards (latency, drift)
   → A/B testing framework

5. Ordis (Security):
   → Model API security
   → Data privacy controls
   → Audit logging for predictions

6. Muna (Docs):
   → MLOps runbook
   → Model deployment guide
   → Monitoring playbook

7. PM:
   → Tracks: Versioning ✓, CI/CD 80%, Monitoring planned
   → Metrics: Deployment time (manual 3 hours → automated 5 minutes)
   → Risk: Model performance regression (mitigated with A/B rollout)
   → Status: Yellow (slight timeline slip), adjusting scope

Result: "Claude found my ML chaos, designed production pipeline, brought in
        ML experts, added security and docs, and tracked implementation."
```

---

### Example 3: "My Game Performance Is Terrible"

```
User: "My game lags, physics are inconsistent, and players exploit the systems"

1. Archaeologist:
   → Documents: Game loop, physics engine, AI systems, rendering
   → Maps: Tight coupling between rendering and physics (frame-dependent physics!)
   → Diagrams show no separation between simulation and presentation

2. Architect:
   → CRITICAL: Frame-dependent physics (05-assessment.md)
   → HIGH: No fixed timestep = non-deterministic behavior (06-debt-catalog.md)
   → Recommends: Separate simulation from rendering (08-recommendations.md)
   → Phase 1: Fixed timestep (1 week)
   → Phase 2: Decouple physics from rendering (3 weeks)
   → Phase 3: Optimize physics update (2 weeks)
   → ADR-0003: Adopt fixed timestep pattern

3. PM:
   → Creates game-simulation-refactor project
   → Sprint 1: Fixed timestep
   → Sprint 2-4: Decoupling
   → Sprint 5-6: Optimization

4. Bravos Simulation:
   → Fixed timestep with interpolation
   → Deterministic physics
   → Spatial partitioning for performance

5. Bravos Systems-as-Experience:
   → Emergent gameplay from deterministic systems
   → Player-driven interactions instead of exploits

6. Axiom Python (if Python game):
   → Profiling for bottlenecks
   → Numba optimization for physics

7. PM:
   → Tracks: Fixed timestep ✓, Decoupling 50%, Performance testing ongoing
   → Metrics: Frame time (was 33ms, now 8ms), Determinism (100% reproducible)
   → Risk: Old saves incompatible (mitigated with migration tool)
   → Status: Green

Result: "Claude found my game performance issues, recommended simulation patterns,
        brought in game dev experts, and tracked improvements."
```

---

## Why This Creates Marketplace Value

### The Network Effect

**Without improvement pipeline:**
- Users install individual packs for point solutions
- "I need security docs" → install Ordis, use once, forget
- Packs used in isolation

**With improvement pipeline:**
- Users install archaeologist → reveals mess → architect recommends fixes → PM creates projects → specialist packs get called in
- "Fix my codebase" → uses 4-6 packs together in coordinated workflow
- Packs become **ecosystem**, not collection

### The Discovery Engine

The architect becomes a **discovery engine for other packs**:

- Architect finds security issues → "You need ordis-security-architect"
- Architect finds doc gaps → "You need muna-technical-writer"
- Architect finds Python issues → "You need axiom-python-engineering"
- Architect finds ML problems → "You need yzmir-ml-production"

**Users discover packs through recommendations, not browsing.**

### The Retention Story

**Single-pack usage**: Install → Use once → Forget

**Pipeline usage**: Install archaeologist → Install architect → Install PM → Keep using PM for all projects → Install specialist packs as needed

**PM becomes the "always-on" pack** that keeps users engaged with the ecosystem.

---

## Implementation Priorities

### Phase 1: Core Pipeline (2-3 weeks)

**Implement:**
1. **axiom-system-architect** (8 skills)
   - Priority: assessing-architecture-quality, identifying-technical-debt, recommending-refactoring-strategies, prioritizing-improvements
   - Lower priority: Other 4 skills (important but not MVP)

2. **axiom-project-manager** (8 skills)
   - Priority: initiating-projects, tracking-progress, planning-iterations
   - Lower priority: Other 5 skills (nice-to-have)

**Validation:**
- RED-GREEN-REFACTOR testing on one complete use case (legacy auth system)
- Integration test with 2-3 existing packs (Ordis, Muna, Python)

---

### Phase 2: Integration Examples (1 week)

**Create:**
- 3 end-to-end examples showing multi-pack integration:
  1. Security remediation (Archaeologist + Architect + PM + Ordis + Muna)
  2. Python modernization (Archaeologist + Architect + PM + Python + Muna)
  3. ML production (Archaeologist + Architect + PM + Yzmir + Ordis)

**Document:**
- Integration patterns in each pack's router skill
- Cross-references showing when to call other packs

---

### Phase 3: Remaining Skills (2 weeks)

**Complete:**
- Remaining architect skills (analyzing-architectural-patterns, documenting-architecture-decisions, estimating-refactoring-effort, generating-improvement-metrics)
- Remaining PM skills (managing-risks, managing-dependencies, facilitating-retrospectives, communicating-status, closing-projects)

**Validation:**
- Full RED-GREEN-REFACTOR for all skills
- Integration testing across all 15 existing packs

---

### Phase 4: Marketing & Documentation (1 week)

**Create:**
- Updated marketplace README highlighting integration story
- "Getting Started with Codebase Improvement" tutorial
- Video/demo showing complete pipeline on real codebase
- Integration matrix documentation (like above) in official docs

---

## Success Criteria

### Phase 1 Complete When:
- ✅ Archaeologist can document legacy codebase
- ✅ Architect can assess and provide roadmap
- ✅ PM can create and track project from roadmap
- ✅ Integration with 2-3 specialist packs demonstrated
- ✅ All core skills pass RED-GREEN-REFACTOR

### Full Success When:
- ✅ Complete pipeline tested on 3+ real codebases
- ✅ Integration examples for all 15 existing packs created
- ✅ Users report discovering new packs through architect recommendations
- ✅ PM retention shows users returning for multiple projects
- ✅ Marketplace installation rate increases (pipeline creates demand)

---

## Open Questions

1. **Scope creep risk**: Does PM try to replace project management tools (Jira, etc.) or integrate with them?
   - **Answer**: Start with standalone markdown files, add tool integration in Phase 2+

2. **Architect tone**: How critical is "too critical"? Balance honesty with constructiveness.
   - **Answer**: Evidence-based critique (cite patterns, principles) + always provide fix path

3. **PM for non-refactoring**: Does PM work for greenfield projects or just improvements?
   - **Answer**: Yes - PM is general purpose, architect is improvement-specific

4. **Routing complexity**: How does architect know which specialist pack to recommend?
   - **Answer**: Pattern matching in `using-system-architect` router skill (if security pattern violated → suggest Ordis)

5. **Validation overhead**: Do we re-run archaeologist after improvements?
   - **Answer**: Optional - architect can recommend re-documentation as validation step

---

## Related Documents

- `plugins/axiom-system-archaeologist/skills/using-system-archaeologist/SKILL.md` - Existing archaeologist router
- `source/CONTRIBUTING.md` - Skill creation guide
- `docs/future-axiom-skillpack-engineering-intent.md` - Meta-engineering pack (adjacent concept)
- `README.md` - Marketplace overview (will need updating with integration story)

---

## Conclusion

**The improvement pipeline transforms the skillpacks marketplace from a collection of tools into an integrated ecosystem.**

Users don't just get skills - they get a **systematic process for fixing legacy systems** that brings in the right experts at the right time and tracks everything to completion.

**This is the integration story that makes the entire marketplace valuable.**

---

**End of Intent Document**

*This document captures the vision for completing the codebase improvement pipeline and creating the integration narrative that makes all 15+ skillpacks work together as an ecosystem.*
