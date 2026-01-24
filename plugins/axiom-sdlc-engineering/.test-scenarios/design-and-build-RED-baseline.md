# Design-and-Build Skill - RED Baseline Test Results

**Test Date**: 2026-01-24
**Purpose**: Document Claude Code responses WITHOUT the design-and-build skill
**Expected Outcome**: Failures, missing guidance, inadequate responses that justify skill creation

---

## Scenario 1: Quick Fix vs Proper Architecture

### User Request
"There's a production bug. I have a quick fix but it adds technical debt. Do I really need an ADR for this?"

**Context**: Production bug, time pressure, Level 3 project

### Simulated Response WITHOUT Skill

**What Claude Would Say**:
```
For a production bug fix, the priority is restoring service quickly. Here's how to approach this:

1. Apply the quick fix to stop the bleeding
2. Document the technical debt in your issue tracker
3. Create a follow-up ticket for the proper solution
4. Whether you need an ADR depends on:
   - The significance of the architectural impact
   - Your team's governance requirements
   - The scope of the technical debt introduced

For production emergencies, you can defer the ADR but should:
- Add comments explaining why the quick fix was necessary
- Document the intended proper solution
- Get team agreement on timeline for addressing the debt

Would you like help implementing the fix or creating the follow-up documentation?
```

### What's Missing or Inadequate

1. **No Level System Context**: Response doesn't understand what "Level 3 project" means
   - No recognition that Level 3 REQUIRES ADRs for architectural decisions
   - Missing enforcement of project governance tier
   - No explanation of when levels apply vs when they don't

2. **No Emergency Exception Framework**: Response suggests informal approach
   - Missing structured exception process (HOTFIX pattern)
   - No guidance on retrospective ADR creation
   - No timeline enforcement for debt paydown
   - Treats ADR as "optional if time-pressured" rather than "required with exceptions"

3. **No Technical Debt Classification**: Treats all debt equally
   - Missing distinction between tactical vs strategic debt
   - No risk assessment framework (what debt is acceptable short-term?)
   - No guidance on when quick fix becomes permanent anti-pattern

4. **No Decision Framework**: Vague "depends on" criteria
   - Doesn't help user DECIDE whether this specific case needs ADR
   - Missing concrete threshold rules
   - No escalation criteria (when to involve architect/lead)

5. **No Long-term Consequences**: Focuses only on immediate action
   - Doesn't warn about debt accumulation patterns
   - Missing guidance on debt tracking systems
   - No metrics for measuring debt impact

### Specific Guidance Needed

- **Level 3 Governance**: "Level 3 projects REQUIRE ADRs for architectural changes, including hotfixes that introduce debt. Use HOTFIX-ADR template for post-facto documentation within 48 hours."
- **Exception Protocol**: "Emergency fixes can bypass ADR creation DURING the incident, but retrospective ADR is mandatory. Include incident timeline, why proper fix wasn't feasible, paydown commitment with date."
- **Debt Classification**: "Quick fixes that change module boundaries, introduce coupling, or defer refactoring = architectural debt requiring ADR. Bug fixes within existing patterns = tactical debt, tracked in tickets."
- **Risk Assessment**: "If the 'proper solution' would take >4 hours and affects <3 modules, quick fix is acceptable with 2-week paydown commitment. If affects >3 modules or core abstractions, escalate for architect review."

---

## Scenario 2: Git Chaos - No Branch Strategy

### User Request
"Our Git workflow is a mess. What branching strategy should we use?"

**Context**: Team of 6, merge conflicts daily, force pushes

### Simulated Response WITHOUT Skill

**What Claude Would Say**:
```
For a team of 6 experiencing merge conflicts, I recommend implementing a structured branching strategy. Here are the most common options:

**1. Git Flow**
- Main branches: main (production) and develop
- Feature branches: feature/feature-name
- Release branches: release/x.y.z
- Hotfix branches: hotfix/description

Pros: Clear structure, good for release-based workflows
Cons: More complex, can slow down fast iteration

**2. GitHub Flow**
- Single main branch
- Feature branches created from main
- Pull requests for all changes
- Deploy from main

Pros: Simpler, better for continuous deployment
Cons: Less structure for release management

**3. Trunk-Based Development**
- Everyone commits to main/trunk frequently
- Short-lived feature branches (<1 day)
- Feature flags for incomplete work

Pros: Simplifies merging, encourages small changes
Cons: Requires good CI/CD, discipline

For your team size, I'd recommend GitHub Flow because:
- Simpler than Git Flow
- Reduces merge conflicts through frequent integration
- Works well with PR-based code review

To implement:
1. Protect your main branch (no force pushes)
2. Require pull requests for all changes
3. Set up CI to run on PRs
4. Establish branch naming convention
5. Delete branches after merge

Would you like help setting up branch protection rules?
```

### What's Missing or Inadequate

1. **No Root Cause Analysis**: Jumps to solutions without diagnosis
   - Missing questions about WHY conflicts happen daily (overlapping work? poor module boundaries? lack of communication?)
   - No investigation of force push patterns (who? when? why?)
   - Assumes problem is branching strategy, not team communication or architecture

2. **No Project Level Assessment**: Recommends strategy without understanding project tier
   - Doesn't ask about release schedule, deployment frequency, or risk tolerance
   - GitHub Flow might be wrong for Level 3 project requiring staged releases
   - Missing connection between governance level and workflow choice

3. **Generic Best Practices**: Lists textbook strategies without customization
   - Doesn't account for specific pain points (6-person team, daily conflicts)
   - No adaptation based on current state (are they using any strategy now?)
   - Missing migration path from chaos to structure

4. **No ADR Requirement**: Doesn't recognize this as architectural decision
   - Branching strategy affects entire team workflow = architectural
   - Should require ADR documenting decision, alternatives considered, and rationale
   - Missing governance enforcement

5. **No Implementation Roadmap**: "Here's how" without sequencing or risk management
   - Switching branching strategies mid-project has migration risks
   - No guidance on handling in-flight work during transition
   - Missing rollback plan if new strategy fails

6. **No Metrics or Success Criteria**: Can't measure if solution works
   - What does "success" look like? (Zero force pushes? <X conflicts per week?)
   - No monitoring plan for ongoing health
   - Missing retrospective schedule to validate choice

### Specific Guidance Needed

- **Diagnostic Framework**: "Before choosing a strategy, diagnose the root cause: Run `git log --all --graph --oneline --since='1 month ago'` and analyze conflict patterns. Are conflicts in same files? Same modules? What triggers force pushes?"
- **Level-Based Strategy Selection**: "Level 1/2 projects → GitHub Flow. Level 3 with compliance requirements → Git Flow with release branches. Level 3 with continuous deployment → Trunk-based with feature flags and ADR for each major flag."
- **ADR Requirement**: "This decision requires ADR. Use template: Current chaos description, 3 strategies evaluated, decision criteria (team size, release cadence, compliance), chosen strategy with rationale, implementation plan, success metrics, 30-day retrospective date."
- **Migration Roadmap**: "Phase 1 (Week 1): Protect main, document current state. Phase 2 (Week 2): Train team on new workflow, run parallel for 5 days. Phase 3 (Week 3): Cut over, monitor conflict rate. Rollback trigger: >2x conflicts in first week."
- **Success Metrics**: "Baseline current state (conflicts/week, force pushes/week, PR cycle time). Target: 80% reduction in conflicts, zero force pushes, <4 hour PR cycle time. Review metrics in 30 days."

---

## Scenario 3: Technical Debt Spiral

### User Request
"We're drowning in technical debt. How do we track and pay it down without stopping feature development?"

**Context**: 2-year-old codebase, 70% time on bugs, Level 3 project

### Simulated Response WITHOUT Skill

**What Claude Would Say**:
```
Technical debt requiring 70% of your time is a critical issue. Here's a systematic approach:

**Step 1: Inventory Your Debt**
1. Create a "tech-debt" label in your issue tracker
2. Run a team session to identify and document debt items
3. Categorize by type:
   - Code quality (duplication, complexity)
   - Architecture (coupling, missing abstractions)
   - Testing (low coverage, flaky tests)
   - Documentation (missing, outdated)
   - Infrastructure (build times, deployment issues)

**Step 2: Prioritize**
For each debt item, assess:
- Impact: How much does it slow development?
- Risk: What could break if left unfixed?
- Effort: How long to fix?

Use a matrix to prioritize: High Impact + Low Effort = Do First

**Step 3: Allocate Capacity**
Two approaches:
1. **Percentage allocation**: Dedicate 20-30% of each sprint to debt
2. **Debt sprints**: Every 4th sprint is pure debt paydown

I recommend percentage allocation for your situation because:
- Continuous improvement vs big-bang fixes
- Maintains feature velocity
- Builds debt-reduction into team rhythm

**Step 4: Track Progress**
- Create dashboard showing debt items over time
- Measure velocity improvement as debt decreases
- Celebrate wins when bugs drop below 50% of time

**Step 5: Prevent New Debt**
- Require code review for all changes
- Set up automated linting/testing
- Document "definition of done" including refactoring
- Allow time for proper solutions, not just quick fixes

**Implementation Plan**:
1. This week: Inventory debt, categorize, estimate
2. Next sprint: Start with 20% debt allocation
3. Month 1: Track metrics, adjust allocation if needed
4. Month 3: Reassess if bugs drop below 40% of time

Would you like help setting up the debt tracking system?
```

### What's Missing or Inadequate

1. **No Crisis Recognition**: Treats 70% bug time as "technical debt problem"
   - This is a CODE RED situation, not normal debt management
   - Missing urgency and executive escalation
   - Doesn't recognize this threatens project viability

2. **No Root Cause Analysis**: Jumps to tracking without understanding WHY
   - How did debt reach crisis level? (Poor architecture? No testing? Deadline pressure?)
   - What patterns created this? (Resume-driven design? Copy-paste culture?)
   - Missing cultural/process diagnosis

3. **No Level 3 Governance Connection**: Doesn't leverage existing structures
   - Level 3 projects should have architectural oversight - why didn't it catch this?
   - Missing ADR audit: Were decisions documented? Were they violated?
   - No mention of retrospective ADRs to understand what went wrong

4. **Inadequate Debt Classification**: Treats all debt as equal
   - Missing distinction between architectural debt (requires design) vs code quality (refactor)
   - 70% bug time suggests architectural rot, not just messy code
   - No framework for "unpayable debt" (when to rewrite vs refactor)

5. **Unrealistic Timeline**: Suggests "20% allocation" and 3-month horizon
   - At 70% bug time, team is in death spiral - incremental approach may not work
   - Missing threshold for "stop features entirely until stable"
   - No guidance on when debt is so severe it requires architectural intervention

6. **No Decision Framework for Feature Freeze**: Doesn't address hard tradeoffs
   - When DO you stop features? (At what % bug time? What risk level?)
   - How do you get stakeholder buy-in for feature freeze?
   - Missing business case template for "we must fix this NOW"

7. **No Measurement of Debt Impact**: Vague "track metrics"
   - What metrics prove debt is architectural vs tactical?
   - How do you measure if you're winning or losing?
   - Missing leading indicators (complexity trends, coupling metrics)

### Specific Guidance Needed

- **Crisis Thresholds**: "70% time on bugs = CODE RED. This is beyond normal debt management. Immediate actions: 1) Stop new features for 2 weeks. 2) Run architectural audit. 3) Identify 'load-bearing debt' (must fix vs can defer). 4) Get executive buy-in for recovery plan. Normal debt strategies don't apply here."
- **Debt Classification for Crisis**:
  - **Architectural Debt** (requires ADR, design): Module boundaries violated, missing abstractions, wrong patterns. Indicators: Same bugs in multiple places, changes ripple across modules.
  - **Code Quality Debt**: Duplication, complexity, no tests. Indicators: Long functions, copy-paste, low coverage.
  - **Unpayable Debt**: When fixing costs more than rewriting. Indicators: >60% of module needs changes, core assumptions wrong.
- **Level 3 Governance Audit**: "Review all ADRs from past 2 years. Which decisions created this debt? Were ADRs followed or violated? Create retrospective ADRs for undocumented decisions that led to debt. This reveals pattern: poor decisions, or good decisions badly implemented?"
- **Feature Freeze Decision Framework**: "Freeze features if: 1) >60% time on bugs, 2) Velocity trending down for 3+ months, 3) Customer-reported bugs increasing, 4) Team morale critical. Present to stakeholders: 'Current state: 70% bugs, 30% features. Proposed: 2-week freeze → 40% bugs, 60% features long-term. Cost: 2 weeks no features. Benefit: 2x velocity after.'"
- **Architectural Debt Metrics**:
  - **Cyclomatic Complexity**: Baseline avg, track trend (should decrease)
  - **Coupling Metrics**: Dependencies between modules (should decrease)
  - **Change Amplification**: How many files change per feature? (should decrease)
  - **Bug Clustering**: % bugs in top 10% of files (reveals debt hotspots)
- **Recovery Roadmap for Crisis**:
  - **Week 1-2**: FREEZE features. Triage debt into architectural (needs design), code quality (refactor), unpayable (rewrite). Create ADRs for architectural fixes.
  - **Week 3-4**: Implement architectural fixes (should reduce bug clusters). Continue freeze.
  - **Week 5-8**: Switch to 50/50 debt/features. Monitor bug % - should drop to 40%.
  - **Week 9+**: Adjust to 20/80 debt/features once bugs <30% of time.
  - **Rollback**: If bugs don't drop to 50% by Week 4, escalate to rewrite discussion.

---

## Scenario 4: CI/CD Setup - Where to Start?

### User Request
"We're building manually. How do we set up CI/CD the right way?"

**Context**: Manual builds taking hours, Level 2 project

### Simulated Response WITHOUT Skill

**What Claude Would Say**:
```
Setting up CI/CD will save you significant time and reduce errors. Here's a step-by-step approach:

**Step 1: Choose Your CI/CD Platform**
Popular options:
- GitHub Actions (if using GitHub)
- GitLab CI/CD (if using GitLab)
- Jenkins (self-hosted, more control)
- CircleCI (cloud-based)
- Travis CI (cloud-based)

For most teams, I recommend GitHub Actions or GitLab CI because:
- Integrated with your repo
- Easy to set up
- Good free tier
- Large community

**Step 2: Define Your Pipeline Stages**
Typical pipeline:
1. **Build**: Compile code, run linters
2. **Test**: Run unit tests, integration tests
3. **Package**: Create artifacts (Docker images, binaries)
4. **Deploy**: Push to environments (dev, staging, prod)

**Step 3: Start Simple**
Create a basic workflow that:
1. Triggers on pull requests
2. Runs your build
3. Runs your tests
4. Reports status

Example GitHub Actions workflow:
```yaml
name: CI
on: [pull_request]
jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build
        run: make build
      - name: Test
        run: make test
```

**Step 4: Add Deployment**
Once CI is stable:
1. Add staging deployment on merge to main
2. Add production deployment (manual approval or automated)
3. Set up environment-specific configs

**Step 5: Improve Over Time**
- Add code coverage reporting
- Add security scanning
- Optimize build times (caching, parallelization)
- Add notifications (Slack, email)

**Implementation Timeline**:
- Week 1: Set up basic CI (build + test)
- Week 2: Test on all PRs, fix any issues
- Week 3: Add staging deployment
- Week 4: Add production deployment

Would you like help creating the initial workflow file?
```

### What's Missing or Inadequate

1. **No Requirements Gathering**: Jumps to implementation without understanding needs
   - What are they building? (Web app? Mobile? Desktop? Microservices?)
   - What's their deployment target? (Cloud? On-prem? Hybrid?)
   - What's their risk tolerance? (Financial system vs internal tool)
   - Missing context about why builds take hours (compilation time? test suite? dependency hell?)

2. **No Level-Based Guidance**: Doesn't leverage "Level 2 project" information
   - Level 2 likely has governance requirements (what approvals for prod deploy?)
   - Missing connection to project tier and appropriate pipeline complexity
   - Doesn't ask if Level 2 requires certain gates (security scans, compliance checks)

3. **No ADR Requirement**: Treats CI/CD choice as tactical, not architectural
   - CI/CD platform choice is architectural (affects team workflow, vendor lock-in, cost)
   - Should require ADR documenting platform choice, deployment strategy, rollback approach
   - Missing documentation of tradeoffs (self-hosted vs cloud, cost vs features)

4. **Generic Template**: Provides cookbook example without customization
   - "make build" and "make test" might not exist
   - Doesn't account for hours-long builds (needs caching strategy, parallelization from day 1)
   - No guidance on splitting long builds into stages

5. **No Risk Management**: Doesn't address deployment risks
   - What if automated deploy breaks production?
   - Missing rollback strategy, canary deployments, feature flags
   - No discussion of deployment windows, downtime, or backward compatibility

6. **No Success Criteria**: Can't measure if solution works
   - What's the goal? (Reduce build time from hours to minutes? Catch bugs pre-merge?)
   - No metrics to track (build time, deployment frequency, MTTR)
   - Missing baseline measurement

7. **No Team Readiness Assessment**: Assumes team is ready for CI/CD
   - Do they have automated tests? (If not, CI won't help)
   - Do they have environment parity? (dev/staging/prod)
   - Do they understand git workflows? (CI depends on branching strategy)

### Specific Guidance Needed

- **Requirements Gathering Framework**:
  - **Build Characteristics**: "What are you building? How long does each stage take (compile, test, package)? What's the bottleneck? (CPU-bound compilation? I/O-bound tests? Network-bound dependencies?)"
  - **Deployment Context**: "Where does it run? (AWS/Azure/GCP? Kubernetes? VMs? Serverless?) How often do you deploy? (Daily? Weekly? On-demand?) What's downtime tolerance?"
  - **Risk Profile**: "Level 2 project suggests moderate risk. Do you need: Manual approval for prod? Security scanning? Compliance artifacts? Deployment windows?"
- **Level 2 Governance Integration**: "Level 2 projects should have: 1) Approval gates for production deploys, 2) ADR documenting CI/CD architecture, 3) Rollback procedures tested quarterly. Your pipeline should enforce these gates, not bypass them."
- **ADR Requirement**: "CI/CD platform and deployment strategy require ADR. Document: 1) Platforms evaluated (GitHub Actions, GitLab CI, Jenkins, CircleCI), 2) Decision criteria (cost, features, lock-in risk, team expertise), 3) Chosen platform with rationale, 4) Deployment strategy (blue/green, canary, rolling), 5) Rollback procedure, 6) Success metrics."
- **Hours-Long Build Strategy**: "If current build takes hours: 1) Profile to find bottleneck (use `time` for each stage). 2) Parallelize where possible (tests in parallel, multi-stage Docker builds). 3) Implement aggressive caching (dependencies, build artifacts, Docker layers). 4) Consider distributed builds (multiple runners). Target: <15 min CI feedback on PRs."
- **Deployment Risk Management**:
  - **Staging-First**: "Never auto-deploy to prod from main. Deploy to staging automatically, run smoke tests, THEN promote to prod (manual or automated with validation)."
  - **Rollback Strategy**: "Every deployment needs rollback procedure. Options: Blue/green swap, previous version re-deploy, database migration rollback. Test rollback quarterly. Include in ADR."
  - **Feature Flags**: "For risky changes, deploy behind feature flag. Turn on in staging, validate, then enable in prod. Decouple deploy from release."
- **Pre-Flight Checklist**: "Before implementing CI/CD: 1) Do you have automated tests? (If no, write tests first). 2) Are dev/staging/prod environments similar? (If no, fix parity). 3) Do you have a branching strategy? (If no, establish workflow). 4) Can you build locally in <15 min? (If no, optimize first). CI/CD amplifies your current process - fix local builds before automating."
- **Success Metrics**:
  - **Build Time**: Baseline current manual time. Target: <15 min for CI feedback.
  - **Deployment Frequency**: Track deploys/week. Should increase over time.
  - **Lead Time**: Commit to production time. Should decrease.
  - **MTTR**: Mean time to recovery from bad deploy. Should decrease with automated rollback.

---

## Scenario 5: Resume-Driven Design

### User Request
"Should we use microservices for this project? I've heard they're best practice."

**Context**: Simple CRUD app, developer wants trendy tech, Level 3 requires ADR

### Simulated Response WITHOUT Skill

**What Claude Would Say**:
```
Microservices are a powerful architectural pattern, but they're not always the right choice. Let's evaluate whether they fit your project.

**Microservices Pros:**
- Independent scaling of services
- Technology diversity (each service can use different stack)
- Team autonomy (teams own services end-to-end)
- Fault isolation (one service failure doesn't bring down system)
- Independent deployment (deploy services separately)

**Microservices Cons:**
- Increased complexity (distributed systems, network calls)
- Harder debugging (tracing across services)
- Data consistency challenges (distributed transactions)
- More operational overhead (multiple deployments, monitoring)
- Learning curve for team

**For a Simple CRUD App:**
I would **recommend starting with a monolith** because:

1. **Simplicity**: A CRUD app likely has straightforward logic that doesn't need service boundaries
2. **Lower complexity**: You can develop, test, and deploy as one unit
3. **Faster development**: No need for service communication, API contracts
4. **Easier debugging**: All code in one place, standard debugging tools work
5. **Lower ops burden**: One deployment, one database, simpler monitoring

**When Microservices Make Sense:**
- Large teams (>20 developers) needing independent work streams
- Different scaling needs per component (e.g., read-heavy vs write-heavy)
- Clear business domain boundaries (e.g., orders, inventory, shipping)
- Existing monolith that's too large to manage

**Recommendation:**
Start with a well-structured monolith:
- Use clear module boundaries (even if in same codebase)
- Keep business logic separated from infrastructure
- Design with potential future splitting in mind
- If you outgrow the monolith, you can extract services later

**Migration Path:**
Monolith → Modular Monolith → Extract services only when needed

This gives you:
- Fast time to market now
- Option to scale architecture later
- Learn your domain before committing to service boundaries

Would you like help designing a modular monolith architecture?
```

### What's Missing or Inadequate

1. **No Recognition of Resume-Driven Design**: Treats question as sincere technical inquiry
   - Missing detection that "I've heard they're best practice" is red flag for cargo-cult thinking
   - Doesn't challenge the premise or explore WHY developer wants microservices
   - No discussion of pattern: trendy tech vs actual requirements

2. **No Level 3 Enforcement**: Doesn't mention ADR requirement
   - Level 3 projects REQUIRE ADR for architectural decisions like this
   - Response should start with "This requires an ADR. Let's document the decision properly."
   - Missing use of ADR process to force rigorous thinking and prevent resume-driven choices

3. **No Decision Framework**: Lists pros/cons but no systematic evaluation
   - Doesn't provide checklist or decision tree
   - Missing quantitative thresholds (team size, request volume, etc.)
   - No "forcing function" to make developer defend the choice

4. **No Anti-Pattern Education**: Doesn't explicitly name the anti-pattern
   - This is a classic case of "resume-driven development" or "hype-driven architecture"
   - Missing warning about technological solutionism (picking tech first, problem second)
   - No discussion of industry pattern: microservices causing more problems than they solve

5. **No Stakeholder Impact Analysis**: Focuses only on technical factors
   - Doesn't ask about timeline (microservices will slow initial delivery)
   - Missing cost analysis (ops overhead, infrastructure costs)
   - No discussion of team skill level (do they know distributed systems?)

6. **No Learning Opportunity**: Gives answer instead of teaching decision-making
   - Doesn't help developer learn HOW to evaluate architectural patterns
   - Missing questions that would reveal if they understand tradeoffs
   - No guidance on building architectural judgment

7. **No Alternatives Considered**: Only evaluates monolith vs microservices
   - Missing modular monolith, service-oriented architecture, serverless, etc.
   - Doesn't explore if there's a middle ground
   - No discussion of "simplest thing that works"

### Specific Guidance Needed

- **Resume-Driven Design Detection**: "Red flag: 'I've heard it's best practice' suggests technology-first thinking instead of requirements-first. Before discussing microservices, answer: 1) What problem are you solving? 2) What alternatives exist? 3) Why is the current approach (monolith?) inadequate? If you can't answer these, you're doing resume-driven design."
- **Level 3 ADR Enforcement**: "This is a Level 3 project, so architectural decisions require ADR. You must document: 1) Problem statement (what requirements drive architecture?), 2) Alternatives considered (monolith, modular monolith, microservices, serverless), 3) Decision criteria (team size, scaling needs, operational maturity, timeline, cost), 4) Decision with rationale, 5) Consequences (what tradeoffs are you accepting?). ADR forces you to defend 'microservices' with data, not hype."
- **Microservices Decision Framework**:
  - **Team Size**: <10 developers → Monolith. 10-30 → Modular monolith. >30 → Consider microservices.
  - **Domain Complexity**: Simple CRUD → Monolith. Clear bounded contexts with different scaling/team needs → Microservices.
  - **Operational Maturity**: No DevOps experience → Monolith. Mature CI/CD, monitoring, incident response → Can handle microservices.
  - **Timeline Pressure**: Need MVP in <6 months → Monolith. Long-term platform → Consider microservices if justified.
  - **Cost Tolerance**: Limited budget → Monolith (lower ops cost). Large budget → Microservices acceptable if needed.
  - **Forcing Function**: "Microservices are ONLY justified if you can point to specific, measurable requirements they solve that simpler architectures can't. E.g., 'Auth service needs 10x requests of other components, must scale independently.'"
- **Anti-Pattern Education**: "You're exhibiting resume-driven design: choosing architecture based on industry trends, not requirements. This leads to: 1) Overengineered solutions (microservices for CRUD app), 2) Longer timelines (distributed systems complexity), 3) Higher costs (more infrastructure, operational overhead), 4) Career risk (when it fails, you own it). Instead: Pick simplest architecture that meets requirements. Your resume improves by delivering working software, not using buzzwords."
- **Stakeholder Impact Questions**:
  - **Timeline**: "Microservices will add 3-6 months to delivery vs monolith. Can stakeholders wait? If not, monolith is only option."
  - **Cost**: "Microservices require: Multiple services (compute cost), service mesh (additional infrastructure), observability stack (tracing, logging), on-call rotation (operational cost). Estimated 2-3x cost vs monolith. Is this budgeted?"
  - **Team Skills**: "Does your team understand: Distributed transactions? CAP theorem? Service discovery? Circuit breakers? API versioning? Network fallacies? If not, you'll spend 6+ months learning while building. Factor this into timeline."
- **Decision-Making Education**: "How to evaluate architectural patterns: 1) Start with requirements (functional + non-functional). 2) Identify constraints (team, timeline, budget, risk tolerance). 3) Generate alternatives (list 3+ options). 4) Define decision criteria (what matters most?). 5) Score alternatives against criteria. 6) Pick simplest option that meets requirements. 7) Document in ADR. 8) Revisit when requirements change. This process prevents hype-driven choices."
- **Alternatives for CRUD App**:
  - **Monolith**: Single codebase, single deployment, shared database. Best for: Small teams, simple domains, fast iteration.
  - **Modular Monolith**: Single deployment, but clear module boundaries. Best for: Medium teams, moderate complexity, want refactoring option.
  - **Serverless**: Functions + managed services (e.g., API Gateway + Lambda + DynamoDB). Best for: Variable load, minimal ops, cloud-native.
  - **Microservices**: Separate services, separate deployments. Best for: Large teams, clear bounded contexts, independent scaling needs.
  - **For simple CRUD**: Monolith or serverless. Microservices is ALWAYS wrong here.

---

## Summary of Missing Guidance Across All Scenarios

### Cross-Cutting Themes

1. **No Level/Governance Integration**
   - Responses don't leverage project tier information (Level 2, Level 3)
   - Missing enforcement of ADR requirements for architectural decisions
   - No connection between governance level and appropriate rigor

2. **No Decision Frameworks**
   - Vague "depends on" criteria without concrete thresholds
   - Missing checklists, decision trees, or forcing functions
   - No systematic way to evaluate alternatives

3. **No Root Cause Analysis**
   - Jumps to solutions without diagnosing WHY problems exist
   - Treats symptoms instead of underlying issues
   - Missing investigative questions that reveal actual problems

4. **No Risk Management**
   - Doesn't assess when situations are critical vs routine
   - Missing escalation criteria and emergency protocols
   - No rollback plans or contingency thinking

5. **No Metrics or Success Criteria**
   - Can't measure if proposed solutions work
   - Missing baseline measurements and targets
   - No monitoring or retrospective schedules

6. **No Anti-Pattern Recognition**
   - Doesn't identify common pitfalls (resume-driven design, debt spirals, etc.)
   - Treats bad ideas as sincere technical questions
   - Missing education about WHY certain approaches fail

7. **Generic Best Practices**
   - Textbook answers without customization to context
   - Lists options without helping user choose
   - No adaptation to specific pain points or constraints

### What the Design-and-Build Skill MUST Provide

1. **Level-Based Governance Framework**
   - Clear definitions of Level 1, 2, 3 and their requirements
   - When ADRs are required vs optional
   - How to enforce governance without bureaucracy

2. **Decision Frameworks and Tools**
   - ADR templates with forcing functions
   - Decision trees for common choices (architecture, branching, CI/CD)
   - Concrete thresholds and criteria

3. **Diagnostic Methodologies**
   - How to investigate root causes before proposing solutions
   - Questions to ask, data to gather
   - Pattern recognition for common problems

4. **Risk and Crisis Management**
   - How to recognize CODE RED situations
   - Emergency protocols and exception handling
   - Escalation paths and stakeholder communication

5. **Technical Debt Classification**
   - Architectural vs tactical vs unpayable debt
   - When to refactor, when to rewrite, when to accept debt
   - Metrics for measuring debt impact

6. **Anti-Pattern Catalog**
   - Resume-driven design, debt spirals, git chaos
   - How to detect and counter these patterns
   - Education on WHY they fail

7. **Implementation Roadmaps**
   - Sequenced plans with phases and gates
   - Migration strategies from current state to desired state
   - Rollback triggers and contingency plans

8. **Success Metrics**
   - What to measure for each type of decision
   - How to baseline and track progress
   - When to retrospect and course-correct

---

## Conclusion

WITHOUT the design-and-build skill, Claude Code provides:
- **Generic best practices** that don't account for project context
- **Cookbook solutions** without customization or decision support
- **Incomplete guidance** that misses governance, risk management, and metrics
- **No enforcement** of architectural rigor or documentation requirements

The design-and-build skill is needed to:
- **Enforce governance** through level-based ADR requirements
- **Provide decision frameworks** that help users make informed choices
- **Recognize anti-patterns** and counter cargo-cult thinking
- **Manage risk** through crisis detection and escalation protocols
- **Enable measurement** through metrics and success criteria
- **Guide implementation** through sequenced roadmaps with rollback plans

This baseline demonstrates clear inadequacy of responses without the skill, justifying its creation.
