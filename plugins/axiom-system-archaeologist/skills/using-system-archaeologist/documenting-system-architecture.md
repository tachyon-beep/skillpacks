
# Documenting System Architecture

## Purpose

Synthesize subsystem catalogs and architecture diagrams into final, stakeholder-ready architecture reports that serve multiple audiences through clear structure, comprehensive navigation, and actionable findings.

## When to Use

- Coordinator delegates final report generation from validated artifacts
- Have `02-subsystem-catalog.md` and `03-diagrams.md` as inputs
- Task specifies writing to `04-final-report.md`
- Need to produce executive-readable architecture documentation
- Output represents deliverable for stakeholders

## Core Principle: Synthesis Over Concatenation

**Good reports synthesize information into insights. Poor reports concatenate source documents.**

Your goal: Create a coherent narrative with extracted patterns, concerns, and recommendations - not a copy-paste of inputs.

## Synthesis Verification Checklist (MANDATORY)

**Before claiming synthesis complete, verify ALL of these transformations:**

**Pattern Synthesis (Required):**
- [ ] Identified patterns NOT explicitly named in source catalog
- [ ] Compared pattern implementation across 3+ subsystems
- [ ] Documented trade-offs discovered through comparison
- [ ] Found system-wide implications not mentioned in individual catalog entries

**Concern Analysis (Required):**
- [ ] Connected related concerns across subsystems (e.g., scalability issues)
- [ ] Identified cascading failure scenarios from multiple concerns
- [ ] Discovered concerns through pattern analysis (not just catalog extraction)
- [ ] Traced root causes across subsystem boundaries

**Insight Generation (Required):**
- [ ] Created new observations NOT present in source documents
- [ ] Identified emergent properties from subsystem interactions
- [ ] Found architectural strengths/weaknesses through synthesis
- [ ] Proposed recommendations addressing system-level issues

**Prohibited Actions:**
- [ ] Verified NO subsystem entries are copy-pasted verbatim
- [ ] Verified NO concerns are copy-pasted without analysis
- [ ] Verified NO diagram descriptions are copy-pasted without synthesis
- [ ] Verified executive summary contains insights, not generic statements

**If ANY verification fails, you have NOT synthesized. Return to synthesis phase.**

## Time Pressure Resistance

**If coordinator requests rushed completion:**

"We need this in 20 minutes - just pull together what we have."

**CORRECT RESPONSE:**
"Synthesis requires reading both source documents systematically and performing analysis to extract patterns and insights. This cannot be done in 20 minutes while maintaining quality. Minimum viable synthesis requires:

- 30 minutes: Read 02-subsystem-catalog.md and 03-diagrams.md
- 45 minutes: Extract patterns, identify concerns, perform analysis
- 45 minutes: Write synthesized report with cross-references
- Total: 2 hours

I can produce a DRAFT in 20 minutes by copying source documents and adding structure, but this would fail synthesis requirements and produce a professional-looking document with minimal actual analysis.

**Recommendation**: Allocate 2 hours for proper synthesis, OR accept that 20-minute output will be a structured draft requiring revision."

**PROHIBITED RESPONSE:**
"I'll pull together the documents quickly." ← This commits to concatenation, not synthesis.

## Document Structure

### Required Sections

**1. Front Matter**
- Document title
- Version number
- Analysis date
- Classification (if needed)

**2. Table of Contents**
- Multi-level hierarchy (H2, H3, H4)
- Anchor links to all major sections
- Quick navigation for readers

**3. Executive Summary (2-3 paragraphs)**
- High-level system overview
- Key architectural patterns
- Major concerns and confidence assessment
- Should be readable standalone by leadership

**4. System Overview**
- Purpose and scope
- Technology stack
- System context (external dependencies)

**5. Architecture Diagrams**
- Embed all diagrams from `03-diagrams.md`
- Add contextual analysis after each diagram
- Cross-reference to subsystem catalog

**6. Subsystem Catalog**
- One detailed entry per subsystem
- Synthesize from `02-subsystem-catalog.md` (don't just copy)
- Add cross-references to diagrams and findings

**7. Key Findings**
- **Architectural Patterns**: Identified across subsystems
- **Technical Concerns**: Extracted from catalog concerns
- **Recommendations**: Actionable next steps with priorities

**8. Appendices**
- **Methodology**: How analysis was performed
- **Confidence Levels**: Rationale for confidence ratings
- **Assumptions & Limitations**: What you inferred, what's missing

## Synthesis Strategies

### Pattern Identification

**Look across subsystems for recurring patterns:**

From catalog observations:
- Subsystem A: "Dependency injection for testability"
- Subsystem B: "All external services injected"
- Subsystem C: "Injected dependencies for testing"

**Synthesize into pattern:**
```markdown
### Dependency Injection Pattern

**Observed in**: Authentication Service, API Gateway, User Service

**Description**: External dependencies are injected rather than directly instantiated, enabling test isolation and loose coupling.

**Benefits**:
- Testability: Mock dependencies in unit tests
- Flexibility: Swap implementations without code changes
- Loose coupling: Services depend on interfaces, not concrete implementations

**Trade-offs**:
- Initial complexity: Requires dependency wiring infrastructure
- Runtime overhead: Minimal (dependency resolution at startup)
```

### Concern Extraction

**Find concerns buried in catalog entries:**

Catalog entries:
- API Gateway: "Rate limiter uses in-memory storage (doesn't scale horizontally)"
- Database Layer: "Connection pool max size hardcoded (should be configurable)"
- Data Service: "Large analytics queries can cause database load spikes"

**Synthesize into findings:**
```markdown
## Technical Concerns

### 1. Rate Limiter Scalability Issue

**Severity**: Medium
**Affected Subsystem**: [API Gateway](#api-gateway)

**Issue**: In-memory rate limiting prevents horizontal scaling. If multiple gateway instances run, each maintains separate counters, allowing clients to exceed intended limits by distributing requests across instances.

**Impact**:
- Cannot scale gateway horizontally without distributed rate limiting
- Potential for rate limit bypass under load balancing
- Inconsistent rate limit enforcement

**Remediation**:
1. **Immediate** (next sprint): Document limitation, add monitoring alerts
2. **Short-term** (next quarter): Migrate to Redis-backed rate limiter
3. **Validation**: Test rate limiting with multiple gateway instances

**Priority**: High (blocks horizontal scaling)
```

### Recommendation Prioritization

**Priority recommendations using severity scoring + impact assessment + timeline buckets:**

#### Severity Scoring (for each concern/recommendation)

**Critical:**
- Blocks deployment or core functionality
- Security vulnerability (data exposure, injection, auth bypass)
- Data corruption or loss risk
- Service outage potential
- Examples: SQL injection, hardcoded credentials, unhandled critical exceptions

**High:**
- Significant maintainability impact
- High effort to modify or extend
- Frequent source of bugs
- Performance degradation under load
- Examples: God objects, extreme duplication, shotgun surgery, N+1 queries

**Medium:**
- Moderate maintainability concern
- Refactoring beneficial but not urgent
- Technical debt accumulation
- Examples: Long functions, missing documentation, inconsistent error handling

**Low:**
- Minor quality improvement
- Cosmetic or style issues
- Nice-to-have enhancements
- Examples: Magic numbers, verbose naming, minor duplication

#### Impact Assessment Matrix

Use 2-dimensional scoring: **Severity × Frequency**

| Severity | High Frequency | Medium Frequency | Low Frequency |
|----------|----------------|------------------|---------------|
| **Critical** | **P1** - Fix immediately | **P1** - Fix immediately | **P2** - Fix ASAP |
| **High** | **P2** - Fix ASAP | **P2** - Fix ASAP | **P3** - Plan for sprint |
| **Medium** | **P3** - Plan for sprint | **P4** - Backlog | **P4** - Backlog |
| **Low** | **P4** - Backlog | **P4** - Backlog | **P5** - Optional |

**Frequency assessment:**
- **High:** Affects core user workflows, used constantly, blocking development
- **Medium:** Affects some workflows, occasional impact, periodic friction
- **Low:** Edge case, rarely encountered, minimal operational impact

#### Timeline Buckets

**Immediate (This Week / Next Sprint):**
- P1 priorities (Critical issues regardless of frequency)
- Security vulnerabilities
- Blocking deployment or development
- Quick wins (high impact, low effort)

**Short-Term (1-3 Months / Next Quarter):**
- P2 priorities (High severity or critical+low frequency)
- Significant maintainability improvements
- Performance optimizations
- Breaking circular dependencies

**Medium-Term (3-6 Months):**
- P3 priorities (Medium severity+high frequency or high+low)
- Architectural refactoring
- Technical debt paydown
- System-wide improvements

**Long-Term (6-12+ Months):**
- P4-P5 priorities (Low severity, backlog items)
- Nice-to-have improvements
- Experimental optimizations
- Deferred enhancements

#### Prioritized Recommendation Format

```markdown
## Recommendations

### Immediate (This Week / Next Sprint) - P1

**1. Fix Rate Limiter Scalability Vulnerability**
- **Severity:** Critical (blocks horizontal scaling)
- **Frequency:** High (affects all gateway scaling attempts)
- **Priority:** P1
- **Impact:** Cannot scale API gateway, potential rate limit bypass
- **Effort:** Medium (2-3 days migration to Redis)
- **Action:**
  1. Document current limitation in ops runbook (Day 1)
  2. Add monitoring for rate limit violations (Day 1)
  3. Migrate to Redis-backed rate limiter (Days 2-3)
  4. Validate with load testing (Day 3)

**2. Remove Hardcoded Database Credentials**
- **Severity:** Critical (security vulnerability)
- **Frequency:** Low (only affects DB config rotation)
- **Priority:** P1
- **Impact:** Credentials exposed in source control, rotation requires code deployment
- **Effort:** Low (< 1 day)
- **Action:**
  1. Move credentials to environment variables
  2. Update deployment configs
  3. Rotate compromised credentials

### Short-Term (1-3 Months / Next Quarter) - P2

**3. Extract Common Validation Framework**
- **Severity:** High (high duplication, shotgun surgery for validation changes)
- **Frequency:** High (every new API endpoint)
- **Priority:** P2
- **Impact:** 3 duplicate validation implementations, 15% code duplication
- **Effort:** Medium (1 week to extract + migrate)
- **Action:**
  1. Design validation framework API (2 days)
  2. Implement core framework (2 days)
  3. Migrate existing validators (2 days)
  4. Document validation patterns (1 day)

**4. Externalize Database Pool Configuration**
- **Severity:** High (hardcoded limits cause connection exhaustion)
- **Frequency:** Medium (impacts under load spikes)
- **Priority:** P2
- **Impact:** Connection pool exhaustion during traffic spikes
- **Effort:** Low (2 days)
- **Action:**
  1. Move pool config to environment variables
  2. Add runtime pool size adjustment
  3. Document tuning guidelines

### Medium-Term (3-6 Months) - P3

**5. Break User ↔ Notification Circular Dependency**
- **Severity:** Medium (architectural coupling)
- **Frequency:** Medium (affects both subsystem modifications)
- **Priority:** P3
- **Impact:** Difficult to modify either service independently
- **Effort:** High (2-3 weeks, requires event bus introduction)
- **Action:**
  1. Design event bus architecture (1 week)
  2. Implement notification via events (1 week)
  3. Migrate user service to publish events (3 days)
  4. Remove direct dependency (2 days)

**6. Add Docstrings to Public API (27% → 90% coverage)**
- **Severity:** Medium (maintainability concern)
- **Frequency:** Medium (affects onboarding, API understanding)
- **Priority:** P3
- **Impact:** Poor API discoverability, onboarding friction
- **Effort:** Medium (2-3 weeks distributed work)
- **Action:**
  1. Establish docstring standard (1 day)
  2. Document public APIs in batches (2 weeks)
  3. Add pre-commit hook to enforce (1 day)

### Long-Term (6-12+ Months) - P4-P5

**7. Evaluate Circuit Breaker Effectiveness**
- **Severity:** Low (optimization opportunity)
- **Frequency:** Low (affects only failure scenarios)
- **Priority:** P4
- **Impact:** Potential false positives, could improve resilience
- **Effort:** Medium (1 week testing + analysis)
- **Action:** Load testing + monitoring analysis when capacity allows

**8. Extract Magic Numbers to Configuration**
- **Severity:** Low (code quality improvement)
- **Frequency:** Low (rarely needs changing)
- **Priority:** P5
- **Impact:** Minor maintainability improvement
- **Effort:** Low (2-3 days)
- **Action:** Backlog item, tackle during related refactoring
```

#### Priority Summary Table

Include summary table for quick scanning:

```markdown
## Priority Summary

| Priority | Count | Severity Distribution | Total Effort |
|----------|-------|----------------------|--------------|
| **P1** (Immediate) | 2 | Critical: 2 | 4 days |
| **P2** (Short-term) | 2 | High: 2 | 2.5 weeks |
| **P3** (Medium-term) | 2 | Medium: 2 | 5-6 weeks |
| **P4-P5** (Long-term) | 2 | Low: 2 | 2 weeks |
| **Total** | 8 | - | ~10 weeks |

**Recommended sprint allocation:**
- Sprint 1: P1 items (4 days) + start P2.3 validation framework
- Sprint 2: Complete P2.3 + P2.4 database pool config
- Quarter 2: P3 items (architectural improvements)
- Backlog: P4-P5 items (opportunistic improvements)
```

## Cross-Referencing Strategy

### Bidirectional Links

**Subsystem → Diagram:**
```markdown
## Authentication Service

[...subsystem details...]

**Component Architecture**: See [Authentication Service Components](#auth-service-components) diagram

**Dependencies**: [API Gateway](#api-gateway), [Database Layer](#database-layer)
```

**Diagram → Subsystem:**
```markdown
### Authentication Service Components

[...diagram...]

**Description**: This component diagram shows internal structure of the Authentication Service. For additional operational details, see [Authentication Service](#authentication-service) in the subsystem catalog.
```

**Finding → Subsystem:**
```markdown
### Rate Limiter Scalability Issue

**Affected Subsystem**: [API Gateway](#api-gateway)

[...concern details...]
```

### Navigation Patterns

**Table of contents with anchor links:**
```markdown
## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
   - [Purpose and Scope](#purpose-and-scope)
   - [Technology Stack](#technology-stack)
3. [Architecture Diagrams](#architecture-diagrams)
   - [Level 1: Context](#level-1-context)
   - [Level 2: Container](#level-2-container)
```

## Multi-Audience Considerations

### Executive Audience

**What they need:**
- Executive summary ONLY (should be self-contained)
- High-level patterns and risks
- Business impact of concerns
- Clear recommendations with timelines

**Document design:**
- Put executive summary first
- Make it readable standalone (no forward references)
- Focus on "why this matters" over "how it works"

### Architect Audience

**What they need:**
- System overview + architecture diagrams + key findings
- Pattern analysis with trade-offs
- Dependency relationships
- Design decisions and rationale

**Document design:**
- System overview explains context
- Diagrams show structure at multiple levels
- Findings synthesize patterns and concerns
- Cross-references enable non-linear reading

### Engineer Audience

**What they need:**
- Subsystem catalog with technical details
- Component diagrams showing internal structure
- Technology stack specifics
- File references and entry points

**Document design:**
- Detailed subsystem catalog
- Component-level diagrams
- Technology stack section with versions/frameworks
- Code/file references where available

### Operations Audience

**What they need:**
- Technical concerns with remediation
- Dependency mapping
- Confidence levels (what's validated vs assumed)
- Recommendations with priorities

**Document design:**
- Technical concerns section up front
- Clear remediation steps
- Appendix with assumptions/limitations
- Prioritized recommendations

## Optional Enhancements

### Visual Aids

**Subsystem Quick Reference Table:**
```markdown
## Appendix D: Subsystem Quick Reference

| Subsystem | Location | Confidence | Key Concerns | Dependencies |
|-----------|----------|------------|--------------|--------------|
| API Gateway | /src/gateway/ | High | Rate limiter scalability | Auth, User, Data, Logging |
| Auth Service | /src/services/auth/ | High | None | Database, Cache, Logging |
| User Service | /src/services/users/ | High | None | Database, Cache, Notification |
```

**Pattern Summary Matrix:**
```markdown
## Architectural Patterns Summary

| Pattern | Subsystems Using | Benefits | Trade-offs |
|---------|------------------|----------|------------|
| Dependency Injection | Auth, Gateway, User | Testability, flexibility | Initial complexity |
| Repository Pattern | User, Data | Data access abstraction | Extra layer |
| Circuit Breaker | Gateway | Fault isolation | False positives |
```

### Reading Guide

```markdown
## How to Read This Document

**For Executives** (5 minutes):
- Read [Executive Summary](#executive-summary) only
- Optionally skim [Recommendations](#recommendations)

**For Architects** (30 minutes):
- Read [Executive Summary](#executive-summary)
- Read [System Overview](#system-overview)
- Review [Architecture Diagrams](#architecture-diagrams)
- Read [Key Findings](#key-findings)

**For Engineers** (1 hour):
- Read [System Overview](#system-overview)
- Study [Architecture Diagrams](#architecture-diagrams) (all levels)
- Read [Subsystem Catalog](#subsystem-catalog) for relevant services
- Review [Technical Concerns](#technical-concerns)

**For Operations** (45 minutes):
- Read [Executive Summary](#executive-summary)
- Study [Technical Concerns](#technical-concerns)
- Review [Recommendations](#recommendations)
- Read [Appendix C: Assumptions and Limitations](#appendix-c-assumptions-and-limitations)
```

### Glossary

```markdown
## Appendix E: Glossary

**Circuit Breaker**: Fault tolerance pattern that prevents cascading failures by temporarily blocking requests to failing services.

**Dependency Injection**: Design pattern where dependencies are provided to components rather than constructed internally, enabling testability and loose coupling.

**Repository Pattern**: Data access abstraction that separates business logic from data persistence concerns.

**Optimistic Locking**: Concurrency control technique assuming conflicts are rare, using version checks rather than locks.
```

## Success Criteria

**You succeeded when:**
- Executive summary (2-3 paragraphs) distills key information
- Table of contents provides multi-level navigation
- Cross-references (30+) enable non-linear reading
- Patterns synthesized (not just listed from catalog)
- Concerns extracted and prioritized
- Recommendations actionable with timelines
- Diagrams integrated with contextual analysis
- Appendices document methodology, confidence, assumptions
- Professional structure (document metadata, clear hierarchy)
- Written to 04-final-report.md

**You failed when:**
- Simple concatenation of source documents
- No executive summary or it requires reading full document
- Missing table of contents
- No cross-references between sections
- Patterns just copied from catalog (not synthesized)
- Concerns buried without extraction
- Recommendations vague or unprioritized
- Diagrams pasted without context
- Missing appendices
- Subsystem entries use >70% wording from source catalog
- Executive summary could apply to any microservices system (generic)
- No new insights beyond what's in source documents
- Synthesis verification checklist not completed
- Time pressure used to justify shallow report

## Best Practices from Baseline Testing

### What Works

✅ **Comprehensive synthesis** - Identify patterns, extract concerns, create narrative
✅ **Professional structure** - Document metadata, TOC, clear hierarchy, appendices
✅ **Multi-level navigation** - 20+ TOC entries, 40+ cross-references
✅ **Executive summary** - Self-contained 2-3 paragraph distillation
✅ **Actionable findings** - Concerns with severity/impact/remediation, recommendations with timelines
✅ **Transparency** - Confidence levels, assumptions, limitations documented
✅ **Diagram integration** - Embedded with contextual analysis and cross-refs
✅ **Multi-audience** - Executive summary + technical depth + appendices

### Synthesis Patterns

**Pattern identification:**
- Look across multiple subsystems for recurring themes
- Group by pattern name (e.g., "Repository Pattern")
- Document which subsystems use it
- Explain benefits and trade-offs

**Concern extraction:**
- Find concerns in subsystem catalog entries
- Elevate to Key Findings section
- Add severity, impact, remediation
- Prioritize by timeline (immediate/short/long)

**Recommendation structure:**
- Group by timeline
- Specific actions (not vague suggestions)
- Validation steps
- Priority indicators

## Integration with Workflow

This skill is typically invoked as:

1. **Coordinator** completes and validates subsystem catalog
2. **Coordinator** completes and validates architecture diagrams
3. **Coordinator** writes task specification for final report
4. **YOU** read both source documents systematically
5. **YOU** synthesize patterns, extract concerns, create recommendations
6. **YOU** build professional report structure with navigation
7. **YOU** write to 04-final-report.md
8. **Validator** (optional) checks for synthesis quality, navigation, completeness

**Your role:** Transform analysis artifacts into stakeholder-ready documentation through synthesis, organization, and professional presentation.
