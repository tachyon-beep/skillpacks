# SDLC Engineering Command

## Description

Routes SDLC-related requests to the appropriate axiom-sdlc-engineering skill based on intent and CMMI maturity level.

## When to Use

Use `/using-sdlc-engineering` or `/sdlc-engineering` when you need guidance on:
- **Process adoption** (implementing CMMI on new or existing projects)
- **Requirements management** (tracking, traceability, change control)
- **Design and architecture** (ADRs, design reviews, coding standards, branching)
- **Quality assurance** (testing strategy, code reviews, verification, validation)
- **Governance and risk** (decision documentation, risk management)
- **Metrics and measurement** (what to measure, dashboards, statistical analysis)
- **Platform integration** (GitHub or Azure DevOps implementation patterns)

## Usage

```bash
/using-sdlc-engineering
```

This command loads the router skill, which will:
1. Detect your project's CMMI maturity level (from CLAUDE.md, conversation, or default to Level 3)
2. Analyze your request intent
3. Route you to the appropriate specialist skill:
   - `lifecycle-adoption` - Adopting CMMI on new or existing projects
   - `requirements-lifecycle` - Requirements development and management (RD + REQM)
   - `design-and-build` - Design, implementation, version control (TS + CM + PI)
   - `quality-assurance` - Testing, code reviews, validation (VER + VAL)
   - `governance-and-risk` - Decisions, ADRs, risk management (DAR + RSKM)
   - `quantitative-management` - Metrics, dashboards, SPC (MA + QPM + OPP)
   - `platform-integration` - GitHub/Azure DevOps setup

## Examples

```bash
# General SDLC question
/using-sdlc-engineering
"How do I set up requirements tracking for my project?"

# Specific process area
/using-sdlc-engineering
"We need to implement code reviews. What's the process?"

# Adoption on existing project
/using-sdlc-engineering
"We have an existing project with no process. How do we adopt CMMI Level 3?"

# Platform-specific
/using-sdlc-engineering
"How do I configure GitHub for CMMI Level 3 traceability?"
```

## CMMI Level Detection

The router automatically detects your target CMMI level:
- **From CLAUDE.md**: `CMMI Target Level: 2` (or 3, or 4)
- **From your message**: "We need CMMI Level 4 for FDA compliance"
- **Default**: Level 3 (if not specified)

**Levels**:
- **Level 2**: Minimal viable (1-5 person teams, ~5% overhead)
- **Level 3**: Organizational standards (5+ person teams, ~10-15% overhead, audit-ready)
- **Level 4**: Statistical process control (regulated industries, ~20-25% overhead)

## Related Skillpacks

For implementation details (not process), see:
- `axiom-python-engineering` - Python testing, linting, type checking
- `ordis-quality-engineering` - E2E testing, performance testing, chaos engineering
- `muna-technical-writer` - Technical documentation writing
- `lyra-ux-designer` - UX design reviews and accessibility

**Separation**: axiom-sdlc-engineering defines **WHAT** to do (process). Domain skillpacks define **HOW** to implement.
