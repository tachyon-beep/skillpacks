# Deliverable Options Menu

Present this menu using **AskUserQuestion tool** after workspace creation.

**Question:** "What deliverables do you need from this architecture analysis?"

---

## Option A: Full Analysis (Comprehensive)

**Recommended for complete understanding**

**Includes:**
- All standard documents (discovery, catalog, diagrams, report)
- Optional: Code quality assessment
- Optional: Architect handover report

**Complexity:** Medium-High (scales with codebase size)

**Best for:** New codebases, major refactoring planning, complete documentation needs

---

## Option B: Quick Overview (Essential)

**Fast turnaround for stakeholder presentations**

**Includes:**
- Discovery findings + high-level diagrams only (Context + Container)
- Executive summary with key findings
- Documented limitations (partial analysis)

**Complexity:** Low-Medium

**Best for:** Initial assessment, stakeholder presentations, time-constrained reviews

---

## Option C: Architect-Ready (Analysis + Improvement Planning)

**Complete analysis with improvement focus**

**Includes:**
- Full analysis (discovery, catalog, diagrams, report)
- Code quality assessment (mandatory for architect)
- Architect handover report with improvement recommendations
- Optional: Integrated architect consultation

**Complexity:** High (includes assessment + recommendations)

**Best for:** Planning refactoring, technical debt assessment, improvement roadmaps

---

## Option D: Custom Selection

**Choose specific documents**

**User selects from:**
- Discovery
- Catalog
- Diagrams (which levels?)
- Report
- Quality
- Handover

**Complexity:** Varies by selection

**Best for:** Updating existing documentation, focused analysis

---

## Option E: Full + Security

**Complete analysis with security surface mapping**

**Includes:**
- Full analysis (discovery, catalog, diagrams, report)
- Security surface mapping with trust boundaries and red flags
- Handoff package for ordis-security-architect

**Complexity:** High (requires security-focused review pass)

**Best for:** Security-sensitive systems, pre-security-review preparation

---

## Option F: Full + Quality

**Complete analysis with test infrastructure and dependencies**

**Includes:**
- Full analysis (discovery, catalog, diagrams, report)
- Test infrastructure analysis (pyramid, coverage gaps, flakiness)
- Dependency analysis (circular deps, layer violations, coupling metrics)
- Handoff package for ordis-quality-engineering

**Complexity:** High (requires quality-focused review pass)

**Best for:** Quality improvement initiatives, test strategy planning

---

## Option G: Comprehensive

**Everything (Full + Security + Quality + Dependencies)**

**Includes:**
- Full analysis (discovery, catalog, diagrams, report)
- Code quality assessment
- Security surface mapping
- Test infrastructure analysis
- Dependency analysis
- Architect handover with all findings

**Complexity:** Very High (full archaeological dig)

**Best for:** Major refactoring, system handover, complete documentation

---

## Documenting User's Choice

After selection, record in coordination plan:

```markdown
## Deliverables Selected: [Option A/B/C/D/E/F/G]

[If Option D, list specific selections]

**Rationale:** [Why user chose this option]
**Timeline target:** [If time-constrained]
**Stakeholder needs:** [If presentation-driven]
```

---

## Optional Steps by Deliverable Choice

| Option | Step 6.5 (Quality) | Step 6.6 (Security) | Step 6.7 (Test Infra) | Step 6.8 (Deps) | Step 11 (Handover) |
|--------|-------------------|--------------------|-----------------------|-----------------|-------------------|
| A | Optional | - | - | - | Optional |
| B | - | - | - | - | - |
| C | **Required** | - | - | - | **Required** |
| D | User choice | User choice | User choice | User choice | User choice |
| E | - | **Required** | - | - | - |
| F | - | - | **Required** | **Required** | - |
| G | **Required** | **Required** | **Required** | **Required** | **Required** |
