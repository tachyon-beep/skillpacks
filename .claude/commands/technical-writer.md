---
description: Use when writing or improving documentation - ADRs, API reference, runbooks, READMEs, architecture docs, security/compliance docs, post-mortems, register review or translation across institutional voices (technical/policy/government/public-facing/executive/academic), fact-checking research papers with dual web-search verification, or surgical edits on large (>=2000 line) files where blast radius matters
---

# Technical Writer Routing

**Different document types and audiences need different craft. Match the situation to the sheet, load only what is needed. Audience (who reads) and register (how text operates) are orthogonal - a document has both. For security CONTENT (threat models, controls) use `/security-architect`; this pack covers documentation STRUCTURE, CLARITY, and REGISTER.**

Use the `using-technical-writer` skill from the `muna-technical-writer` plugin to route to the right specialist sheet. Content authority lives in `plugins/muna-technical-writer/skills/using-technical-writer/SKILL.md` - this wrapper is a thin pointer.

## Sheets (under `using-technical-writer/`)

### Core (universal documentation craft)
- **documentation-structure** - ADR (Context/Decision/Consequences), API reference patterns, runbook templates, README structure, architecture docs
- **clarity-and-style** - active voice, concrete examples, progressive disclosure, audience adaptation, step-by-step prose
- **diagram-conventions** - C4 model, system diagrams, data-flow, consistent notation
- **documentation-testing** - verify accuracy, completeness, findability, validate examples and links

### Register (institutional voice)
- **editorial-registers** - six registers (technical / policy / government / public-facing / executive / academic), register relationships, custom-register template

### Extension (specialised contexts)
- **security-aware-documentation** - sanitising sensitive examples, classification marking, redaction discipline
- **incident-response-documentation** - post-mortem templates, RCA structure, timeline docs
- **itil-and-governance-documentation** - ITIL processes, change management, governance frameworks
- **operational-acceptance-documentation** - SSP, SAR, POA&M, government authorization artifacts

## Commands

- `/muna-technical-writer:write-docs` - write documentation using proven patterns (ADRs, API reference, runbooks, READMEs)
- `/muna-technical-writer:create-adr` - create an Architecture Decision Record documenting a technology or design choice
- `/muna-technical-writer:review-docs` - review documentation for clarity, structure, completeness, audience fit
- `/muna-technical-writer:review-style` - review document style against a target writing register (auto-detects register when none specified)
- `/fact-check <file-paths>` - dual-verified research paper fact-checking via the leaf `fact-checking` skill (expensive; invoke explicitly)

## Agents

- `complex-writer` (opus) - surgical edits on large (>=2000 line) files; survey -> pre-work assessment (complexity/blast-radius/risk/mitigations) -> caller confirms -> plan -> edit
- `complex-reviewer` (opus) - independent verification of a complex-writer edit; zero-hit checks, structural integrity, completeness
- `editorial-reviewer` (sonnet) - register detection / review / translation across the six registers
- `doc-critic` (sonnet) - broad documentation review (clarity, structure, completeness, audience fit)
- `structure-analyst` (haiku) - mechanical structural analysis of an existing document set

All agents follow the SME Agent Protocol with Confidence / Risk / Information Gaps / Caveats sections.

## Routing quick reference

| Task                                            | Load |
|-------------------------------------------------|------|
| "Why did we choose X?"                          | documentation-structure (ADR) |
| "Document this API"                             | documentation-structure + clarity-and-style |
| "Deployment runbook"                            | documentation-structure + clarity-and-style |
| "README" (complex)                              | documentation-structure |
| "Security docs"                                 | `/security-architect` (content) + documentation-structure + clarity-and-style |
| "Compliance package" (SSP/SAR/POA&M)            | operational-acceptance-documentation + documentation-structure |
| "Post-mortem"                                   | incident-response-documentation + documentation-structure + clarity-and-style |
| "Is this in the right register?"                | editorial-registers + editorial-reviewer agent (review mode) |
| "Rewrite for [register]"                        | editorial-registers + editorial-reviewer agent (translate mode) |
| "Fact-check this paper"                         | `/fact-check <paths>` |
| "Edit / rename across this large file"          | complex-writer + complex-reviewer agents |

## Cross-references

- Security CONTENT (threat models, controls, compliance mapping) -> `/security-architect`
- Multi-document derivation, propagation, audit -> `/wiki-manager`
- Document visual design, PDF, Typst, Pandoc -> `/document-designer`
- Code-architecture critique that the docs describe -> `/system-architect`
