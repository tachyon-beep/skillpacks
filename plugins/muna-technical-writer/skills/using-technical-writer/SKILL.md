---
name: using-technical-writer
description: "Use when documentation accuracy, executable procedures, institutional register, sensitive information, or source-to-document traceability needs deliberate review."
---

# Technical Documentation

Routine documentation can be written directly. Use this pack for source accuracy,
reader task completion, institutional artifacts, and repeatable verification.

## Workflow

1. Identify the reader's task, requested artifact, and authoritative sources.
   Infer clear intent; ask only when missing facts affect the result.
2. Read the relevant code/configuration, current docs and decisions. Do not invent
   why a decision was made, what an API supports, or who approved it.
3. Write the smallest document that lets that reader act. Separate instructions,
   reference facts, rationale, and unresolved proposals when their use differs.
4. Preserve exact consequential claims, names, units, interfaces, and approval
   status. Label new inference rather than silently inserting it into a summary.
5. Verify touched examples, commands, links/anchors, and prerequisites against the
   stated environment. Use a clean walkthrough when installation or first-use
   behavior is the claim. Record unrun checks and environment limits.
6. For broad structural edits, trace affected headings, cross-references, numbered
   clauses and terms before editing; inspect the final diff and downstream links.
   Review depth follows blast radius and consequence, not file line count.

## Optional references

References are beside this file. Retrieve only guidance that changes the result.

| Need | Reference |
|---|---|
| Executable examples, accuracy, task walkthrough | [documentation-testing.md](documentation-testing.md) |
| Technical/policy/government/public/executive/academic register | [editorial-registers.md](editorial-registers.md) |
| Incidents, timelines, post-mortems | [incident-response-documentation.md](incident-response-documentation.md) |
| Authorization evidence and SSP/SAR/POA&M artifacts | [operational-acceptance-documentation.md](operational-acceptance-documentation.md) |
| Sensitive examples and classification | [security-aware-documentation.md](security-aware-documentation.md) |
| Institutional change/governance artifacts | [itil-and-governance-documentation.md](itil-and-governance-documentation.md) |
| Document templates or notation conventions | [documentation-structure.md](documentation-structure.md), [clarity-and-style.md](clarity-and-style.md), [diagram-conventions.md](diagram-conventions.md) |

Use the sibling `fact-checking` skill when external claims need verification.
Security content can use `ordis-security-architect`; multi-document lineage uses
`muna-wiki-management`; Pandoc/Typst production uses `muna-document-designer`.
Do not require those packs for ordinary writing or block on their absence.

## Roles and output

`doc-critic`, `structure-analyst`, and `editorial-reviewer` are optional focused
reviewers. `complex-writer` and `complex-reviewer` help with broad documentation
edits when independent verification is useful. Engineering workflows own source
code changes. An explicit rewrite/translation request supplies authorization;
no second confirmation or paired review is automatically required.

Return the document or a concise prioritized review with cited evidence. For a
review, give the factual defect, reader consequence, and correction. Keep actual
source behavior, local verification, stakeholder approval, and publication status
separate. Do not turn every edit into a course, checklist, or audit report.
