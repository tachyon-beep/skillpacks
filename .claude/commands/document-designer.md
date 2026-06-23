---
description: Use when creating professional documents, PDF reports, Typst templates, Pandoc conversions, or any task where document visual design quality matters - typography, layout, branded systems, format conversion
---

# Document Designer Routing

**Documents are communication artifacts. Content determines what you say; design determines whether anyone reads it.** When the default formatting isn't good enough — when the document needs to *look* professional with intentional typography, layout, and visual hierarchy — load this pack.

Use the `using-document-designer` skill from the `muna-document-designer` plugin. Content authority lives in `plugins/muna-document-designer/skills/using-document-designer/SKILL.md` — this wrapper is a thin pointer.

## When to Use

- Creating PDF reports, proposals, whitepapers, or specifications
- Designing Typst templates for repeatable document production
- Converting Markdown/DOCX to professionally formatted PDF
- Building branded document systems (letterheads, report templates, slide decks)
- User mentions: "PDF", "Typst", "Pandoc", "template", "format", "layout", "typography"

**Don't use for**: plain text content writing (use `/technical-writer`), web page design (use `/ux-designer`), or data visualization.

## Commands

- `/muna-document-designer:design-document` — produce a professionally designed document via Pandoc and/or Typst, selecting tooling appropriate to the document class and branding constraints

## Agents

- `document-designer` — specialist subagent combining graphic-design sensibility with typesetting engineering (Typst, Pandoc filters, typography, layout systems, branded templates). An executor: it builds and renders documents, surfaces design assumptions (e.g. ambiguous brand palette), and runs a MANDATORY build-and-verify loop on table layouts before delivering.

## Cross-references

- Document content authoring (prose, structure, register) → `/technical-writer`
- Static website design or developer-docs site → `/site-designer`
- UI/UX design (screen surfaces, accessibility) → `/ux-designer`
- Wiki/multi-document set governance → `/wiki-manager`
