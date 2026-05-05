---
name: using-document-designer
description: Use when creating professional documents, PDF reports, Typst templates, Pandoc conversions, or any task where document visual design quality matters
---

# Using Document Designer

## Overview

This skill routes you to professional document design capabilities using Pandoc and Typst. Use it when the task requires more than content — when the document needs to **look** professional, with intentional typography, layout, and visual hierarchy.

**Core Principle**: Documents are communication artifacts. Content determines what you say; design determines whether anyone reads it.

## When to Use

Load this skill when:
- Creating PDF reports, proposals, whitepapers, or specifications
- Designing Typst templates for repeatable document production
- Converting Markdown/DOCX to professionally formatted PDF
- Building branded document systems (letterheads, report templates, slide decks)
- User mentions: "PDF", "Typst", "Pandoc", "template", "format", "layout", "typography"
- Any task where default formatting isn't good enough

**Don't use for**: Plain text content writing (use `muna-technical-writer`), web page design (use `lyra-ux-designer`), or data visualization.

## How This Pack Works

This pack provides a **document-designer agent** — a specialist subagent that combines graphic design sensibility with typesetting engineering. The agent handles:

| Capability | Details |
|------------|---------|
| **Typst templates** | Page layout, fonts, colors, headers/footers, cover pages, callouts |
| **Pandoc pipelines** | Format conversion, custom templates, Lua filters, citation processing |
| **Document types** | Reports, proposals, specs, resumes, brochures, slide decks |
| **Design patterns** | Accent bars, sidebars, pull quotes, data cards, section dividers |
| **Quality assurance** | Typography, table verification, font checks, accessibility |

## Toolchain Versions

This pack targets:

- **Typst >= 0.14.0** (tagged-PDF-by-default, PDF/UA-1, `image(alt:)`, `pdf.artifact`, fixed CMYK image export). Several reference sheets and the agent assume 0.14+ semantics. Older 0.11–0.13 docs work for most examples but lose accessibility features and need manual `context` syntax checks.
- **Pandoc >= 3.2** for the `--pdf-engine=typst` pipeline. Earlier versions have rougher Typst output and inconsistent admonition handling.

Pin these at the top of any new template, e.g. `// Requires Typst >= 0.14.0` in a comment, so future readers know what they're compiling against.

## Typst Universe Packages

The Typst package registry at <https://typst.app/universe> is the modern way to add non-trivial functionality. Use `#import "@preview/<name>:<version>"` with a pinned version — naked names break when packages drift. Verify the latest version on Universe before pinning.

Canonical packages by use case:

| Package | Use case | Notes |
|---------|----------|-------|
| `@preview/cetz` | Drawing, plots, charts, geometry | TikZ-inspired API; the workhorse for diagrams in Typst |
| `@preview/fletcher` | Node-and-arrow diagrams, commutative diagrams, flowcharts | Built on cetz |
| `@preview/touying` | Modern presentation slides — themes, animations, speaker notes, PDF export | Active and recommended for slides |
| `@preview/polylux` | Slides — older, still maintained, simpler model | Use touying for new decks |
| `@preview/valkyrie` | Type-safe argument validation for template authors | Useful when shipping a parameterised template |
| `@preview/glossarium` | Glossaries and acronym lists | |
| `@preview/physica` / `@preview/mitex` | Physics math notation; LaTeX-math passthrough | |

**Discovery pattern**: when the user asks for a capability that isn't a built-in (graph diagrams, citation styles, CV templates, slide themes), search Universe first. `typst init @preview/<template-name>` scaffolds a full project from a template package.

**Note on `tablex`**: Typst 0.11+ absorbed most tablex functionality into native `table()`. Prefer the built-in unless you specifically need tablex's `fit-spans` or arbitrary line extension.

## Reference Sheets

Reference sheets are located in the **same directory** as this SKILL.md file. Load them when the user's task matches:

| Reference | When to Load |
|-----------|-------------|
| [academic-papers.md](academic-papers.md) | Journal papers, conference submissions, theses — citations, author blocks, two-column layouts, theorem environments |
| [accessible-documents.md](accessible-documents.md) | Any public-facing document — tagged PDF, screen readers, contrast, alt text, PDF/UA compliance |
| [data-heavy-documents.md](data-heavy-documents.md) | Documents dominated by tables, charts, metrics — dashboard layouts, landscape pages, large table handling |
| [multilingual-documents.md](multilingual-documents.md) | RTL scripts, CJK typography, bilingual layouts, mixed-script font fallback chains |
| [print-production.md](print-production.md) | Documents for commercial printing — bleed, crop marks, binding margins, CMYK, image DPI |
| [standards-and-specifications.md](standards-and-specifications.md) | ISO/NIST/RFC-style specs, compliance docs, classification markings, conformance clauses |

To load: Read the reference sheet from this skill's directory and include its guidance when dispatching the document-designer agent. Multiple sheets can be combined (e.g., accessible + print-production for a printed public document).

## Quick Start

### Creating a New Document

Ask for what you need — the document-designer agent will be dispatched:

```
"Create a professional PDF report template for our quarterly reviews"
"I need a Typst template with our brand colors (#1a365d, #e53e3e) and Inter font"
"Convert this markdown spec to a polished PDF with table of contents"
```

### Scaffolding from a Template Package

```bash
# List candidate templates on https://typst.app/universe and pick one
typst init @preview/charged-ieee   # IEEE conference template
typst init @preview/modern-cv      # CV/resume template

# Then customise — most template packages expose a #show: <template-fn>.with(...) entrypoint
```

### Common Workflows

**From scratch (Pure Typst):**
Best when you need fine control over every layout element. The agent designs the template, populates content, and compiles to PDF.

**From existing content (Pandoc → Typst):**
Best when you have Markdown/DOCX content that needs professional formatting. The agent creates a custom Pandoc template and conversion pipeline.

**Reusable template design:**
Best when you need a template for repeated use. The agent designs a parameterized Typst template with metadata-driven customization.

## Relationship to Other Muna Skills

| Need | Use |
|------|-----|
| Writing content (ADRs, runbooks, READMs) | `muna-technical-writer` |
| Document set governance and consistency | `muna-wiki-management` |
| Audience-testing a document suite | `muna-panel-review` |
| **Making documents look professional** | **`muna-document-designer`** (this pack) |

The typical flow: write content with `muna-technical-writer`, then format it with `muna-document-designer`.
