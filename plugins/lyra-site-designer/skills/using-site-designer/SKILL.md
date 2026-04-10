---
name: using-site-designer
description: Use when designing or implementing static websites for developer tools, open-source projects, or technical documentation — information architecture, HTML/CSS, design tokens, developer UX patterns
---

# Using Site Designer

## Overview

This skill routes you to static site design and implementation capabilities. Use it when building websites for developer tools, open-source projects, or documentation-heavy sites where content quality and developer UX matter more than marketing aesthetics.

**Core Principle**: Developer tool sites succeed on substance — clear docs, working examples, and fast navigation. Not hero images and testimonials.

## When to Use

Load this skill when:
- Planning site structure or information architecture for a project
- Implementing HTML/CSS for a developer-facing site
- Setting up design tokens and theming (dark/light mode)
- Building documentation site layouts with sidebars, breadcrumbs, search
- Choosing and configuring a static site generator (Hugo, Astro, Eleventy)
- User mentions: "website", "homepage", "landing page", "docs site", "static site"

**Don't use for**: General UI/UX design principles (use `lyra-ux-designer`), document formatting/PDF output (use `muna-document-designer`), or web API backend development (use `axiom-web-backend`).

## How This Pack Works

This pack provides a **site-designer agent** — a specialist subagent that combines information architecture, CSS craftsmanship, and developer UX expertise. The agent handles:

| Capability | Details |
|------------|---------|
| **Information architecture** | Site maps, navigation strategy, audience-specific paths |
| **HTML/CSS implementation** | Semantic markup, CSS custom properties, responsive layouts |
| **Design tokens** | Color palettes, typography, spacing scales, dark/light mode |
| **Developer UX** | Code blocks with copy, anchor links, version selectors, search |
| **Static site tooling** | Hugo, Astro, Eleventy configuration and templating |
| **Accessibility** | WCAG AA compliance, keyboard nav, screen reader support |

## Reference Sheets

Reference sheets are located in the **same directory** as this SKILL.md file. Load them when the user's task matches:

| Reference | When to Load |
|-----------|-------------|
| [code-block-patterns.md](code-block-patterns.md) | Code blocks with copy-to-clipboard, syntax highlighting, language tabs, terminal styling, heading anchors |
| [documentation-sites.md](documentation-sites.md) | Multi-page documentation with sidebar navigation, versioning, search, API reference layouts |
| [landing-pages.md](landing-pages.md) | Project homepages and landing pages — content-first design for open-source and developer tools |
| [responsive-patterns.md](responsive-patterns.md) | Making layouts work across screen sizes — sidebar collapse, mobile nav, responsive tables and code blocks |
| [static-site-tooling.md](static-site-tooling.md) | Choosing and configuring Hugo, Astro, Eleventy, or plain HTML — build pipelines and deployment |
| [theming-and-tokens.md](theming-and-tokens.md) | Design token systems, dark/light mode, CSS custom properties, color palette derivation |

To load: Read the reference sheet from this skill's directory and include its guidance when dispatching the site-designer agent. Multiple sheets can be combined (e.g., documentation-sites + theming-and-tokens + responsive-patterns for a themed responsive docs site).

## Quick Start

### Designing a New Site

Ask for what you need — the site-designer agent will be dispatched:

```
"Plan the site structure for our open-source CLI tool"
"Create a docs site layout with sidebar navigation"
"Set up design tokens with dark mode for our project site"
"Build a landing page for our Python library"
```

### Common Workflows

**New project site:**
Start with information architecture (who are the audiences, what content exists), then design tokens, then implement pages incrementally.

**Documentation site:**
Load the documentation-sites reference sheet. Focus on navigation patterns, content hierarchy, and cross-linking before any visual design.

**Adding theming:**
Load the theming-and-tokens reference sheet. Define CSS custom properties, implement dark/light toggle, derive the full palette from brand colors.

## Relationship to Other Skills

| Need | Use |
|------|-----|
| UI/UX design principles and patterns | `lyra-ux-designer` |
| Professional PDF/document formatting | `muna-document-designer` |
| Writing documentation content | `muna-technical-writer` |
| Web API backend development | `axiom-web-backend` |
| **Building the website itself** | **`lyra-site-designer`** (this pack) |

The typical flow: design with `lyra-ux-designer`, write content with `muna-technical-writer`, build the site with `lyra-site-designer`.
