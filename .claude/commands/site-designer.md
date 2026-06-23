---
description: Use when designing or implementing static websites for developer tools, open-source projects, or technical documentation - information architecture, HTML/CSS, design tokens, developer UX patterns, docs-first frameworks (Hugo, Astro, Starlight, VitePress, Docusaurus)
---

# Site Designer Routing

**Developer tool sites succeed on substance — clear docs, working examples, and fast navigation. Not hero images and testimonials.** When building a website for a developer tool, an open-source project, or a documentation-heavy site where content quality and developer UX matter more than marketing aesthetics, load this pack.

Use the `using-site-designer` skill from the `lyra-site-designer` plugin. Content authority lives in `plugins/lyra-site-designer/skills/using-site-designer/SKILL.md` — this wrapper is a thin pointer.

## When to Use

- Planning site structure or information architecture for a project
- Implementing HTML/CSS for a developer-facing site
- Setting up design tokens and theming (dark/light mode)
- Building documentation site layouts with sidebars, breadcrumbs, search
- Choosing and configuring a static site generator or docs-first framework (Hugo, Astro, Eleventy, Starlight, VitePress, Docusaurus 3)
- User mentions: "website", "homepage", "landing page", "docs site", "static site"

**Don't use for**: general UI/UX design principles (use `/ux-designer`), document formatting/PDF output (use `/document-designer`), or web API backend development (use `/web-backend`).

## Commands

- `/lyra-site-designer:design-site` — design or implement a static website for a developer tool or documentation site (IA, page layouts, design tokens, HTML/CSS implementation)

## Agents

- `site-designer` — specialist subagent combining information architecture, CSS craftsmanship, and developer UX expertise (design tokens, theming systems, docs-first frameworks, content strategy for developer audiences). A producer agent that generates IA, CSS, and layout directly against the pack's quality checklists.

## Cross-references

- General UI/UX design across surfaces (visual, IA, interaction, accessibility) → `/ux-designer`
- Document/PDF visual design (Typst, Pandoc, branded templates) → `/document-designer`
- Web backend / API design → `/web-backend`
- Documentation content authoring (READMEs, ADRs, runbooks, register translation) → `/technical-writer`
