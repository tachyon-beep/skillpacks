---
name: using-site-designer
description: "Use when building or maintaining static developer-tool, open-source, or documentation sites, including navigation, versioned URLs, code examples, theming, and build or deployment pipelines."
---

# Developer and Documentation Sites

Use the project's existing framework, design system, and publication conventions.
This pack contributes docs-site contracts and implementation recipes; ordinary
HTML/CSS questions can be answered directly.

## Workflow

1. Inspect the content, audiences, existing routes, framework version, build
   command, hosting configuration, and design tokens relevant to the request.
2. Make the important task easy to reach: installation, a working example,
   reference lookup, or the requested next step. Preserve established URLs or
   supply redirects when reorganizing published content.
3. Implement semantic HTML and responsive navigation with keyboard/focus behavior,
   readable code/table overflow, clear version selection, and theme contrast.
4. Reuse project tokens. New tokens should encode semantic roles and remain
   coherent across light/dark themes; verify contrast rather than infer it from
   a color space or palette name.
5. Run the actual build and relevant link/route checks. Inspect representative
   narrow/wide and light/dark pages in a browser when visual behavior changes.
   Exercise mobile navigation and code-copy controls when touched.
6. Keep local build, preview, deployment, and live acceptance evidence distinct.
   Follow the user's authorization and the platform's deployment controls.

## Optional recipes

All references are beside this file. Retrieve only the section needed, and check
framework/API details against the installed version or official documentation.
Examples are starting points, not prevalidated production components.

| Need | Reference |
|---|---|
| IA, navigation, versions, redirects, search | [documentation-sites.md](documentation-sites.md) |
| Code copy, language tabs, anchors | [code-block-patterns.md](code-block-patterns.md) |
| Docs-first framework, build, hosting | [static-site-tooling.md](static-site-tooling.md) |
| Token architecture and theme switching | [theming-and-tokens.md](theming-and-tokens.md) |
| Responsive docs layouts and overflow | [responsive-patterns.md](responsive-patterns.md) |
| Content-first project homepage | [landing-pages.md](landing-pages.md) |

Use native site/browser tooling when it fits the project. The `site-designer`
agent is optional for a bounded implementation task, not a required intermediary.
Application interaction research belongs to `lyra-ux-designer`; content accuracy
and executable documentation to `muna-technical-writer`.

## Completion evidence

Report the artifact or changed pages, build/check results, inspected states,
redirects if any, and material remaining limitations. A successful compiler run
alone does not verify navigation, accessibility, or deployed behavior.
