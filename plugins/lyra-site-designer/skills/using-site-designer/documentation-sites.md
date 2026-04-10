# Documentation Site Design

Reference sheet for multi-page documentation sites with structured navigation, versioning, and search. Load this when building sites whose primary purpose is technical documentation — API references, user guides, specification browsers, or knowledge bases.

## When to Use This Reference

- Building a documentation site with 10+ pages and sidebar navigation
- Implementing versioned documentation (multiple releases)
- Designing API reference layouts
- Creating tutorial/guide navigation flows
- Any site where "finding the right page fast" is the core UX requirement

## Information Architecture Patterns

### The Four Documentation Types

Structure content around the Diátaxis framework — four types that serve different user needs:

| Type | User Need | Format | Example |
|------|----------|--------|---------|
| **Tutorials** | Learning by doing | Step-by-step, ordered | "Build your first app" |
| **How-to guides** | Solving a specific problem | Goal-oriented, practical | "How to configure auth" |
| **Reference** | Looking up exact details | Precise, complete, dry | "API method signatures" |
| **Explanation** | Understanding concepts | Discursive, contextual | "How the type system works" |

Don't mix these types on a single page. A tutorial that stops to explain theory loses the reader. A reference page that tries to teach loses the expert.

### Navigation Hierarchy

```
Docs/
├── Getting Started/          # Tutorial — sequential, ordered
│   ├── Installation
│   ├── Quick Start
│   └── First Project
├── Guides/                   # How-to — independent, task-oriented
│   ├── Configuration
│   ├── Authentication
│   ├── Deployment
│   └── Migration
├── Reference/                # Reference — exhaustive, alphabetical/logical
│   ├── CLI Commands
│   ├── Configuration Options
│   ├── API Reference
│   └── Error Codes
└── Concepts/                 # Explanation — contextual, interconnected
    ├── Architecture
    ├── Security Model
    └── Design Decisions
```

### Sidebar Design

- **Collapsible sections**: Group pages into sections, collapse by default for large doc sets
- **Current page indicator**: Bold or highlighted text + accent bar on active page
- **Section ordering**: Tutorials first (new users), then guides, then reference, then concepts
- **Depth limit**: Max 3 levels in sidebar (section → page → subpage). Deeper nesting belongs in on-page ToC
- **Mobile**: Sidebar becomes an off-canvas drawer or hamburger menu

### On-Page Table of Contents

For long pages (> 3 screens), add a right-side "On this page" ToC:

```html
<aside class="page-toc" aria-label="On this page">
  <h2>On this page</h2>
  <nav>
    <ul>
      <li><a href="#section-1">Section 1</a></li>
      <li><a href="#section-2">Section 2</a>
        <ul>
          <li><a href="#subsection-2-1">Subsection 2.1</a></li>
        </ul>
      </li>
    </ul>
  </nav>
</aside>
```

Use `IntersectionObserver` to highlight the current section as the user scrolls.

## Versioned Documentation

### Version Selector

Place a version selector in the header or sidebar. Pattern:

```html
<select class="version-select" aria-label="Documentation version">
  <option value="/docs/v2/" selected>v2.0 (latest)</option>
  <option value="/docs/v1/">v1.x</option>
  <option value="/docs/v0/">v0.x (legacy)</option>
</select>
```

### URL Strategy

- **Versioned**: `/docs/v2/getting-started/` — each version is a separate directory
- **Latest alias**: `/docs/latest/` → redirects to current version
- **Unversioned for stable content**: `/blog/`, `/community/` don't need versioning

### Deprecation Notices

For old versions, add a banner at the top of every page:

```html
<div class="version-banner version-banner--old" role="alert">
  You're viewing docs for v1.x.
  <a href="/docs/latest/">Switch to the latest version →</a>
</div>
```

## Search

### Static Search (No Server)

For static sites, use client-side search. Options:

| Tool | Index Size | Features | Best For |
|------|-----------|----------|----------|
| **Pagefind** | Tiny (~1% of content) | Fast, multilingual, zero-config | Most sites |
| **Lunr.js** | Moderate | Customizable, mature | Smaller sites |
| **FlexSearch** | Small | Very fast, fuzzy matching | Performance-critical |
| **Algolia DocSearch** | External | Powerful, free for OSS | Large/popular projects |

Pagefind is the default recommendation — it indexes at build time, produces tiny bundles, and requires zero JavaScript beyond the search UI.

### Search UX

- `Ctrl+K` / `Cmd+K` keyboard shortcut to open search
- Results show page title, section heading, and content excerpt
- Results are grouped by section (Guides, Reference, etc.)
- Search input is always accessible from the header

## API Reference Layout

### Method/Function Entries

```
┌─────────────────────────────────────────────┐
│ get_user(user_id, include_profile=False)     │
│                                              │
│ Retrieve a user by their unique identifier.  │
│                                              │
│ Parameters:                                  │
│   user_id (str) — The user's unique ID       │
│   include_profile (bool) — Whether to        │
│     include the full profile. Default: False  │
│                                              │
│ Returns:                                     │
│   User — The user object                     │
│                                              │
│ Raises:                                      │
│   NotFoundError — If user_id doesn't exist   │
│                                              │
│ Example:                                     │
│   ┌──────────────────────────────────┐       │
│   │ user = client.get_user("abc123") │       │
│   └──────────────────────────────────┘       │
│                                              │
│ Since: v1.2.0                                │
└─────────────────────────────────────────────┘
```

### Styling API Entries

```css
.api-entry {
  border-left: 3px solid var(--color-accent);
  padding: var(--space-3);
  margin-bottom: var(--space-4);
}

.api-signature {
  font-family: var(--font-code);
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--color-primary);
}

.api-params dt {
  font-family: var(--font-code);
  font-weight: 600;
}

.api-params dd {
  margin-left: var(--space-4);
  margin-bottom: var(--space-2);
}
```

## Previous/Next Navigation

At the bottom of every documentation page, show previous and next links:

```html
<nav class="page-nav" aria-label="Page navigation">
  <a href="/docs/prev-page/" class="page-nav__prev">
    <span class="page-nav__label">Previous</span>
    <span class="page-nav__title">Previous Page Title</span>
  </a>
  <a href="/docs/next-page/" class="page-nav__next">
    <span class="page-nav__label">Next</span>
    <span class="page-nav__title">Next Page Title</span>
  </a>
</nav>
```

## Quality Additions for Documentation Sites

In addition to the base quality checklist, verify:
- [ ] Sidebar navigation reflects the actual page hierarchy
- [ ] Current page is highlighted in sidebar
- [ ] On-page ToC generates from headings (not hardcoded)
- [ ] Previous/next links follow the logical reading order
- [ ] Search indexes all content and returns relevant results
- [ ] Version selector switches correctly and preserves page path where possible
- [ ] Old version pages show deprecation banners
- [ ] API reference entries have consistent formatting (all have parameters, returns, examples)
- [ ] Code examples in docs actually work (test them)
- [ ] Broken internal links are caught (use a link checker in CI)
