# Responsive Layout Patterns

Reference sheet for making developer sites work across screen sizes — from wide desktop monitors to mobile phones. Load this when implementing layouts that need to adapt, especially sidebar+content documentation layouts.

## When to Use This Reference

- Implementing responsive documentation layouts (sidebar collapses on mobile)
- Building navigation that works on both desktop and mobile
- Creating responsive tables, code blocks, and other content that tends to overflow
- Any page where mobile usability is a requirement

## Container Queries — Default to These

For component-level layout (cards, code blocks, callouts, sidebars, ToC entries), use **container queries**. They reflow based on the *available width of the parent container*, not the viewport. A 280px-wide sidebar slot needs to look the same whether the viewport is 1024px or 1920px — viewport media queries can't tell those apart, but `@container` can. Container queries are Baseline 2023 (Chrome 105, Safari 16, Firefox 110).

```css
/* Mark which elements are query containers */
.docs-content,
.feature-grid,
.api-entry {
  container-type: inline-size;
}

/* Style a card based on the *card's* available width, not the viewport */
.feature-card {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--space-3);
}

@container (min-width: 28rem) {
  .feature-card {
    grid-template-columns: auto 1fr;  /* icon next to text once there's room */
  }
}

@container (min-width: 40rem) {
  .feature-card {
    padding: var(--space-5);
  }
}
```

For **page-level** breakpoints (header layout, body padding, when the sidebar exists at all), viewport media queries are still the right tool — that's a viewport question.

## Viewport Breakpoint System

Use a mobile-first approach with consistent breakpoints for page-level layout decisions:

```css
:root {
  --bp-sm: 640px;    /* small phones → large phones */
  --bp-md: 768px;    /* phones → tablets */
  --bp-lg: 1024px;   /* tablets → laptops */
  --bp-xl: 1280px;   /* laptops → desktops */
}

/* Mobile first: base styles are for small screens */
/* Then add complexity as screens get larger */
@media (min-width: 768px) { /* tablet and up */ }
@media (min-width: 1024px) { /* laptop and up */ }
@media (min-width: 1280px) { /* desktop */ }
```

## Fluid Typography with `clamp()`

Hero titles and section headings benefit from typography that scales smoothly between breakpoints rather than stepping at viewport thresholds. `clamp(min, preferred, max)` is Baseline 2020:

```css
:root {
  /* Fluid heading scale: scales from min to max across viewport range */
  --text-fluid-h1: clamp(2rem,    1.4rem + 3vw, 3.5rem);
  --text-fluid-h2: clamp(1.5rem,  1.1rem + 2vw, 2.25rem);
  --text-lead:     clamp(1.05rem, 0.95rem + 0.5vw, 1.25rem);
}

.hero h1 { font-size: var(--text-fluid-h1); line-height: 1.1; }
```

## Native CSS Nesting

Native CSS nesting (Baseline 2023) keeps related rules visually grouped without a preprocessor. Combined with container queries it makes component CSS read top-to-bottom:

```css
.feature-card {
  padding: var(--space-3);
  border: 1px solid var(--color-border);

  & h3 {
    margin-top: 0;
    color: var(--color-primary);
  }

  @container (min-width: 28rem) {
    padding: var(--space-5);
  }
}
```

## Stateful Parent Styling with `:has()`

`:has()` lets a parent style itself based on what's inside it — Baseline 2023 (Safari 15.4, Chrome 105, Firefox 121). The single most useful application in docs sites is highlighting a sidebar section that contains the active page:

```css
/* Sidebar section that contains the current page gets emphasis */
.docs-sidebar li:has(> a[aria-current="page"]) {
  background: var(--color-bg-subtle);
  border-left: 3px solid var(--color-accent);
}

/* Parent details element of the active page is auto-expanded
   (server-rendered: add `open` attribute when section contains current page).
   :has() lets the styling complement the markup without extra classes. */
.docs-sidebar details:has(a[aria-current="page"]) summary {
  font-weight: 600;
  color: var(--color-text);
}

/* Code block container that has a copy button reserves space for it */
.code-block:has(.code-block__copy) pre {
  padding-right: var(--space-8);
}
```

## Documentation Layout (Sidebar + Content + ToC)

### Desktop (> 1024px): Three-column

```css
.docs-layout {
  display: grid;
  grid-template-columns: var(--sidebar-width) 1fr auto;
  gap: 0;
  max-width: 90rem;
  margin: 0 auto;
}

.docs-sidebar {
  position: sticky;
  top: var(--header-height);
  height: calc(100vh - var(--header-height));
  overflow-y: auto;
  padding: var(--space-4);
  border-right: 1px solid var(--color-border);
}

.docs-content {
  max-width: var(--content-width);
  padding: var(--space-4) var(--space-6);
}

.docs-toc {
  position: sticky;
  top: var(--header-height);
  width: 14rem;
  padding: var(--space-4);
  height: calc(100vh - var(--header-height));
  overflow-y: auto;
}
```

### Tablet (768px-1024px): Two-column, no ToC

```css
@media (max-width: 1024px) {
  .docs-layout {
    grid-template-columns: var(--sidebar-width) 1fr;
  }

  .docs-toc {
    display: none;  /* or move to top of content as a collapsible */
  }
}
```

### Mobile (< 768px): Single column, off-canvas sidebar

Use the **native `popover` attribute** (Baseline 2024 — Chrome 114, Safari 17, Firefox 125). The browser handles the open/close state, the backdrop, focus management, light-dismiss on outside click, and Escape-to-close — replacing what used to be ~30 lines of custom JS plus a focus-trap library.

```html
<button popovertarget="docs-sidebar"
        aria-label="Toggle navigation"><!-- hamburger icon --></button>

<aside id="docs-sidebar" popover class="docs-sidebar">
  <!-- nav links -->
</aside>
```

```css
@media (max-width: 768px) {
  .docs-layout {
    grid-template-columns: 1fr;
  }

  /* The :popover-open pseudo-class targets the open state without JS */
  .docs-sidebar:popover-open {
    position: fixed;
    inset: var(--header-height) auto 0 0;
    width: min(80%, 320px);
    margin: 0;
    border: 0;
    border-right: 1px solid var(--color-border);
    background: var(--color-bg);
    box-shadow: 2px 0 8px rgb(0 0 0 / 0.15);
  }

  /* Native backdrop — replaces the .docs-overlay element */
  .docs-sidebar::backdrop {
    background: rgb(0 0 0 / 0.3);
  }
}
```

That's the whole pattern. No `is-open` class plumbing, no Escape handler, no focus trap, no overlay element. The browser handles all of it. (For browsers without popover support, the button is still focusable and the aside still renders — you can layer a small JS fallback if your audience requires it.)

## Responsive Header/Navigation

### Desktop: Horizontal nav links

```css
.site-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 var(--space-4);
  height: var(--header-height);
  border-bottom: 1px solid var(--color-border);
}

.site-nav {
  display: flex;
  gap: var(--space-4);
}

.site-nav a {
  color: var(--color-text);
  text-decoration: none;
  font-size: var(--text-sm);
  font-weight: 500;
}
```

### Mobile: Hamburger menu

```css
@media (max-width: 768px) {
  .site-nav {
    display: none;
  }

  .site-nav.is-open {
    display: flex;
    flex-direction: column;
    position: absolute;
    top: var(--header-height);
    left: 0;
    right: 0;
    background: var(--color-bg);
    border-bottom: 1px solid var(--color-border);
    padding: var(--space-3);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  }
}
```

## Responsive Content Elements

### Tables

Tables are the most common overflow problem on mobile:

```css
/* Wrap tables in a scrollable container */
.table-wrapper {
  overflow-x: auto;
  margin: var(--space-4) 0;
}

/* Visual hint that table scrolls */
.table-wrapper {
  background:
    linear-gradient(to right, var(--color-bg) 30%, transparent),
    linear-gradient(to left, var(--color-bg) 30%, transparent),
    linear-gradient(to right, rgba(0,0,0,0.1), transparent 15px),
    linear-gradient(to left, rgba(0,0,0,0.1), transparent 15px);
  background-position: left, right, left, right;
  background-size: 40px 100%, 40px 100%, 15px 100%, 15px 100%;
  background-repeat: no-repeat;
  background-attachment: local, local, scroll, scroll;
}
```

### Code Blocks

```css
/* Code blocks scroll horizontally, don't wrap */
.code-block pre {
  overflow-x: auto;
}

/* On very small screens, reduce code font size slightly */
@media (max-width: 640px) {
  .code-block code {
    font-size: 0.8rem;
  }
}
```

### Images

```css
/* Images never overflow their container */
img {
  max-width: 100%;
  height: auto;
}

/* But don't stretch small images */
.docs-content img {
  max-width: min(100%, 800px);
}
```

## Reduced Motion

Respect users who prefer reduced motion:

```css
@media (prefers-reduced-motion: reduce) {
  * {
    transition-duration: 0.01ms !important;
    animation-duration: 0.01ms !important;
  }

  .docs-sidebar {
    transition: none;
  }
}
```

## Testing Checklist

Verify at these widths:
- [ ] **375px** (iPhone SE) — smallest common phone
- [ ] **414px** (iPhone 14) — standard phone
- [ ] **768px** (iPad portrait) — tablet breakpoint
- [ ] **1024px** (iPad landscape) — sidebar transition point
- [ ] **1440px** (standard laptop) — typical desktop
- [ ] **1920px** (large monitor) — content shouldn't stretch to fill

At each width, check:
- [ ] No horizontal scrollbar on the page (content elements may scroll internally)
- [ ] Navigation is accessible (hamburger on mobile, links on desktop)
- [ ] Sidebar collapses and expands correctly
- [ ] Tables and code blocks scroll horizontally within their containers
- [ ] Text is readable without zooming (minimum 16px body text)
- [ ] Touch targets are at least 44×44px on mobile
