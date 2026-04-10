# Responsive Layout Patterns

Reference sheet for making developer sites work across screen sizes — from wide desktop monitors to mobile phones. Load this when implementing layouts that need to adapt, especially sidebar+content documentation layouts.

## When to Use This Reference

- Implementing responsive documentation layouts (sidebar collapses on mobile)
- Building navigation that works on both desktop and mobile
- Creating responsive tables, code blocks, and other content that tends to overflow
- Any page where mobile usability is a requirement

## Breakpoint System

Use a mobile-first approach with consistent breakpoints:

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

```css
@media (max-width: 768px) {
  .docs-layout {
    grid-template-columns: 1fr;
  }

  .docs-sidebar {
    position: fixed;
    left: -100%;
    top: var(--header-height);
    width: 80%;
    max-width: 320px;
    z-index: 100;
    background: var(--color-bg);
    transition: left 0.2s ease;
    border-right: 1px solid var(--color-border);
    box-shadow: 2px 0 8px rgba(0,0,0,0.1);
  }

  .docs-sidebar.is-open {
    left: 0;
  }

  /* Overlay behind sidebar */
  .docs-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.3);
    z-index: 99;
  }

  .docs-overlay.is-open {
    display: block;
  }
}
```

### Sidebar Toggle

```html
<button class="sidebar-toggle" aria-label="Toggle navigation" aria-expanded="false">
  <svg><!-- hamburger icon --></svg>
</button>
```

```javascript
const toggle = document.querySelector('.sidebar-toggle');
const sidebar = document.querySelector('.docs-sidebar');
const overlay = document.querySelector('.docs-overlay');

function openSidebar() {
  sidebar.classList.add('is-open');
  overlay.classList.add('is-open');
  toggle.setAttribute('aria-expanded', 'true');
  // Trap focus inside sidebar
  sidebar.querySelector('a, button')?.focus();
}

function closeSidebar() {
  sidebar.classList.remove('is-open');
  overlay.classList.remove('is-open');
  toggle.setAttribute('aria-expanded', 'false');
  toggle.focus();
}

toggle.addEventListener('click', () => {
  sidebar.classList.contains('is-open') ? closeSidebar() : openSidebar();
});

overlay.addEventListener('click', closeSidebar);

// Close on Escape
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && sidebar.classList.contains('is-open')) {
    closeSidebar();
  }
});
```

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
  -webkit-overflow-scrolling: touch;
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
  -webkit-overflow-scrolling: touch;
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
