# Theming & Design Tokens

Reference sheet for defining design token systems, implementing dark/light mode, and building CSS custom property architectures. Load this when setting up a site's visual foundation or implementing theme switching.

## When to Use This Reference

- Defining a color palette, typography, and spacing system for a new site
- Implementing dark mode / light mode toggle
- Creating CSS custom properties architecture
- Deriving a full palette from brand anchor colors
- Any task where design consistency depends on shared token values

## Design Token Architecture

### Token Layers

Organize tokens in three layers:

```css
:root {
  /* Layer 1: Primitives — raw values, never used directly in components */
  --navy-50: #f0f4f8;
  --navy-100: #d9e2ec;
  --navy-200: #bcccdc;
  --navy-500: #627d98;
  --navy-700: #334e68;
  --navy-900: #102a43;

  --teal-400: #0fa89e;
  --teal-600: #0a6e72;
  --teal-800: #044e54;

  /* Layer 2: Semantic — role-based aliases, theme-dependent */
  --color-bg: var(--navy-50);
  --color-bg-subtle: white;
  --color-text: var(--navy-900);
  --color-text-muted: var(--navy-500);
  --color-primary: var(--navy-700);
  --color-accent: var(--teal-600);
  --color-border: var(--navy-200);
  --color-link: var(--teal-600);
  --color-link-hover: var(--teal-800);
  --color-bg-code: var(--navy-100);
  --color-bg-code-block: var(--navy-900);
  --color-code-text: var(--navy-100);

  /* Layer 3: Component — specific to a component (optional, define as needed) */
  --sidebar-bg: var(--color-bg-subtle);
  --sidebar-border: var(--color-border);
  --nav-bg: var(--color-primary);
  --nav-text: white;
}
```

### Why Three Layers?

- **Primitives** define the full color ramp — they never change between themes
- **Semantic tokens** map primitives to roles — these swap between light and dark
- **Component tokens** are optional — only create them when a component needs to deviate from semantic defaults

### Generating a Palette from Anchor Colors

Given a primary color (e.g., `#1a365d` navy) and an accent (e.g., `#2b6cb0` blue):

1. **Generate a full ramp** for each: 50 (lightest) through 900 (darkest), 10 steps
2. **Derive grays** from the primary hue (tinted grays look more cohesive than pure gray)
3. **Add functional colors**: success (green), warning (amber), error (red), info (blue)
4. **Test contrast**: every text/background combination must pass WCAG AA

```css
/* Tinted grays derived from navy hue (210°) */
--gray-50: #f7f8fa;   /* warm, not clinical */
--gray-100: #ebedf0;
--gray-200: #d1d5db;
--gray-400: #9ca3af;
--gray-600: #4b5563;
--gray-800: #1f2937;
--gray-900: #111827;

/* Functional colors */
--color-success: #059669;
--color-warning: #d97706;
--color-error: #dc2626;
--color-info: #2563eb;
```

## Dark Mode Implementation

### CSS Approach: prefers-color-scheme + Toggle

```css
/* Default: light mode */
:root {
  --color-bg: #ffffff;
  --color-bg-subtle: #f7f8fa;
  --color-text: #1a1a2e;
  --color-text-muted: #4a5568;
  --color-border: #e2e8f0;
  --color-bg-code: #f0f2f5;
  --color-bg-code-block: #1e293b;
  --color-code-text: #e2e8f0;
}

/* System preference: dark */
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --color-bg: #0d1117;
    --color-bg-subtle: #161b22;
    --color-text: #e6edf3;
    --color-text-muted: #8b949e;
    --color-border: #30363d;
    --color-bg-code: #1c2128;
    --color-bg-code-block: #0d1117;
    --color-code-text: #e6edf3;
  }
}

/* Explicit dark mode (toggle override) */
:root[data-theme="dark"] {
  --color-bg: #0d1117;
  --color-bg-subtle: #161b22;
  --color-text: #e6edf3;
  --color-text-muted: #8b949e;
  --color-border: #30363d;
  --color-bg-code: #1c2128;
  --color-bg-code-block: #0d1117;
  --color-code-text: #e6edf3;
}
```

### Toggle JavaScript

```javascript
// Theme toggle with localStorage persistence
const toggle = document.querySelector('.theme-toggle');
const root = document.documentElement;

// Load saved preference
const saved = localStorage.getItem('theme');
if (saved) {
  root.setAttribute('data-theme', saved);
}

toggle.addEventListener('click', () => {
  const current = root.getAttribute('data-theme');
  const isDark = current === 'dark' ||
    (!current && window.matchMedia('(prefers-color-scheme: dark)').matches);
  const next = isDark ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
});
```

### Preventing Flash of Wrong Theme (FOWT)

Add a blocking script in `<head>` before any CSS loads:

```html
<script>
  (function() {
    var saved = localStorage.getItem('theme');
    if (saved) {
      document.documentElement.setAttribute('data-theme', saved);
    }
  })();
</script>
```

This runs synchronously before paint, preventing a flash of light mode for dark mode users.

## Typography Tokens

### System Font Stacks

```css
:root {
  /* Body: system UI fonts */
  --font-body: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;

  /* Code: monospace stack */
  --font-code: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', ui-monospace,
               'SF Mono', 'Roboto Mono', Menlo, Consolas, monospace;

  /* Headings: same as body by default, override if using a display font */
  --font-heading: var(--font-body);
}
```

### Type Scale

Use a consistent ratio. 1.25 (major third) works well for documentation:

```css
:root {
  --text-xs: 0.75rem;    /* 12px — labels, badges */
  --text-sm: 0.875rem;   /* 14px — captions, metadata */
  --text-base: 1rem;     /* 16px — body text */
  --text-lg: 1.125rem;   /* 18px — lead paragraphs */
  --text-xl: 1.25rem;    /* 20px — H4 */
  --text-2xl: 1.5rem;    /* 24px — H3 */
  --text-3xl: 1.875rem;  /* 30px — H2 */
  --text-4xl: 2.25rem;   /* 36px — H1 */
  --text-5xl: 3rem;      /* 48px — hero titles */
}
```

### Line Height and Spacing

```css
:root {
  --leading-tight: 1.25;   /* headings */
  --leading-normal: 1.6;   /* body text */
  --leading-relaxed: 1.75; /* long-form reading */

  --tracking-tight: -0.025em;  /* large headings */
  --tracking-normal: 0;
  --tracking-wide: 0.025em;    /* small caps, labels */
}
```

## Spacing Scale

```css
:root {
  /* 4px base, powers of 2 */
  --space-px: 1px;
  --space-0: 0;
  --space-1: 0.25rem;   /* 4px */
  --space-2: 0.5rem;    /* 8px */
  --space-3: 0.75rem;   /* 12px */
  --space-4: 1rem;      /* 16px */
  --space-5: 1.5rem;    /* 24px */
  --space-6: 2rem;      /* 32px */
  --space-8: 3rem;      /* 48px */
  --space-10: 4rem;     /* 64px */
  --space-12: 6rem;     /* 96px */
}
```

## Contrast Verification

### Quick Contrast Check

Before finalizing any color combination, verify:

| Usage | Minimum Ratio | Test With |
|-------|--------------|-----------|
| Body text on background | 4.5:1 | `--color-text` on `--color-bg` |
| Muted text on background | 4.5:1 | `--color-text-muted` on `--color-bg` |
| Link text on background | 4.5:1 | `--color-link` on `--color-bg` |
| Code text on code background | 4.5:1 | Both inline and block |
| Nav text on nav background | 4.5:1 | `--nav-text` on `--nav-bg` |
| All of the above in dark mode | 4.5:1 | Repeat every check |

### Common Failures

- Light gray text on white background (looks fine on a retina display, fails on cheaper screens)
- Accent color as text on dark background (teal/green often fails)
- White text on light accent background (buttons, badges)
- Muted text in dark mode (easy to make it too dim)

## Quality Additions for Theming

In addition to the base quality checklist, verify:
- [ ] All color combinations pass WCAG AA in both light and dark mode
- [ ] Dark mode toggle persists preference via localStorage
- [ ] No flash of wrong theme on page load (blocking script in `<head>`)
- [ ] System preference (`prefers-color-scheme`) is respected when no explicit choice is saved
- [ ] Semantic tokens are used in CSS, not raw color values
- [ ] Images/diagrams are readable in both modes (consider `filter: invert()` for simple diagrams)
- [ ] Syntax highlighting theme changes between modes (light theme for light mode, dark for dark)
- [ ] Focus indicators are visible in both modes
- [ ] Borders and dividers are visible in both modes
