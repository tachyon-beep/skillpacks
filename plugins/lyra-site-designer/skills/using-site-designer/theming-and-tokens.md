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

Organize tokens in three layers. Use **OKLCH** for primitives — it is perceptually uniform, so equal lightness steps in the ramp look like equal steps to the eye, and the hue stays stable across the ramp. `oklch()` is Baseline 2023 (Chrome 111, Safari 15.4, Firefox 113).

```css
:root {
  color-scheme: light dark;

  /* Layer 1: Primitives — raw values, never used directly in components.
     OKLCH(L C H): L = 0..1 lightness, C = chroma, H = hue degrees.
     Equal L steps = perceptually equal lightness steps. */
  --navy-50:  oklch(0.97 0.012 250);
  --navy-100: oklch(0.93 0.025 250);
  --navy-200: oklch(0.86 0.045 250);
  --navy-500: oklch(0.55 0.090 250);
  --navy-700: oklch(0.38 0.080 250);
  --navy-900: oklch(0.20 0.045 250);

  --teal-400: oklch(0.70 0.110 195);
  --teal-600: oklch(0.55 0.110 195);
  --teal-800: oklch(0.38 0.080 195);

  /* Layer 2: Semantic — role-based aliases, theme-dependent.
     light-dark(lightValue, darkValue) picks the right side based on
     :root color-scheme. Replaces a duplicated dark-mode token block. */
  --color-bg:            light-dark(oklch(0.99 0.003 250), oklch(0.18 0.020 250));
  --color-bg-subtle:     light-dark(oklch(0.97 0.005 250), oklch(0.22 0.020 250));
  --color-text:          light-dark(var(--navy-900),       oklch(0.93 0.015 250));
  --color-text-muted:    light-dark(var(--navy-500),       oklch(0.70 0.020 250));
  --color-primary:       light-dark(var(--navy-700),       oklch(0.78 0.060 250));
  --color-accent:        light-dark(var(--teal-600),       var(--teal-400));
  --color-border:        light-dark(var(--navy-200),       oklch(0.30 0.020 250));
  --color-link:          var(--color-accent);
  /* color-mix derives hover from link without committing a second token value */
  --color-link-hover:    color-mix(in oklch, var(--color-link) 80%, var(--color-text) 20%);
  --color-bg-code:       light-dark(var(--navy-100),       oklch(0.25 0.020 250));
  --color-bg-code-block: light-dark(var(--navy-900),       oklch(0.15 0.018 250));
  --color-code-text:     light-dark(oklch(0.95 0.010 250), oklch(0.93 0.015 250));

  /* Layer 3: Component — specific to a component (optional, define as needed) */
  --sidebar-bg: var(--color-bg-subtle);
  --sidebar-border: var(--color-border);
  --nav-bg: var(--color-primary);
  --nav-text: light-dark(white, var(--navy-900));
}
```

### Why Three Layers?

- **Primitives** define the full color ramp — they never change between themes
- **Semantic tokens** map primitives to roles — `light-dark()` swaps the value per scheme
- **Component tokens** are optional — only create them when a component needs to deviate from semantic defaults

### Why OKLCH + `light-dark()`?

- **OKLCH ramps are perceptually uniform.** Equal `L` steps look like equal lightness steps; equal `C` steps look like equal saturation steps. Hex ramps drift through perceptual lightness in the middle of the ramp, which is why so many hex palettes fail WCAG mid-ramp and need manual fixups.
- **`light-dark()` halves the dark-mode block.** One declaration site per token instead of two, so the "I updated the light value but forgot the dark value" bug class disappears. It activates from the `color-scheme` property — set `color-scheme: light dark` on `:root` and the engine resolves which side to use.
- **`color-mix(in oklch, ...)`** derives hover/active states from a base color without committing a separate token, keeping the palette compact and the relationships visible.

### Legacy hex fallback

If you must support browsers that pre-date Baseline 2023 (older Safari TPs, downstream products on locked Chromium forks), define hex aliases first and use `@supports` to layer the OKLCH/`light-dark()` ramp on top. For most public dev-tool sites — which track evergreen browsers — this isn't needed.

### Generating a Palette from Anchor Colors

Given a primary color (e.g., navy at `oklch(0.38 0.08 250)`) and an accent (e.g., teal at `oklch(0.55 0.11 195)`):

1. **Generate a full OKLCH ramp** for each: hold hue (`H`) and chroma (`C`) roughly fixed and step `L` from ~0.97 down to ~0.18, 9–10 steps.
2. **Derive grays** from the primary hue with very low chroma (`C ≈ 0.005–0.015`) — tinted grays read as cohesive with the brand instead of clinical.
3. **Add functional colors**: success (green ~140°), warning (amber ~75°), error (red ~25°), info (blue ~250°). Use the same `L`/`C` envelope so they sit at consistent visual weight.
4. **Test contrast**: every text/background combination must pass WCAG AA. OKLCH lightness is *not* the same as WCAG luminance — a contrast checker is still required.

```css
/* Tinted grays derived from navy hue (~250°) */
--gray-50:  oklch(0.98 0.005 250);
--gray-100: oklch(0.94 0.008 250);
--gray-200: oklch(0.86 0.012 250);
--gray-400: oklch(0.66 0.015 250);
--gray-600: oklch(0.48 0.018 250);
--gray-800: oklch(0.28 0.018 250);
--gray-900: oklch(0.18 0.015 250);

/* Functional colors — equal L envelope so they read at consistent weight */
--color-success: oklch(0.58 0.13 145);
--color-warning: oklch(0.72 0.15  75);
--color-error:   oklch(0.58 0.18  25);
--color-info:    oklch(0.58 0.16 245);
```

## Dark Mode Implementation

### CSS Approach: `color-scheme` + `light-dark()` + Toggle Override

The semantic-token block above already uses `light-dark()`, so per-scheme values exist exactly once. To make the toggle override the system preference, switch the `color-scheme` property at the root — `light-dark()` honours it automatically:

```css
:root {
  /* System preference is the default; engine maps light-dark() accordingly */
  color-scheme: light dark;
}

/* Explicit user choice (toggle) overrides system preference.
   No need to redeclare every token — light-dark() reads color-scheme. */
:root[data-theme="light"] { color-scheme: light; }
:root[data-theme="dark"]  { color-scheme: dark; }
```

This is the canonical 2024+ pattern: the only thing the toggle changes is `color-scheme`. Every token automatically picks the correct side of its `light-dark()` declaration.

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
