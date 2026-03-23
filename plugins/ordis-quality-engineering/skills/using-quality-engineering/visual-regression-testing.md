---
name: visual-regression-testing
description: Use when testing UI changes, preventing visual bugs, setting up screenshot comparison, handling flaky visual tests, testing responsive layouts, or choosing visual testing tools (Percy, Chromatic, BackstopJS) - provides anti-flakiness strategies and component visual testing patterns
---

# Visual Regression Testing

## Overview

**Core principle:** Visual regression tests catch UI changes that automated functional tests miss (layout shifts, styling bugs, rendering issues).

**Rule:** Visual tests complement functional tests, don't replace them. Test critical pages only.

## Visual vs Functional Testing

| Aspect | Functional Testing | Visual Regression Testing |
|--------|-------------------|---------------------------|
| **What** | Behavior (clicks work, data saves) | Appearance (layout, styling) |
| **How** | Assert on DOM/data | Compare screenshots |
| **Catches** | Logic bugs, broken interactions | CSS bugs, layout shifts, visual breaks |
| **Speed** | Fast (100-500ms/test) | Slower (1-5s/test) |
| **Flakiness** | Low | High (rendering differences) |

**Use both:** Functional tests verify logic, visual tests verify appearance

---

## Tool Selection Decision Tree

| Your Need | Team Setup | Use | Why |
|-----------|------------|-----|-----|
| **Component testing** | React/Vue/Angular | **Chromatic** | Storybook integration, CI-friendly |
| **Full page testing** | Any framework | **Percy** | Easy setup, cross-browser |
| **Self-hosted** | Budget constraints | **BackstopJS** | Open source, no cloud costs |
| **Playwright-native** | Already using Playwright | **Playwright Screenshots** | Built-in, no extra tool |
| **Budget-free** | Small projects | **Playwright + pixelmatch** | DIY, full control |

**First choice for teams:** Chromatic (components) or Percy (pages)

**First choice for individuals:** Playwright + pixelmatch (free, simple)

---

## Basic Visual Test Pattern (Playwright)

```javascript
import { test, expect } from '@playwright/test';

test('homepage visual regression', async ({ page }) => {
  await page.goto('https://example.com');

  // Wait for page to be fully loaded
  await page.waitForLoadState('networkidle');

  // Take screenshot
  await expect(page).toHaveScreenshot('homepage.png', {
    fullPage: true,  // Capture entire page, not just viewport
    animations: 'disabled',  // Disable animations for stability
  });
});
```

**First run:** Creates baseline screenshot
**Subsequent runs:** Compares against baseline, fails if different

---

## Anti-Flakiness Strategies

**Visual tests are inherently flaky. Reduce flakiness with these techniques:**

### 1. Disable Animations

```javascript
test('button hover state', async ({ page }) => {
  await page.goto('/buttons');

  // Disable ALL animations/transitions
  await page.addStyleTag({
    content: `
      *, *::before, *::after {
        animation-duration: 0s !important;
        transition-duration: 0s !important;
      }
    `
  });

  await expect(page).toHaveScreenshot();
});
```

---

### 2. Mask Dynamic Content

**Problem:** Timestamps, dates, random data cause false positives

```javascript
test('dashboard', async ({ page }) => {
  await page.goto('/dashboard');

  await expect(page).toHaveScreenshot({
    mask: [
      page.locator('.timestamp'),  // Hide timestamps
      page.locator('.user-avatar'),  // Hide dynamic avatars
      page.locator('.live-counter'),  // Hide live updating counters
    ],
  });
});
```

---

### 3. Wait for Fonts to Load

**Problem:** Tests run before web fonts load, causing inconsistent rendering

```javascript
test('typography page', async ({ page }) => {
  await page.goto('/typography');

  // Wait for fonts to load
  await page.evaluate(() => document.fonts.ready);

  await expect(page).toHaveScreenshot();
});
```

---

### 4. Freeze Time

**Problem:** "Posted 5 minutes ago" changes every run

```javascript
import { test } from '@playwright/test';

test('posts with timestamps', async ({ page }) => {
  // Mock system time
  await page.addInitScript(() => {
    const fixedDate = new Date('2025-01-13T12:00:00Z');
    Date = class extends Date {
      constructor() {
        super();
        return fixedDate;
      }
      static now() {
        return fixedDate.getTime();
      }
    };
  });

  await page.goto('/posts');
  await expect(page).toHaveScreenshot();
});
```

---

### 5. Use Test Data Fixtures

**Problem:** Real data changes (new users, products, orders)

```javascript
test('product catalog', async ({ page }) => {
  // Seed database with fixed test data
  await seedDatabase([
    { id: 1, name: 'Widget', price: 9.99 },
    { id: 2, name: 'Gadget', price: 19.99 },
  ]);

  await page.goto('/products');
  await expect(page).toHaveScreenshot();
});
```

---

## Component Visual Testing (Storybook + Chromatic)

### Storybook Story

```javascript
// Button.stories.jsx
import { Button } from './Button';

export default {
  title: 'Components/Button',
  component: Button,
};

export const Primary = {
  args: {
    variant: 'primary',
    children: 'Click me',
  },
};

export const Disabled = {
  args: {
    variant: 'primary',
    disabled: true,
    children: 'Disabled',
  },
};

export const LongText = {
  args: {
    children: 'This is a very long button text that might wrap',
  },
};
```

---

### Chromatic Configuration

```javascript
// .storybook/main.js
module.exports = {
  stories: ['../src/**/*.stories.@(js|jsx|ts|tsx)'],
  addons: ['@storybook/addon-essentials', '@chromatic-com/storybook'],
};
```

```yaml
# .github/workflows/chromatic.yml
name: Chromatic

on: [push]

jobs:
  chromatic:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Required for Chromatic

      - name: Install dependencies
        run: npm ci

      - name: Run Chromatic
        uses: chromaui/action@v1
        with:
          projectToken: ${{ secrets.CHROMATIC_PROJECT_TOKEN }}
```

**Benefits:**
- Isolates component testing
- Tests all states (hover, focus, disabled)
- No full app deployment needed

---

## Responsive Design Testing

**Test multiple viewports:**

```javascript
const viewports = [
  { name: 'mobile', width: 375, height: 667 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'desktop', width: 1920, height: 1080 },
];

viewports.forEach(({ name, width, height }) => {
  test(`homepage ${name}`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await page.goto('https://example.com');

    await expect(page).toHaveScreenshot(`homepage-${name}.png`);
  });
});
```

---

## Threshold Configuration

**Allow small pixel differences (reduces false positives):**

```javascript
await expect(page).toHaveScreenshot({
  maxDiffPixels: 100,  // Allow up to 100 pixels to differ
  // OR
  maxDiffPixelRatio: 0.01,  // Allow 1% of pixels to differ
});
```

**Thresholds:**
- **Exact match (0%):** Critical branding pages (homepage, landing)
- **1-2% tolerance:** Most pages (handles minor font rendering differences)
- **5% tolerance:** Pages with dynamic content (dashboards with charts)

---

## Updating Baselines

**When to update:**
- Intentional UI changes
- Design system updates
- Framework upgrades

**How to update:**

```bash
# Playwright: Update all baselines
npx playwright test --update-snapshots

# Percy: Accept changes in web UI
# Visit percy.io, review changes, click "Approve"

# Chromatic: Accept changes in web UI
# Visit chromatic.com, review changes, click "Accept"
```

**Process:**
1. Run visual tests
2. Review diffs manually
3. Approve if changes are intentional
4. Investigate if changes are unexpected

---

## Anti-Patterns Catalog

### ❌ Testing Every Page

**Symptom:** Hundreds of visual tests for every page variant

**Why bad:**
- Slow CI (visual tests are expensive)
- High maintenance (baselines update frequently)
- False positives from minor rendering differences

**Fix:** Test critical pages only

**Criteria for visual testing:**
- Customer-facing pages (homepage, pricing, checkout)
- Reusable components (buttons, forms, cards)
- Pages with complex layouts (dashboards, admin panels)

**Don't test:**
- Internal admin pages with frequent changes
- Error pages
- Pages with highly dynamic content

---

### ❌ No Flakiness Prevention

**Symptom:** Visual tests fail randomly

```javascript
// ❌ BAD: No stability measures
test('homepage', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveScreenshot();
  // Fails due to: animations, fonts not loaded, timestamps, etc.
});
```

**Fix:** Apply all anti-flakiness strategies

```javascript
// ✅ GOOD: Stable visual test
test('homepage', async ({ page }) => {
  await page.goto('/');

  // Disable animations
  await page.addStyleTag({ content: '* { animation: none !important; }' });

  // Wait for fonts
  await page.evaluate(() => document.fonts.ready);

  // Wait for images
  await page.waitForLoadState('networkidle');

  await expect(page).toHaveScreenshot({
    animations: 'disabled',
    mask: [page.locator('.timestamp')],
  });
});
```

---

### ❌ Ignoring Baseline Drift

**Symptom:** Baselines diverge between local and CI

**Why it happens:**
- Different OS (macOS vs Linux)
- Different browser versions
- Different screen resolutions

**Fix:** Always generate baselines in CI

```yaml
# .github/workflows/update-baselines.yml
name: Update Visual Baselines

on:
  workflow_dispatch:  # Manual trigger

jobs:
  update:
    runs-on: ubuntu-latest  # Same as test CI
    steps:
      - uses: actions/checkout@v3

      - name: Update snapshots
        run: npx playwright test --update-snapshots

      - name: Commit baselines
        run: |
          git config user.name "GitHub Actions"
          git add tests/**/*.png
          git commit -m "Update visual baselines"
          git push
```

---

### ❌ Using Visual Tests for Functional Assertions

**Symptom:** Only visual tests, no functional tests

```javascript
// ❌ BAD: Only checking visually
test('login form', async ({ page }) => {
  await page.goto('/login');
  await expect(page).toHaveScreenshot();
  // Doesn't verify login actually works!
});
```

**Fix:** Use both

```javascript
// ✅ GOOD: Functional + visual
test('login form functionality', async ({ page }) => {
  await page.goto('/login');
  await page.fill('#email', 'user@example.com');
  await page.fill('#password', 'password123');
  await page.click('button[type="submit"]');

  // Functional assertion
  await expect(page).toHaveURL('/dashboard');
});

test('login form appearance', async ({ page }) => {
  await page.goto('/login');

  // Visual assertion
  await expect(page).toHaveScreenshot();
});
```

---

## CI/CD Integration

### GitHub Actions (Playwright)

```yaml
# .github/workflows/visual-tests.yml
name: Visual Tests

on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Playwright
        run: |
          npm ci
          npx playwright install --with-deps

      - name: Run visual tests
        run: npx playwright test tests/visual/

      - name: Upload failures
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: visual-test-failures
          path: test-results/
```

---

## Bottom Line

**Visual regression tests catch UI bugs that functional tests miss. Test critical pages only, apply anti-flakiness strategies religiously.**

**Best practices:**
- Use Chromatic (components) or Percy (pages) for teams
- Use Playwright + pixelmatch for solo developers
- Disable animations, mask dynamic content, wait for fonts
- Test responsive layouts (mobile, tablet, desktop)
- Allow small thresholds (1-2%) to reduce false positives
- Update baselines in CI, not locally

**If your visual tests are flaky, you're doing it wrong. Apply flakiness prevention first, then add tests.**
