# Landing Page Design for Developer Tools

Reference sheet for project homepages and landing pages. Load this when building the front door to an open-source project, developer tool, or technical framework — the page that converts visitors into users.

## When to Use This Reference

- Building the homepage for a developer tool or library
- Creating a "what is this" page for an open-source project
- Designing a project overview page that links out to docs, source, and community
- Any page whose job is to explain what the project does and why someone should care

## Design Philosophy

Developer landing pages fail in two predictable ways:

1. **Too corporate**: Hero images, vague taglines, "request a demo" buttons. Developers close the tab.
2. **Too sparse**: Just a README rendered as HTML. No structure, no navigation, no reason to stay.

The sweet spot: **substance with structure**. Show what the tool does (code examples), why it matters (concrete benefits), and how to start (install command). Skip everything else.

## Page Structure

### The Content-First Landing Page

```
1. Header bar (logo, nav links, dark mode toggle)
2. Value statement (1-2 sentences: what it does + for whom)
3. Install command (copy-to-clipboard)
4. Primary CTAs (Get Started, View on GitHub)
5. Feature highlights (3-4 cards, concrete not abstract)
6. Quick example (real code showing actual usage)
7. Social proof (GitHub stars, downloads, adopter logos — only if genuine)
8. Footer (license, links, version)
```

### Section-by-Section Guidance

#### Value Statement

**One sentence** that tells a developer what this is and whether they should care.

Good:
- "A static type checker for Python that catches bugs before your code runs."
- "Compile-time memory safety without garbage collection."
- "Fast, opinionated code formatter for JavaScript."

Bad:
- "Empowering developers to build the future of secure, scalable applications." (means nothing)
- "The modern platform for enterprise-grade solutions." (buzzword soup)

```html
<section class="hero">
  <h1>Project Name</h1>
  <p class="hero__tagline">
    One clear sentence about what it does and for whom.
  </p>
</section>
```

#### Install Command

The most important interactive element on the page. Make it prominent, make it copyable:

```html
<div class="install-command">
  <code>pip install projectname</code>
  <button class="copy-btn" aria-label="Copy install command">
    <!-- clipboard icon -->
  </button>
</div>
```

For multi-platform tools, use tabs:

```html
<div class="install-tabs" role="tablist">
  <button role="tab" aria-selected="true">pip</button>
  <button role="tab">conda</button>
  <button role="tab">docker</button>
</div>
<div class="install-panel" role="tabpanel">
  <code>pip install projectname</code>
</div>
```

#### Feature Highlights

3-4 cards maximum. Each card:
- **Concrete title** (not "Powerful" or "Flexible")
- **One sentence** explaining the specific capability
- **Optional**: small icon or emoji, link to relevant docs page

```html
<section class="features">
  <div class="feature-card">
    <h3>Static Analysis</h3>
    <p>Catches type errors at build time, before your code reaches production.</p>
  </div>
  <!-- ... -->
</section>
```

Good feature titles: "Incremental Compilation", "Zero-Config Formatting", "SARIF Output"
Bad feature titles: "Powerful", "Flexible", "Enterprise-Ready", "Next-Generation"

#### Quick Example

Show real code that a developer can immediately understand:

```html
<section class="example">
  <h2>Quick Example</h2>
  <div class="code-block">
    <div class="code-block__header">
      <span class="code-block__lang">python</span>
      <button class="copy-btn">Copy</button>
    </div>
    <pre><code class="language-python">
# Real, runnable example — not pseudocode
from project import check

results = check("my_module.py")
for issue in results:
    print(f"{issue.file}:{issue.line} — {issue.message}")
    </code></pre>
  </div>
</section>
```

Rules for examples:
- Must be real, runnable code (not pseudocode)
- Under 15 lines — show the happy path, not every option
- Include output if it makes the value clearer
- Link to "more examples" page for depth

#### Social Proof

Only include if genuine. Options:
- GitHub stars badge (if > 1k)
- Download count (if > 10k)
- Adopter logos (only with permission, and only recognizable ones)
- Quotes from real users (with attribution, not anonymous)

Skip this section entirely if the project is new. An empty social proof section is worse than none.

## Styling Patterns

### Feature Card Grid

```css
.features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: var(--space-4);
  padding: var(--space-5) 0;
}

.feature-card {
  padding: var(--space-4);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: var(--color-bg-subtle);
}

.feature-card h3 {
  margin-top: 0;
  color: var(--color-primary);
}
```

### Install Command Styling

```css
.install-command {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-3);
  background: var(--color-bg-code);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-family: var(--font-code);
  font-size: 1.1rem;
}

.install-command .copy-btn {
  background: none;
  border: none;
  cursor: pointer;
  color: var(--color-text-muted);
  padding: var(--space-1);
}

.install-command .copy-btn:hover {
  color: var(--color-accent);
}
```

### Hero Section

```css
.hero {
  text-align: center;
  padding: var(--space-6) var(--space-3);
  max-width: var(--content-width);
  margin: 0 auto;
}

.hero h1 {
  font-size: 2.5rem;
  margin-bottom: var(--space-2);
}

.hero__tagline {
  font-size: 1.25rem;
  color: var(--color-text-muted);
  max-width: 36rem;
  margin: 0 auto var(--space-4);
}
```

## What NOT to Include

- Pricing tables (this is for open-source / developer tools)
- Newsletter signup (unless the project has a genuine newsletter)
- Animated backgrounds, particle effects, or WebGL demos
- Carousel / slider components (nobody reads slide 2)
- "Trusted by 10,000+ developers" without verifiable data
- Screenshots of the tool in a Mac window chrome mockup (just show the output)

## Quality Additions for Landing Pages

In addition to the base quality checklist, verify:
- [ ] Value statement is under 2 sentences and mentions what the tool does
- [ ] Install command is prominent and copyable
- [ ] Feature cards use concrete titles (not adjectives)
- [ ] Code example is real and runnable
- [ ] Page loads and renders fully without JavaScript
- [ ] Primary CTA (Get Started) is visible without scrolling
- [ ] No broken links to docs, GitHub, or external resources
- [ ] Page looks good at both 1440px and 375px widths
