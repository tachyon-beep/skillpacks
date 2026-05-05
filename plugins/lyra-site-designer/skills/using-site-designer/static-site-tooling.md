# Static Site Tooling

Reference sheet for choosing and configuring static site generators. Load this when the user needs to set up the build toolchain for a documentation or project site.

## When to Use This Reference

- Choosing between general-purpose generators (Hugo, Astro, Eleventy) and docs-first frameworks (Starlight, VitePress, Docusaurus)
- Configuring a static site generator for a new project
- Setting up build pipelines, templating, and deployment
- Migrating between static site generators

## Decision Matrix

### General-Purpose Generators

| Factor | Hugo | Astro | Eleventy | Plain HTML |
|--------|------|-------|----------|------------|
| **Speed** | Fastest builds | Fast | Fast | N/A |
| **Language** | Go templates | JS/TS | JS | N/A |
| **Learning curve** | Moderate | Moderate | Low | None |
| **Templating** | Go html/template | JSX/Astro | Nunjucks/Liquid/etc | Manual |
| **Content** | Markdown + front matter | Markdown + MDX | Markdown + data files | Manual |
| **JS framework support** | None (by design) | React/Vue/Svelte islands | None (by design) | Manual |
| **Best for** | Blogs, custom layouts | Content + interactive bits | Simple sites, blogs | < 5 pages |
| **Binary dependency** | Yes (Go) | Yes (Node) | Yes (Node) | None |
| **Ecosystem** | Themes, modules | Components, integrations | Plugins | N/A |

### Docs-First Frameworks (Recommended for Documentation Sites)

For a project's docs site, prefer a purpose-built docs framework over rolling your own with a general-purpose generator. They ship sidebar nav, search, dark mode, MDX, and accessible defaults out of the box.

| Factor | Starlight (Astro) | VitePress | Docusaurus 3 |
|--------|-------------------|-----------|--------------|
| **Underlying** | Astro | Vite + Vue | React |
| **Sidebar / ToC / search** | Built in (Pagefind) | Built in (local + Algolia) | Built in (Algolia) |
| **Dark mode** | Built in | Built in | Built in |
| **MDX / components** | Yes (any framework via Astro islands) | Yes (Vue components) | Yes (React components) |
| **i18n** | Yes | Yes | Yes (mature) |
| **Versioned docs** | Manual | Manual | Built in |
| **Best for** | Most modern OSS dev-tool docs in 2025 | Vue / Vite ecosystem tools | Larger projects needing versioning + React ecosystem |
| **Caveat** | Younger ecosystem | Vue idioms even if you don't write Vue | Heavier bundle, React-first |

### Quick Recommendations

- **< 5 pages, no blog**: Plain HTML/CSS. No build step needed.
- **Modern OSS dev-tool docs (default pick in 2025)**: **Starlight** on Astro. Sidebar/search/i18n/dark-mode all built in; tunes well; stays out of the way.
- **Project lives in the Vue/Vite world**: **VitePress**. Same idioms as the rest of your stack.
- **Versioned docs with sizable React component library**: **Docusaurus 3**. Built-in versioning is the differentiator.
- **Blog + docs in one**: **Hugo** (built-in taxonomies, RSS, pagination) or **Astro** with content collections.
- **Project site with marketing page + interactive demos**: **Astro** (island architecture — static by default, JS where needed).
- **Must avoid Node.js**: **Hugo** (single Go binary).
- **Maximum simplicity, no docs framework conventions**: **Eleventy** with Nunjucks.

> Jekyll is omitted — still works, but for new dev-tool sites in 2025 the docs-first frameworks above ship with strictly more out of the box and don't carry the Ruby toolchain.

## Hugo Setup

### Project Structure

```
site/
├── config.toml              # Site configuration
├── content/                  # Markdown content
│   ├── _index.md            # Homepage
│   ├── docs/
│   │   ├── _index.md        # Docs section page
│   │   ├── getting-started.md
│   │   └── configuration.md
│   └── blog/
│       └── 2026-01-release.md
├── layouts/                  # Templates
│   ├── _default/
│   │   ├── baseof.html      # Base template
│   │   ├── single.html      # Single page
│   │   └── list.html        # Section listing
│   ├── partials/
│   │   ├── header.html
│   │   ├── footer.html
│   │   └── sidebar.html
│   └── docs/
│       └── single.html      # Docs-specific template
├── static/                   # Static assets (copied as-is)
│   ├── css/
│   ├── js/
│   └── images/
└── themes/                   # Optional theme
```

### Key Configuration

```toml
# config.toml
baseURL = "https://example.dev/"
languageCode = "en"
title = "Project Name"

[markup]
  [markup.highlight]
    style = "github-dark"     # Syntax highlighting theme
    lineNos = false
  [markup.goldmark]
    [markup.goldmark.renderer]
      unsafe = true           # Allow raw HTML in markdown

[menu]
  [[menu.main]]
    name = "Docs"
    url = "/docs/"
    weight = 1
  [[menu.main]]
    name = "Blog"
    url = "/blog/"
    weight = 2

[params]
  description = "One-line project description"
  github = "https://github.com/org/repo"
```

### Build Commands

```bash
# Development server with live reload
hugo server -D                 # -D includes drafts

# Production build
hugo --minify                  # outputs to public/

# Build with specific base URL
hugo --baseURL="https://example.dev/" --minify
```

## Astro Setup

### Project Structure

```
site/
├── astro.config.mjs          # Astro configuration
├── src/
│   ├── layouts/
│   │   ├── Base.astro        # Base HTML shell
│   │   └── Docs.astro        # Docs layout with sidebar
│   ├── pages/
│   │   ├── index.astro       # Homepage
│   │   └── docs/
│   │       └── [...slug].astro  # Dynamic docs pages
│   ├── content/
│   │   ├── config.ts         # Content collection schema
│   │   └── docs/             # Markdown content
│   ├── components/
│   │   ├── Header.astro
│   │   ├── Sidebar.astro
│   │   └── CodeBlock.astro
│   └── styles/
│       └── global.css
├── public/                    # Static assets
└── package.json
```

### Key Configuration

```javascript
// astro.config.mjs
import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://example.dev',
  output: 'static',           // fully static output
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
    },
  },
});
```

### Build Commands

```bash
# Development server
npx astro dev

# Production build
npx astro build                # outputs to dist/

# Preview production build locally
npx astro preview
```

## Eleventy Setup

### Project Structure

```
site/
├── .eleventy.js              # Configuration
├── src/
│   ├── _includes/            # Layouts and partials
│   │   ├── base.njk          # Base layout
│   │   ├── docs.njk          # Docs layout
│   │   ├── header.njk
│   │   └── footer.njk
│   ├── _data/                # Global data files
│   │   └── site.json         # Site metadata
│   ├── docs/                 # Docs content
│   │   ├── docs.json         # Default front matter for section
│   │   ├── getting-started.md
│   │   └── configuration.md
│   ├── index.md              # Homepage
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
└── package.json
```

### Key Configuration

```javascript
// .eleventy.js
module.exports = function(eleventyConfig) {
  // Copy static assets
  eleventyConfig.addPassthroughCopy("src/css");
  eleventyConfig.addPassthroughCopy("src/js");

  // Syntax highlighting
  const syntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");
  eleventyConfig.addPlugin(syntaxHighlight);

  return {
    dir: {
      input: "src",
      output: "_site",
      includes: "_includes",
      data: "_data"
    },
    markdownTemplateEngine: "njk"
  };
};
```

### Build Commands

```bash
# Development server
npx @11ty/eleventy --serve

# Production build
npx @11ty/eleventy              # outputs to _site/

# Build with path prefix
npx @11ty/eleventy --pathprefix="/docs/"
```

## Deployment

### Static File Hosting Options

| Host | Free Tier | Custom Domain | Build Integration | Best For |
|------|----------|---------------|-------------------|----------|
| **GitHub Pages** | Yes | Yes | GitHub Actions | GitHub-hosted projects |
| **Cloudflare Pages** | Yes | Yes | Git integration | Performance-critical |
| **Netlify** | Yes | Yes | Git integration | Easy setup |
| **Vercel** | Yes | Yes | Git integration | Astro/Next.js projects |
| **Self-hosted (Caddy/Nginx)** | N/A | Yes | Manual/CI | Full control |

### GitHub Actions Build (first-party Pages flow)

Use GitHub's first-party `actions/configure-pages` + `actions/upload-pages-artifact` + `actions/deploy-pages` flow. This is the supported path since 2023; the older third-party `peaceiris/actions-gh-pages` is no longer recommended.

```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages
on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      # Pick ONE of the build steps:
      - name: Build (Astro / Starlight)
        run: npm ci && npx astro build
      # - name: Build (Hugo)
      #   run: hugo --minify
      # - name: Build (Eleventy)
      #   run: npm ci && npx @11ty/eleventy
      # - name: Build (VitePress)
      #   run: npm ci && npx vitepress build docs
      - uses: actions/upload-pages-artifact@v3
        with:
          path: ./dist  # or ./public (Hugo) or ./_site (Eleventy) or ./docs/.vitepress/dist (VitePress)

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

### Caching Headers

For static sites behind a reverse proxy:

```
# Caddy
example.dev {
    root * /var/www/site
    file_server

    # Cache static assets aggressively
    @static path *.css *.js *.png *.jpg *.svg *.woff2
    header @static Cache-Control "public, max-age=31536000, immutable"

    # Don't cache HTML (so deployments take effect immediately)
    @html path *.html
    header @html Cache-Control "public, max-age=0, must-revalidate"
}
```

## Build-Time Optimizations

- **CSS**: Inline critical CSS in `<head>`, load the rest async or at end of body
- **Images**: Use `<picture>` with WebP/AVIF sources, appropriate `srcset` for responsive sizes
- **Fonts**: Use `font-display: swap`, preload critical fonts, prefer system font stacks
- **HTML**: Minify in production builds (Hugo does this with `--minify`)
- **Search index**: Generate at build time (Pagefind runs post-build)

## Quality Additions for Tooling

In addition to the base quality checklist, verify:
- [ ] Build completes without warnings
- [ ] All markdown content renders correctly (tables, code blocks, images)
- [ ] Internal links resolve (run a link checker: `htmltest`, `lychee`, or similar)
- [ ] RSS feed generates correctly (if applicable)
- [ ] Sitemap generates correctly
- [ ] 404 page exists and is styled
- [ ] Production build output is clean (no draft content, no dev artifacts)
- [ ] Build is reproducible (same output from clean checkout)
