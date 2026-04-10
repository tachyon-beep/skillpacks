# Static Site Tooling

Reference sheet for choosing and configuring static site generators. Load this when the user needs to set up the build toolchain for a documentation or project site.

## When to Use This Reference

- Choosing between Hugo, Astro, Eleventy, Jekyll, or plain HTML
- Configuring a static site generator for a new project
- Setting up build pipelines, templating, and deployment
- Migrating between static site generators

## Decision Matrix

### Choosing a Generator

| Factor | Hugo | Astro | Eleventy | Plain HTML |
|--------|------|-------|----------|------------|
| **Speed** | Fastest builds | Fast | Fast | N/A |
| **Language** | Go templates | JS/TS | JS | N/A |
| **Learning curve** | Moderate | Moderate | Low | None |
| **Templating** | Go html/template | JSX/Astro | Nunjucks/Liquid/etc | Manual |
| **Content** | Markdown + front matter | Markdown + MDX | Markdown + data files | Manual |
| **JS framework support** | None (by design) | React/Vue/Svelte islands | None (by design) | Manual |
| **Best for** | Docs sites, blogs | Content + interactive bits | Simple sites, blogs | < 5 pages |
| **Binary dependency** | Yes (Go) | Yes (Node) | Yes (Node) | None |
| **Ecosystem** | Themes, modules | Components, integrations | Plugins | N/A |

### Quick Recommendations

- **< 5 pages, no blog**: Plain HTML/CSS. No build step needed.
- **Documentation site (10-100 pages)**: Hugo or Eleventy. Hugo for speed, Eleventy for flexibility.
- **Project site with interactive elements**: Astro (island architecture — static by default, JS where needed).
- **Blog + docs**: Hugo (built-in taxonomy, RSS, pagination).
- **Maximum simplicity**: Eleventy with Nunjucks templates.
- **Must avoid Node.js**: Hugo (single Go binary).

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
│       └── 2024-01-release.md
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

### GitHub Actions Build

```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: |
          # Hugo
          hugo --minify
          # OR Eleventy
          npm ci && npx @11ty/eleventy
          # OR Astro
          npm ci && npx astro build
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./public  # or ./_site or ./dist
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
