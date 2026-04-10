# Code Block & Developer UX Patterns

Reference sheet for implementing code block presentation, copy-to-clipboard, syntax highlighting, language tabs, and other interactive patterns developers expect on technical sites. Load this when building pages with significant code content.

## When to Use This Reference

- Implementing syntax-highlighted code blocks
- Adding copy-to-clipboard functionality
- Building tabbed code examples (multiple languages/platforms)
- Styling terminal/command-line output
- Any page where code presentation quality affects the user experience

## Code Block Anatomy

A well-designed code block has:

```
┌──────────────────────────────────────────┐
│ Python                          [Copy]   │  ← header bar
├──────────────────────────────────────────┤
│ from project import Client               │
│                                          │  ← code content
│ client = Client(api_key="...")            │
│ result = client.analyze("input.py")      │
│ print(result.summary)                    │
└──────────────────────────────────────────┘
```

### HTML Structure

```html
<div class="code-block">
  <div class="code-block__header">
    <span class="code-block__lang">python</span>
    <button class="code-block__copy" aria-label="Copy code">
      <svg><!-- clipboard icon --></svg>
      <span class="code-block__copy-text">Copy</span>
    </button>
  </div>
  <pre><code class="language-python">from project import Client

client = Client(api_key="...")
result = client.analyze("input.py")
print(result.summary)</code></pre>
</div>
```

### CSS

```css
.code-block {
  border: 1px solid var(--color-border);
  border-radius: 8px;
  overflow: hidden;
  margin: var(--space-4) 0;
}

.code-block__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--space-1) var(--space-3);
  background: var(--color-bg-code);
  border-bottom: 1px solid var(--color-border);
  font-size: var(--text-sm);
}

.code-block__lang {
  color: var(--color-text-muted);
  font-family: var(--font-code);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-size: var(--text-xs);
}

.code-block__copy {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  background: none;
  border: none;
  color: var(--color-text-muted);
  cursor: pointer;
  padding: var(--space-1);
  border-radius: 4px;
  font-size: var(--text-sm);
}

.code-block__copy:hover {
  color: var(--color-accent);
  background: var(--color-bg-subtle);
}

.code-block pre {
  margin: 0;
  padding: var(--space-3);
  overflow-x: auto;
  background: var(--color-bg-code-block);
  color: var(--color-code-text);
}

.code-block code {
  font-family: var(--font-code);
  font-size: 0.875rem;
  line-height: 1.6;
}
```

### Copy-to-Clipboard JavaScript

```javascript
document.querySelectorAll('.code-block__copy').forEach(button => {
  button.addEventListener('click', async () => {
    const code = button.closest('.code-block').querySelector('code').textContent;
    try {
      await navigator.clipboard.writeText(code);
      button.querySelector('.code-block__copy-text').textContent = 'Copied!';
      setTimeout(() => {
        button.querySelector('.code-block__copy-text').textContent = 'Copy';
      }, 2000);
    } catch (err) {
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = code;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      button.querySelector('.code-block__copy-text').textContent = 'Copied!';
      setTimeout(() => {
        button.querySelector('.code-block__copy-text').textContent = 'Copy';
      }, 2000);
    }
  });
});
```

## Language Tabs

For examples that show the same concept in multiple languages:

```html
<div class="code-tabs" role="tablist" aria-label="Code examples">
  <button role="tab" aria-selected="true" aria-controls="panel-python"
          id="tab-python">Python</button>
  <button role="tab" aria-selected="false" aria-controls="panel-js"
          id="tab-js">JavaScript</button>
  <button role="tab" aria-selected="false" aria-controls="panel-go"
          id="tab-go">Go</button>
</div>

<div class="code-panel" role="tabpanel" id="panel-python"
     aria-labelledby="tab-python">
  <div class="code-block">
    <pre><code class="language-python">client = Client(api_key="...")</code></pre>
  </div>
</div>

<div class="code-panel" role="tabpanel" id="panel-js"
     aria-labelledby="tab-js" hidden>
  <div class="code-block">
    <pre><code class="language-javascript">const client = new Client({ apiKey: '...' });</code></pre>
  </div>
</div>
```

### Tab Switching JavaScript

```javascript
document.querySelectorAll('.code-tabs').forEach(tablist => {
  const tabs = tablist.querySelectorAll('[role="tab"]');
  const panels = tablist.parentElement.querySelectorAll('[role="tabpanel"]');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      // Deactivate all
      tabs.forEach(t => t.setAttribute('aria-selected', 'false'));
      panels.forEach(p => p.hidden = true);

      // Activate selected
      tab.setAttribute('aria-selected', 'true');
      const panel = document.getElementById(tab.getAttribute('aria-controls'));
      panel.hidden = false;
    });

    // Keyboard navigation
    tab.addEventListener('keydown', (e) => {
      const index = Array.from(tabs).indexOf(tab);
      let next;
      if (e.key === 'ArrowRight') next = tabs[(index + 1) % tabs.length];
      if (e.key === 'ArrowLeft') next = tabs[(index - 1 + tabs.length) % tabs.length];
      if (next) { next.focus(); next.click(); }
    });
  });

  // Persist tab preference across page navigation
  const saved = localStorage.getItem('code-tab-pref');
  if (saved) {
    const match = Array.from(tabs).find(t => t.textContent.trim() === saved);
    if (match) match.click();
  }
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      localStorage.setItem('code-tab-pref', tab.textContent.trim());
    });
  });
});
```

## Terminal / Command-Line Blocks

For shell commands, distinguish input from output:

```html
<div class="code-block code-block--terminal">
  <div class="code-block__header">
    <span class="code-block__lang">Terminal</span>
    <button class="code-block__copy" aria-label="Copy command">Copy</button>
  </div>
  <pre><code><span class="terminal-prompt">$</span> pip install projectname
<span class="terminal-output">Successfully installed projectname-1.0.0</span>

<span class="terminal-prompt">$</span> projectname --version
<span class="terminal-output">projectname 1.0.0</span></code></pre>
</div>
```

```css
.terminal-prompt {
  color: var(--color-accent);
  user-select: none;  /* don't include $ when copying */
}

.terminal-output {
  color: var(--color-text-muted);
}
```

**Important**: The copy button should copy only the commands (lines starting with `$`), not the prompt character or output lines.

## Inline Code

```css
/* Inline code in body text */
:not(pre) > code {
  font-family: var(--font-code);
  font-size: 0.875em;  /* relative to surrounding text */
  padding: 0.15em 0.35em;
  background: var(--color-bg-code);
  border-radius: 3px;
  word-break: break-word;  /* prevent long identifiers from overflowing */
}
```

## Syntax Highlighting

### Build-Time (Preferred)

Generate highlighted HTML at build time — zero client-side JS needed:

| Tool | Integration | Theme Format |
|------|------------|-------------|
| **Shiki** | Astro, custom build | VS Code themes |
| **Chroma** | Hugo built-in | Chroma styles |
| **Prism** | Build plugin or client-side | CSS themes |
| **Highlight.js** | Build plugin or client-side | CSS themes |

Shiki produces the best output (VS Code-quality highlighting) and works at build time with Astro or as a standalone transformer.

### Client-Side (Fallback)

If build-time highlighting isn't possible:

```html
<!-- In <head> -->
<link rel="stylesheet" href="/css/prism-theme.css">

<!-- Before </body> -->
<script src="/js/prism.js"></script>
```

### Theme Switching

For dark/light mode, you need two syntax themes:

```css
/* Light mode syntax theme */
:root {
  --syntax-keyword: #d73a49;
  --syntax-string: #032f62;
  --syntax-comment: #6a737d;
  --syntax-function: #6f42c1;
  --syntax-number: #005cc5;
}

/* Dark mode syntax theme */
:root[data-theme="dark"] {
  --syntax-keyword: #ff7b72;
  --syntax-string: #a5d6ff;
  --syntax-comment: #8b949e;
  --syntax-function: #d2a8ff;
  --syntax-number: #79c0ff;
}
```

## Heading Anchor Links

Every heading should be deep-linkable:

```html
<h2 id="configuration">
  Configuration
  <a href="#configuration" class="heading-anchor" aria-label="Link to this section">#</a>
</h2>
```

```css
.heading-anchor {
  color: var(--color-text-muted);
  text-decoration: none;
  margin-left: var(--space-2);
  opacity: 0;
  transition: opacity 0.15s;
}

h2:hover .heading-anchor,
h3:hover .heading-anchor,
.heading-anchor:focus {
  opacity: 1;
}
```

## Quality Additions for Code-Heavy Pages

In addition to the base quality checklist, verify:
- [ ] All code blocks have syntax highlighting applied
- [ ] Copy button works and provides feedback ("Copied!")
- [ ] Terminal blocks exclude the prompt character from copy
- [ ] Language tabs persist preference across pages (localStorage)
- [ ] Inline code doesn't overflow its container (word-break applied)
- [ ] Code font is monospaced and sized proportionally (0.875em of surrounding text)
- [ ] Horizontal scrollbar appears for long lines (not wrapping)
- [ ] Syntax theme matches light/dark mode
- [ ] Code blocks are accessible (can be navigated to and read by screen readers)
