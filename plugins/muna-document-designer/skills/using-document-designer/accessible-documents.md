# Accessible Document Design

Reference sheet for creating documents that are usable by people with disabilities — including screen reader users, people with low vision, color vision deficiency, and cognitive or motor impairments. Load this alongside the document-designer agent when document accessibility matters, which should be most of the time.

## When to Use This Reference

- Any document intended for public distribution
- Documents required to meet accessibility standards (Section 508, EN 301 549, WCAG)
- Documents that will be read by people using assistive technology
- Documents for organizations with accessibility policies
- When the user mentions "accessible", "screen reader", "PDF/UA", "Section 508", or "WCAG"

## Core Principles

1. **Perceivable**: Information must be presentable in ways all users can perceive — not solely through color, not solely through vision, not solely through hearing
2. **Operable**: Navigation must work for keyboard and assistive technology users
3. **Understandable**: Content and layout must be predictable and readable
4. **Robust**: Documents must work across different assistive technologies

## Document Structure (Tagged PDF)

### Why Tags Matter

Screen readers navigate PDFs using the tag tree — a structural representation of the document's logical reading order. An untagged PDF is essentially an image to a screen reader.

Typst generates tagged PDF output. Key considerations:

- **Heading hierarchy**: Use `= Heading 1`, `== Heading 2`, etc. — never skip levels (no jumping from H1 to H3)
- **Reading order**: The source order in Typst determines the tag tree order. Decorative elements placed with `place()` may appear out of reading order
- **Lists**: Use Typst's list syntax (`-`, `+`, `1.`) — don't fake lists with manual bullets and line breaks
- **Tables**: Use Typst's `table()` function with `table.header()` — screen readers need header cell association

### Heading Structure

```typst
// CORRECT: sequential heading levels
= Chapter Title          // H1
== Section               // H2
=== Subsection           // H3

// WRONG: skipped level
= Chapter Title          // H1
=== Subsection           // H3 — screen reader users lose context
```

### Table Accessibility

```typst
// CORRECT: explicit header row
#table(
  columns: 3,
  table.header([*Name*], [*Role*], [*Email*]),
  [Alice], [Engineer], [alice\@example.com],
  [Bob], [Designer], [bob\@example.com],
)

// Complex tables with row headers
// Ensure the first column acts as a row header when appropriate
// Screen readers need to associate data cells with both row and column headers
```

## Color & Contrast

### Minimum Contrast Ratios (WCAG 2.1)

| Element | Level AA | Level AAA |
|---------|---------|----------|
| Normal text (< 18pt) | 4.5:1 | 7:1 |
| Large text (≥ 18pt or 14pt bold) | 3:1 | 4.5:1 |
| UI components, graphical objects | 3:1 | 3:1 |

### Testing Contrast

Use the relative luminance formula or an online tool. Common failures:

| Combination | Ratio | Verdict |
|------------|-------|---------|
| Black on white | 21:1 | Passes all levels |
| Dark gray (#333) on white | 12.6:1 | Passes all levels |
| Medium gray (#767676) on white | 4.5:1 | Passes AA normal text |
| Light gray (#999) on white | 2.8:1 | Fails all levels |
| White on yellow (#ffd700) | 1.3:1 | Fails — nearly invisible |

### Color Independence

Never convey information through color alone:

```typst
// WRONG: status indicated only by color
#text(fill: red, "Failed")
#text(fill: green, "Passed")

// CORRECT: color + text/shape indicator
#text(fill: rgb("#e53e3e"), [✗ Failed])
#text(fill: rgb("#38a169"), [✓ Passed])

// CORRECT: color + pattern in charts
// Use distinct patterns (solid, dashed, dotted) alongside colors
```

### Color Vision Deficiency (CVD) Safe Palettes

Avoid red-green distinctions. Prefer palettes distinguishable under all common forms of CVD:

```typst
// CVD-safe categorical palette (Wong, 2011)
#let cvd-orange = rgb("#E69F00")
#let cvd-sky-blue = rgb("#56B4E9")
#let cvd-green = rgb("#009E73")
#let cvd-yellow = rgb("#F0E442")
#let cvd-blue = rgb("#0072B2")
#let cvd-vermillion = rgb("#D55E00")
#let cvd-purple = rgb("#CC79A7")
```

## Typography for Readability

### Font Selection

- **Body text**: Sans-serif fonts are generally more readable on screen; serif fonts work well in print. Avoid decorative or thin-weight fonts for body text.
- **Minimum size**: 11-12pt for print body text, never below 9pt for any content (including footnotes)
- **Line height**: 1.4-1.6× font size for body text — cramped leading impairs readability for everyone
- **Line length**: 45-75 characters per line is optimal. Wider lines cause tracking errors; narrower lines cause excessive saccades.

### Paragraph Styling

```typst
// Accessible body text defaults
#set text(size: 11pt, lang: "en")
#set par(
  leading: 0.8em,        // ~1.5× line height
  spacing: 1.2em,        // clear paragraph separation
  justify: false,        // ragged-right is more readable for dyslexic readers
)
```

### Avoid These

- **All caps for body text**: Harder to read (letter shape recognition is reduced)
- **Italic for long passages**: Harder to read; use for emphasis only
- **Justified text in narrow columns**: Creates uneven word spacing ("rivers")
- **Text over images**: Destroys contrast — always use a solid or semi-opaque overlay

## Alternative Text

### Images and Figures

Every non-decorative image needs alternative text. In Typst:

```typst
// Figure caption serves as the accessible description
#figure(
  image("architecture.png", width: 80%,
    alt: "System architecture showing three microservices connected via message queue"
  ),
  caption: [System architecture overview.],
)

// Decorative images should have empty alt text
// (Typst doesn't yet have a standard way to mark images as decorative —
//  use an empty alt string and note this limitation)
```

### Charts and Data Visualizations

Charts need both:
1. **Alt text**: Brief description of what the chart shows
2. **Data table**: The underlying data in an accessible table format, either inline or in an appendix

### Complex Diagrams

For flowcharts, architecture diagrams, and other complex visuals:
- Provide a text description of the relationships and flow
- Consider a simplified version alongside the detailed one
- Link to a text-based alternative (e.g., a bulleted list of connections)

## Navigation

### Table of Contents

- Generate a ToC for any document longer than 5 pages
- ToC entries should be navigable links in the PDF
- Ensure ToC entries match actual heading text exactly

### Bookmarks

Typst generates PDF bookmarks from headings automatically. Verify:
- All major sections appear in the bookmark panel
- Bookmark hierarchy matches heading hierarchy
- Bookmark text is meaningful (not "Section 1" but the actual title)

### Page Numbers

- Include page numbers on every page (except cover)
- Match the numbering system in the ToC
- Consider both printed page numbers and logical page labels

## Language Declaration

```typst
// Set the document language for screen reader pronunciation
#set text(lang: "en")  // or "fr", "de", "es", etc.

// For multilingual documents, set per-block
#block(text(lang: "fr")[Bonjour, comment allez-vous?])
```

## PDF/UA Compliance

PDF/UA (Universal Accessibility, ISO 14289) is the accessibility standard for PDF. Key requirements:

1. **All content is tagged** — no untagged text or images
2. **Reading order is logical** — matches visual order
3. **All images have alt text** — or are marked as decorative (artifacts)
4. **Language is declared** — both document-level and for language changes
5. **Headings are properly nested** — no skipped levels
6. **Tables have headers** — associated with data cells
7. **Links have descriptive text** — not "click here"
8. **Color is not the sole means** of conveying information
9. **Document title is set** in metadata

```typst
// Set document metadata
#set document(
  title: "Quarterly Performance Report",
  author: "Planning Division",
)
```

## Pandoc Considerations

- Pandoc generates tagged PDF when using `--pdf-engine=typst`
- Set `lang` in YAML front matter for language declaration
- Use Pandoc's figure syntax with alt text: `![Alt text](image.png "Optional title")`
- `--toc` generates a linked table of contents
- `--metadata=title:"Document Title"` sets the PDF title metadata

## Quality Additions for Accessible Documents

In addition to the base quality checklist, verify:
- [ ] All headings follow sequential hierarchy (no skipped levels)
- [ ] All images have meaningful alt text (or are marked decorative)
- [ ] Color contrast meets WCAG AA minimum (4.5:1 for text, 3:1 for large text)
- [ ] No information is conveyed solely through color
- [ ] Document language is set in metadata
- [ ] Table headers are properly marked with `table.header()`
- [ ] Reading order in the tag tree matches visual reading order
- [ ] Document title is set in PDF metadata
- [ ] Body text is at least 11pt with adequate line spacing
- [ ] Links have descriptive text (not bare URLs or "click here")
- [ ] Table of contents is present and linked for documents > 5 pages
- [ ] PDF bookmarks are generated and correctly nested
