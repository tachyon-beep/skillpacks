# Data-Heavy Document Design

Reference sheet for documents dominated by tables, charts, figures, and quantitative content. Load this alongside the document-designer agent when the document has more data than prose — dashboards, data appendices, audit reports, inventory listings, or any document where table and figure layout is the primary design challenge.

## When to Use This Reference

- Documents where tables occupy more than 30% of page area
- Reports with large data appendices or inventory tables
- Dashboard-style PDF summaries with metrics and charts
- Documents requiring landscape pages for wide tables
- Any document where "the tables look broken" is the likely failure mode

## Table Design Patterns

### Column Width Strategy

The most common source of visual defects in data-heavy documents. Follow this decision tree:

1. **Identify the widest content** in each column (header or body, whichever is longer)
2. **Assign minimum widths** based on content type:
   - Short values (Yes/No, numbers, status codes): 5-8%
   - Identifiers, enum values, short labels: 12-20%
   - Sentences or descriptions: 30-50%
   - Single words or abbreviations: 8-12%
3. **Verify totals** sum to 100% or less (Typst distributes remainder)
4. **Build and visually verify** — column widths that look right in source often break in render

### Large Table Handling

```typst
// Always make large tables breakable
#set table(stroke: none)

#figure(
  block(breakable: true,
    table(
      columns: (15%, 25%, 40%, 20%),
      // Repeat header on each page
      table.header(
        [*ID*], [*Category*], [*Description*], [*Status*],
      ),
      // ... rows
    )
  ),
  caption: [Inventory of findings.],
)
```

### Alternating Row Styling

```typst
// Zebra striping for readability
#let zebra-table(headers, ..rows) = {
  let data = rows.pos()
  table(
    columns: headers.len(),
    fill: (_, y) => if y == 0 { rgb("#2c5282") }
      else if calc.odd(y) { rgb("#f7fafc") }
      else { white },
    table.header(..headers.map(h => text(fill: white, weight: "bold", h))),
    ..data.flatten()
  )
}
```

### Compact Table Typography

```typst
// Data tables should use smaller text than body
#show figure.where(kind: table): set text(size: 9pt)
#show figure.where(kind: table): set par(justify: false)  // never justify in tables
```

## Figure & Chart Layout

### Figure Sizing Strategy

| Content Type | Width | Height Constraint | Notes |
|-------------|-------|-------------------|-------|
| Wide charts (bar, line, area) | 90-100% | Max 40% page height | Landscape page if > 6 columns |
| Tall diagrams (flowcharts, trees) | 60-80% | Max 70% page height | Break into parts if taller |
| Small icons/badges | 15-30% | Natural height | Float alongside text |
| Screenshots | 80-90% | Max 50% page height | Add border for definition |
| Comparison grids | 100% | Variable | Use subfigure layout |

### Landscape Pages for Wide Content

```typst
// Rotate a single page to landscape
#page(flipped: true)[
  #figure(
    table(
      columns: (8%, 12%, 15%, 15%, 15%, 10%, 10%, 15%),
      // ... wide table with many columns
    ),
    caption: [Full comparison matrix — landscape for readability.],
  )
]
```

### Figure Numbering and Cross-References

```typst
// Separate counters for figures and tables
// (Typst handles this automatically with figure kinds)

// Reference figures by label
See @fig:overview for the architecture diagram.

#figure(
  image("overview.png", width: 80%),
  caption: [System architecture overview.],
) <fig:overview>
```

## Dashboard-Style Layouts

### Metric Cards

```typst
#let metric-card(label, value, trend: none, color: rgb("#2c5282")) = {
  box(
    width: 100%, inset: 12pt, radius: 6pt,
    stroke: 0.5pt + luma(200),
    fill: color.lighten(95%),
    [
      #text(size: 9pt, fill: luma(100), upper(label))
      #v(4pt)
      #text(size: 24pt, weight: "bold", fill: color, value)
      #if trend != none {
        h(8pt)
        text(size: 10pt, fill: if trend.starts-with("+") { rgb("#38a169") } else { rgb("#e53e3e") }, trend)
      }
    ]
  )
}

// Usage: 3-up metric row
#grid(columns: (1fr, 1fr, 1fr), gutter: 12pt,
  metric-card("Total Issues", "142", trend: "+12"),
  metric-card("Resolved", "98", trend: "+8"),
  metric-card("Open", "44", trend: "-4", color: rgb("#e53e3e")),
)
```

### Summary + Detail Pattern

For reports that need both overview and drill-down:

1. **Summary page**: Metric cards, key charts, executive findings (1-2 pages)
2. **Detail sections**: Full data tables, per-category breakdowns
3. **Data appendix**: Raw tables, complete listings, reference data

Use section divider pages between summary and detail to create visual separation.

## Pandoc Considerations

- Pandoc-generated column widths from Markdown pipe tables are almost always wrong — always override in the Typst template
- For very large tables, consider generating Typst source directly rather than going through Pandoc
- Use `--columns=120` or similar to prevent Pandoc from wrapping table content prematurely
- Long tables in Markdown may need manual page break hints in the template

## Quality Additions for Data-Heavy Documents

In addition to the base quality checklist, verify:
- [ ] Every table is breakable (`breakable: true`) — test with enough rows to trigger a page break
- [ ] Table headers repeat on continuation pages
- [ ] No column content overflows — verify by reading the rendered PDF
- [ ] Landscape pages are used for tables wider than portrait margins allow
- [ ] All figures have explicit width AND height constraints (not just width)
- [ ] Figure/table numbering is sequential and unbroken
- [ ] Data appendix tables use compact typography (8-9pt)
- [ ] Zebra striping or horizontal rules aid row tracking in long tables
- [ ] Numeric columns are right-aligned, text columns are left-aligned
- [ ] Units are stated in column headers, not repeated in every cell
