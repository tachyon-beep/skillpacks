# Standards & Specification Document Design

Reference sheet for formatting formal specifications, compliance documents, and standards-style publications. Load this alongside the document-designer agent when the document must conform to standards-body conventions.

## When to Use This Reference

- Formatting documents that follow ISO, NIST, RFC, or similar standards conventions
- Creating specification documents with normative/informative language distinctions
- Documents requiring formal conformance clauses, definition tables, or cross-reference systems
- Corporate compliance or governance documents with classification markings and document control

## Document Styling Patterns

### Specification Documents (ISO/NIST/RFC Style)
- **Heading hierarchy**: Strictly numbered (1, 1.1, 1.1.1) — no decorative styling, clarity over aesthetics
- **Normative language**: Terms like "SHALL", "MUST", "SHOULD", "MAY" styled distinctly (often small caps or bold) per RFC 2119
- **Cross-reference formatting**: Internal references use section numbers ("see Section 3.2"), not page numbers
- **Definition tables**: Term/definition pairs with consistent column widths, alphabetized or grouped by domain
- **Conformance clauses**: Clearly separated sections listing testable requirements, often with requirement IDs (e.g., REQ-001)
- **Annexes**: Normative vs informative annexes distinguished in heading (e.g., "Annex A (normative)")

### Technical Reports with Compliance Elements
- **Executive summaries**: Structured findings with severity/priority indicators
- **Risk matrices**: Color-coded likelihood × impact grids with consistent legend
- **Findings tables**: ID, description, severity, status, recommendation columns — often the most table-heavy document type
- **Evidence sections**: Cross-references to supporting artifacts, audit trail formatting

### Corporate Governance Documents
- **Classification markings**: Header/footer banners for sensitivity level (e.g., PUBLIC, INTERNAL, CONFIDENTIAL)
- **Document control blocks**: Version, date, author, reviewer, approver in a front-matter table
- **Revision history tables**: Version, date, author, changes — typically on page 2 or in an annex
- **Distribution lists**: Formatted tables of recipients and their roles
- **Branded headers/footers**: Organization logo, document ID, page numbering, classification level

## Typst Implementation Notes

### Normative Language Highlighting

```typst
// Highlight RFC 2119 keywords
#show regex("(SHALL|MUST|SHOULD|MAY|SHALL NOT|MUST NOT|SHOULD NOT)"): set text(
  weight: "bold",
  font: "small-caps"  // or use features: ("smcp",)
)
```

### Classification Banner

```typst
// Repeating classification marking in header
#set page(header: align(center,
  block(fill: rgb("#fbbf24"), width: 100%, inset: 4pt,
    text(weight: "bold", size: 8pt, "INTERNAL — OFFICIAL USE ONLY")
  )
))
```

### Document Control Table

```typst
#let doc-control(title, version, status, date, author, approver) = {
  block(stroke: 0.5pt + luma(180), inset: 12pt, width: 100%)[
    #grid(columns: (1fr, 1fr),
      [*Document:* #title], [*Version:* #version],
      [*Status:* #status], [*Date:* #date],
      [*Author:* #author], [*Approved by:* #approver],
    )
  ]
}
```

### Requirement ID Formatting

```typst
// Styled requirement identifiers
#let req(id) = box(
  fill: luma(240), inset: (x: 4pt, y: 2pt), radius: 2pt,
  text(font: "monospace", size: 9pt, weight: "bold", id)
)

// Usage: #req("REQ-042") The system shall...
```

## Pandoc Considerations

- Use `--number-sections` for automatic section numbering
- YAML metadata for document control fields: `title`, `version`, `status`, `classification`, `author`, `date`
- Lua filters can enforce normative language highlighting and requirement ID formatting
- CSL styles exist for standards citations (ISO 690, etc.)

## Quality Additions for Standards Documents

In addition to the base quality checklist, verify:
- [ ] All normative terms (SHALL, MUST, etc.) are used consistently per RFC 2119
- [ ] Section numbering is unbroken and sequential
- [ ] All cross-references resolve (no "see Section ?" broken links)
- [ ] Requirement IDs are unique and sequential
- [ ] Classification markings appear on every page including cover
- [ ] Document control block is complete (no blank fields)
- [ ] Normative and informative annexes are clearly distinguished
