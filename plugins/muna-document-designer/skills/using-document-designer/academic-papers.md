# Academic & Research Paper Design

Reference sheet for formatting academic papers, journal submissions, conference proceedings, and research documents. Load this alongside the document-designer agent when the document targets an academic audience or follows scholarly conventions.

## When to Use This Reference

- Formatting research papers for journal or conference submission
- Creating thesis or dissertation templates
- Documents with heavy citation usage and bibliographies
- Preprint or working paper formatting
- Any document requiring author affiliations, abstracts, or academic metadata

## Document Styling Patterns

### Journal-Style Papers
- **Title block**: Title, author names, affiliations, correspondence email, ORCID links — often with footnote-style affiliation markers (superscript numbers or symbols)
- **Abstract**: Indented or boxed block, typically 150-300 words, sometimes with keyword list below
- **Two-column body**: Common for conference papers (IEEE, ACM style) — requires careful figure/table placement
- **Single-column body**: Common for preprints (arXiv style) and many journals — wider margins, more readable
- **Section numbering**: Usually flat (1, 2, 3) or two-level (1.1, 1.2) — deeper nesting is discouraged

### Citation & Bibliography
- **In-text citations**: Numeric [1], author-year (Smith, 2024), or author-number (Smith [1]) depending on style
- **Bibliography formatting**: Consistent entry format per style guide (APA, IEEE, Chicago, Vancouver)
- **Cross-references**: "Figure 1", "Table 2", "Equation (3)" — always capitalized when named
- **Footnotes vs endnotes**: Field-dependent; humanities prefer footnotes, sciences prefer endnotes or none

### Mathematical Content
- **Equation numbering**: Right-aligned, sequential or by section (1.1, 1.2)
- **Inline vs display math**: Display for important results, inline for variables and short expressions
- **Theorem environments**: Theorem, Lemma, Corollary, Proof — each with distinct styling (often italic body, bold label)
- **Notation tables**: Symbol/meaning pairs, typically in an early section or appendix

### Figures & Tables
- **Captions**: Figures captioned below, tables captioned above (academic convention)
- **Figure numbering**: Sequential (Figure 1, 2, 3) or by section (Figure 1.1, 1.2)
- **Subfigures**: (a), (b), (c) labels with a shared caption
- **Table notes**: Footnotes within the table environment, using symbols (*, †, ‡) not numbers

### Front & Back Matter
- **Acknowledgments**: Funding sources, collaborator thanks — before references
- **Author contributions**: CRediT taxonomy or prose description
- **Data availability**: Statement about where data/code can be accessed
- **Supplementary material**: Appendices with separate numbering (A, B, C)
- **Conflict of interest**: Disclosure statement, often templated

## Typst Implementation Notes

### Author Affiliation Block

```typst
#let author-block(authors) = {
  // authors: array of (name: str, affil: int, email: str)
  let names = authors.map(a =>
    [#a.name#super[#a.affil]]
  )
  align(center)[
    #names.join(", ")
  ]
}

#let affiliations(affils) = {
  // affils: array of (id: int, institution: str)
  align(center, text(size: 9pt,
    affils.map(a => [#super[#a.id] #a.institution]).join(linebreak())
  ))
}
```

### Abstract Box

```typst
#let abstract-block(content) = block(
  width: 85%,
  inset: (x: 0pt, y: 8pt),
  [
    #align(center, text(weight: "bold", size: 10pt, "Abstract"))
    #v(4pt)
    #set text(size: 9.5pt)
    #set par(justify: true)
    #content
  ]
)
```

### Two-Column Layout with Full-Width Elements

```typst
// Switch to two columns for body
#show: columns.with(2, gutter: 12pt)

// Full-width figure spanning both columns
#place(top + center, scope: "parent", float: true,
  figure(
    image("wide-figure.png", width: 100%),
    caption: [A wide figure spanning both columns.]
  )
)
```

### Theorem Environments

```typst
#let theorem-counter = counter("theorem")

#let theorem(title: none, body) = block(
  width: 100%, inset: (x: 0pt, y: 6pt),
  {
    theorem-counter.step()
    [*Theorem #context theorem-counter.display().*]
    if title != none [ _(#title)_ ]
    [ ]
    emph(body)
  }
)

#let proof(body) = block(
  width: 100%, inset: (x: 0pt, y: 6pt),
  [_Proof._ #body #h(1fr) $square$]
)
```

## Pandoc Considerations

- Use `--citeproc` with appropriate CSL style files for citation formatting
- `--bibliography=refs.bib` for BibTeX/BibLaTeX source files
- `--number-sections` for automatic section numbering
- CSL styles available for most journals at the [CSL Style Repository](https://github.com/citation-style-language/styles)
- YAML metadata fields: `title`, `author` (array with `name`, `affiliation`), `abstract`, `keywords`, `date`, `bibliography`
- For two-column output, configure in the Typst template rather than Pandoc options

## Quality Additions for Academic Papers

In addition to the base quality checklist, verify:
- [ ] All citations resolve — no "[?]" or missing references
- [ ] Bibliography entries are complete (no missing years, publishers, or page numbers)
- [ ] Figure captions are below figures, table captions are above tables
- [ ] Equation numbers are sequential and all referenced equations have numbers
- [ ] Author affiliations match the correct superscript markers
- [ ] Abstract is within the target word count for the venue
- [ ] Acknowledgments include all required funding disclosures
- [ ] Page/word count is within submission limits (if applicable)
- [ ] Fonts embed correctly in PDF (no substitution warnings)
