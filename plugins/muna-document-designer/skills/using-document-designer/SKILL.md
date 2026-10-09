---
name: using-document-designer
description: "Use when producing or validating Pandoc/Typst documents, reusable templates, print-ready PDFs, multilingual layouts, or accessible publication artifacts."
---

# Pandoc and Typst Document Production

Use this pack for a selected production toolchain and its output constraints.
Native document/PDF/presentation tools may be more suitable for routine file
creation; do not introduce Typst or a delegated agent without a task reason.

## Production workflow

1. Inspect source content, requested format, audience, branding, page geometry,
   existing template and installed Pandoc/Typst versions. Identify print,
   multilingual, citation and accessibility requirements that affect production.
2. Reuse the existing toolchain where it meets the brief. For new packages/templates,
   check official documentation or Typst Universe and pin compatible versions.
   Record required compiler/package versions with the template.
3. Preserve content, citations, data, and semantic structure during conversion.
   Keep decorative elements distinct from reading-order content. Use real headings,
   lists, figures and table headers rather than visual imitations.
4. Compile, render and inspect representative pages plus known layout stressors:
   long headings, tables, code, captions, page breaks and mixed scripts. Verify
   fonts/fallback and text extraction when they affect the output.
5. For accessibility, request supported tagged/conformance output, run appropriate
   external validation and inspect reading order/alt text. Test with actual
   assistive technology where acceptance requires it; a compiler flag is not
   complete accessibility evidence.
6. For commercial print, confirm trim/bleed/gutter/color/profile requirements with
   the receiving printer and check the exported artifact against those values.

## Optional production references

Read only the section governing the artifact. References are beside this file.
Version tables and examples are starting points to verify, not live guarantees.

| Requirement | Reference |
|---|---|
| Citations, journal layouts, theses | [academic-papers.md](academic-papers.md) |
| Tagged PDF, reading order and validation | [accessible-documents.md](accessible-documents.md) |
| Dense tables, charts, landscape pages | [data-heavy-documents.md](data-heavy-documents.md) |
| RTL/CJK, mixed script and fallback | [multilingual-documents.md](multilingual-documents.md) |
| Bleed, marks, binding and color | [print-production.md](print-production.md) |
| Numbered clauses, conformance and markings | [standards-and-specifications.md](standards-and-specifications.md) |

Use `muna-technical-writer` for content evidence or `muna-wiki-management` for
source lineage only when the task crosses those boundaries. The
`document-designer` agent is an optional production role.

## Deliver

Provide editable sources, the exported artifact, reproducible build command and
required versions/assets. Report visual inspection, validation results and any
unverified print/accessibility requirements. Keep successful export distinct from
printer acceptance, standards conformance and human approval.
