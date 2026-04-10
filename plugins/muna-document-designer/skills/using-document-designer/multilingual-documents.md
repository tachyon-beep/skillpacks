# Multilingual & Internationalized Document Design

Reference sheet for documents involving multiple languages, scripts, or writing directions. Load this alongside the document-designer agent when the document contains non-Latin scripts, right-to-left text, CJK content, bilingual layouts, or must handle mixed-script typography correctly.

## When to Use This Reference

- Documents containing right-to-left (RTL) scripts: Arabic, Hebrew, Farsi, Urdu
- CJK (Chinese, Japanese, Korean) content with distinct typographic conventions
- Bilingual or multilingual parallel layouts (side-by-side or alternating)
- Documents requiring script-specific font fallback chains
- Any document where mixed scripts appear in the same paragraph or table

## Script-Specific Typography

### Right-to-Left (RTL) Scripts

- **Page mirroring**: RTL documents flip the entire page layout — binding on the right, page numbers on the left, marginalia swapped
- **Bidirectional (bidi) text**: Mixed LTR/RTL in the same paragraph requires proper Unicode bidi algorithm support
- **Numerals**: Arabic-script documents may use Eastern Arabic numerals (٠١٢٣) or Western Arabic (0123) — confirm with the audience
- **Punctuation**: Some punctuation marks mirror in RTL context (parentheses, brackets)
- **Tables**: Column order may reverse in fully RTL documents — header reads right-to-left

### CJK Typography

- **No word spacing**: CJK text has no spaces between words — line breaking occurs at any character boundary (with exceptions)
- **Punctuation width**: CJK punctuation is full-width; avoid double-width gaps when CJK punctuation meets Latin punctuation
- **Ruby text (furigana)**: Phonetic annotations above/beside kanji — requires special layout support
- **Vertical text**: Traditional CJK can be set vertically (top-to-bottom, right-to-left columns) — Typst has limited support, plan accordingly
- **Font size**: CJK characters are visually denser; body text may need to be 1-2pt larger than equivalent Latin text for comparable readability
- **Line height**: CJK text typically needs more generous leading (1.6-1.8× vs 1.4-1.5× for Latin)

### Indic Scripts (Devanagari, Tamil, Bengali, etc.)

- **Conjuncts and ligatures**: Complex shaping rules — font must support the target script's OpenType features
- **Hanging characters**: Some scripts have characters that extend below the baseline more than Latin descenders — increase line spacing
- **Numerals**: Each script has its own numeral set; decide whether to use native or Arabic numerals

## Bilingual & Parallel Layouts

### Side-by-Side (Two-Column Bilingual)

```typst
// Parallel bilingual: left column = primary language, right = translation
#let bilingual-page(primary, secondary) = {
  grid(
    columns: (1fr, 1fr),
    gutter: 16pt,
    primary,
    secondary,
  )
}

// With a vertical divider
#let bilingual-page-divided(primary, secondary) = {
  grid(
    columns: (1fr, auto, 1fr),
    gutter: 8pt,
    primary,
    line(length: 100%, angle: 90deg, stroke: 0.5pt + luma(200)),
    secondary,
  )
}
```

### Alternating Paragraphs

```typst
// Alternating: primary paragraph followed by translation in smaller/italic text
#let bilingual-block(primary, translation) = {
  block(inset: (bottom: 8pt), primary)
  block(
    inset: (left: 12pt, bottom: 16pt),
    text(size: 9pt, style: "italic", fill: luma(80), translation)
  )
}
```

### Translation Footnotes

For documents that are primarily one language with occasional translated terms:

```typst
// Inline with footnote translation
#let translated(term, translation) = [#term#footnote[#term: #translation]]
```

## Font Fallback Chains

### Building Multi-Script Chains

```typst
// Latin + Arabic fallback
#set text(
  font: ("Inter", "Noto Sans Arabic"),
  lang: "en",  // primary language
)

// Latin + CJK fallback
#set text(
  font: ("Inter", "Noto Sans CJK SC"),  // SC = Simplified Chinese
  lang: "en",
)

// Latin + Devanagari
#set text(
  font: ("Inter", "Noto Sans Devanagari"),
  lang: "en",
)
```

### Per-Region CJK Font Selection

```typst
// Simplified Chinese
#set text(font: ("Inter", "Noto Sans CJK SC"), lang: "zh", region: "cn")

// Traditional Chinese (Taiwan)
#set text(font: ("Inter", "Noto Sans CJK TC"), lang: "zh", region: "tw")

// Japanese
#set text(font: ("Inter", "Noto Sans CJK JP"), lang: "ja")

// Korean
#set text(font: ("Inter", "Noto Sans CJK KR"), lang: "ko")
```

### Checking Font Coverage

Always verify font availability before committing to a fallback chain:

```bash
# List available fonts matching a pattern
typst fonts | grep -i "noto"
typst fonts | grep -i "arabic"
```

If required fonts aren't installed, either:
1. Install them (Noto font family covers most scripts)
2. Fall back to system defaults and document the limitation
3. Embed fonts via Typst's font path mechanism

## RTL Document Setup

```typst
// Full RTL document
#set text(lang: "ar", dir: rtl)
#set page(
  margin: (
    // Mirrored margins for RTL binding
    inside: 3cm,   // binding side (right in RTL)
    outside: 2cm,
    top: 2.5cm,
    bottom: 2.5cm,
  ),
)

// RTL header with page number on the left
#set page(header: context {
  let num = counter(page).display()
  grid(columns: (auto, 1fr, auto),
    num, [], text(size: 9pt, "عنوان المستند")
  )
})
```

## Pandoc Considerations

- Set `lang` and `dir` in YAML front matter for correct text direction
- Use `--pdf-engine=typst` for best multilingual support (better than LaTeX for many scripts)
- Pandoc handles bidi text via Unicode bidi algorithm — verify output carefully
- For CJK, ensure `--pdf-engine-opt` passes the correct font to Typst
- BibTeX/CSL can handle multilingual bibliographies but may need script-specific sorting rules

## Quality Additions for Multilingual Documents

In addition to the base quality checklist, verify:
- [ ] Font fallback chain covers all scripts used in the document — no tofu (□) characters
- [ ] RTL text flows correctly — verify by reading rendered output, not source
- [ ] Bidi text at script boundaries renders correctly (e.g., Arabic text containing English brand names)
- [ ] CJK punctuation doesn't create double-width gaps at script boundaries
- [ ] Line height accommodates the tallest script in use (CJK and Indic need more)
- [ ] Page layout mirrors correctly for RTL documents (binding, page numbers, marginalia)
- [ ] Hyphenation is enabled for each language (`#set text(lang: "xx")`)
- [ ] Parallel bilingual columns maintain rough alignment — paragraphs don't drift apart
- [ ] Numerals are consistent (don't mix Eastern Arabic and Western Arabic in the same document unless intentional)
