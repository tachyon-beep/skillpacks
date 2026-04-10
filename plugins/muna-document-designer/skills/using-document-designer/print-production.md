# Print Production Design

Reference sheet for documents destined for professional printing — offset, digital, or large-format. Load this alongside the document-designer agent when the document will be physically printed and the user needs bleed, crop marks, binding margins, color management, or imposition.

## When to Use This Reference

- Documents going to a commercial printer (offset or digital press)
- Print-ready PDF preparation with bleed and crop marks
- Booklet or bound document formatting with gutter margins
- Color-critical documents requiring CMYK consideration
- Large-format prints (posters, banners) with specific DPI requirements
- Any document where the user mentions "print-ready", "bleed", "crop marks", or "binding"

## Page Geometry for Print

### Bleed

Bleed is the area beyond the trim edge where ink extends — prevents white edges if the cut is slightly off. Standard bleed is 3mm (0.125in).

```typst
// A4 with 3mm bleed on all sides
#set page(
  width: 210mm + 6mm,    // A4 width + 2× bleed
  height: 297mm + 6mm,   // A4 height + 2× bleed
  margin: (
    top: 2.5cm + 3mm,    // margin + bleed offset
    bottom: 2.5cm + 3mm,
    left: 2.5cm + 3mm,
    right: 2.5cm + 3mm,
  ),
)

// Full-bleed background element
#place(top + left, dx: -3mm, dy: -3mm,
  rect(width: 210mm + 6mm, height: 40mm, fill: rgb("#2c5282"))
)
```

### Binding & Gutter Margins

For bound documents, the inside margin (gutter) must be wider to account for the binding eating into the page:

```typst
// Asymmetric margins for perfect binding
#set page(
  margin: (
    inside: 3cm,    // binding side — wider
    outside: 2cm,   // outer edge — standard
    top: 2.5cm,
    bottom: 2.5cm,
  ),
)
```

| Binding Type | Extra Gutter | Total Inside Margin |
|-------------|-------------|-------------------|
| Saddle stitch (stapled) | 3-5mm | ~2.5cm |
| Perfect binding (glued) | 6-10mm | ~3cm |
| Spiral/wire binding | 10-15mm | ~3.5cm |
| Case binding (hardcover) | 8-12mm | ~3cm |

### Common Print Sizes

| Name | Dimensions | Common Use |
|------|-----------|-----------|
| A4 | 210 × 297mm | Standard international |
| US Letter | 8.5 × 11in (216 × 279mm) | Standard North America |
| A5 | 148 × 210mm | Booklets, handbooks |
| A3 | 297 × 420mm | Posters, foldouts |
| US Legal | 8.5 × 14in (216 × 356mm) | Legal documents |
| Tabloid/Ledger | 11 × 17in (279 × 432mm) | Posters, foldouts |
| Business card | 3.5 × 2in (89 × 51mm) | Cards |
| DL envelope | 110 × 220mm | Envelope inserts |

## Crop Marks & Registration

Crop marks show the printer where to trim. Registration marks help align color plates.

```typst
// Manual crop marks at page corners
#let crop-length = 8mm
#let crop-offset = 3mm  // = bleed distance

#let crop-marks() = {
  let mark(x, y, dx, dy) = place(
    top + left,
    dx: x, dy: y,
    line(length: crop-length, angle: if dx != 0pt { 0deg } else { 90deg },
         stroke: 0.25pt + black)
  )
  // Top-left corner
  mark(-crop-offset - crop-length, -crop-offset, 1pt, 0pt)
  mark(-crop-offset, -crop-offset - crop-length, 0pt, 1pt)
  // ... repeat for other three corners
}
```

Note: Many print shops prefer receiving PDF without crop marks and adding their own during prepress. Ask the printer.

## Color Management

### CMYK Considerations

Screen colors (RGB) don't map 1:1 to print colors (CMYK). Key issues:

- **Bright blues and greens** are the most affected — RGB blues look duller in CMYK
- **Pure black text**: Use K-only black (0,0,0,100) not rich black (mixing all inks) for body text — prevents registration issues
- **Rich black for large areas**: Use (40,30,20,100) or similar for large black backgrounds — K-only black looks washed out at large scale
- **Total ink coverage**: Most printers limit to 300% total (C+M+Y+K) — exceeding causes drying problems

Typst works in RGB natively. For print-critical work:

```typst
// Define colors with print awareness
// These RGB values approximate their CMYK equivalents well
#let print-blue = rgb("#1a4d80")    // avoids the oversaturated RGB blue trap
#let print-red = rgb("#c0392b")     // printable red, not screen-bright
#let text-black = rgb("#000000")    // maps to K-only in most PDF workflows
```

### Spot Colors

For brand-critical colors (e.g., Pantone), note the Pantone reference in comments. Typst can't embed spot color definitions, but the comment helps the print shop:

```typst
// Brand blue: Pantone 2955 C — closest RGB approximation
#let brand-blue = rgb("#003057")
```

## Image Requirements

### Resolution

| Output | Minimum DPI | Recommended DPI |
|--------|------------|----------------|
| Offset print (photos) | 300 | 300-350 |
| Offset print (line art) | 600 | 1200 |
| Digital print | 200 | 300 |
| Large format (posters) | 100-150 | 150 |
| Screen/web | 72-96 | 150 |

### Image Format Preferences

| Format | Use For | Notes |
|--------|---------|-------|
| PDF (vector) | Charts, diagrams, logos | Best quality, resolution-independent |
| SVG (vector) | Charts, diagrams, logos | Good, but verify Typst rendering |
| PNG | Screenshots, raster art with transparency | Lossless, large files |
| JPEG | Photographs | Lossy, use quality 85%+ for print |
| TIFF | Professional photography | Preferred by some print shops, large files |

```typst
// Verify image resolution is sufficient for print
// A 1000px wide image at 300 DPI prints at ~85mm (3.3in)
// Formula: print_width_mm = (pixels / DPI) × 25.4

// Don't stretch small images — set explicit width limits
#image("photo.jpg", width: 80mm)  // let height follow aspect ratio
```

## Imposition (Multi-Page Sheets)

For booklet printing where multiple pages are arranged on a single sheet:

- **Saddle-stitch booklets**: Pages must be imposed in printer-spread order (not reader-spread)
- **Page count**: Must be a multiple of 4 for saddle-stitch (pad with blank pages if needed)
- **Creep compensation**: Inner pages of thick booklets extend past outer pages at the trim — shift content inward progressively

Most imposition is handled by the print shop's prepress software. Deliver single-page PDFs in reader order unless the printer requests otherwise.

## Preflight Checklist

Before sending to a printer, verify:

### Geometry
- [ ] Page size matches the intended trim size (not trim + bleed, unless printer requests it)
- [ ] Bleed extends 3mm (or as specified) beyond trim on all sides where content reaches the edge
- [ ] Gutter margin accounts for binding type
- [ ] No critical content within 5mm of trim edge (safety margin)

### Images
- [ ] All images are at least 300 DPI at their printed size
- [ ] No images are upscaled beyond their native resolution
- [ ] Vector graphics (logos, charts) are embedded as PDF/SVG, not rasterized

### Color
- [ ] No oversaturated RGB colors that will shift dramatically in CMYK
- [ ] Body text uses pure black, not rich black
- [ ] Large black areas use rich black
- [ ] Total ink coverage doesn't exceed printer's limit (typically 300%)

### Fonts
- [ ] All fonts are embedded in the PDF (Typst does this by default)
- [ ] No font substitution warnings during compilation
- [ ] Body text is at least 8pt (smaller risks legibility issues in print)

### Pagination
- [ ] Page count is a multiple of 4 if saddle-stitched
- [ ] Blank pages are intentionally placed (typically verso of half-title, before new chapters)
- [ ] Page numbers are consistent and match table of contents

### File Delivery
- [ ] PDF version is compatible with printer's workflow (PDF/X-1a or PDF/X-4 preferred for print)
- [ ] File is named clearly with version/date
- [ ] Accompanying spec sheet states: trim size, bleed, binding type, paper stock, color mode, quantity
