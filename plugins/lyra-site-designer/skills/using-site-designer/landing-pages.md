# Content-First Developer Homepage

Use when a developer-tool or open-source homepage needs a clear working entry
point. Preserve brand/product choices; this is a useful style, not a universal
rule that every site must reject imagery or testimonials.

## Content contract

- State the problem and who the project helps in concrete terms.
- Show a realistic minimal example or outcome with the prerequisites it needs.
- Make install/try, documentation, source and relevant release/support status
  easy to find. Verify commands and links against the actual project.
- Explain differentiating capabilities and limitations without invented metrics,
  customers, endorsements or unsupported superiority claims.
- Let secondary detail follow the main task rather than obscure it.

Use the existing design tokens and semantic HTML. Inspect keyboard/focus behavior,
small-screen wrapping, light/dark contrast and code overflow. Animation is optional
and should respect reduced-motion preferences; it must not delay access to content.

## Validation and handoff

Run the site's actual build and relevant route/link checks. Exercise the install
copy action and primary navigation. Report what was locally previewed versus
published. Supporting recipes: [code-block-patterns.md](code-block-patterns.md),
[theming-and-tokens.md](theming-and-tokens.md),
[responsive-patterns.md](responsive-patterns.md).
