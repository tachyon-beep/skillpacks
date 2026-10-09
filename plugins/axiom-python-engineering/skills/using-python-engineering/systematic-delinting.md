# Repair lint diagnostics in bounded groups

## Bound the repair

Read repository lint configuration and run the configured command in the intended environment. Capture the baseline. Group diagnostics by root cause and affected boundary; fix a small representative instance before applying a mechanical transformation broadly.

## Preserve semantics

- Inspect every automated fix that changes control flow, imports, exception handling, resource lifetime or public API.
- An unused import can have registration side effects; confirm intent before removing it.
- Do not replace explicit error handling with an assertion for external data.
- Simplifying a condition can change evaluation order, truthiness or side effects.
- Avoid unrelated formatting/configuration changes that obscure review.
- A suppression needs a specific rule, scope and reason grounded in the project's policy. Disabling a category is not evidence that its findings were false.
- Generated or vendored code may need an established exclusion rather than hand edits; verify ownership first.

## Verify and report

Rerun affected lint and meaningful behavior checks after each coherent group. Broaden only if the change crosses callers or shared configuration. Report repaired categories, remaining diagnostics and justified exclusions. No day-by-day schedule, zero-warning quota or tool migration is required unless the user/project explicitly sets one.
