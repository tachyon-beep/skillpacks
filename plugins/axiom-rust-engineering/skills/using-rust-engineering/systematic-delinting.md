# Resolve Clippy diagnostics without hiding defects

## Establish scope

Read lint levels, toolchain and feature/target policy; capture the actual configured command and baseline. Group diagnostics by shared cause and fix one representative case before a broad mechanical edit.

## Review fixes

- A suggested rewrite can change allocation, drop timing, short-circuit evaluation or panic behavior. Read the affected callers/invariant.
- Avoid broad `allow` attributes and crate-wide category disables. A narrow justified allowance should identify the tool limitation or intentional contract.
- Do not replace error handling with `unwrap` merely to simplify syntax.
- Lifetime/ownership suggestions need to preserve identity and resource behavior, not just compilation.
- Generated/vendored code needs an ownership-aware policy rather than hand edits.
- Keep unrelated formatting and configuration migration out of a focused repair.

Rerun affected diagnostics and meaningful behavior tests. Expand to a workspace/feature matrix only when the changed boundary requires it. Report remaining diagnostics and intentional allowances; no fixed iteration count or zero-warning quota is implied.
