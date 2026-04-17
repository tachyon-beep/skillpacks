# Systematic Delinting (Rust/Clippy)

## Overview

**Core Principle:** Fix warnings systematically, NEVER silently suppress them. Delinting is about making existing code compliant with Rust's quality standards through minimal, focused changes. It is NOT refactoring.

Clippy is a superset of `rustc` warnings. Where the compiler catches hard errors and a handful of style nudges, Clippy covers hundreds of additional categories: correctness hazards that compile fine but are almost certainly wrong, performance opportunities, complexity smells, and idiomatic patterns the Rust community considers best practice. The distinction matters: `rustc` denies compilation on errors; Clippy is advisory by default. Delinting is the process of moving your codebase from "advisory tolerated" to "enforced clean."

**Clippy's category hierarchy**: Every lint lives in one of nine categories — `correctness`, `suspicious`, `style`, `complexity`, `perf`, `pedantic`, `restriction`, `nursery`, and `cargo`. Default `cargo clippy` enables `correctness`, `suspicious`, `style`, `complexity`, and `perf`. Promotion to `pedantic` or `nursery` is an explicit opt-in.

**This skill teaches the PROCESS of delinting, not just how to configure Clippy.**

## When to Use

Use this skill when:

- Codebase has 500+ Clippy warnings
- Enabling `clippy::pedantic` on legacy code produces hundreds of new diagnostics
- Inheriting a project that was developed without Clippy in CI
- Team wants to gate `cargo clippy -- -Dwarnings` in CI but can't pass today
- Need to reduce lint debt incrementally without blocking ongoing development
- Want a systematic commit history that shows warning count decreasing category by category

**Symptoms triggering this skill:**
- "500 clippy warnings, where do I start?"
- "How do I get `clippy::pedantic` working without mass-suppression?"
- "Legacy code, no CI lint check — need to clean it up"
- "How do I fix warnings without accidentally refactoring?"
- "Team added `#![allow(clippy::all)]` to shut up CI — that's wrong, right?"

## When NOT to Use

- **New projects**: route to [project-structure-and-tooling.md](project-structure-and-tooling.md) — configure Clippy correctly from day one instead of retroactively cleaning it up.
- **Fewer than ~50 warnings**: just fix them directly; a systematic process has overhead that isn't justified at small scale.
- **Refactoring work**: delinting makes minimal, mechanical changes to satisfy lints. Reducing a `too_many_arguments` lint by redesigning a builder API is refactoring — create a separate ticket.
- **Enabling `clippy::restriction` globally**: restriction lints enforce style choices that are not universally correct (e.g., `clippy::panic`, `clippy::unwrap_used`). Enabling them wholesale is almost always a mistake; they require item-level opt-in, not crate-level.

## The Methodology

### Discover → Categorize → Fix One Category → Verify → Commit → Repeat

The process is identical to Python's ruff-based delinting in spirit; only the tool names change.

**Step 1: Discover — capture baseline**

```bash
# Full warning count (default clippy lints)
cargo clippy 2>&1 | grep "^warning" | wc -l

# See warnings grouped by lint name
cargo clippy 2>&1 | grep "^\s*= help: for further" -A1 | \
  grep "clippy::" | sort | uniq -c | sort -rn

# Save full baseline for reference
cargo clippy 2>&1 > clippy-baseline.txt
```

Commit `clippy-baseline.txt` so progress is measurable.

**Step 2: Categorize — sort warnings by effort**

| Tier | Effort | Examples | Strategy |
|------|--------|----------|----------|
| Mechanical auto-fix | Very low | `needless_return`, `redundant_field_names`, `map_flatten` | `cargo clippy --fix` per lint |
| Mechanical manual | Low | `or_fun_call`, `clone_on_copy`, `needless_collect` | Pattern replace, file-by-file |
| Requires thought | Medium | `too_many_arguments`, `large_enum_variant`, `module_name_repetitions` | Case-by-case review |
| Defers to refactor | High | `cognitive_complexity`, `too_many_lines`, design lints | Create ticket, defer |

**Step 3: Fix one category or lint**

Pick the highest-count, lowest-effort lint. Fix it everywhere. Do not mix lint categories in a single commit.

```bash
# Fix one specific lint across the codebase
cargo clippy --fix --allow-dirty -- -A clippy::all -W clippy::needless_return

# Review before staging
git diff

# Run tests
cargo test

# Commit with count
git commit -m "fix(clippy): remove needless_return (42 warnings)"
```

**Step 4: Verify**

```bash
cargo clippy -- -W clippy::LINT_NAME 2>&1 | grep "^warning" | wc -l
# should be 0 for the fixed lint
cargo test  # never skip
```

**Step 5: Commit**

One lint type per commit. Include before/after warning counts in the commit message.

**Step 6: Repeat**

```bash
# Updated baseline
cargo clippy 2>&1 | grep "^warning" | wc -l
```

### Rule-by-Lint vs File-by-File

**Rule-by-lint** (recommended for 100+ warnings):
- Small, focused commits that are trivially reviewable
- Progress is visible in warning counts
- Easy to revert a single category if it causes test failures
- Ideal for correctness, perf, and style categories

**File-by-file** (alternative for modular codebases):
- Fix all warnings in one module, commit, move to next
- Works well when the team has file ownership
- Better when warnings are sparse and spread across many lint types

**Hybrid**: Use rule-by-lint for auto-fixable lints (Phase 1), then file-by-file for remaining mechanical fixes (Phase 2).

## Clippy Categories

Understanding which category a lint belongs to determines fix priority.

### `correctness` — deny by default

These lints identify code that is **almost certainly wrong**. Clippy denies them by default (same level as compiler errors). Fix these first, always.

```bash
cargo clippy -- -Dclippy::correctness
```

Examples: `eq_op` (comparing a value to itself), `absurd_extreme_comparisons` (u8 < 0), `iter_next_loop` (for x in iter.next()), `wrong_transmute`.

If `correctness` lints fire in your codebase, treat them as bugs.

### `suspicious` — warn by default

These lints identify code that is **probably wrong or confusing** even if technically valid.

Examples: `suspicious_arithmetic_impl`, `misrefactored_assign_op`, `suspicious_map`. (`cast_lossless` is **not** a suspicious lint — it was moved to the `pedantic` group, which is allow-by-default; you only see it if you opt into `-W clippy::pedantic`.)

Fix suspicious lints in the same sprint as correctness.

### `style` — warn by default

Idiomatic Rust patterns. The fixes are mechanical and the code is correct without them, but the canonical form is clearer.

Examples: `needless_return`, `redundant_field_names`, `let_and_return`, `match_like_matches_macro`.

High volume, low risk. Fix with `cargo clippy --fix` where available.

### `complexity` — warn by default

Code that is more complex than necessary.

Examples: `too_many_arguments` (>7 params), `cognitive_complexity`, `needless_collect`, `manual_flatten`.

Some are mechanical (`needless_collect`), some require architectural thought (`too_many_arguments`). Separate them during triage.

### `perf` — warn by default

Performance opportunities that are almost always improvements.

Examples: `or_fun_call`, `redundant_clone`, `unnecessary_to_owned`, `iter_overeager_cloned`.

Fix these early — they're usually local changes with clear wins.

### `pedantic` — off by default

Stricter style and correctness lints that are not universally agreed upon. Some are opinionated (e.g., `doc_markdown`, `must_use_candidate`). Enable deliberately after clearing default lints.

```rust
#![warn(clippy::pedantic)]
```

### `restriction` — off by default, use sparingly

Blanket restrictions on specific constructs (`unwrap_used`, `panic`, `print_stdout`). Not universally correct. Enable per-item with `#[allow]` on specific well-understood exceptions, not at crate level.

### `nursery` — off by default, unstable

Lints still being refined. May produce false positives. Enable with caution; expect occasional noise.

```rust
#![warn(clippy::nursery)]
```

### `cargo` — off by default

Lint your `Cargo.toml`: duplicate dependencies, multiple versions, wildcard dependencies.

```bash
cargo clippy -- -Wclippy::cargo
```

### Priority Order for Systematic Fixing

```
correctness → suspicious → perf → complexity → style → pedantic
```

Leave `restriction` and `nursery` until the codebase is clean on everything above.

## Applying Category Promotion

### Default Lints First

Before enabling any non-default category, your codebase must be clean against default Clippy:

```bash
cargo clippy -- -Dwarnings
# Must exit 0
```

Do not promote to pedantic until this passes in CI with zero exceptions.

### Enabling `clippy::pedantic`

Add at the crate root (`src/lib.rs` or `src/main.rs`):

```rust
#![warn(clippy::pedantic)]
```

Or, to deny (stricter):

```rust
#![deny(clippy::pedantic)]
```

Run the baseline:

```bash
cargo clippy 2>&1 | grep "clippy::pedantic\|clippy::" | sort | uniq -c | sort -rn
```

Expect a wave of new lints. Do not allow them all. Fix the top 5 by count, then re-evaluate.

**Pedantic lints that are usually right to fix:**
- `clippy::must_use_candidate` — add `#[must_use]` to functions whose output should not be ignored
- `clippy::missing_errors_doc` — document the error conditions in `/// # Errors`
- `clippy::missing_panics_doc` — document panic conditions in `/// # Panics`
- `clippy::redundant_closure_for_method_calls` — `|x| x.foo()` → `.foo` as a method reference

**Pedantic lints that often need per-site `#[allow]`:**
- `clippy::module_name_repetitions` — sometimes the repetition is intentional naming
- `clippy::wildcard_imports` — `use super::*;` in test modules is widely accepted
- `clippy::too_many_lines` — a function can legitimately be long if it's a `match` dispatch

### Enabling `clippy::nursery`

Use only if you want early access to experimental lints. Add a comment:

```rust
// nursery: experimental lints, may produce false positives
#![warn(clippy::nursery)]
```

If a nursery lint fires and the fix is unclear, `#[allow]` with a comment is acceptable — nursery lints have known rough edges. Revisit when the lint graduates.

### Backing Off Promotion

If enabling `pedantic` produces 300+ new warnings on a large codebase, a staged approach works:

```rust
// Stage 1: warn on pedantic
#![warn(clippy::pedantic)]
// Stage 2: deny pedantic after cleanup
#![deny(clippy::pedantic)]
```

Track warning count by category daily. Never allow the count to increase.

## Common Clippy Lints and Fixes

### 1. `clippy::needless_collect`

**Why it fires:** A collection is built and immediately iterated, with no intermediate need for ownership.

```rust
// Fires: collects then counts
let v: Vec<_> = iter.filter(|x| x > 0).collect();
let count = v.len();

// Fix: use the iterator directly
let count = iter.filter(|x| x > 0).count();
```

**Auto-fix:** `cargo clippy --fix -- -W clippy::needless_collect`

### 2. `clippy::or_fun_call`

**Why it fires:** A function call is used as the argument to `.or()` / `.unwrap_or()`, but the function is called eagerly — even when not needed.

```rust
// Fires: String::new() called unconditionally
let name = opt_name.unwrap_or(String::new());

// Fix: use lazy variant
let name = opt_name.unwrap_or_default();

// Or when a function has side effects or cost:
let name = opt_name.unwrap_or_else(|| compute_default_name());
```

**Auto-fix:** Available for simple cases.

### 3. `clippy::redundant_clone`

**Why it fires:** A value is cloned but the clone is immediately used in a position where the original could be moved or borrowed.

```rust
// Fires: s is cloned but original is not used after
fn process(s: String) -> String {
    let copy = s.clone(); // redundant if s not used after
    transform(copy)
}

// Fix: move s directly
fn process(s: String) -> String {
    transform(s)
}
```

**Auto-fix:** `cargo clippy --fix -- -W clippy::redundant_clone`

### 4. `clippy::too_many_arguments`

**Why it fires:** A function has more than 7 parameters (configurable).

```rust
// Fires: 8 parameters
fn create_user(
    id: u64, name: String, email: String, role: Role,
    active: bool, created_at: DateTime, timezone: Tz, locale: Locale,
) -> User { ... }
```

**Fix options** (choose based on context — this requires thought, not mechanical fixing):

```rust
// Option A: introduce a config/builder struct
struct CreateUserParams {
    id: u64,
    name: String,
    email: String,
    role: Role,
    active: bool,
    created_at: DateTime,
    timezone: Tz,
    locale: Locale,
}

fn create_user(params: CreateUserParams) -> User { ... }
```

This is light refactoring — keep in a separate commit from mechanical lint fixes.

### 5. `clippy::module_name_repetitions`

**Why it fires:** A type exported from module `parser` is named `ParserError` — the word "parser" appears in both the module path and the type name, so users write `parser::ParserError`.

```rust
// In module `parser`: fires
pub struct ParserError { ... }

// Fix: rename to drop the module prefix
pub struct Error { ... }
// Users write: parser::Error — unambiguous in context
```

**When to allow:** When the type is re-exported at a higher level and the name becomes ambiguous without the prefix, use `#[allow(clippy::module_name_repetitions)]` on the item with a comment explaining the re-export scenario.

### 6. `clippy::missing_errors_doc`

**Why it fires:** A public function returns `Result<T, E>` but its doc comment has no `# Errors` section.

```rust
// Fires: no Errors section
/// Reads the configuration file.
pub fn read_config(path: &Path) -> Result<Config, ConfigError> { ... }

// Fix: add the section
/// Reads the configuration file.
///
/// # Errors
///
/// Returns `ConfigError::Io` if the file cannot be opened or read.
/// Returns `ConfigError::Parse` if the TOML is malformed.
pub fn read_config(path: &Path) -> Result<Config, ConfigError> { ... }
```

**Lint group:** `clippy::pedantic` — enable when documentation standards are being enforced.

### 7. `clippy::clone_on_ref_ptr`

**Why it fires:** Calling `.clone()` on an `Arc<T>` or `Rc<T>` is cheap (just an atomic increment), but it is semantically distinct from cloning the data. The lint encourages explicit `Arc::clone(&x)` to make the operation visible.

```rust
// Fires (clippy::restriction, not default)
let handle = arc_value.clone();

// Fix: use the associated function form
let handle = Arc::clone(&arc_value);
```

This lint is in `clippy::restriction`. Enable it if your codebase uses `Arc` heavily and you want the operation to be visually distinct.

### 8. `clippy::large_enum_variant`

**Why it fires:** One variant of an enum is much larger than the others, making every instance of the enum as large as the largest variant — wasting stack space for the common case.

```rust
// Fires: Large variant dominates
enum Message {
    Ping,                      // 0 bytes
    Pong,                      // 0 bytes
    Data([u8; 4096]),          // 4096 bytes — all variants now pay this cost
}

// Fix: box the large variant
enum Message {
    Ping,
    Pong,
    Data(Box<[u8; 4096]>),    // only allocates when Data is used
}
```

Check whether the large data is actually moved frequently; sometimes the right fix is to split the enum.

### 9. `clippy::unnecessary_wraps`

**Why it fires:** A function always returns `Ok(...)` or `Some(...)` — the wrapper type is never needed.

```rust
// Fires: always returns Some
fn find_default(items: &[Item]) -> Option<&Item> {
    Some(&items[0]) // panics if empty, and always Some — wrong signature
}

// Fix A: return the type directly
fn find_default(items: &[Item]) -> &Item {
    &items[0]
}

// Fix B: handle the empty case properly
fn find_default(items: &[Item]) -> Option<&Item> {
    items.first()
}
```

### 10. `clippy::match_wildcard_for_single_variants`

**Why it fires:** A `match` arm uses `_` as a wildcard but could name the remaining variant explicitly — losing exhaustiveness protection when new variants are added.

```rust
// Fires: wildcard hides future-variant risk
match status {
    Status::Active => do_active(),
    Status::Inactive => do_inactive(),
    _ => do_default(),  // silently catches any new Status variants
}

// Fix: name the remaining variant
match status {
    Status::Active => do_active(),
    Status::Inactive => do_inactive(),
    Status::Pending => do_default(),
}
// Now adding Status::Archived causes a compile error — caught at the right place
```

### 11. `clippy::redundant_field_names`

**Why it fires:** A struct literal uses field initialization where the field name and the local variable name are identical — shorthand init syntax is available.

```rust
// Fires
let name = compute_name();
let age = compute_age();
let user = User { name: name, age: age };

// Fix
let user = User { name, age };
```

**Auto-fix:** `cargo clippy --fix -- -W clippy::redundant_field_names`

### 12. `clippy::unwrap_used` (restriction)

**Why it fires:** `unwrap()` on `Option` or `Result` panics on `None`/`Err`. In production code this is a latent panic hazard.

```rust
// Fires (clippy::restriction)
let config = load_config().unwrap();

// Fix: propagate with ?
let config = load_config()?;

// Fix: provide a meaningful default
let config = load_config().unwrap_or_default();

// Fix: give a useful panic message when the invariant MUST hold
let config = load_config().expect("config file must exist at startup");
```

**Scope guidance:** Enable `clippy::unwrap_used` only in library crates or via per-function `#[deny]`. Test modules should retain `#[allow(clippy::unwrap_used)]` — panicking in tests is acceptable and expected.

## `#[allow(...)]` — When It Is Justified

### Narrow Scope Only

Never add `#[allow]` at the crate level to silence a lint that should be fixed. The allowed scope should be the smallest item where the lint fires: a single function, a single `impl` block, or a single `match` arm.

```rust
// BAD: crate-level suppression hides all future firings
#![allow(clippy::too_many_arguments)]

// BAD: module-level suppression is too broad
#[allow(clippy::too_many_arguments)]
mod legacy {
    fn old_api(...) { ... }
    fn another_fn(...) { ... }  // also silenced even if it's fine
}

// GOOD: item-level suppression with justification comment
/// Legacy C-FFI bridge — argument list mirrors the C API exactly.
/// Wrapped by `DeviceConfig::from_raw()` for new callers.
/// TODO: remove after `legacy-c-bridge` feature is deleted.
#[allow(clippy::too_many_arguments)]
pub fn raw_device_init(
    id: u32, addr: u64, size: usize, flags: u32,
    timeout_ms: u32, retry: u8, mode: u8, reserved: u32,
) -> i32 { ... }
```

### Required Elements of a Justified `#[allow]`

1. **Justification comment** on the line above the allow attribute explaining _why_ the lint is wrong here.
2. **Narrowest possible scope** — prefer per-function over per-module; never per-crate.
3. **TODO if temporary** — if the suppression is waiting on a refactor, add a `// TODO(#issue): remove when ...` comment.

### Code Review Discipline

Every `#[allow(clippy::...)]` that reaches code review must carry a comment. PRs that add allows without justification should be blocked at review.

### Legitimate Uses

- `#[allow(clippy::module_name_repetitions)]` on a type that is re-exported and would be ambiguous without the prefix.
- `#[allow(clippy::wildcard_imports)]` in a `#[cfg(test)] mod tests` block — `use super::*;` is idiomatic in tests.
- `#[allow(clippy::too_many_arguments)]` on a C-FFI bridge function that must mirror a C signature.
- `#[allow(clippy::cast_possible_truncation)]` on a cast that you have verified is bounded, with a comment showing the proof.

## Measuring Progress

### Count Warnings

```bash
# Total default-lint warnings
cargo clippy 2>&1 | grep "^warning\[" | wc -l

# Warnings that would block -Dwarnings
cargo clippy -- -Dwarnings 2>&1 | grep "^warning\[" | wc -l

# Warnings by lint name (requires parsing)
cargo clippy --message-format=json 2>/dev/null | \
  python3 -c "
import sys, json
counts = {}
for line in sys.stdin:
    try:
        m = json.loads(line)
        if m.get('reason') == 'compiler-message':
            code = (m.get('message', {}).get('code') or {}).get('code', '')
            if code.startswith('clippy::'):
                counts[code] = counts.get(code, 0) + 1
    except: pass
for k, v in sorted(counts.items(), key=lambda x: -x[1]):
    print(f'{v:4d}  {k}')
"
```

### Track Trend Across Commits

```bash
# Record count after each lint-fix commit
git log --oneline --grep="clippy" | while read sha msg; do
    count=$(git show $sha:clippy-baseline.txt 2>/dev/null | wc -l || echo "?")
    echo "$sha $count $msg"
done
```

Or maintain a `clippy-progress.csv`:

```
date,total_warnings,fixed_lint,warnings_in_lint
2026-04-01,487,baseline,0
2026-04-02,445,needless_return,42
2026-04-03,421,redundant_field_names,24
2026-04-04,399,or_fun_call,22
```

### CI Gate Progression

Move from advisory to enforced in stages:

**Stage 1: Warn (advisory)**
```toml
# .cargo/config.toml
[target.'cfg(all())']
rustflags = ["-W", "clippy::all"]
```

**Stage 2: Ratchet (count must not increase)**

In CI, compare warning count to the committed baseline:

```bash
BASELINE=$(cat clippy-baseline.txt | grep "^warning\[" | wc -l)
CURRENT=$(cargo clippy 2>&1 | grep "^warning\[" | wc -l)
if [ "$CURRENT" -gt "$BASELINE" ]; then
  echo "Clippy warnings increased: $BASELINE -> $CURRENT"
  exit 1
fi
echo "Clippy: $CURRENT warnings (baseline: $BASELINE)"
```

**Stage 3: Deny (all default lints)**
```bash
cargo clippy -- -Dwarnings
```

**Stage 4: Deny pedantic**
```bash
cargo clippy -- -Dwarnings -Dclippy::pedantic
```

### Commit Message Convention

```
fix(clippy): <lint-name> — <brief description>

Lint: clippy::<lint_name>
Before: <N> warnings
After: <M> warnings
Method: [auto-fix | manual pattern | file-by-file]
```

Example:

```
fix(clippy): needless_collect — remove unnecessary intermediate Vecs

Lint: clippy::needless_collect
Before: 487 warnings total (42 needless_collect)
After: 445 warnings total
Method: cargo clippy --fix -- -A clippy::all -W clippy::needless_collect
```

## Anti-Patterns

### 1. Global `#[allow(clippy::all)]` to "clean up" CI output

```rust
// BAD: suppresses every current and future lint silently
#![allow(clippy::all)]
```

**Why wrong:** This is not delinting. It hides every correctness hazard, performance opportunity, and bug pattern Clippy would find — now and in the future. The output looks clean but the code is not. Any new code added to the crate is also silently exempted. CI now provides zero signal.

**The fix:** Remove the allow, run `cargo clippy 2>&1 | grep "^warning" | wc -l` to see the true baseline, and begin systematic per-lint fixing.

### 2. `cargo clippy --fix` with uncommitted changes

```bash
# BAD workflow
# (you have unstaged edits in 5 files)
cargo clippy --fix --allow-dirty
git add .
git commit -m "fix lint"
```

**Why wrong:** `--allow-dirty` applies machine-generated edits on top of hand-written changes. The diff is now a mixture of two concerns. If the auto-fix breaks something, isolating the regression is hard. If you need to revert the lint fix, you also lose your manual changes.

**The fix:** Always start from a clean working tree. Commit your in-progress work first (even as a WIP commit), then run `--fix`, review `git diff`, run `cargo test`, and commit the lint fix separately.

### 3. Fixing clippy by aggressively refactoring

```rust
// clippy::too_many_arguments fires on this function
fn process(a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32) -> u32 { ... }

// BAD: during a "delinting" PR, you rewrite the entire module to use a builder pattern,
// change the public API, and restructure three other files.
```

**Why wrong:** Delinting PRs should be mechanically reviewable. When you intermix refactoring, reviewers cannot distinguish "Clippy forced this change" from "engineer chose this change." Refactoring introduces risk — test coverage may not catch all regressions. Mixing the two makes bisecting a regression difficult.

**The fix:** In the delinting PR, add `#[allow(clippy::too_many_arguments)]` with a comment referencing the refactoring ticket. Fix the lint properly in the refactoring PR where the structural change is the explicit intent.

### 4. Enabling `clippy::pedantic` and immediately suppressing half of it

```rust
// BAD: add pedantic, then add a wall of allows at the crate root
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::cast_precision_loss)]
// ... 12 more allows
```

**Why wrong:** If you suppress 12 of the 20 pedantic lints at the crate level, you have not adopted pedantic — you have added noise to the crate root and given future contributors the impression that pedantic is "configured" when it is effectively disabled for the hardest lints.

**The fix:** Enable pedantic one lint at a time. Fix each lint category completely before adding the next. A crate with three pedantic lints actually enforced is more valuable than a crate with all pedantic lints added but half suppressed.

### 5. Adding `#[allow]` without a comment explaining why

```rust
// BAD: unexplained suppression
#[allow(clippy::too_many_arguments)]
pub fn configure_device(id: u32, addr: u64, size: usize, flags: u32,
                        timeout: u32, retry: u8, mode: u8, tag: u8) -> Result<Device> { ... }
```

**Why wrong:** Future maintainers cannot tell whether this `allow` is intentional (e.g., mirrors a C API), temporary (waiting for a builder refactor), or legacy neglect. They have no signal to help them decide whether to remove the `allow` when refactoring.

**The fix:** Every `#[allow]` must carry a comment:

```rust
// Mirrors the C `device_init()` ABI exactly; argument order is fixed by the hardware spec.
// Wrapped by `Device::open()` for all Rust-native callers.
// TODO(#412): remove this binding after FFI layer is replaced.
#[allow(clippy::too_many_arguments)]
pub fn configure_device(...) -> Result<Device> { ... }
```

## Checklist

**Before starting:**
- [ ] Working tree is clean (`git status`)
- [ ] `cargo test` passes on HEAD
- [ ] Baseline captured: `cargo clippy 2>&1 > clippy-baseline.txt && git add clippy-baseline.txt && git commit -m "chore: capture clippy baseline"`
- [ ] Total warning count recorded

**For each lint type:**
- [ ] Identify lint name and count: `cargo clippy 2>&1 | grep "clippy::LINT_NAME" | wc -l`
- [ ] Determine auto-fixable vs manual
- [ ] If auto-fixable: `cargo clippy --fix -- -A clippy::all -W clippy::LINT_NAME`
- [ ] If manual: fix file-by-file or pattern-by-pattern
- [ ] Review diff: `git diff`
- [ ] Run tests: `cargo test`
- [ ] Commit with lint name and before/after counts
- [ ] Update progress tracking

**Category sequence:**
- [ ] `correctness` lints: zero tolerance, fix first
- [ ] `suspicious` lints: treat as near-bugs
- [ ] `perf` lints: usually local changes, fix early
- [ ] `complexity` lints: separate mechanical from architectural
- [ ] `style` lints: high volume, auto-fixable, fix in bulk
- [ ] `pedantic` lints: opt-in, fix one lint group at a time

**After each phase:**
- [ ] `cargo clippy -- -Dwarnings` passes (or count has decreased)
- [ ] `cargo test` still passes
- [ ] Progress CSV updated
- [ ] No `#[allow]` added without a justification comment

**CI gate progression:**
- [ ] Stage 1: advisory (warn only, no CI failure)
- [ ] Stage 2: ratchet (count must not increase vs. baseline)
- [ ] Stage 3: deny default lints (`-Dwarnings`)
- [ ] Stage 4: deny pedantic (`-Dwarnings -Dclippy::pedantic`)

## Related Skills

- [project-structure-and-tooling.md](project-structure-and-tooling.md) — configure Clippy in `Cargo.toml` and `.cargo/config.toml` from day one; CI pipeline setup for `cargo clippy -- -Dwarnings`
- [modern-rust-and-editions.md](modern-rust-and-editions.md) — 2024 edition lint changes; lints that fire only on pre-2021 code patterns
- [error-handling-patterns.md](error-handling-patterns.md) — `clippy::unwrap_used`, `clippy::expect_used`, `clippy::missing_errors_doc` context; when `expect()` is preferable to `?`
- [testing-and-quality.md](testing-and-quality.md) — `#[allow(clippy::unwrap_used)]` in test modules; Clippy in integration test crates; `cargo-nextest` with lint gates
- [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — `clippy::redundant_clone`, `clippy::unnecessary_to_owned`, `clippy::borrow_deref_ref` fix patterns
- [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) — `clippy::trait_duplication_in_bounds`, `clippy::type_repetition_in_bounds`, `clippy::default_trait_access`
- [async-and-concurrency.md](async-and-concurrency.md) — `clippy::async_yields_async`, Clippy lints specific to async code; `cargo clippy` in async contexts
- [performance-and-profiling.md](performance-and-profiling.md) — `clippy::perf` category deep dive; when Clippy's perf suggestion conflicts with profiling data
- [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) — `clippy::restriction` lints in `unsafe` blocks; `clippy::undocumented_unsafe_blocks`
- [ai-ml-and-interop.md](ai-ml-and-interop.md) — Clippy configuration for mixed Rust/Python interop crates; `pyo3` and `ndarray` lint patterns
