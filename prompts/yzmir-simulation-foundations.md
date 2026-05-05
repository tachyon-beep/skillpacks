# Refresh: yzmir-simulation-foundations

**Verdict:** HIGH / M effort. **Investigation required before edits** — this is structural breakage, not staleness.

## Context

- Pack path: `/home/john/skillpacks/plugins/yzmir-simulation-foundations/`
- Full review: `/tmp/skillpack-refresh-review/yzmir-simulation-foundations.md`
- Purpose: simulation math (ODEs, integrators, stability, control, numerics, chaos, stochastic).

## Why refresh

Reviewer found **structural defects**, not just staleness:
- `feedback-control-theory.md` is essentially missing — 113 lines with no PID implementation.
- 6 of 8 specialist sheets are truncated at the top with no titles or overviews.

Underlying math is evergreen — once content is restored, the pack should be solid.

## Step 0 — investigate before refreshing

This is mandatory. Don't start writing replacement content until you know what was lost.

```bash
cd /home/john/skillpacks
git log --all --oneline -- plugins/yzmir-simulation-foundations/skills/feedback-control-theory/SKILL.md
git log --all --oneline -- 'plugins/yzmir-simulation-foundations/skills/*/SKILL.md' | head -50
```

For each truncated sheet, run:

```bash
git log --all --diff-filter=D --summary -- <path>
git log --all -p -- <path> | head -300
```

Branches in this repo include worktrees; check `git branch -a` and `git stash list`.

## Possible outcomes

1. **Content was deleted** — recover from git history, verify it's still accurate, restore.
2. **Content was never written** — reviewer found stubs from initial scaffolding. Decide pack-by-pack: write proper content, or remove the skill entry from the router.
3. **Content was moved** — find current location, fix links.

Surface the outcome to the user before proceeding to Step 1.

## Step 1 — refresh (after investigation)

For sheets that need writing:

- `feedback-control-theory/SKILL.md` — needs full PID implementation, transfer functions, root-locus reading, stability margins, anti-windup, derivative kick.
- Any sheet with missing titles/overviews — restore the standard SKILL.md skeleton: frontmatter, H1, "When to use", "Core concepts", "Recipes", "Anti-patterns".

Additive opportunities (only after restoration):
- Differentiable simulation pointers (JAX/diffrax, neural ODEs) where relevant.
- Cross-pack references to `yzmir-pytorch-engineering` for autograd-through-simulation.

## Scope — DO NOT

- Do not rewrite math that's already correct in surviving sheets.
- Do not introduce JAX/PyTorch dependencies into foundational math sheets — keep them library-agnostic.

## Acceptance criteria

1. Every SKILL.md has a proper title, frontmatter, and overview.
2. `feedback-control-theory/SKILL.md` contains a working PID example (textbook form).
3. Router skill links to every specialist sheet that exists; entries for non-existent sheets removed.
4. `plugin.json` version bumped (minor).

## Process

1. Read `/tmp/skillpack-refresh-review/yzmir-simulation-foundations.md`.
2. Run Step 0 git archaeology. Report findings before writing.
3. After user confirms the path forward, restore or rewrite.
4. Verify by reading every restored sheet end-to-end.
5. Bump version.

## Constraints

- No fabrication of equations or derivations — cite or omit.
- No expansion beyond what was originally scoped (no new sheets unless restoring).
