---
description: Interactive elicitation of the lattice's tier set for a static analyzer. Walks the user through the trust hierarchy their codebase actually has — what data is untrusted, what sanitises it, what the analyzer should distinguish — and produces a tier-set proposal with an explicit join semilattice, monotonicity claims, and finite-height argument suitable for emission as `02-abstract-domain-spec.md`. Cross-validates against the `taint-lattice-design.md` reference sheet's anti-patterns ("boolean lattice masquerading as tiers"; "tiers added without justification"). Feeds the design pass; does not write code.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[analyzer_name_or_path]"
---

# Design Tier Model Command

You are eliciting the tier set for a static analyzer's lattice. The output is a draft of `02-abstract-domain-spec.md` for the user to review and adopt as part of the `analyzer-engineering/` artifact set. This command does NOT write code; it produces design content. It is the highest-leverage point in the analyzer's design — every later choice (rules, callgraph rung, stubs, suppressions) sits downstream of the tier set.

## Invocation Path

`/design-tier-model` is a Claude Code slash command. It complements `using-static-analysis-engineering` by structuring the lattice-design conversation explicitly. Use this command when:

- You are about to write `02-abstract-domain-spec.md` from scratch.
- An existing analyzer has a "boolean lattice masquerading as tiers" and you are refactoring `02-` to be honest.
- Two stakeholders disagree about the tier set and the disagreement is values-not-vocabulary.

For a quick design pass without elicitation, use `using-static-analysis-engineering`'s `taint-lattice-design.md` directly.

## Preconditions

The command takes a single optional argument: the analyzer name or path.

```bash
INPUT="${ARGUMENTS}"
TARGET_DIR=""

if [ -z "${INPUT}" ]; then
  # Use AskUserQuestion to collect:
  # "Which analyzer is this tier model for? Provide a name or a path containing analyzer-engineering/."
  :
fi
```

Read `analyzer-engineering/00-scope-and-targets.md` if present (so the elicitation aligns with the declared analyzer scope and tier). If absent, proceed with elicitation but output a draft `00-` first.

If `analyzer-engineering/02-abstract-domain-spec.md` already exists and the user did not pass `--replace`, ask via AskUserQuestion:

1. **Refine** — read existing tier set, walk through it, propose changes.
2. **Replace** — start from blank.
3. **Compare** — produce an alternative; let the user diff.

## Workflow

The elicitation has six rounds. Each round is an AskUserQuestion call with structured options or a free-text answer; each round writes a fragment of the draft `02-`.

### Round 1 — What property does the analyzer track?

Ask the user, in their own words: "What is the analyzer trying to detect?" Options to surface:

- Untrusted data reaching dangerous sinks (taint analysis).
- Data leaving authorised tenant boundaries (capability / multi-tenant).
- Privileged data being logged or returned to unprivileged callers (information-flow).
- Resources held without release (linear / affine).
- Custom — free text.

The answer determines the lattice's *purpose*. A "taint analyzer" and a "capability analyzer" have lattices with different shapes; conflating them is the most common source of "the lattice doesn't fit our problem".

### Round 2 — What are the distinguishable values along the property axis?

Free-text elicitation: "List every distinguishable value the analyzer should track. Examples for taint: untrusted, authenticated, sanitised, sanitised-for-HTML, sanitised-for-SQL, structured-data-from-trusted-source, ..."

Push the user past binary thinking. Most "we need 2 tiers" arguments collapse on questioning into "we need 4 tiers and we hadn't realised". Ask:

- "If a value passes the SQL sanitiser but not the HTML sanitiser, is it the same as one that passed both? If not, you need at least three tiers, not two."
- "Is the union of two distinct sanitisers a new tier, or an existing one?"

Record the candidate tier set as a flat list. The order doesn't matter yet.

### Round 3 — What is the partial order?

For every pair of tiers, ask whether one is "stricter than" the other. The answer to (`tier_a`, `tier_b`) is one of:

- `a ≤ b` — `a` is at least as restrictive as `b`; flow from `a` is acceptable wherever `b` is.
- `b ≤ a` — symmetric.
- Incomparable — neither implies the other; they live in distinct sub-orders.

Construct the Hasse diagram. Common shapes:

- **Linear chain** (`untrusted ≤ sanitised ≤ trusted`) — the simplest; appropriate when the property is one-dimensional.
- **Lattice with multiple sanitisers** — `untrusted` at bottom; `sanitised-for-HTML`, `sanitised-for-SQL` incomparable; `sanitised-for-both` joins them; `trusted` at top.
- **Multi-axis** — taint × capability × privilege. Each axis is its own chain; the lattice is the product. The lattice height multiplies; ensure finiteness still holds.

Validate: every two tiers must have a least upper bound (join). If not, the candidate is not a lattice — either add the join, or revise the partial order, or accept that the analyzer's domain is a join-semilattice (which is fine; state it).

### Round 4 — Define the join

For each pair of tiers, the join is what the analyzer assumes when both flow into the same point (e.g., a phi node, a function whose return tier depends on both branches).

Common join: "least upper bound by safety". `join(untrusted, trusted) = untrusted`-like (the more dangerous of the two; or `top` if the lattice has one). For sanitisers: `join(sanitised-for-HTML, sanitised-for-SQL) = `?

- If `sanitised-for-both` is in the lattice, use it.
- If it isn't, the join is `min(sanitised-for-HTML, sanitised-for-SQL)` — neither sanitiser was applied on this combined path; effectively `untrusted-for-anything-but-an-HTML-or-SQL-context-respectively`. Most analyzers conservatively join down to a coarser tier.

Write the join table explicitly. A lattice without an explicit join is a soup.

### Round 5 — Monotonicity and finite height

For each rule's transfer function and for the join itself, assert monotonicity: if `a ≤ b`, then `transfer(a) ≤ transfer(b)`. The user states this as a property; the analyzer's correctness depends on it.

Finite height: count the tiers along the longest chain. State the number explicitly. If you cannot, the lattice has potentially infinite chains; either prove finite chains argument by argument or apply widening at depth `k`.

This is the consistency gate's check 3 (Lattice well-formedness) and check 4 (Termination proof grounding). The output of this round is the per-pair monotonicity claim plus the height bound.

### Round 6 — Anti-pattern checks

Run the user's tier set against four anti-patterns from `taint-lattice-design.md`:

1. **Boolean lattice masquerading as tiers.** If every join collapses to "top" (i.e., the lattice is effectively `bottom`/`top` with intermediate tiers that never resolve), the lattice is boolean in disguise. Surface this; the user must either justify intermediate tiers (the rules consume them) or simplify to two tiers.
2. **Tiers added without justification.** A tier exists if and only if some rule consumes it differently from its neighbours. If no rule does, the tier is dead weight. Ask, for every tier: "what rule fires differently when this tier is observed?"
3. **Sanitisers as tiers vs sanitisers as transitions.** Some sanitiser shapes are tiers (`sanitised-for-HTML`); some are transitions (`apply_html_escape` shifts a value from one tier to another but is not a tier itself). Confusing the two creates spurious tier explosions. Walk through the user's list and classify each as tier-or-transition.
4. **Tiers without tests.** Every tier must have a TP and TN fixture in the test corpus. If a tier has neither, it cannot be falsified; demote or delete.

If any anti-pattern fires, write up the failure and propose a remediation. Do not paper over.

## Output

Emit `analyzer-engineering/02-abstract-domain-spec.md` with:

```
# 02 — Abstract Domain Spec (DRAFT)

## Property tracked
[from Round 1]

## Tier set
[the final list, with names]

## Partial order
[Hasse diagram in ASCII; the partial order is `≤` between named tiers]

## Join table
[every pair of tiers and their join]

## Meet table
[if relevant; not all analyzers use meet]

## Monotonicity claims
[per transfer function: claim, with sketch of why]

## Finite-height argument
[the longest-chain count, OR the widening operator and depth bound]

## Soundness/completeness statement
[which side the analyzer errs on; rationale]

## Test corpus seeds
[for each tier: at least one TP and one TN fixture, named]

## Open questions
[anything Round 6 surfaced that the user deferred]

## Cross-references
- Drives `03-inference-pipeline-spec.md`'s termination proof
- Drives `04-rule-plugin-spec.md`'s rule consumes-this-tier semantics
- Drives `05-false-positive-economics.md`'s "FP often = lattice imprecision" remediation
- Drives `07-callgraph-construction.md`'s `top` value at unresolved callsites
- Drives `08-cross-module-flow.md`'s default stub return tier
```

The output is a *draft*. The user must review the join table and monotonicity claims before adopting; this command captures the design decisions, it does not validate them mathematically.

If the user adopts the draft, prompt them to bump `02-` semver appropriately (initial version `1.0.0`, or major bump if replacing an existing tier set), and to re-run any downstream artifacts that depend on the lattice (consistency gate's coordinated re-emission rules in the router).

## Failure modes

| Symptom | Cause | Response |
|---------|-------|----------|
| User cannot articulate the property in Round 1 | The analyzer's purpose is unclear; a tier model is premature | Halt; surface the gap; suggest the user revisit `00-scope-and-targets.md` |
| Round 2 produces 30+ tiers | Confusion between tiers and transitions, or the analyzer is being asked to track too much | Push back to Round 1; consider splitting into two analyzers each with its own lattice |
| Join table has incomparable pairs without a common upper bound | The candidate is a poset, not a lattice | Either add the join, accept join-semilattice, or revise the partial order |
| Monotonicity claims hand-waved in Round 5 | Some transfer function is non-monotonic; analyzer will not terminate or will be unsound | Re-design the transfer function; flag in `03-` |
| All anti-pattern checks fire in Round 6 | The lattice is junk; the user is asking the analyzer to track informal categories | Halt; advise reducing scope; do not produce a lattice that will not survive contact with rules |

## Cross-References

- Router: `using-static-analysis-engineering` — where the spike is defined; this command is its tier-design step
- `taint-lattice-design.md` — the reference sheet; anti-patterns drawn from it
- `/scaffold-analyzer` — runs after this command; consumes `02-` directly
- `/design-rule-set` — runs after both; rules are written against the lattice this command settles
- Cross-pack: `ordis-security-architect:design-controls` — when the lattice tiers map to security control families
