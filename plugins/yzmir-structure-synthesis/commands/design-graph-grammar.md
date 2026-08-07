---
description: Design a typed graph grammar from requirements - operator whitelist, ceilings, validity rules, and a staged expansion plan
allowed-tools: ["Read", "Write", "Bash", "Skill"]
argument-hint: "<domain-description> [--start-level=1|2|3]"
---

# Design Graph Grammar

Turn a requirements description (operator set, budgets, interface contract) into a complete typed grammar specification: an operator whitelist with typing/shape rules, node/edge/parameter/memory ceilings, and a staged expansion plan gated on reliability metrics.

## Core Principle

**The grammar you ship on day one should be the smallest one that covers the requirement — not the largest one you can imagine covering it.** Every operator and every unit of topology freedom is something `structural-verification.md` must check, `canonicalisation-and-normal-forms.md` must normalize, and `equivalence-detection-and-semantic-hashing.md` must hash correctly, for every candidate, forever. Start at the bottom of the staged-levels ladder in `typed-graph-grammars.md`; expand only when a reliability gate says the current level is healthy.

## Process

### Step 1: Elicit the Requirements

Before writing anything, pin down:

1. **What is the candidate for?** (a subgraph inserted into a host network; a program fragment over a typed IR; a molecule graph; etc.) — general terms, not naming the answer (see `conditioning-on-context-and-contracts.md` on why the request itself must not name the answer).
2. **The interface contract**: required input/output shape, arity, and any invariant the candidate's boundary must satisfy (e.g., shape-preserving insertion).
3. **The budget**: target parameter count, compute, and memory ceilings — as hard numbers, not "small."
4. **What must never appear**: any operation that is unsafe, unverifiable, or outside the intended domain (side-effecting operations, host-state access, anything the interface contract doesn't cover).
5. **Existing operator vocabulary**, if any — reuse names already established in the target codebase rather than inventing new ones.

If any of these is missing, ask before drafting rather than guessing — a grammar built on a guessed budget or guessed interface contract has to be redone, not adjusted.

### Step 2: Choose the Starting Level

Per `typed-graph-grammars.md`'s staged model:

- **Level 1 (universal envelope)** — default choice for a new domain with no prior track record. A single fixed parametric form (e.g., a low-rank update or small gated block); the "grammar" is really just the form's internal parameter space.
- **Level 2 (constrained genotype)** — appropriate when the requirement genuinely needs structural choice (width, stage count, activation family) but the overall topology can stay fixed. Enumerate the knob space explicitly and report its size — see `typed-graph-grammars.md`'s worked example for how to size it.
- **Level 3 (typed DAG)** — only start here if there's a specific, stated reason levels 1–2 cannot express the requirement. Default to starting lower and let `search-space-evolution-and-explosion-control.md`'s reliability gate justify moving up, rather than starting here because it's more general.

State the chosen level and the reasoning in the output — don't silently pick one.

### Step 3: Draft the Operator Whitelist and Typing Rules

For each operator: name, input arity, output arity, the shape-transformation rule (a function from input shape(s) to output shape), and whether it belongs to the grammar's declared identity-function subset (relevant to `canonicalisation-and-normal-forms.md`'s pruning rule — most operators are not identity operators; only declare the ones that provably compute the identity function under some parameterization).

### Step 4: Set the Ceilings

Four independent numbers, each justified against the budget from Step 1 — not derived from each other:

- Max nodes
- Max edges (independent of node count — see `typed-graph-grammars.md`'s RED scenario for why a node-only ceiling is insufficient)
- Max parameters
- Max memory (at a stated batch size)

### Step 5: Declare the Interface Contract and Forbidden Operations

State the exact input/output shape relationship the candidate's boundary must satisfy, and enumerate anything explicitly forbidden even if it would otherwise fit the whitelist (e.g., an operator that exists in the broader vocabulary but is unsafe for this domain).

### Step 6: Write the Staged Expansion Plan

For each level beyond the starting one: what would justify moving to it, and what the reliability-gate thresholds should be (rejection rate, duplicate rate, verification p95 latency — see `search-space-evolution-and-explosion-control.md`). Do not pre-approve the expansion; state the conditions under which it becomes reasonable to *propose*.

## Output Format

```markdown
# Graph Grammar: [domain]

## Requirements Summary
[What the candidate is for, interface contract, budget, forbidden operations — as stated by the user, not invented]

## Starting Level
[1, 2, or 3, with the reasoning]

## Operator Whitelist
| Operator | Input arity | Output arity | Shape rule | Identity op? |
|---|---|---|---|---|
| ... | ... | ... | ... | yes/no |

## Ceilings
- Max nodes: N
- Max edges: N
- Max parameters: N
- Max memory: N (at batch size B)

## Interface Contract
[Exact boundary shape/arity relationship]

## Forbidden Operations
[Explicitly excluded, with why]

## Staged Expansion Plan
| Next level | Trigger condition | Reliability-gate thresholds |
|---|---|---|
| ... | ... | rejection ≤ X%, duplicate ≤ Y%, verify p95 ≤ Zs |

## Open Questions
[Anything the requirements didn't specify and had to be assumed — flagged for the user to confirm]
```

## Anti-Patterns to Avoid While Drafting

| Temptation | Why to resist it |
|---|---|
| Starting at level 3 "to be safe" | More expressiveness without a track record just means more untested space; see `typed-graph-grammars.md` |
| Declaring an operator "probably an identity op" without a proof | This is exactly the bug in `canonicalisation-and-normal-forms.md`'s RED scenario — only declare identity-op membership when it's provable |
| Setting only a node ceiling because "edges scale with nodes anyway" | They don't reliably — see the dense-graph counterexample in `typed-graph-grammars.md` |
| Naming the interface contract by operator type rather than shape | That's naming the answer — see `conditioning-on-context-and-contracts.md` |

## Reference

For the full grammar-design discipline:
```
Load skill: yzmir-structure-synthesis:using-structure-synthesis
Then read: typed-graph-grammars.md
```

For sizing the staged expansion plan's gate thresholds:
```
Then read: search-space-evolution-and-explosion-control.md
```

For the interface contract's design:
```
Then read: conditioning-on-context-and-contracts.md
```
