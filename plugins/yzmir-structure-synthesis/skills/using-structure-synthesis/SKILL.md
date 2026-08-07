---
name: using-structure-synthesis
description: Use when a model's OUTPUT is a graph or structure rather than a label or scalar - generating neural-architecture candidates, program-synthesis over typed IRs, molecule or circuit graphs, or any typed DAG emitted by a generator. Covers typed graph grammars with node/edge/cost ceilings, deterministic and latent-conditioned generation, best-of-K pools, mutation/recombination over lineages, canonicalisation to normal forms, graph isomorphism and equivalence detection, semantic hashing, and mode-collapse diagnosis. Use for "graph grammar", "DAG generation", "canonicalise a graph", "graph isomorphism", "semantic hash", "duplicate rate after canonicalisation", "constrained decoding" for structured outputs, or "evolutionary search over graphs". Also use to audit a generator/verifier split for the generator grading its own examination.
---

# Using Structure Synthesis

## Overview

**A structure generator that also judges its own output is not a generator — it is a policy wearing a generator's clothes.**

This pack is for the class of system where a model's output is not a number or a label but an *object with topology*: a candidate neural-network subgraph, a program in a typed intermediate representation, a molecule graph, a circuit netlist, a level layout graph — anything a generator emits as a small directed graph over a constrained vocabulary of typed operators. The generation problem and the verification problem look similar (both touch the same graph) but they answer different questions, and collapsing them is the single most common failure mode this pack exists to prevent:

- **Generation answers "what structures are plausible."** A generator proposes. It may propose badly. It may mode-collapse. It may hallucinate topology that violates the grammar. None of that is fatal on its own.
- **Structural verification answers "is this candidate legal."** Shape-correct, cycle-free, contract-respecting, within budget. This is checkable by rule, not by taste, and it must not consume any signal about whether the candidate is *useful*.
- **Utility judgement answers "does this candidate help."** That question belongs to whatever evaluates the candidate against real outcomes — a counterfactual trial, a benchmark, a downstream reward. **This pack does not answer it and does not ship the machinery to answer it.**

The failure this pack is built around: a generator that filters its own best-of-K pool by predicted task performance is grading its own examination. A verifier that rejects a legal candidate because it looks unpromising has let policy leak into what should be a rule-driven gate. Both failures are invisible in a demo and catastrophic at scale, because they silently narrow the search space to whatever the *current* model already believes is good — which is exactly the thing generation was supposed to explore past.

Key tensions this pack resolves, sheet by sheet: *expressiveness vs. verifiability* (bigger grammars generate more, cost more to check), *validity-by-construction vs. generate-then-verify* (cheaper to decode legally, but you still need a verifier), *raw diversity vs. functional diversity* (a pool that looks different byte-for-byte may be one candidate after canonicalisation), *simple generator vs. simple in the wrong sense* (start deterministic/latent-conditioned; escalate to flows or diffusion only when that demonstrably fails).

## When to Use

Use this pack when:

- A model or search process must emit a **graph, program, or structured artifact** — not a scalar, not free text — and that artifact has to satisfy typing, shape, or resource constraints.
- You are designing the **grammar** a generator may emit from: operator whitelist, node/edge/parameter ceilings, staged expressiveness levels.
- You need to **canonicalise** generated structures to a normal form so that "same structure, different serialization" collapses to one identity.
- You need to detect whether two generated structures are **equivalent** (isomorphic under the grammar's semantics), or assign a **semantic hash** that is stable across serialization order and generator/library versions.
- You suspect **mode collapse**: a best-of-K pool that looks diverse in raw syntax but is one candidate after canonicalisation.
- You are choosing a **generation strategy**: deterministic direct generation, latent-conditioned sampling, stochastic best-of-K, mutation/recombination of an archived lineage, or (last resort) a flow/diffusion model.
- You are designing the **learning objective** for the generator itself: reconstruction, functional-effect matching, ranking, contrastive learning from failures — and you need the generator/judge boundary to survive that training.
- You are auditing an existing generator+verifier pipeline for **generator/judge conflation**, canonicalisation that silently changes semantics, or a hash that is not actually stable.

Do **not** use this pack when:

- You are selecting *among existing, human-authored* architecture families (ResNet vs. Transformer vs. Mamba) rather than generating novel topology → `yzmir-neural-architectures`.
- You are building the *growable network substrate itself* — FSM lifecycle, gradient isolation, alpha blending mechanics for a structure once it has been chosen → `yzmir-dynamic-architectures`. This pack produces the candidate structure; that pack embodies it.
- You are designing the *controller* that decides WHEN and WHERE to request a new structure, and how to score the causal effect of admitting it → `yzmir-morphogenetic-rl`. That pack owns the request and the admission decision; this pack owns only what happens between "a request was issued" and "a canonical, verified candidate exists."
- You need the **statistics** of comparing a generated candidate's pool against a no-op baseline (independent unit, paired tests, winner's curse, abstention calibration) → `yzmir-counterfactual-statistics`.
- You are decomposing a **staged human/LLM procedure** (a wizard, a curriculum, an approval pipeline) into stages and decision points → `axiom-procedural-architecture`. That pack's "graph" is a control-flow decomposition of expert work; this pack's graph is a typed computational structure a model emits as output.
- You are analyzing or writing rules for an existing static analyzer over *already-written* programs → `axiom-static-analysis-engineering`. That pack verifies code humans wrote; this pack verifies candidates a generator proposed, before anything is compiled or run.

## Start Here

If you are designing a structure-synthesis pipeline from scratch:

1. Read `typed-graph-grammars.md` — define the operator whitelist, ceilings, and staged expressiveness levels *before* choosing how to generate within them. Foundation; every other sheet assumes a grammar exists.
2. Read `graph-representations-for-generation.md` — pick the encoding the generator actually emits (op sequence, edge list, latent code) and know what it buys and costs in validity.
3. Read `validity-by-construction-vs-post-hoc.md` — decide how much illegality you prevent by construction vs. catch after the fact. Either way, a verifier is mandatory; this sheet tells you what it must still check.
4. Read `canonicalisation-and-normal-forms.md` and `equivalence-detection-and-semantic-hashing.md` together — these are the technical core. Get the canonical form and the hash right before anything downstream depends on candidate identity.
5. Read `structural-verification.md` — the full legality gate: shapes, cycles, contracts, trainability, forbidden operations.
6. Read `generation-strategies.md` — now choose deterministic, latent-conditioned, best-of-K, or mutation/recombination, informed by what the grammar and verifier can actually support.
7. Use the **Routing** table below for the rest (diversity, objectives, lineage, explosion control) as they become relevant.

Steps 1–5 are the spine: grammar, representation, validity discipline, canonical identity, verification. Skipping to generation strategy before the spine exists produces a generator with no stable notion of what it produced.

## How to Access Reference Sheets

All reference sheets are in the same directory as this `SKILL.md`. When you see a link like `[canonicalisation-and-normal-forms.md](canonicalisation-and-normal-forms.md)`, read the file from the same directory.

## Pipeline Position

This pack sits between "something decided a new structure is worth requesting" and "something will compile, execute, or judge the result." It owns generation and structural verification. It does not own the request, the compilation, or the judgement.

```
Upstream: whatever decides WHEN/WHERE a structure is needed
  (a NAS search loop, a morphogenetic RL controller, a program-
   synthesis driver, a molecule-design campaign)
        │
        │  a typed request: operator budget + interface contract
        │  ── NEVER a blueprint naming the answer ──
        ▼
┌───────────────────────────────────────────────────────────────┐
│                yzmir-structure-synthesis                        │
│                                                                  │
│  GENERATE ("imagines")            VERIFY & CANONICALISE          │
│  grammar · representation    →    ("permits")                    │
│  generation strategy ·            validity · canonical form ·    │
│  conditioning · diversity ·       equivalence class ·            │
│  lineage/mutation                 semantic hash                  │
│                                                                  │
│  A generator that filters its own pool by predicted utility     │
│  has crossed into the next box without permission.              │
└───────────────────────────────────────────────────────────────┘
        │
        │  a canonical, verified candidate (or ranked-by-nothing pool)
        ▼
Downstream: compiles/lowers, executes, and judges
  (a compiler or lowering pass; a counterfactual trial whose
   statistics are yzmir-counterfactual-statistics's job; an
   admission authority this pack must never become)
```

| Question | Pack |
|----------|------|
| "What operators/topology may a generator emit at all?" | **this pack** — `typed-graph-grammars` |
| "Is this candidate legal — shapes, cycles, contracts?" | **this pack** — `structural-verification` |
| "Are these two candidates the same structure?" | **this pack** — `equivalence-detection-and-semantic-hashing` |
| "How do I choose among existing architecture families?" | `yzmir-neural-architectures` |
| "How does a chosen structure actually get embodied and blended into a live network?" | `yzmir-dynamic-architectures` |
| "When/where should a new structure even be requested?" | `yzmir-morphogenetic-rl` |
| "Is this candidate's measured effect real, or noise?" | `yzmir-counterfactual-statistics` |

## Specialist Skills Catalog

**Grammar and representation (define the space before generating in it):**

| Sheet | Concern |
|-------|---------|
| `typed-graph-grammars` | Operator whitelist, typing/shape rules, node/edge/parameter/memory ceilings, staged expressiveness levels |
| `graph-representations-for-generation` | Encodings a generator can emit; round-trip fidelity; validity-friendly vs. validity-hostile choices |

**Generation:**

| Sheet | Concern |
|-------|---------|
| `generation-strategies` | Deterministic, latent-conditioned, stochastic best-of-K; the escalation ladder to flows/diffusion |
| `conditioning-on-context-and-contracts` | Conditioning on diagnostic context, interface contracts, and budgets without naming the answer |

**Validity, canonical identity, verification (the technical core):**

| Sheet | Concern |
|-------|---------|
| `validity-by-construction-vs-post-hoc` | Constrained decoding vs. generate-then-verify; why the verifier is mandatory either way |
| `canonicalisation-and-normal-forms` | Semantics-preserving normal form; idempotence as the hard test |
| `equivalence-detection-and-semantic-hashing` | Practical isomorphism, equivalence classes, hash design and its failure modes |
| `structural-verification` | The full legality gate: shapes, cycles, contracts, trainability, forbidden operations |

**Diversity, learning, and lineage:**

| Sheet | Concern |
|-------|---------|
| `diversity-and-mode-collapse` | Measuring diversity in canonical/functional space; duplicate rate as the honest metric |
| `learning-objectives-for-generators` | Reconstruction, functional-effect matching, ranking, contrastive learning — without crossing into self-judging |
| `lineage-mutation-and-recombination` | Parent selection, grammar-respecting mutation, crossover, provenance |

**Scaling the search space:**

| Sheet | Concern |
|-------|---------|
| `search-space-evolution-and-explosion-control` | Growing the grammar safely: gates, ceilings, the verification-cost curve |

**Catalogue:**

| Sheet | Concern |
|-------|---------|
| `synthesis-anti-patterns` | The eight recurring ways generation, verification, and judgement re-merge |

## Routing

| Symptom or question | Primary sheet |
|---------------------|---------------|
| "What operators/topology can the generator even produce?" | `typed-graph-grammars` |
| "What format does the generator actually emit — tokens, adjacency, latent?" | `graph-representations-for-generation` |
| "Should I use constrained decoding or generate-then-verify?" | `validity-by-construction-vs-post-hoc` |
| "My canonicaliser 'simplified' a node that wasn't actually redundant" | `canonicalisation-and-normal-forms` |
| "Two candidates should be the same structure but hash differently" | `equivalence-detection-and-semantic-hashing` |
| "The hash changed when I reordered edges / upgraded a library" | `equivalence-detection-and-semantic-hashing` |
| "How do I check a candidate is shape-correct and has no forbidden ops?" | `structural-verification` |
| "My best-of-K pool of 32 canonicalises down to 3 distinct candidates" | `diversity-and-mode-collapse` |
| "Raising sampling temperature but the pool isn't getting more diverse" | `generation-strategies`, `diversity-and-mode-collapse` |
| "What loss should the generator train against?" | `learning-objectives-for-generators` |
| "The generator is filtering its own candidates by predicted score" | `learning-objectives-for-generators`, `synthesis-anti-patterns` |
| "How do I mutate/recombine successful past candidates?" | `lineage-mutation-and-recombination` |
| "Should I let the grammar cover more operators?" | `search-space-evolution-and-explosion-control` |
| "Verification is getting slower every time we extend the grammar" | `search-space-evolution-and-explosion-control` |
| "I need to condition generation on a diagnostic signal without naming the fix" | `conditioning-on-context-and-contracts` |
| "Is my generator/verifier split actually sound?" | `synthesis-anti-patterns` |
| "Is this candidate's effect real, or should I have used a no-op baseline?" | → `yzmir-counterfactual-statistics` |
| "Set up the RL controller deciding when to request a structure" | → `yzmir-morphogenetic-rl` |
| "Embody an accepted candidate in a live network" | → `yzmir-dynamic-architectures` |

### Specialist Agents

- **`agent: structure-synthesis-architect`** — Forward-design SME: grammar, representation, generation strategy, and learning-objective selection for a given synthesis problem. Invoked via the `Agent` tool.
- **`agent: synthesis-integrity-reviewer`** — Critic SME: hunts generator/judge conflation, canonicalisation semantic drift, syntax-space diversity claims, and hash instability in a design or codebase. A zero-findings run is treated as an audit defect, not a clean bill of health. Invoked via the `Agent` tool.

### Specialist Commands

- **`/design-graph-grammar`** — From requirements (operator set, budgets, interface contract) to a typed grammar spec with ceilings, validity rules, and a staged expansion plan.
- **`/scaffold-structure-generator`** — Scaffold a generator + verifier + canonicaliser skeleton with failing-first tests: round-trip, canonicalisation idempotence, equivalent-graph hash equality, non-equivalent-graph hash separation.
- **`/audit-canonicalisation`** — Adversarially review a canonicaliser/hasher for semantic drift, non-idempotence, hash instability, and equivalence false positives/negatives, with severity-rated findings.

**Agents vs. skills:** Skills *design* the grammar, generator, and verifier. Agents *audit or critique* an existing design or implementation. Load a skill when designing; dispatch an agent when reviewing.

## Common Multi-Skill Scenarios

### Scenario: Greenfield structure-synthesis pipeline

1. `typed-graph-grammars` — Operator whitelist and ceilings first
2. `graph-representations-for-generation` — Pick the emission format
3. `validity-by-construction-vs-post-hoc` — Decide the validity strategy
4. `canonicalisation-and-normal-forms` + `equivalence-detection-and-semantic-hashing` — Canonical identity and hashing, together
5. `structural-verification` — The full legality gate
6. `generation-strategies` — Now choose how the generator actually samples
7. `learning-objectives-for-generators` — Train it, keeping generator/judge separate

### Scenario: Best-of-K pool looks suspiciously uniform

1. `diversity-and-mode-collapse` — Measure duplicate rate *after* canonicalisation, not before
2. `equivalence-detection-and-semantic-hashing` — Confirm the hash/canonicalisation pipeline is actually distinguishing distinct structures
3. `learning-objectives-for-generators` — Audit for a collapsed objective (no explicit latent, no min-over-K term)

### Scenario: Auditing an existing generator + verifier

1. `synthesis-anti-patterns` — Run the eight-pattern checklist first
2. `canonicalisation-and-normal-forms` — Idempotence test: does `canon(canon(x)) == canon(x)`?
3. `equivalence-detection-and-semantic-hashing` — Equal-semantics-same-hash and distinct-semantics-different-hash, both directions
4. `structural-verification` — Does the verifier consume any utility/reward/provenance signal? It must not.
5. → dispatch `synthesis-integrity-reviewer` for a full pass with severity ratings

### Scenario: Grammar keeps growing, verification is getting slow / unreliable

1. `search-space-evolution-and-explosion-control` — Is there a reliability gate before each expansion?
2. `structural-verification` — Measure the verification-cost curve against grammar size
3. `typed-graph-grammars` — Consider staged levels instead of one flat grammar

### Scenario: Choosing a generation strategy for a new domain

1. `generation-strategies` — Start at the bottom of the escalation ladder: deterministic or latent-conditioned
2. `conditioning-on-context-and-contracts` — What must the request constrain vs. leave open?
3. `lineage-mutation-and-recombination` — If an archive of past candidates exists, consider mutation/recombination before a from-scratch generator

## Decision Tree

```
Designing the pipeline from scratch?
├─ Yes → typed-graph-grammars → graph-representations-for-generation
│        → validity-by-construction-vs-post-hoc
│        → canonicalisation-and-normal-forms + equivalence-detection-and-semantic-hashing
│        → structural-verification → generation-strategies
│        → learning-objectives-for-generators
└─ No  → continue

Best-of-K pool looks uniform / duplicated?        → diversity-and-mode-collapse
Two candidates that should match don't (or        → equivalence-detection-and-semantic-hashing
  vice versa)?
Hash changed after a reorder / library upgrade?   → equivalence-detection-and-semantic-hashing
Canonicaliser deleted something it shouldn't have? → canonicalisation-and-normal-forms
Generator filtering its own pool by predicted     → learning-objectives-for-generators,
  score?                                             synthesis-anti-patterns
Grammar growing without a reliability gate?       → search-space-evolution-and-explosion-control
Need mutation/recombination over an archive?      → lineage-mutation-and-recombination
Auditing an existing pipeline end-to-end?         → synthesis-anti-patterns
                                                      → dispatch synthesis-integrity-reviewer
Need the request format itself designed?          → conditioning-on-context-and-contracts
Comparing a candidate's measured effect to a       → yzmir-counterfactual-statistics
  no-op baseline?
Deciding WHEN/WHERE to request a structure?       → yzmir-morphogenetic-rl
Embodying an accepted candidate in a live network? → yzmir-dynamic-architectures
```

## Rationalization Resistance

| Rationalization | Reality | Counter-guidance |
|-----------------|---------|------------------|
| "The generator should just skip proposing candidates it thinks are bad" | That is the generator judging its own examination; predicted-bad candidates are exactly the ones the judge should see, so the pool's failure modes are observable | See `learning-objectives-for-generators` and `synthesis-anti-patterns` |
| "The verifier can down-rank ugly-looking candidates, not just reject illegal ones" | Ranking-by-taste is utility judgement wearing a verifier's badge | See `structural-verification` — verification is legality only |
| "Our best-of-K=32 pool is obviously diverse, look at the raw graphs" | Raw-syntax diversity is cheap and misleading; canonicalise first, then measure | See `diversity-and-mode-collapse` |
| "`nx.is_isomorphic()` is good enough to check if two candidates match" | Bare topology-only isomorphism ignores operator labels; two structurally-identical graphs with completely different operators compare as isomorphic | See `equivalence-detection-and-semantic-hashing` |
| "A canonicaliser that removes any degree-(1,1) node is just cleaning up pass-throughs" | Not every single-in/single-out node is an identity function; deleting a real nonlinearity changes semantics | See `canonicalisation-and-normal-forms` |
| "Refining node signatures from their inputs is enough to canonicalise" | Nodes distinguished only by their *downstream* role stay tied forever, and raw labels then leak into the canonical form — one structure, several identities | See `canonicalisation-and-normal-forms` RED Scenario 2 |
| "The hash library's default is fine, we don't need our own serialization" | Hash libraries change their output across versions and are frequently attribute-blind by default; an unpinned hash is not a stable identity | See `equivalence-detection-and-semantic-hashing` |
| "We'll just let the grammar cover a few more operators, it's a small change" | Grammar growth is combinatorial in verification cost; "small" additions have caused search and verification to become intractable in this exact failure mode before | See `search-space-evolution-and-explosion-control` |
| "Constrained decoding means we don't need a separate verifier" | Constrained decoding narrows the search but rarely proves every legality property (cross-node contracts, global budgets); the verifier is still required | See `validity-by-construction-vs-post-hoc` |
| "Diffusion/flow models are strictly more powerful, let's start there" | The first generator should be small and deterministic or latent-conditioned; flows/diffusion are justified only by a demonstrated coverage failure of the simpler model | See `generation-strategies` |
| "Mutating winners only is more efficient than mutating the whole archive" | Winners-only mutation is survivorship bias applied to search; failed and rejected lineages carry information a winners-only archive discards | See `lineage-mutation-and-recombination` |

### Red Flags Checklist

- [ ] **Generator filters its own pool**: any code path where the generator drops or re-ranks candidates using a predicted-utility signal
- [ ] **Verifier reads reward, future utility, or candidate provenance** in its accept/reject decision
- [ ] **Canonicaliser changes non-equivalent semantics**: a "simplification" that isn't provably identity-preserving
- [ ] **Canonicaliser splits one structure into several identities**: signature refinement that reads predecessors only, letting raw node IDs leak into the "canonical" form for branch-and-merge graphs
- [ ] **Diversity measured on raw syntax**: duplicate rate reported before canonicalisation
- [ ] **Hash has no version field**: no way to detect that a library upgrade silently changed hash values
- [ ] **Equivalence check is bare, unlabeled isomorphism**: `nx.is_isomorphic()` (or equivalent) with no node/edge attribute matching
- [ ] **Grammar expanded with no reliability gate**: new operators added without a staged-expressiveness review
- [ ] **Validity left entirely post-hoc** when cheap constrained decoding could have prevented most illegal candidates
- [ ] **Mode collapse hidden behind syntactic variation**: a pool that looks different but canonicalises to one structure

## Integration with Other Skillpacks

### Dynamic architectures (`yzmir-dynamic-architectures`)

That pack covers the growable network substrate an accepted candidate is embodied into — FSM lifecycle, gradient isolation, alpha blending. This pack covers everything upstream of that: inventing and legality-checking the candidate structure itself.

### Morphogenetic RL (`yzmir-morphogenetic-rl`)

That pack's controller decides WHEN and WHERE a new structure should be requested, and judges the causal effect of admitting one. This pack answers what happens between "a request was issued" and "a canonical, verified candidate exists" — it never decides whether to admit anything.

### Counterfactual statistics (`yzmir-counterfactual-statistics`)

Once a candidate (or pool) exists, deciding whether its measured effect is real — the independent unit, paired tests against a no-op anchor, the winner's curse in best-of-K screening — is that pack's job, not this one's.

### Neural architectures (`yzmir-neural-architectures`)

That pack selects among existing, human-authored architecture families for a given modality and constraint set. This pack generates *novel* topology from a grammar as a model's own output.

### Procedural architecture (`axiom-procedural-architecture`)

That pack decomposes staged human/LLM/team procedures into stages and decision points — a different sense of "graph" (control flow over expert work, not a typed computational structure a model emits).

### Other packs

| Request | Primary pack |
|---------|--------------|
| Static analysis of already-written code | `axiom-static-analysis-engineering` |
| PyTorch implementation of the generator network itself | `yzmir-pytorch-engineering` |
| Training the generator faster (mixed precision, FSDP) | `yzmir-training-optimization` |
| Deploying an accepted structure's compiled artifact | `yzmir-ml-production` |
| Determinism of the generation process under fixed seed/latent | `axiom-determinism-and-replay` |

## Quick Reference

| Need | Use this |
|------|----------|
| Define the operator whitelist and ceilings | `typed-graph-grammars` |
| Pick what format the generator emits | `graph-representations-for-generation` |
| Choose deterministic / latent / best-of-K / mutation | `generation-strategies` |
| Design the request without naming the answer | `conditioning-on-context-and-contracts` |
| Decide constrained-decoding vs. generate-then-verify | `validity-by-construction-vs-post-hoc` |
| Get the canonical normal form right | `canonicalisation-and-normal-forms` |
| Detect equivalence / build a stable semantic hash | `equivalence-detection-and-semantic-hashing` |
| Run the full legality gate | `structural-verification` |
| Measure diversity honestly, diagnose collapse | `diversity-and-mode-collapse` |
| Choose the generator's training objective | `learning-objectives-for-generators` |
| Mutate/recombine an archived lineage | `lineage-mutation-and-recombination` |
| Grow the grammar without an explosion | `search-space-evolution-and-explosion-control` |
| Check the eight recurring failure patterns | `synthesis-anti-patterns` |
| Scaffold a greenfield generator+verifier | command: `/scaffold-structure-generator` |
| Design a grammar from requirements | command: `/design-graph-grammar` |
| Audit an existing canonicaliser/hasher | command: `/audit-canonicalisation` |
| Forward-design a synthesis pipeline | agent: `structure-synthesis-architect` |
| Critique an existing pipeline for integrity | agent: `synthesis-integrity-reviewer` |

## The Bottom Line

**Generation, structural verification, and utility judgement are three different questions, and the moment one component answers more than one of them, the search space silently narrows to whatever the system already believes is good.** This pack designs the first two — a generator that may propose badly, and a verifier that checks legality by rule and consumes no utility signal — and gives them a stable shared vocabulary: a canonical normal form and a semantic hash that agree exactly when two structures mean the same thing. It deliberately does not answer whether a candidate is any good. That question, and the machinery to answer it safely, belongs downstream.

---

## Reference Sheets

After routing, load the appropriate specialist sheet:

**Grammar and representation:**

1. [typed-graph-grammars.md](typed-graph-grammars.md) — Operator whitelist, typing/shape rules, ceilings, staged expressiveness levels
2. [graph-representations-for-generation.md](graph-representations-for-generation.md) — Encodings a generator can emit; round-trip fidelity

**Generation:**

3. [generation-strategies.md](generation-strategies.md) — Deterministic, latent-conditioned, best-of-K; escalation to flows/diffusion
4. [conditioning-on-context-and-contracts.md](conditioning-on-context-and-contracts.md) — Conditioning on context, contracts, and budgets

**Validity, canonical identity, verification:**

5. [validity-by-construction-vs-post-hoc.md](validity-by-construction-vs-post-hoc.md) — Constrained decoding vs. generate-then-verify
6. [canonicalisation-and-normal-forms.md](canonicalisation-and-normal-forms.md) — Semantics-preserving normal form; idempotence
7. [equivalence-detection-and-semantic-hashing.md](equivalence-detection-and-semantic-hashing.md) — Isomorphism, equivalence classes, hash design
8. [structural-verification.md](structural-verification.md) — The full legality gate

**Diversity, learning, lineage:**

9. [diversity-and-mode-collapse.md](diversity-and-mode-collapse.md) — Canonical/functional diversity; duplicate rate
10. [learning-objectives-for-generators.md](learning-objectives-for-generators.md) — Training objectives; generator/judge separation
11. [lineage-mutation-and-recombination.md](lineage-mutation-and-recombination.md) — Archive-driven mutation and crossover

**Scaling the search space:**

12. [search-space-evolution-and-explosion-control.md](search-space-evolution-and-explosion-control.md) — Growing the grammar safely

**Catalogue:**

13. [synthesis-anti-patterns.md](synthesis-anti-patterns.md) — The eight recurring failure patterns
