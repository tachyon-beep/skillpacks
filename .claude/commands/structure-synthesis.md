---
description: Generative models whose outputs are graphs - typed DAG grammars, deterministic/latent-conditioned generation, best-of-K pools, mutation and recombination over lineages, canonicalisation to normal forms, equivalence detection, semantic hashing; keeps generation, structural verification, and utility judgement separate
---

# Structure Synthesis Routing

**For neural architecture search, program synthesis over typed IRs, or molecule/circuit generation where a model's output is a graph — not for training networks on data (that's the rest of `yzmir`) and not for selecting among existing, human-authored architecture families (`/neural-architectures`).**

Use the `using-structure-synthesis` skill from the `yzmir-structure-synthesis` plugin to route to the right specialist sheet.

## Sheets

**Grammar and representation:**
- **typed-graph-grammars** - operator whitelist, typing/shape rules, node/edge/parameter/memory ceilings, staged expressiveness levels
- **graph-representations-for-generation** - encodings a generator can emit; round-trip fidelity

**Generation:**
- **generation-strategies** - deterministic, latent-conditioned, best-of-K; escalation ladder to flows/diffusion
- **conditioning-on-context-and-contracts** - conditioning on diagnostic context, interface contracts, budgets without naming the answer

**Validity, canonical identity, verification (the technical core):**
- **validity-by-construction-vs-post-hoc** - constrained decoding vs. generate-then-verify
- **canonicalisation-and-normal-forms** - semantics-preserving normal form; idempotence as the hard test
- **equivalence-detection-and-semantic-hashing** - practical isomorphism, equivalence classes, hash design
- **structural-verification** - the full legality gate: shapes, cycles, contracts, trainability, forbidden ops

**Diversity, learning, lineage:**
- **diversity-and-mode-collapse** - canonical/functional diversity; duplicate rate as the honest metric
- **learning-objectives-for-generators** - training objectives; generator/judge separation
- **lineage-mutation-and-recombination** - archive-driven mutation and crossover

**Scaling:**
- **search-space-evolution-and-explosion-control** - growing the grammar safely: gates, ceilings, cost curve

**Catalogue:**
- **synthesis-anti-patterns** - the eight recurring failure patterns

## Commands

- `/design-graph-grammar` - requirements → typed grammar spec with ceilings, validity rules, staged expansion plan
- `/scaffold-structure-generator` - generator + verifier + canonicaliser skeleton with failing-first tests
- `/audit-canonicalisation` - adversarial review of a canonicaliser/hasher, severity-rated findings

## Agents

- `structure-synthesis-architect` - forward-design SME: grammar, representation, generation strategy, objectives
- `synthesis-integrity-reviewer` - critic SME: hunts generator/judge conflation and canonicalisation drift; zero-findings is treated as an audit defect

## Cross-references

- Selecting among existing, human-authored architecture families → `/neural-architectures`
- Embodying an accepted candidate in a live network (FSM, gradient isolation, alpha blending) → `/dynamic-architectures`
- Deciding WHEN/WHERE to request a structure, and admission decisions → `/morphogenetic-rl`
- Statistics of comparing a candidate's measured effect to a no-op baseline → `/counterfactual-statistics`
- Staged human/LLM procedure decomposition (a different sense of "graph") → `/procedural-architecture`
