---
description: Formalise the experimental design layer alone — the hypothesis and its mode (hypothesis-driven vs hypothesis-forming, recorded before execution), the factors and levels, the comparison structure including the mandatory no-intervention arm, the ancestor and shared-input identifiers that make pairing explicit, the declared unit of analysis, and the design strategies attached as separate instances because EXPO declares them disjoint. The stage where unverified terms and unsupportable statistical claims are most likely to enter.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write"]
argument-hint: "[path_to_experiment_or_pipeline]"
---

# Formalise Design Command

You are formalising the design layer only — the part that determines whether the record can support a causal claim later. Measurement, provenance, and validation are separate stages.

Read `skills/using-experiment-formalisation/controls-counterfactuals-and-replication.md` and `skills/using-experiment-formalisation/expo-verified-inventory.md` first.

## What this stage must produce

1. **The hypothesis**, as a falsifiable proposition — not a goal, not a description. If nobody can state what observation would refute it, that is the finding.
2. **The mode**: `GalileanExperiment` (hypothesis-driven) or `BaconianExperiment` (hypothesis-forming), **recorded before execution**. Exploratory work is a declared mode, not a defect; drifting from exploratory to hypothesis-driven after seeing results is.
3. **Factors and levels**: `ExperimentalFactor` (< `Variable`) with `FactorLevel`. Note `IndependentVariable` is an *attribute* (< `Independence` < `AttributeOfVariable`), not a variable type.
4. **Target variables**: `TargetVariable`.
5. **The comparison structure**, with the no-intervention arm as the null `FactorLevel` — `Treated_Untreated` < `ComparisonControl_TargetGroups`.
6. **Pairing identifiers**: the common ancestor state and the shared input sequence, per arm.
7. **The declared unit of analysis** — naming which identifier defines independence.
8. **Design strategies as several instances.**

## The disjointness check — run it

`Treated_Untreated`, `DoseResponse`, `TimeCourse`, `Normal_Disease`, `GeneKnock-in` and `GeneKnock-out` are declared pairwise `owl:disjointWith`; `ComparisonControl_TargetGroups` is disjoint with both `PairedComparison*Groups` classes; `PairedComparison` is disjoint with `QualityControl`.

A design with a no-intervention arm **and** a graded strength **and** a staged schedule needs **several `ExperimentalDesignStrategy` instances** attached to one `ExperimentalDesign`. Asserting one instance as several of these makes the ontology inconsistent. Consult `skills/using-experiment-formalisation/expo-owl-inventory.json` → `disjoint_with`, **checking both directions** — the map records each axiom under one class only, so a single key lookup silently passes.

## The three invariants this command enforces

- **Presence** — a no-intervention arm exists for every comparison.
- **Pairing** — arms record their ancestor state and shared inputs.
- **Eligibility to win** — the outcome vocabulary can represent *the no-intervention arm was better*. Check the enum. If it is `{adopt, defer}`, the apparatus cannot record its own failure, and that is a critical finding.

## Dispatch

```
Task(subagent_type="experiment-formalisation-architect",
     description="Formalise design layer for <scope>",
     prompt="Formalise the DESIGN LAYER ONLY per controls-counterfactuals-and-replication.md.
     Produce: hypothesis with falsifiability check; mode; factors/levels/target variables
     using OWL fragments verified against expo-owl-inventory.json; comparison structure
     with the mandatory no-intervention arm; ancestor and shared-input identifiers;
     declared unit of analysis; design strategies as SEPARATE instances checked against
     the disjointness axioms. Report any of the three invariants that fail.
     Do not formalise measurement, provenance, or validation.
     SYSTEM: <file list>")
```

## Output

The design-layer formalisation, the invariant results, and any confounding warning — if intervention strength and a scaffold withdrawal move on the same schedule, the effects are confounded and the design is multi-factor. EXPO names the resulting defect: `FaultyComparison` (< `ResultError`, a sibling of `ErrorOfConclusion`).

For what may be concluded from the resulting structure, hand off to `/counterfactual-statistics`.

## Cross-references

- `controls-counterfactuals-and-replication.md` — the comparison vocabulary and the three invariants
- `expo-verified-inventory.md` — the disjointness axioms and Galilean/Baconian modes
- `experiment-formalisation-architect` agent — the designer this command dispatches
- `/formalise-experiment` — the full staged workflow
- `/counterfactual-statistics` — what may be concluded from the structure this produces
