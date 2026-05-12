---
description: Analyst pipeline — apply queueing-theory, discrete-event-simulation, or workflow-net reasoning when structural review alone is insufficient.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit"]
argument-hint: "[procedure_text_or_file] [analysis_question]"
---

# Analyze Procedure Command

<!-- sections: When to Use · Required Input · Pipeline · Output Format · Hand-off -->

You are running the analyst pipeline for `axiom-procedural-architecture`: a structurally-sound procedure + an analysis question → a scoped report covering bottleneck identification, simulation recommendations, or soundness verdicts. Not for first-pass structural critique (use `/review-decomposition`); not for producing a decomposition (use `/decompose-procedure`); not for continuous-time dynamics outside procedure shape (use `/simulation-foundations`).

## When to Use

Use when you have a structurally-sound procedure — one that has passed or would pass `/review-decomposition` — and you need to reason beyond structure: about throughput capacity, queue saturation, stochastic sensitivity, formal soundness under regulatory or safety constraints, or a mismatch between the procedure's current notation and the question being asked.

Do **not** use for first-pass structural critique of a new decomposition → `/review-decomposition`. Do **not** use for continuous-time dynamics or content-domain simulation that lives outside the shape of the procedure itself → `/simulation-foundations`.

## Required Input

Before running, collect both:

1. **Procedure under analysis** — the procedure in any reasonable format (inline text, markdown, file path, stage list). It should already be structurally sound; if it has unresolved structural findings, run `/review-decomposition` first.
2. **Analysis question** — a concrete statement of what you need to know. Examples:
   - "What is the maximum sustainable throughput given N concurrent workers?"
   - "Where does the process saturate under bursty arrival patterns?"
   - "Is this workflow sound — can every token reach the final state without deadlock?"
   - "We modeled this as a flowchart but the question is really about state invariants — are we in the right notation?"

If either is missing, ask before proceeding. An underspecified question produces an underspecified report.

## Pipeline

Route on the analysis question. Read the sheet before applying its guidance — do not rely on memory.

**Capacity / bottleneck question** — the question is about throughput, utilisation, queue length, service rates, or where work piles up under load.
→ Read and apply **`queueing-theory-for-procedures.md`**.
Identify the bottleneck stage using utilisation analysis (U = λ/μ per stage). Compute queue build-up (Little's Law: L = λW). Rank stages by utilisation. Emit bottleneck identification + remediation options (parallelise, reorder, split, cache exit artifacts).

**Stochastic / non-Markovian / sensitivity question** — arrival patterns are bursty or non-Poisson; service times are highly variable; you need confidence intervals or sensitivity analysis across parameter ranges.
→ Read and apply **`discrete-event-simulation-for-procedures.md`**.
Model the procedure as a DES event graph. Run parameter sweeps across the sensitivity range specified in the analysis question. Emit simulation recommendations + sensitivity findings with confidence bounds.

**Soundness / regulatory / safety-critical question** — the question is about deadlock freedom, liveness, reachability, compliance with a regulatory requirement, or whether every execution path terminates correctly.
→ Read and apply **`process-algebra-and-workflow-nets.md`**.
Translate the procedure into a workflow net (or process-algebra expression). Run soundness checks: every token can reach the final marking; no dead transitions; no deadlocks; every place is bounded. Emit soundness verdict + violation evidence (reachability counter-examples, dead transition list, or liveness failures).

**Abstraction-mismatch question** — the procedure is described in one notation (e.g., flowchart) but the analysis question requires a different model (e.g., state machine, decision table).
→ Read and apply **`flow-vs-state-vs-decision-modeling.md`**.
Identify the mismatch between the current notation and what the question requires. Map the procedure into the appropriate model. Emit a re-modeling recommendation and, where feasible, a translated representation ready for the target analysis.

If the analysis question spans more than one route, run the primary route first; note the secondary route and run it after.

## Output Format

A self-contained analysis report scoped to the question. Structure depends on the route taken:

**Bottleneck report (queueing route):**
```
Bottleneck Stage: <name>
Utilisation: <U = λ/μ>
Queue build-up (Little's Law): L = <value> at <arrival rate>
Ranked stages by utilisation: [table]
Remediation options:
  1. <option> — expected utilisation after change
  2. <option> — expected utilisation after change
```

**Simulation report (DES route):**
```
Model: <event-graph summary>
Parameters swept: <range>
Key findings:
  - <metric>: mean=<x>, 95th-pct=<y>, sensitivity to <param>=<z>
Recommendations:
  1. <recommendation>
```

**Soundness report (workflow-nets route):**
```
Verdict: sound | unsound | conditionally sound
Checks run: deadlock-free, live, bounded, proper completion
Violations (if any):
  - <violation type>: <evidence — reachability path or dead transition>
Remediation: <structural change to restore soundness>
```

**Re-modeling report (abstraction-mismatch route):**
```
Current notation: <type>
Required notation: <type>
Mismatch: <description>
Translated representation: <inline or file reference>
Ready for: <target analysis>
```

## Hand-off

When the analysis question crosses a boundary this pack does not own, name the correct pack and stop — do not attempt to answer the out-of-scope part.

Read **`procedural-boundary-and-handoffs.md`** for the full boundary catalog. Common boundary cases:

- **Continuous-time dynamics** (ODEs, physical simulation, agent-based modelling of content-domain entities) → `/simulation-foundations`
- **Structural decomposition review** (grain, MECE branching, dependency ordering) → `/review-decomposition`
- **Content-domain capacity** (hardware sizing, network throughput independent of procedure shape) → out of pack scope; name the relevant infrastructure or performance-engineering resource

State clearly: "This question crosses into [domain]. Use [pack/resource] for that part. The within-boundary portion of the analysis is above."
