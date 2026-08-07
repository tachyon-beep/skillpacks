---
description: Forward-design a typed contract suite for a system's subsystem boundaries — record schemas with explicit absence encoding and one authority per record class, fail-closed versioning rules, deterministic resolver contract, blinded views, canonical-identity plan, versioned policy records, import-direction gate, and the contract-test plan. Elicits the boundary inventory if absent, then dispatches the contract-suite-architect agent and returns the full design package with per-decision confidence and risk.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[design_doc_path_or_system_description]"
---

# Design Contract Suite Command

You are producing a complete contract-suite design for a system, via the `contract-suite-architect` agent. The output is a design package an implementer can build directly — not code, not a review of existing code (that is `/review-contracts`).

## Step 1 — Assemble the input package

The architect needs a boundary inventory. Gather it before dispatching:

1. If the argument is a path (design doc, HLD, README), READ it and extract: subsystems, who produces what for whom, the record classes or data flows named, any invariants/ADRs/compliance constraints cited, any blinding or redaction requirements, and the fleet/upgrade reality.
2. If the argument is a prose description, use it directly.
3. If neither yields a boundary inventory, ask the user for exactly these (one round, all at once):
   - the subsystems and who talks to whom;
   - the record classes (or flows) crossing each boundary;
   - which fields can be unmeasured or optional, per record;
   - who must not see what, if anyone;
   - whether producers and consumers upgrade together or a mixed fleet persists.
4. For brownfield systems, Glob/Grep for existing contract definitions (`contracts/`, `schemas/`, `*.proto`, `*_pb2.py`, dataclass modules named for records) and include their paths in the dispatch — the architect must read them rather than design against an imagined state.

## Step 2 — Dispatch the architect

```
Task(subagent_type="contract-suite-architect",
     description="Design contract suite for <system>",
     prompt="Design the complete contract suite for the following system.
     Follow your design steps 1-10 and produce the full output format,
     including the catalogue self-check and per-decision confidence/risk.

     BOUNDARY INVENTORY:
     <extracted or elicited inventory>

     SOURCE DOCUMENTS (read these first):
     <paths, with the anchors/sections that matter>

     EXISTING CONTRACTS (brownfield; read before designing):
     <paths or 'none — greenfield'>

     CONSTRAINTS AND INVARIANTS CITED BY THE USER:
     <verbatim list, using the user's own identifiers>")
```

## Step 3 — Present the package

Return the architect's design package intact — do not summarize away the authority table, the evolution rules, or the catalogue self-check; those are the deliverable. Prepend a five-line orientation: system, record-class count, the highest-risk design decisions, and the open questions requiring the user.

## Verification

Before presenting, check the package contains every section of the architect's output format (authority table → catalogue self-check) plus the four SME protocol sections. If a section is missing, send the agent back for it rather than presenting an incomplete package.

Recommend as next steps: resolve open questions → implement contracts and tests → run `/review-contracts` as the independent audit.

## Cross-references

- `using-contract-engineering` — router; the failure-mode catalogue the self-check runs against
- `contract-suite-architect` agent — the designer this command dispatches
- `/review-contracts` — independent audit after implementation
- `/audit-contract-drift` — ongoing drift sweep once the suite is live
