---
name: conditioning-on-context-and-contracts
description: "Use when designing the request a generator conditions on - diagnostic context, interface contracts, and budgets - and when a request field is at risk of naming the answer instead of constraining what may be built, which quietly turns generation into rubber-stamping a pre-chosen structure."
---

# Conditioning on Context and Contracts

## When to Use

- Designing the schema of the request that triggers generation
- Reviewing whether a request field biases the generator toward a specific answer rather than constraining the space
- Debugging "the generator always produces roughly the same family of structure regardless of context"
- Deciding what diagnostic signal, budget, or contract a generator should condition on

This sheet is about the *request* — what goes in before generation starts. For what the generator does with it, see `generation-strategies.md`. For what the grammar bounds regardless of any request, see `typed-graph-grammars.md`.

## Core Principle

**A request constrains what may be built. It must never name the answer.** The distinction is not stylistic — a request field that names a specific operator family, module type, or topology pattern converts the generator into a lookup table keyed by that field, whatever else the request contains. Diagnostic context, budgets, and interface contracts are legitimate because they describe *the problem*, not *the solution*. A field like "preferred family" or "suggested operator" describes the solution before generation has even started, and the generator's apparent choice becomes a rubber stamp.

This matters because the entire value of generation over blueprint selection is that the structure is derived from the actual deficit or task, not from what a human (or an earlier, possibly wrong, decision) guessed the answer should be. A request that leaks the answer collapses that value silently — the pipeline still runs, samples still come out, and the generator looks like it's doing its job while actually just reflecting back whatever the request told it to.

## What a Request May Legitimately Contain

| Field class | Legitimate content | Not legitimate |
|---|---|---|
| **Diagnostic context** | A measured signal about the current deficit — where it is, what shape it has, what evidence supports it | A pre-diagnosed *fix* for the deficit |
| **Interface contract** | Required input/output shape relationship, arity, insertion point | A named module type that would satisfy the contract |
| **Budget** | Parameter/compute/memory ceilings for this request | A budget scoped specifically to fit one particular answer |
| **Request reason** | Why this request was issued, for audit — read by humans, excluded from the generator's decision logic | A field the generator is expected to branch its topology choice on |

The test for any candidate field: **could a human read this field and correctly guess the intended answer without knowing anything else about the domain?** If yes, the field names the answer and does not belong in the request.

## The RED Scenario: A Field That Names the Answer

```python
from dataclasses import dataclass

@dataclass
class RequestWRONG:
    diagnostic_context: dict
    parameter_budget: int
    preferred_operator_family: str   # <-- names the answer

def generate_WRONG(request: RequestWRONG) -> str:
    return request.preferred_operator_family  # obediently follows the named field
```

```python
ctx_low = {"dominant_frequency_band": "low"}
ctx_high = {"dominant_frequency_band": "high"}

wrong_low = generate_WRONG(RequestWRONG(ctx_low, 1000, preferred_operator_family="attention"))
wrong_high = generate_WRONG(RequestWRONG(ctx_high, 1000, preferred_operator_family="attention"))
assert wrong_low == wrong_high == "attention"
print(f"WRONG: low-signal -> {wrong_low}, high-signal -> {wrong_high} (context is ignored entirely)")
```

Both requests produce `"attention"` regardless of what the diagnostic context actually says, because the request told the generator what to build and the generator complied. This is not a hypothetical failure mode of a badly-written generator — it is the *correct* behavior of a generator given a request that names the answer. The bug is in the schema, not the model.

### The GREEN Fix: Condition on Signal, Not on a Pre-Named Answer

```python
from dataclasses import dataclass

@dataclass
class RequestGREEN:
    diagnostic_context: dict
    parameter_budget: int
    interface_contract: dict

def generate_GREEN(request: RequestGREEN) -> str:
    signal = request.diagnostic_context.get("dominant_frequency_band", "low")
    return "attention" if signal == "high" else "convolution"

green_low = generate_GREEN(RequestGREEN(ctx_low, 1000, interface_contract={"shape_preserving": True}))
green_high = generate_GREEN(RequestGREEN(ctx_high, 1000, interface_contract={"shape_preserving": True}))
assert green_low != green_high
print(f"GREEN: low-signal -> {green_low}, high-signal -> {green_high} (context actually drives the choice)")
```

The GREEN request contains no field a human could read to correctly guess the family in advance — only a diagnostic signal, a budget, and a contract. The generator's choice is now actually a function of context, and testing whether that function is a *good* one (not just a responsive one) is exactly the job that belongs to the downstream evaluator, never to the request schema itself.

## Interface Contracts: Constrain the Boundary, Not the Interior

An interface contract declares what a candidate's boundary must look like — typically a shape-preserving insertion requirement (output shape equals input shape, so the candidate can be spliced into an existing computation without touching anything else) — without constraining what happens between the boundary and the output. This is the difference between "the candidate must accept and return a `(batch, 64)` tensor" (legitimate — it's a boundary condition any correct candidate must satisfy) and "the candidate must be a convolution that accepts and returns a `(batch, 64)` tensor" (illegitimate — the second clause names the answer). `structural-verification.md` checks the boundary contract on every candidate; the request is what declares it.

## Red Flags Checklist

- [ ] **Any request field that, read alone, tells a human what the generator will produce**
- [ ] **A field documented as "just a hint" that the generator's actual output distribution shows it never ignores**
- [ ] **Interface contract specifies an operator identity rather than a boundary shape/arity requirement**
- [ ] **Diagnostic context computed downstream of a decision about what fix to apply** — context must precede and be independent of the answer, not rationalize one already chosen
- [ ] **Budget scoped to fit one specific candidate** rather than declared independently of what gets generated

## Cross-References

- **How the generator actually uses this request to sample**: `generation-strategies.md`
- **The boundary contract this sheet's interface clause feeds into verification**: `structural-verification.md`
- **The grammar the request's budget must stay within**: `typed-graph-grammars.md`
- **The full anti-pattern catalogue, including request-schema leaks**: `synthesis-anti-patterns.md`
