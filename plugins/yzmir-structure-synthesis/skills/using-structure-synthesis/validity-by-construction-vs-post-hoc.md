---
name: validity-by-construction-vs-post-hoc
description: Use when deciding how much illegality to prevent by grammar-constrained decoding versus catching after generation with a verifier - the cost/coverage tradeoff, and why a post-hoc structural verifier remains mandatory even when decoding is fully constrained.
---

# Validity by Construction vs. Post-Hoc

## When to Use

- Designing a generator's decoding loop and deciding what to enforce at each step vs. after the full candidate exists
- Debugging "the decoder only offers legal tokens, so why did an illegal candidate still come out"
- Deciding whether a new grammar constraint belongs in the decode-time mask or the post-generation verifier
- Reviewing a claim that "we don't need a verifier, decoding is already constrained"

This sheet is about *where* legality is enforced. For *what* must be enforced regardless of where, see `structural-verification.md`. For the grammar being enforced, see `typed-graph-grammars.md`.

## Core Principle

**Constrained decoding narrows the search space at generation time; it does not replace the verifier — it changes how much work the verifier has left to do.** The confusion between these two is a specific, recurring mistake: each individual decoding step being locally legal does not imply the completed candidate is globally legal. Some properties are genuinely local (an operator drawn from the whitelist) and belong in the decode-time mask. Others are genuinely global (a running budget total, a shape contract that depends on the very last node chosen) and either require carrying extra state through decoding, or cannot be enforced until the candidate is complete — in which case a post-hoc check is not optional, it is the only place that property can be checked at all.

## The Tradeoff

| | Constrained decoding | Post-hoc verification |
|---|---|---|
| **When it runs** | During generation, per token/node | After the full candidate exists |
| **What it can prevent** | Any property checkable from the prefix generated so far | Any property, including ones that need the whole candidate |
| **Cost** | Paid on every step, even for candidates that would have been fine anyway | Paid once per completed candidate |
| **Coverage of local properties** | Prevents illegal choices before they're made — no wasted generation | Catches them, but only after the candidate is fully built |
| **Coverage of global properties** | Requires carrying running state (budget spent so far) through the whole decode | Naturally sees the whole candidate, checks any global property directly |

Neither column is free, and neither column is sufficient alone. A well-designed pipeline uses constrained decoding for what it's cheap and effective at (whitelist membership, per-step shape compatibility, running-budget tracking) and keeps the post-hoc verifier for everything else, including as a backstop for the properties decoding is supposed to guarantee — because decode-time constraints have bugs too.

## The RED Scenario: "Every Step Was Legal" Is Not "The Candidate Is Legal"

A decoder that only checks whitelist membership per token, with no budget tracking:

```python
OP_PARAMS = {"linear_16x16": 272, "linear_32x32": 1056, "relu": 0}
WHITELIST = list(OP_PARAMS)

def decode_LOCAL_ONLY(preferred_order, n_steps):
    """Every token offered is drawn from the whitelist -- locally legal at
    every step -- but nothing tracks the running parameter total."""
    return [preferred_order[i % len(preferred_order)] for i in range(n_steps)]
```

Every single token this emits passes the whitelist check. The completed sequence — five ops, three of them the larger linear layer — carries 3168 parameters against a declared ceiling of 3000. **No individual decoding decision was illegal, and the candidate is still illegal.** A pipeline that trusts "decoding is constrained, so I don't need to check the output" ships this candidate straight past a check that was never actually run.

### The GREEN Fix: Carry the Budget Through Decoding, Verify the Candidate Anyway

```python
def decode_BUDGET_AWARE(preferred_order, n_steps, max_params):
    """The running budget is part of the decode-time mask: an op is only
    offered if choosing it cannot push the total over the ceiling."""
    seq, spent = [], 0
    for i in range(n_steps):
        candidate = preferred_order[i % len(preferred_order)]
        if spent + OP_PARAMS[candidate] > max_params:
            legal_now = [op for op in WHITELIST if spent + OP_PARAMS[op] <= max_params]
            if not legal_now:
                break
            candidate = min(legal_now, key=lambda op: OP_PARAMS[op])
            if spent + OP_PARAMS[candidate] > max_params:
                break
        spent += OP_PARAMS[candidate]
        seq.append(candidate)
    return seq

MAX_PARAMS = 3000
PREFERRED = ["linear_32x32", "relu", "linear_32x32", "relu", "linear_32x32"]

local_seq = decode_LOCAL_ONLY(PREFERRED, n_steps=5)
local_total = sum(OP_PARAMS[op] for op in local_seq)
aware_seq = decode_BUDGET_AWARE(PREFERRED, n_steps=5, max_params=MAX_PARAMS)
aware_total = sum(OP_PARAMS[op] for op in aware_seq)

assert local_total > MAX_PARAMS       # locally-legal-only decode overshoots
assert aware_total <= MAX_PARAMS      # budget-aware decode respects the ceiling
print(f"local-only total={local_total} (over budget) | budget-aware total={aware_total} (within budget)")
```

Carrying the budget through decoding fixes *this* property. It does not fix every global property a grammar might declare — a shape contract requiring the final node's output to match the declared input shape, for instance, can depend on a choice made at the very last step in a way no earlier mask can fully anticipate without lookahead the decoder doesn't have. That is why `structural-verification.md`'s full checklist still runs on every completed candidate, even one produced by budget-aware constrained decoding. Constrained decoding reduces how often the verifier finds something wrong; it does not reduce the verifier's job description.

## Rationalization Resistance

| Rationalization | Reality |
|---|---|
| "Every token our decoder offers is grammar-legal, so the output must be legal" | Local legality at every step does not compose into global legality for properties that are sums, contracts, or lookahead-dependent |
| "We don't need a verifier, decoding already handles it" | Decoding at best matches the properties someone remembered to encode into the mask; the verifier is the place those omissions get caught before they cost anything downstream |
| "Constrained decoding is strictly better, so more constraints in the mask is always good" | Every constraint carried through decoding adds state and cost to every step, even for candidates that would never have violated it; put only what's cheap and genuinely local in the mask |
| "The verifier is redundant if decoding already enforces the same rule" | Decode-time masks have their own bugs (see the RED scenario); a verifier that re-checks a decode-time-enforced property is a regression test for the decoder, not redundant work |
| "This property is global, so it can only be checked post-hoc, no point trying to constrain it in decoding" | Some global properties (running totals, prefix-derivable contracts) can be carried through decoding cheaply — check before assuming it can't be done |

## Red Flags Checklist

- [ ] **A decoder claimed to "guarantee validity" with no post-hoc verifier anywhere downstream**
- [ ] **Budget or contract properties left unchecked** in a decode-time mask that only tracks whitelist membership
- [ ] **No test that decode-time constraints and post-hoc verification agree** on the same candidates
- [ ] **A global property enforced only post-hoc when it could have been carried through decoding cheaply**, wasting generation compute on candidates rejected late
- [ ] **The verifier skipped for "trusted" generation paths** (e.g., a hand-tuned deterministic generator) — trust is not a substitute for the check

## Cross-References

- **The full legality gate that runs regardless of how constrained decoding was**: `structural-verification.md`
- **The grammar and ceilings both decode-time and post-hoc checks enforce**: `typed-graph-grammars.md`
- **The representations a decoder emits, and what they make easy or hard to constrain**: `graph-representations-for-generation.md`
- **How this tradeoff interacts with generation strategy choice**: `generation-strategies.md`
