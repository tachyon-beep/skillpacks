---
name: deterministic-resolution
description: "Use when one record is derived from others across a boundary \u2014 a request resolved from an intent plus constraints, a decision computed from evidence, a config materialised from layered policies \u2014 and the derivation must be reproducible, auditable, or replayable. Covers pure resolvers over recorded inputs, canonical outputs, covert-channel closure (serialisation, aliases, ordering, batch position), narrow-only authority, and eliminating hidden resolver state."
---

# Deterministic Resolution

**A resolver is a pure function from recorded inputs to one canonical output. If replaying the recorded inputs cannot reproduce the output bit-for-bit, the resolution didn't record its inputs — it consumed something it never admitted to.**

## When this earns its cost

Read this sheet when:

- A component derives one contract record from others: intent + policy + constraints → request; evidence + thresholds → decision; layered configs → effective config.
- An audit, replay, or debugging session must reconstruct *why* an output happened from stored records alone.
- Two "equivalent" inputs produce different outputs and nobody can say why.
- A resolver has begun to accumulate preferences, heuristics, or "smart" behavior.
- Downstream behavior seems to depend on things that shouldn't matter — field order, batch position, alias choice, serialization quirks.

## The contract of a resolver

```text
resolve(recorded_input_1, recorded_input_2, ..., resolver_version) -> canonical_output
```

Four properties, each load-bearing:

1. **Pure.** No clock, no RNG, no environment, no network, no process-local caches or memory of previous calls. Everything the resolver reads arrives as an argument, and every argument is a record that was (or will be) durably stored *before* the output is acted on.
2. **Total over declared inputs, closed over everything else.** Inputs it doesn't declare, it cannot see — there is no "also peeks at the live config" path.
3. **Canonical.** Equivalent inputs produce *one* output, byte-identical after canonical serialization. Not "semantically the same" — identical.
4. **Versioned.** The resolver's own version is an input and is stamped on the output. Change the resolution logic → bump the version → old outputs remain interpretable under the version that produced them (`versioned-policy-parameters.md`).

The payoff compounds: reproducibility gives you audit ("show me why this request was granted" = re-run the function), regression testing (golden inputs → golden outputs), and drift detection (re-resolve stored inputs; any diff means the resolver changed without a version bump).

## Hidden state: the respectable defect

Nobody writes `random()` in a resolver. What they write is *memory*:

```python
# Looks disciplined — records what it used, handles absence explicitly.
# Is not a resolver — it's a stateful process.
class Decider:
    _last_seen: dict[str, tuple[int, float]] = {}   # carry-forward cache

    def resolve(self, metrics: MetricsRecord) -> Decision:
        if isinstance(metrics.queue_depth, Absent):
            prior = self._last_seen.get(metrics.host_id)   # <-- hidden input
            ...
```

The carry-forward *policy* may be entirely legitimate (bounded staleness, recorded age — see `silent-default-elimination.md`). The defect is *where the prior value lives*: in process memory, consumed by the resolution but recorded nowhere. Replay the stored `MetricsRecord` through a fresh process and you get a different decision. The decision's stored `inputs` block says what was used — but you cannot *reconstruct* it, only trust it.

The fix is mechanical: promote the hidden state to a record.

```python
def resolve(metrics: MetricsRecord,
            carry: CarryForwardState,      # explicit, versioned, stored record
            policy: PolicyRecord) -> tuple[Decision, CarryForwardState]:
    ...  # returns the decision AND the successor state, both stored
```

Now resolution is a fold over stored records: `state_n+1, decision_n = resolve(input_n, state_n, policy)`. Anything the resolver needs across calls — caches, cursors, exponential averages, rate-limit windows — either becomes an explicit input/output record or the resolver may not use it. "Where does this value live when the process dies?" is the audit question; if the answer is "nowhere," the resolver is consuming an input it never admitted to.

## Canonical output: closing the covert channels

"Equivalent inputs → one output" fails in practice through channels nobody designed:

| Channel | Failure shape | Closure |
|---|---|---|
| **Serialization variance** | Same input as JSON-with-spaces vs. without, map ordering, float formatting → different bytes → different downstream hash or cache key | Canonical serialization at the boundary: sorted keys, fixed float encoding, no insignificant whitespace; resolve from the *parsed* form, hash the *canonical* form (`canonical-identity.md`) |
| **Aliases** | Two names for one thing (`"us-east"` vs `"use1"`, dual keys during migration) resolve differently or hash differently | Alias→canonical mapping applied at parse time, before resolution; the resolver never sees an alias |
| **Field ordering** | Resolver iterates a mapping; first-match logic makes output depend on insertion order | Canonicalise ordering before iteration; property-test with permuted inputs |
| **Irrelevant variation** | Batch position ("candidate 3 of 12"), request arrival order, sibling contents leak into per-item output | Per-item resolution takes only that item's declared inputs; batch metadata is execution accounting, not resolver input |
| **Free-text fields** | A "notes"/"context" string smuggles preferences the schema forbids as typed fields | No free text into resolvers — closed enums and typed fields only (`blinding-by-construction.md`) |

The covert-channel test is behavioral, not structural: **construct two inputs the contract defines as equivalent, and assert byte-identical outputs.** Every row above is a property test (`contract-testing.md`): permute field order, swap aliases, reposition within a batch, vary serialization — output unchanged.

Why this is worth property-test rigor: any channel through which irrelevant variation reaches the output is a channel through which an upstream author can *steer* the output while staying schema-legal. Covert channels aren't just nondeterminism — they are unauthorized authority.

## Narrow-only authority

A resolver typically combines a *request for authority* (an intent) with *grants of authority* (envelopes, policies, region constraints). The rule:

**The output's authority is at most the intersection of its inputs' authorities. A resolver may narrow; it may never widen.**

- Budgets: `resolved_budget = min(requested, granted, capacity)` — never a default that exceeds a grant, never "requested was absent so use the generous default."
- Permissions: the resolved set is a subset of every input's permitted set; an option absent from any input's allow-list cannot appear in the output.
- Escalation is not resolution: if the intent asks for more than the grants allow, the resolver **fails closed or narrows visibly** (recording that it narrowed, and from what) — it does not quietly upgrade the grant, and there is no code path by which any input can make the output *more* permissive than another input allows.

Widening hides in defaults: an absent optional field that resolves to a permissive value is the silent-default defect operating on authority, the one place it's most expensive. Absent authority resolves to *no* authority.

The property test: for every input record, mutate it toward more-restrictive; the output must be equal or more restrictive on every authority axis. Authority tests (`contract-testing.md`) additionally prove forbidden fields cannot arrive at all.

## Resolvers do not grow preferences

A resolver applies *recorded policy* to *recorded inputs*. The failure mode is accretion: a tie-break "for stability," a heuristic "just for the edge case," a vendor hint honored from a passthrough field — each one a preference living in code, unversioned, invisible to audit.

The test for any proposed resolver change: **is this mechanism, or is this policy?** Mechanism (parsing, validation, canonicalisation, intersection, deterministic tie-breaking by canonical identity) belongs in the resolver. Policy (which candidate wins, what threshold applies, what's preferred) belongs in a versioned policy record the resolver takes *as input* — so that changing the preference is a recorded policy-version event, not a code edit that silently re-decides history. Even tie-breaking must be by a canonical, content-derived ordering — never by arrival order, iteration order, or "first match wins" (all covert channels).

## Rationalizations

| Rationalization | Reality |
|---|---|
| "The cache is just an optimization" | A cache the resolution *reads through* is an input. Either its contents are reconstructible from recorded inputs (pure memoization — fine) or it's hidden state (defect). |
| "We record what the resolver used, that's auditable" | Recorded-but-not-reconstructible is testimony, not evidence. The bar is replay: stored inputs → identical output. |
| "The heuristic is tiny and obviously right" | Then it's tiny, obviously right *policy* — put it in the policy record with a version. In code it's an unversioned preference that re-decides history on every deploy. |
| "Output differs only in formatting" | Formatting differences become hash differences become cache misses and false non-equivalence. Canonical means byte-identical. |
| "The batch index is harmless metadata" | If it reaches the resolver, someone can steer output by reordering a batch while every record stays schema-legal. Execution metadata stays outside. |
| "Failing closed on an over-broad intent breaks callers" | The caller asked for authority nobody granted. Narrow visibly or refuse loudly — the quiet upgrade is a privilege escalation with logs that say everything is fine. |

## Red flags

- A resolver that is a class with mutable fields surviving between calls; module-level dicts; `functools.lru_cache` over anything time- or state-dependent.
- `datetime.now()`, `random`, environment reads, or config-file reads inside resolution.
- Output hashes that differ for inputs the contract calls equivalent; tie-breaks by iteration or arrival order.
- `min()` missing where a budget crosses the boundary; any default that is more permissive than an explicit value could be.
- A resolver diff that changes *which output wins* without a resolver- or policy-version bump.
- Free-text or passthrough fields flowing into resolution.

## Quick reference

| Property | Enforcement |
|---|---|
| Purity | All inputs are stored records passed as arguments; no clock/RNG/env/global state |
| Reconstructibility | Replay test in CI: stored inputs → byte-identical output |
| Cross-call state | Explicit state record in and out (a fold), stored with the outputs |
| Canonicality | Canonical serialization; alias resolution at parse; equivalence property tests (permute, alias-swap, reposition) |
| Narrow-only | Intersection semantics on every authority axis; absent authority = no authority; monotonicity property test |
| Preference hygiene | Policy in versioned records as input; resolver code is mechanism only; resolver_version stamped on output |

## Cross-references

- `canonical-identity.md` — canonical serialization and content-addressed hashing of resolver inputs and outputs.
- `versioned-policy-parameters.md` — policy records as resolver inputs; binding the version in force.
- `silent-default-elimination.md` — absence in resolver inputs; why absent never resolves to a permissive value.
- `contract-testing.md` — equivalence-invariance property tests, authority tests, replay tests.
- `blinding-by-construction.md` — keeping forbidden influence out of the resolver's input schema entirely.
- Cross-pack: `axiom-determinism-and-replay` — whole-system record/replay discipline; this sheet is the contract-layer special case.
