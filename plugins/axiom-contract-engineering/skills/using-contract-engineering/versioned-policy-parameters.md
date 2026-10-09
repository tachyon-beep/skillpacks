---
name: versioned-policy-parameters
description: "Use when a threshold, weight, operating point, tie-break preference, hysteresis band, staleness budget, or escalation criterion lives as a code constant or config literal \u2014 or when retuning one, when a decision record carries a schema version but nothing says which rules produced it, when an incident tempts a formula edit to cope with a missing input, when admission and retention gates on the same quantity flap, or when two decision paths each hold their own copy of \"the\" weight. Covers policy records, decision-side version binding, veto-versus-weight parameters, deliberate hysteresis, and shared-term identity."
---

# Versioned Policy Parameters

**Every parameter that shapes a decision — thresholds, weights, operating points, hysteresis bands, imputation rules — lives in a versioned policy record, and every decision binds the exact policy version in force. A threshold in a code constant is an unversioned policy that silently re-decides history on every deploy.**

## When this earns its cost

Read this sheet when:

- A decision boundary is a module constant: `CPU_HIGH = 0.80`, `MIN_QUALITY = 0.62`, `ESCALATE_ABOVE = 3`.
- Someone is retuning a threshold and the diff touches only a number.
- A decision record carries `schema_version` and nothing that says *which rules* produced it — the most common near-miss in this whole pack.
- An incident is in progress and the proposed fix changes the decision formula (renormalise the surviving weights, substitute an estimate for the missing input) rather than the data.
- The same quantity gates both admission and retention, and things are flapping at the boundary.
- Two code paths price the same cost term and you cannot say whether their coefficients are equal on purpose.

## The two baseline failures

Both are produced by careful engineers who are already doing everything else right.

**Failure 1 — the schema-versioned, policy-unversioned decision.** The record is disciplined by every measure this pack usually asks for, and is still uninterpretable a month later:

```python
# policy.py — "config", nobody calls it policy
CPU_HIGH = 0.80          # scale up above this
CPU_LOW = 0.30           # scale down below this
LATENCY_WEIGHT = 0.6

@dataclass(frozen=True)
class ScaleDecision:
    schema_version: int  # 3        <-- versioned wire format...
    host_id: str
    cpu_util: float      # 0.85
    action: str          # "scale_up"
    # ...and nothing here says 0.80 was the line. The record looks versioned.
```

Six weeks later `CPU_HIGH` is 0.85 and someone asks whether the 0.85 reading in that record was a breach. Nobody can answer it from the record. The wire format was versioned; the *rule* was not. Git archaeology across a deploy timeline is not an audit trail — and it fails outright the moment a rollback, a canary, or a per-tenant override means two thresholds were live at once.

**Failure 2 — the incident-pressure formula edit.** A weighted score loses one of its inputs, so the on-call renormalises over the survivors:

```python
# WRONG — shipped as a hotfix, no version marker anywhere:
present = {k: w for k, w in WEIGHTS.items() if k in signals}
total = sum(present.values())
score = sum(signals[k] * w / total for k, w in present.items())   # renormalised
```

This is not a bug fix. It is a new decision rule: under the old policy a missing input dragged the score down, under the new one it is imputed at the mean of its peers. Records emitted before and after are labelled identically, compare as if commensurable, and disagree systematically. The renormalisation may well be the right call — but it is a **policy change under incident pressure**, which is exactly the moment the version bump is skipped and exactly the moment it matters most.

```python
# RIGHT — the substitution is policy, so it gets a version and lands on the output:
policy = registry.get("autoscale", version=7)   # v7 declares renormalise_on_absent=True
score, subs = score_with(signals, policy)
decision = ScaleDecision(..., policy_version=policy.policy_version, substitutions=subs)
```

## 1. Policy as a record

A policy is a frozen, identified, versioned record — the same class of object as any other contract record in this pack, with the same lifecycle (`definition-lifecycle.md`: draft → approved → locked; nothing decides under a draft).

```python
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

class ParamKind(str, Enum):
    VETO = "veto"        # adjudicated first; never traded against benefit
    WEIGHT = "weight"    # participates in the tradable comparison
    BAND = "band"        # hysteresis / separation between two operating points

class Bound(str, Enum):
    """Which direction a VETO binds. Without this, the adjudicator hardcodes
    one comparison and a floor veto (minimum safety coverage, minimum eval
    score) can only be expressed by branching on the parameter's *name* —
    the exact defect §5 forbids."""
    CEILING = "ceiling"  # trips when the metric exceeds the value
    FLOOR = "floor"      # trips when the metric falls below the value

class Authority(str, Enum):
    """Closed set — who may retune this parameter. Free text here would be a
    covert channel into anything that reads it (blinding-by-construction.md)."""
    SAFETY_BOARD = "safety_board"
    COMPLIANCE = "compliance"
    SERVICE_OWNER = "service_owner"

@dataclass(frozen=True)
class Param:
    value: float
    kind: ParamKind
    authority: Authority  # who owns this operating point
    basis: str            # what the number was derived from (e.g. "p99 noise, 30d")
    bound: Bound | None = None  # direction, for VETO params only

    def __post_init__(self) -> None:
        # Illegal states unrepresentable: a veto without a direction cannot be
        # adjudicated; a direction on a non-veto is meaningless.
        if (self.kind is ParamKind.VETO) != (self.bound is not None):
            raise ValueError(f"bound is mandatory for VETO and forbidden otherwise")

@dataclass(frozen=True)
class PolicyRecord:
    policy_id: str               # "autoscale.cpu"      — stable identity
    policy_version: int          # 7                    — bumped on ANY value change
    lifecycle: str               # "locked"             — see definition-lifecycle.md
    params: Mapping[str, Param]  # the parameters themselves

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))
```

The deciders take it as an **input**, never by import:

```python
# BEFORE — the decider imports its own rules; the rules are invisible to callers,
# untestable except by monkeypatching, and unversioned.
from .policy import CPU_HIGH, CPU_LOW

def decide(m: MetricsRecord) -> ScaleDecision:
    if m.cpu_util > CPU_HIGH:
        return ScaleDecision(action="scale_up", ...)
    if m.cpu_util < CPU_LOW:
        return ScaleDecision(action="scale_down", ...)
    return ScaleDecision(action="hold", ...)

# AFTER — policy arrives as an argument; the decider is mechanism only.
def decide(m: MetricsRecord, policy: PolicyRecord) -> ScaleDecision:
    high = policy.params["cpu_high"].value
    low = policy.params["cpu_low"].value
    action = "scale_up" if m.cpu_util > high else "scale_down" if m.cpu_util < low else "hold"
    return ScaleDecision(
        schema_version=3,
        host_id=m.host_id,
        cpu_util=m.cpu_util,
        action=action,
        policy_version=policy.policy_version,   # the rule in force, on the record
        resolver_version=DECIDER_VERSION,       # the code that applied it
    )
```

That is the concrete form of `deterministic-resolution.md`'s rule that a resolver holds mechanism and takes policy as input. The import was not a shortcut — it was an undeclared input, and an undeclared input is unreplayable.

## 2. Decisions bind the version in force

`schema-versioning-and-evolution.md` ends by requiring that every derived record carry the versions of what it consumed. Policy is one of those things:

- **One `policy_version` per policy consumed.** A decision reading an autoscale policy and a tenant-quota policy carries both, keyed by `policy_id` — `policies={"autoscale.cpu": 7, "quota.tenant": 2}`. A single scalar field cannot represent two policies and will be wrong the first time a second one appears.
- **`policy_version` and `resolver_version` are different axes.** `resolver_version` answers *what code combined the inputs*; `policy_version` answers *what numbers it combined them with*. Retuning a threshold bumps the policy only; changing the comparison structure bumps the resolver only. Collapsing them into one field makes every retune look like a code change and vice versa.

What binding buys, and what nothing else buys:

- **Historical interpretability.** "Was 0.85 a breach?" is answered from the record alone: look up policy v7, read `cpu_high`. Archived decisions are read under *their own* recorded policy version — the same rule `schema-versioning-and-evolution.md` applies to archived payloads.
- **Rollback and A/B become record swaps.** Two policy versions can be live simultaneously across cohorts because each decision says which one it used. With constants, "which threshold was live" is a property of the deploy, so an A/B needs a fork and a rollback needs a deploy.
- **Audits answer from records.** "What rule was in force at 14:02?" has an answer that does not involve reconstructing a deploy timeline.

Policy versions inherit the reader discipline of schema versions, and for the same reasons: **a `policy_version` is never defaulted at read time** (a record without one raises `ContractViolation` at the boundary, it is not "presumably the current policy"), and **an unrecognised policy version fails closed** — the same typed refusal, rather than an interpretation under today's numbers. Most important, **version-in-name-only applies to policy**: editing `cpu_high` from 0.80 to 0.85 without a bump makes v7 name two different rule sets, and — exactly as with schemas — there is no repair. Every record stamped v7 is permanently ambiguous.

## 3. What counts as policy

The test is blunt: **does it change which output wins?** If yes, it is policy and it is versioned. This is wider than the list of things that look like thresholds:

| Parameter | Why it is policy |
|---|---|
| Thresholds, operating points | Directly move the decision boundary |
| Weights, scoring coefficients | Change the ranking of candidates |
| Tie-break preference | Chooses the winner in the exact cases nobody notices |
| Imputation / renormalisation rules for missing inputs | Change what the score *means* when an input is absent |
| Staleness budgets | Decide whether an old value still counts as evidence |
| Escalation criteria, review-queue routing | Determine what a human ever sees |
| Hysteresis bands | Determine whether the system settles or churns (§4) |

The trap is the fourth row, and it arrives under pressure. `silent-default-elimination.md` establishes that an absent input must not become a plausible value; this sheet supplies the other half. Substituting or renormalising for a missing input is a **semantic change to the output**, so:

1. It requires a **policy version bump**, not a formula edit — the same edit shipped as a hotfix leaves two incompatible rule sets sharing one label.
2. The substitution must be **recorded on the output**: which inputs were absent, which rule applied, what value was substituted. A score that was 40% imputed and a score that was fully measured must not be indistinguishable downstream.

```python
class SubstitutionRule(str, Enum):
    """Closed set, extended only by a policy version that declares the new rule."""
    RENORMALISE_SURVIVORS = "renormalise_survivors"
    PEER_MEAN = "peer_mean"
    CARRY_FORWARD_BOUNDED = "carry_forward_bounded"

@dataclass(frozen=True)
class Substitution:
    field: str
    rule: SubstitutionRule
    substituted: float

# decision.substitutions == () means fully measured. It is never absent from the record.
```

If the incident genuinely cannot wait for a policy review, the correct emergency move is to cut a new policy version with the emergency rule and mark its lifecycle accordingly — a fast bump, not an absent one. The bump is cheap; the ambiguity is permanent.

## 4. Deliberate asymmetry: hysteresis is a parameter, not an accident

When one quantity gates both **admission** (let it in) and **retention** (keep it), a single threshold guarantees churn. Any candidate sitting near the line crosses it on noise alone, so the system admits, evicts, and readmits the same item indefinitely — burning the full admission cost each cycle and producing a decision log that looks busy and decides nothing.

The fix is a deliberate asymmetry: the admit threshold sits **above** the retain threshold by an explicit band, and the band is sized against **measured** noise in the quantity, not guessed.

```python
@dataclass(frozen=True)
class GateOutcome:
    admitted: bool
    policy_version: int

def gate(score: float, currently_held: bool, policy: PolicyRecord) -> GateOutcome:
    retain = policy.params["retain_at"].value       # 0.55
    band = policy.params["hysteresis_band"].value   # 0.07, basis: "2x p95 tick-to-tick noise"
    admit = retain + band                           # 0.62 — derived, never a second constant
    threshold = retain if currently_held else admit
    return GateOutcome(admitted=score >= threshold, policy_version=policy.policy_version)
```

Three properties make this discipline rather than a magic number:

- **The band is a first-class versioned parameter** with a recorded `basis` (the noise measurement it was sized against). Retuning it is a policy version bump like any other, and the basis is what a future reviewer needs to judge whether the size is still right.
- **Only one of the pair is stored.** Storing `admit_at = 0.62` and `retain_at = 0.55` as independent parameters lets them drift into an inverted or zero band — derive the admit point from retain + band so the asymmetry cannot be edited away by touching one number.
- **The asymmetry is declared, not emergent.** The defect this replaces is two teams independently picking constants for "their" gate, and the gap between them being whatever the arithmetic happened to produce — sometimes zero, occasionally negative (retain stricter than admit, which evicts everything it just admitted).

The same shape covers any admit/keep pair: an ML evaluation harness admitting candidate models into a serving pool and deciding which to retain, a fraud-review queue escalating a case and deciding when to release it, an autoscaler adding and removing capacity.

## 5. Non-negotiable gates before tradable utilities

Some parameters are **veto operating points**: a tail-risk limit, a safety bound, a compliance ceiling. They are not terms in a weighted sum, however large the coefficient — a big weight can always be outvoted by a big enough measured benefit, and "the model is so much better that it justifies exceeding the bound" is precisely the sentence the veto exists to make unsayable.

So vetoes are adjudicated **lexicographically**: every veto is evaluated first, and only candidates that survive all of them enter the cost/benefit comparison. Crucially, the decider does not *know* which parameter is the safety bound — the record says, via `ParamKind`:

```python
def adjudicate(candidate: Candidate, policy: PolicyRecord) -> Verdict:
    # Stage 1 — vetoes, lexicographic, no trading. Purely mechanical: the decider
    # branches on kind, never on a parameter's identity.
    for name, p in sorted(policy.params.items()):
        if p.kind is not ParamKind.VETO:
            continue
        measured = candidate.metrics[name]
        tripped = measured > p.value if p.bound is Bound.CEILING else measured < p.value
        if tripped:                       # direction comes from the record, not the code
            return Verdict(admitted=False, vetoed_by=name,
                           veto_authority=p.authority,
                           policy_version=policy.policy_version)

    # Stage 2 — only survivors are scored. Weights may trade freely against each
    # other. "retain_at" is the §4 operating point, kind=BAND-derived — it is a
    # threshold the score is compared against, so it is excluded from the WEIGHT
    # sum by its kind, not by its name.
    score = sum(candidate.metrics[n] * p.value
                for n, p in policy.params.items() if p.kind is ParamKind.WEIGHT)
    return Verdict(admitted=score >= policy.params["retain_at"].value,
                   score=score, policy_version=policy.policy_version)
```

If the decider ever reads `if name == "tail_risk_limit"`, the veto has moved back into code — an unversioned preference, which `deterministic-resolution.md` names as the defect. Keep it structural: the record classifies, the code obeys.

The **owning authority** of each veto is recorded on the parameter, because a veto that anyone may retune is not a veto. Record it as provenance for audit and for the review gate on version bumps — drawn from a closed vocabulary, and *not* as something the decider reads to modulate its behaviour, which would be a covert channel (`blinding-by-construction.md`). Downgrading a parameter from `VETO` to `WEIGHT` is the highest-scrutiny change in a policy diff: it converts a bound into a price.

## 6. Shared terms share weights

When two decision paths price the same cost term — compute spend appearing in both an admission gate and a retention gate, latency budget in both routing and shedding — they reference **the same versioned parameter**, not two copies of the same number.

```python
# WRONG — one concept, two homes. They agree today.
admission_policy.params["compute_cost_weight"] = Param(0.30, ...)
retention_policy.params["compute_cost_weight"] = Param(0.30, ...)

# RIGHT — one parameter, referenced twice.
COST = PolicyRecord(policy_id="cost.compute", policy_version=4, ...)
# both gates consume COST and both stamp policies={"cost.compute": 4, ...}
```

Two silently different copies of one weight is the **dual-source-of-truth defect applied to policy**, with the usual property: the copies agree at commit time and diverge the first time one is tuned, and the divergence has no error signal — both paths keep deciding, just incoherently, and the system prices the same resource at two rates depending on which door you came through.

Divergence is permitted, but only as a **declared** structural difference: if retention weights compute cost differently because switching cost is already sunk, that is a real argument, and it lives as a named, separate parameter with its `basis` stating the structural reason. The rule is not "never differ" — it is that *sameness is the default and difference is documented*, so that a reviewer can tell an intentional asymmetry from an unnoticed drift.

## Rationalizations

| Rationalization | Reality |
|---|---|
| "The threshold is in config, that's basically versioned" | Config with no version stamped on the *decision* is a global mutable read. The record must say which value was in force, or nobody can reconstruct the decision — including you, next quarter. |
| "Git history tells us what the threshold was" | Only if exactly one value was live at a time, deploy timestamps map cleanly to decision timestamps, and there were no canaries, rollbacks, or per-tenant overrides. That has never been true of a real fleet. |
| "It's just a tuning change, not a semantic change" | Tuning *is* the semantic change — it moves which output wins. If retuning didn't change outcomes there'd be no reason to do it. Bump. |
| "The renormalisation just keeps it working with the missing input" | It changes what the score means: absent-drags-down became absent-imputed-at-peer-mean. Records before and after are labelled the same and are not comparable. Policy bump plus the substitution recorded on the output. |
| "One threshold is simpler than two; hysteresis is over-engineering" | One threshold is admit→evict→readmit flapping at the boundary, which costs more than the band ever saves. Size the band against measured noise and store it as a parameter. |
| "The safety bound has a very high weight, that's equivalent" | A weight can be outvoted; that is what a weight is. Lexicographic veto before any comparison, marked as `VETO` on the record, with a recorded owning authority. |
| "Both teams use 0.30, it's the same weight" | It is two weights that currently agree. Reference one parameter, or write down the structural reason they differ. |
| "We'll re-run the old decisions under the new policy to compare" | Fine as a *new* labelled analysis; fatal if it overwrites history. Recomputed outputs stamped with the new policy version are records of a simulation, never a correction of what was decided. |

## Red flags

- A decision boundary as a module constant, class attribute, or literal in a comparison — anything the decider imports rather than receives.
- A decision record with `schema_version` and no `policy_version` (or a single `policy_version` where two policies were consumed).
- A diff that changes a threshold, weight, or imputation rule without touching a policy version constant.
- A formula edit landing during an incident: renormalisation, substitution, "temporarily ignore this term".
- Admission and retention thresholds stored as two independent parameters — or set equal, or with retain stricter than admit.
- A hysteresis band with no recorded basis, or one whose value nobody can trace to a noise measurement.
- A backfill or re-run that writes recomputed decisions over stored ones.
- The same weight name defined in two policy records; a decider branching on a parameter's *name* to decide whether it is a hard bound.

## Quick reference

| Situation | Discipline |
|---|---|
| Threshold / weight / operating point | `Param` inside a versioned `PolicyRecord`; decider takes the record as an input |
| Emitting a decision | Stamp `policy_version` per `policy_id` consumed, plus `resolver_version` for the code |
| Retuning a number | Policy version bump; old records stay interpretable under their recorded version |
| Reading a stored decision | Resolve its recorded policy version; unknown version fails closed; never defaulted |
| Missing input, need a substitute | Policy version bump declaring the rule + substitution recorded on the output |
| Admission and retention on one quantity | Store `retain_at` + versioned `hysteresis_band` with measured basis; derive `admit_at` |
| Safety / tail-risk / compliance bound | `ParamKind.VETO`, adjudicated lexicographically before scoring, owning authority recorded |
| One cost term, two decision paths | One shared parameter; divergence only with a documented structural reason |
| Comparing old decisions under a new policy | New labelled analysis records; never overwrite the originals |

## Cross-references

- `deterministic-resolution.md` — resolvers take policy as an input and hold mechanism only; `resolver_version` versus `policy_version`.
- `schema-versioning-and-evolution.md` — the version-gate discipline this sheet applies to policy; version-in-name-only; reading archived records under their own recorded version.
- `silent-default-elimination.md` — why imputation and renormalisation for an absent input are policy changes rather than parsing fixes.
- `definition-lifecycle.md` — draft → approved → locked for a policy version; nothing decides under a draft policy.
- `contract-testing.md` — fixtures pinned to a policy version; hysteresis anti-flap property tests; veto-precedence tests.
- `blinding-by-construction.md` — the veto's owning authority is recorded provenance, never an input the decider reads.
