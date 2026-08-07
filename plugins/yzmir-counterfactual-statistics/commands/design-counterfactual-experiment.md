---
description: Turn a research question plus constraints into a committed pre-registered analysis plan - unit, pairing, splits, endpoints, horizon, N, tests, corrections, and abort/success criteria
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit", "Skill", "Task"]
argument-hint: "[research question, or path to a design doc / eval harness]"
---

# Design Counterfactual Experiment

Produce `preregistration.yaml` — a plan complete enough that someone else could run the experiment and get the same analysis. The deliverable is a **committed file**, not a chat summary; a plan that lives only in a conversation cannot be shown to predate the data.

## Core Principle

**Every decision this command makes is one that would otherwise be made after seeing results.** The order below is the dependency order: the unit determines `n`, the matching determines `sd_d`, `sd_d` and `δ` determine the fleet, and the fleet size determines whether the question is answerable at all. Running it backwards — picking a fleet size first and deriving a claim to fit — is the failure this command exists to prevent.

## Phase 0: Fact-Finding

Do not design from the prompt alone. Read what exists:

- The eval harness: how branches are forked, what they share, what the results schema looks like.
- Any prior trial results — they are the pilot data for `sd_d`, whether or not they were called a pilot.
- The cost model, budget, or SLO documents that would justify a `δ_min`.
- Existing `preregistration.yaml`, design docs, or ADRs.

State what you found and what you had to assume. **If no pilot data exists, say so** — the fleet size will be a guess with a stated range, and that is worth knowing before compute is spent.

## Phase 1: The Unit (blocking)

Load [statistical-units-and-clustering.md](../skills/using-counterfactual-statistics/statistical-units-and-clustering.md).

Apply the four questions: what is shared, what varies only inside, what a new draw costs, what the claim generalises over. Emit:

- `unit.definition` in one sentence
- `unit.id_column`
- `unit.repeated_measures` — the factors that are explicitly *not* units
- the rationale, phrased as "an additional independent observation requires …"

**Do not proceed until the unit is named.** Every later number is denominated in it.

## Phase 2: Pairing and the Matching Contract

Load [common-random-numbers-and-matching.md](../skills/using-counterfactual-statistics/common-random-numbers-and-matching.md).

Produce the matching contract as an explicit checklist with a yes/no per row for *this* harness: starting state, future input sequence, stochastic-regularisation draws, resource budget, evaluation protocol, numerics/hardware, cost accounting. Any "no" is either a design change or a stated caveat that widens the interval.

Then specify:

- the control: a no-op branch scoring exactly zero by construction, plus the assertion that proves it
- the RNG substream plan (hash-derived, keyed by unit for shared purposes)
- the stream-digest CI assertion

## Phase 3: Splits

Load [grouped-splits-and-leakage.md](../skills/using-counterfactual-statistics/grouped-splits-and-leakage.md).

Enumerate every stage that *makes a decision* and assign it a data role. Emit role weights, the stable-hash assignment method and salt, and the consumption assertions. If the fleet is too small for four roles, choose cross-fitting or an explicit merge — and **write down the claim downgrade the merge implies**.

## Phase 4: Endpoint and Utility

Load [effect-sizes-and-cost-charged-utility.md](../skills/using-counterfactual-statistics/effect-sizes-and-cost-charged-utility.md).

- **One** primary endpoint: metric + horizon + population + estimand.
- Cost weights `λ, μ, ν` with their derivation (budget shadow price, willingness-to-pay, or revealed).
- `δ_min` — the smallest utility worth acting on — derived from the cost model, **not** from any pilot estimate.
- Secondary endpoints, explicitly marked descriptive.

## Phase 5: Horizon

Load [horizon-choice-and-divergence-noise.md](../skills/using-counterfactual-statistics/horizon-choice-and-divergence-noise.md).

If pilot data spans multiple horizons, compute the `d_z(H)` curve and pick the plateau centre. If not, specify the pilot that will choose it, and note that the horizon is provisional until then. Run the divergence-growth check on any available data — a `sd_d` that grows much faster than `√H` means the matching is broken and Phase 2 is not finished.

## Phase 6: Fleet Size

Load [power-and-sample-size-for-paired-designs.md](../skills/using-counterfactual-statistics/power-and-sample-size-for-paired-designs.md).

Compute `n` at the **80% upper confidence limit** of `sd_d`, not its point estimate. Add the cost of any planned correction. Then confront the budget honestly:

- `n` affordable → proceed.
- `n` unaffordable → present the options in this order: **reduce `sd_d`** (fix the matching — usually the cheapest fix by far), narrow the claim to the MDE the budget supports, blinded internal pilot re-sizing, or do not run. Do not quietly shrink `n` and proceed.

Always report the MDE at the affordable fleet size, whatever is decided. It is the claim the budget actually buys.

## Phase 7: Tests, Corrections, Looks

Load [paired-comparison-methods.md](../skills/using-counterfactual-statistics/paired-comparison-methods.md) and [multiple-comparisons-and-sequential-testing.md](../skills/using-counterfactual-statistics/multiple-comparisons-and-sequential-testing.md).

- Aggregation rule: how each unit collapses to one number.
- Primary test plus the bootstrap cross-check.
- Family size; the correction, or the argument that one primary endpoint makes it unnecessary.
- Interim look schedule by *information fraction*, spending function, and a **futility boundary** (cheap, and it saves compute).
- Exclusion and failure-scoring rules, decided now: failures score a declared penalty and keep the pair; they are never silently dropped.

## Phase 8: Success and Abort Criteria

Both, explicitly:

- `ship_if` — usually "CI lower bound above `δ_min`", not "p < 0.05".
- `abandon_if` — the condition under which the answer is no. **A plan with no abandon condition is a search, not a test.**
- `inconclusive` — what gets reported otherwise (a null with its MDE).

## Phase 9: Emit and Commit

Write `preregistration.yaml` using the template in [preregistration-and-exploratory-vs-confirmatory.md](../skills/using-counterfactual-statistics/preregistration-and-exploratory-vs-confirmatory.md). Then:

```bash
git add preregistration.yaml && git commit -m "prereg: <study id>"
python -c "from freeze import freeze_preregistration; print(freeze_preregistration())"
```

Record the sha256, commit hash, and UTC timestamp in the study record. **Launch nothing until this is committed** — the commit is the evidence that the plan predates the data.

## Output Contract

1. **Fact-finding summary** — what was read, what was assumed, what is missing.
2. **`preregistration.yaml`** — written to disk, not pasted.
3. **The fleet-size conversation** — `δ_min`, `sd_d` and its provenance, `n` at the UCL, what the budget affords, and the MDE at that size.
4. **Open decisions** — anything that needs a human call (cost weights, `δ_min`, whether to run at all), stated as a question, not silently defaulted.
5. **Next actions** — commit the plan; wire the CI assertions; run the pilot if the horizon is still provisional.

## Anti-Patterns This Command Prevents

`AP-01` unit never named · `AP-09` unmatched branches · `AP-13` fleet sized from an inflated pilot · `AP-14` post-hoc plan · `AP-19` no control branch — see [anti-pattern-catalogue.md](../skills/using-counterfactual-statistics/anti-pattern-catalogue.md).

## When Not to Use This Command

- The work is exploratory and you know it — say so, ring-fence report-role units, and skip to the search. This command's output is what the *next* stage needs.
- The experiment is already running → `/analyze-paired-trial` or `/audit-experiment-statistics`.
- The question is a general A/B or observational-causal one → `yzmir-experimentation` *(planned)*.
