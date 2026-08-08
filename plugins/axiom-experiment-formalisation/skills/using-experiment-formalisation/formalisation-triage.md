# Formalisation Triage

**The most valuable output of this pack is sometimes a one-page note saying "don't."** Formalisation has a real, recurring cost — a second model to maintain, version, and keep synchronised with the contracts — and it pays that cost whether or not anyone ever queries it.

This sheet decides three things, in order: **whether** to formalise, **how much**, and **when to stop**.

---

## Gate 1 — Should you formalise at all?

### Formalise when at least one is true

- **The grammar will be reused.** The same experimental structure recurs across many runs, and people need to agree on what its parts mean.
- **Multiple parties must agree.** More than one team, service, or organisation reads these records, and misinterpretation is expensive.
- **Something external demands it.** A collaborator's schema, an archive's submission requirements, a regulator, a journal, a data-sharing mandate. **This is the strongest reason and the easiest to verify — name the consumer.**
- **Compliance must be machine-checked.** "Every result records the regime it was produced under" needs to be a query, not a convention.
- **The cost of a grammar error is high.** Results pooled across incomparable regimes; a claim published that the record cannot support.

### Do not formalise when

- **The experiment is a one-off.** Write it down in prose. Prose is a fine formalism for something that happens once.
- **The grammar is still moving.** If what counts as an arm, a result, or a decision changed twice this quarter, formalising now buys you a migration every time it changes again. This is **premature generalisation**, and it is the most expensive failure in this pack. Wait for stability.
- **There is no consumer.** Nobody has asked a question the current records cannot answer. Formalising "for FAIR" or "for interoperability" with no named counterparty produces triples nobody queries — **RDF cosplay** — at full maintenance cost.
- **The typed records already answer every Must competency question.** Then the answer is a short note saying so, and the honest thing is to write it.
- **The real problem is enforcement, not description.** "Our telemetry sometimes reports zero when it means unmeasured" is a contract problem. An ontology cannot fix it. Load `/contract-engineering` instead.

### The honest exit

The rule, whose gate lives in [`competency-questions-first.md`](competency-questions-first.md): **at least one Must question must be one you cannot answer correctly today.** If every Must question is already answerable from the existing records, formalisation is buying nothing.

Say so, name the questions, and stop. That is a successful use of this pack.

---

## Gate 2 — Which tier?

### Tier 1 is the default

**Tier 1: competency questions + a verified mapping + a JSON-LD context over records you already emit + the gap register + the sync check.**

You get: verified vocabulary, an explicit grammar, executable competency questions, interoperable output, and a durable record of what your terms mean.

You do not get: OWL reasoning, SHACL shapes, a triple store. **Most projects never needed them.**

Tier 1's defining property is that **producers do not change.** The context describes records that already exist. If the formalisation is abandoned in six months, nothing breaks — the cost was bounded and the artifacts remain readable.

### Tier 2 requires positive justification

**Tier 2: an OWL module, SHACL shapes, reasoner checks in CI, and usually a graph store.**

Permitted only when **all four** hold:

1. **The grammar is stable enough to freeze.** No structural change in the last three months, and none foreseen.
2. **A consumer will actually query it.** Named. A person, a service, or an external party — not "future researchers."
3. **Drift cost is assigned to an owner.** Someone maintains the sync check and fixes it when it fails. Unowned, Tier 2 rots into a second wrong source of truth.
4. **Tier 1 cannot answer the Must questions.** Concretely: you need entailment, or closed-world shape validation across a whole graph, or federation with an external ontology.

**Fewer than four: stay at Tier 1.** Tier 1 is not a stepping stone you are expected to outgrow. It is where most formalisations should permanently live.

### Tier comparison

| | Tier 1 | Tier 2 |
|---|---|---|
| Artifacts | CQs, mapping, gap register, JSON-LD context, sync check | + OWL module, SHACL shapes, reasoner CI |
| Producers change? | No | Usually no, but shapes constrain them in CI |
| Ongoing cost | Low — update the mapping when contracts change | Real — versioning, shapes, profile conformance, an owner |
| Failure if abandoned | Stale doc | A second source of truth that disagrees with the first |
| Buys | Shared meaning, executable questions, interoperable output | Entailment, whole-graph validation, federation |

---

## The ROI heuristic

When the gates leave you undecided, ask what the formalisation *prevents*, and price it.

```
Formalising pays when:

  (probability of the grammar error) × (cost when it happens) × (number of chances)
      >
  (build cost) + (maintenance cost per change × expected changes)
```

You will not have precise numbers. You will usually have enough to see which side is larger by an order of magnitude — and if it is close, the answer is Tier 1, because Tier 1's right-hand side is small.

**Errors worth pricing** (each is a real, recurring incident class):

- Results pooled across incomparable regimes or scaffold states.
- A claim published that the record cannot actually support.
- An artifact integrated that is not the one that was verified.
- Absence read as zero by a downstream consumer.
- A decision that cannot be re-derived because the policy version was not recorded.

Notice that **the contract layer prevents several of these more cheaply than the ontology does.** If your list is dominated by those, the correct output of triage is "fix the contracts first, formalise later" — and that ordering is itself a finding worth reporting.

---

## Gate 3 — When to stop

Formalisation has no natural endpoint, so declare one in advance.

**You are done when every Must competency question has a passing query with a positive control, the sync check is green in CI, and the gap register has no open rows.** Write that down before starting; otherwise "done" becomes "we stopped."

Stop-work signals — each means go back a gate:

| Signal | Read it as |
|---|---|
| The gap register grows every sprint | The grammar was not stable. Premature generalisation. Pause and revisit Gate 1. |
| Nobody has run a competency query in a month | No consumer. Drop to Tier 1 or stop. |
| The sync check has been failing and muted | Formalisation debt. Fix or delete — a muted check is worse than none. |
| You are modelling for a "Later" question | Out of scope by construction. |
| Terms are being minted faster than they are being reused | Sources are not being searched. Back to the mint ladder. |
| Someone proposes putting a shape on a request path | Layering inversion. See [`the-projection-law.md`](the-projection-law.md). |

---

## Triage output

Whatever the verdict, produce a short record — it is the artifact that stops this being re-litigated every quarter:

- **Verdict**: don't formalise / Tier 1 / Tier 2.
- **Why**, against the gates above.
- **The named consumer**, or an explicit statement that there is none.
- **The Must questions**, and which are unanswerable today.
- **If Tier 2**: which of the four conditions hold, and who owns the drift.
- **The stop condition.**
- **The revisit trigger** — what would change the verdict.

A "don't formalise" record with a revisit trigger is a genuinely valuable artifact. It stops the same discussion recurring, and it names the condition under which the answer changes.
