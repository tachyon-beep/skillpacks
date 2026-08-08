# Competency Questions First

**Write the questions the formalisation must answer before you name a single class. They are the requirements, and they are the acceptance tests.**

This is the discipline that separates a formalisation from a vocabulary exercise. Without it you get a plausible class hierarchy that nobody can query, nobody consumes, and nobody can tell is finished — because "finished" was never defined.

## What a competency question is

A **competency question** is a question the formalisation must be able to answer, stated in the language of the domain, specific enough to become an executable query.

| Not a competency question | Why | Competency question |
|---|---|---|
| "We need to model experiments." | A topic, not a question. | "Which experiments tested factor F at more than two levels?" |
| "The ontology should support provenance." | A capability claim with no test. | "What did result R descend from, transitively, back to the initial state?" |
| "Users should be able to find things." | No answer shape. | "List every retained run in which the no-intervention arm won, with its ancestor state id." |
| "Is the data FAIR?" | Unfalsifiable as posed. | "For dataset D, which unit does each measurement carry, and which measurements are absent with a stated reason?" |

**The test:** could two people disagree about whether a given answer is correct? If yes, it is not yet a competency question. Sharpen until the answer is a set of rows.

## Where they come from

Not from your imagination. From people who will actually ask them:

1. **The audit question.** What will someone ask when they doubt a result? *"Which policy version was in force when this was decided?"*
2. **The reproduction question.** What does a stranger need in order to re-run this? *"What were the inputs, the code version, and the seed?"*
3. **The comparison question.** What must be true for this claim to mean anything? *"Was there a no-intervention arm, and did it share an ancestor?"*
4. **The debugging question.** What will you ask at 3am? *"Which runs consumed a result flagged incomplete?"*
5. **The reporting question.** What goes in the paper, the dashboard, the quarterly review? *"What fraction of candidates were rejected at each stage?"*
6. **The regret question.** What did you wish you could ask last time and could not? This one is usually the best source.

**Interview for these; do not invent them.** Ten minutes with whoever gets paged, whoever writes the paper, and whoever will be asked to justify a decision produces a better set than a day of modelling.

## The rule that makes this bite

**Every class, property, and mint must be traceable to at least one competency question.**

A term that answers no question is not in scope. This single rule does more to control ontology sprawl than any amount of modelling taste — it converts "should we model this?" from an aesthetic argument into a lookup.

Keep the trace explicit. A two-column table (term → the questions it serves) costs nothing and is the first thing a reviewer should read. When someone later proposes a new class, they must name the question; when a question is retired, the terms it alone justified become candidates for deprecation.

## From question to test

A competency question becomes an acceptance test in three steps:

1. **Write the question** in domain language.
2. **Write the query** — SPARQL over your graph, or a plain function over your records if you are at Tier 1. Both count. The point is executability, not RDF.
3. **Write the expected answer shape** — not the exact rows, but what a correct answer looks like: how many columns, what each means, and critically **what a suspicious answer would be**.

Step 3 is where most of the value hides. "Returns zero rows" is a passing test for a query that is silently broken, and it is a passing test for a system that has genuinely never recorded a failure — and those two situations require opposite responses.

**Guard against vacuity.** For each competency question, also record:

- a **positive control** — data you know should match, so an empty result proves the query wrong rather than the world empty;
- a **negative control** where meaningful — data you know should *not* match, so a query that matches everything is caught.

A competency-question suite without controls is validation theatre. See [`validation-and-conformance.md`](validation-and-conformance.md).

## Sizing the set

**Ten to twenty questions is a healthy first set.** Fewer than five and you have not interviewed anyone. More than forty and you are describing a product rather than scoping a formalisation — split it.

Prioritise ruthlessly:

| Priority | Test |
|---|---|
| **Must** | If the formalisation cannot answer this, it has failed. Usually audit and comparison questions. |
| **Should** | Real value, but the formalisation is useful without it. |
| **Later** | Genuinely wanted, no consumer yet. Record it; do not model for it. |

**"Later" questions do not justify classes.** This is where premature generalisation enters — modelling for a question nobody is asking yet, then maintaining that model through every schema change for years. Write the question down and leave the term unminted.

## The gate

Run this before modelling begins, and again before anyone writes OWL:

- [ ] Between 5 and 40 questions, each answerable as a set of rows.
- [ ] Each sourced from a named person or a named consumer, not invented.
- [ ] Each prioritised Must / Should / Later.
- [ ] Every **Must** question has a drafted query and an expected answer shape.
- [ ] Every Must question has a positive control; those that can, have a negative control.
- [ ] **At least one Must question cannot be answered correctly today** — otherwise the formalisation is buying nothing and you should stop. This is the honest exit.
- [ ] Every proposed term traces to at least one question.

**That second-to-last item is the one to take seriously.** If the existing typed records already answer every Must question — whether correctly, or not at all — the correct output of this pack is a short note saying so. See [`formalisation-triage.md`](formalisation-triage.md).

## Worked example

A generic paired-branch intervention pipeline. Six questions, their queries' shape, and what they force into the model:

| # | Question | Answer shape | Forces into the model |
|---|---|---|---|
| 1 | Was candidate C judged against a no-intervention arm sharing its ancestor state? | one row: arm id, ancestor id, boolean match | ancestor-state identity; arm role |
| 2 | How many independent units support the claim that intervention I helps? | one row: n, plus the identity of the unit that n counts | declared unit of analysis |
| 3 | Which policy version was in force for decision D, and what evidence did it consume? | one row: policy version; N evidence ids | policy versioning; decision→evidence links |
| 4 | Which measurements in run R were absent, and why? | N rows: field, absence reason | four-state absence encoding |
| 5 | What fraction of candidates were rejected at each stage? | N rows: stage, count, fraction | retention of failures; stage vocabulary |
| 6 | Show every run where the no-intervention arm won. | N rows: run id, margin | outcome vocabulary that can express it |

Six questions; each forces one or two structural commitments, and nothing enters the model without one. Note that question 6 forces the model to be *capable of recording its own failure* — a property no amount of class-hierarchy design would have produced on its own.

**That is what competency questions are for.** They surface the structural commitments that matter, and they do it before anyone has become attached to a diagram.
