# Adapting This Pack for Your Own Needs

**This pack encodes one opinionated route through a large landscape. Your domain will differ, and the pack is meant to be edited.** This sheet says which parts are load-bearing, which are defaults you should expect to change, and how to change them without breaking the discipline.

## What not to change

Three things carry the pack's value. Alter them and you have a different, weaker pack.

| Keep | Why |
|---|---|
| **Law 1 — verified vocabulary only** | Domain-independent. Every ontology community has plausible-sounding non-existent terms, and every model generates them fluently. |
| **Law 2 — the projection law, with its sync check** | Applies wherever a description layer sits over an enforcement layer. The check is the part that makes it real. |
| **Competency questions before terms** | The only thing that makes "finished" and "adequate" definable. |

Everything below is a default.

## Swapping the upper ontology

The pack teaches SUMO because **EXPO is built on SUMO**. If you replace it:

| Replacing with | Changes | Stays the same |
|---|---|---|
| **BFO** | `sumo-upper-binding.md` is rewritten around continuant/occurrent; the category-error table maps to BFO's distinctions (`Process` → occurrent, `Plan` → generically dependent continuant, `Attribute` → quality); `experimentalControl` disappears — check OBI instead. | Category discipline, the four-way test, the "does it have temporal parts?" heuristic. |
| **DOLCE** | Descriptive rather than realist framing; roles become considerably easier (DOLCE has real machinery for them). | Everything else. |
| **None** | Delete the binding sheet; keep the category-error table as a checklist. | The discipline. **This is a legitimate and common choice** — most Tier 1 formalisations need category hygiene, not an import. |

**Do not bind to two.** Cross-walking upper ontologies is a research project.

## Swapping the experiment model

EXPO is the default because nothing has replaced it at that level of generality. Reasonable substitutions:

| If your domain is | Use instead | What changes |
|---|---|---|
| **Life science / biomedical** | **OBI** (BFO-anchored, maintained, populated) | `expo-verified-inventory.md` becomes an OBI inventory. The verification discipline is unchanged — OBI has real IRIs you can resolve, so verification is *easier*. Keep the do-not-cite habit. |
| **Multi-study programmes with tabular data** | **ISA** (Investigation / Study / Assay) | The I/S/A containment replaces EXPO's tree. Lighter, tool-supported, less expressive. |
| **Clinical trials** | **CDISC** standards (SDTM/CDASH/ODM — now a row in the prior-art map), plus a protocol registry | Regulated context: registration, protocol amendment history, and pre-specification obligations dominate. Add a sheet; do not improvise. |
| **Software A/B testing** | Usually **nothing** — a typed experiment record | Assignment mechanism, exposure logging, and guardrail metrics matter more than an ontology. Tier 1 or don't formalise. |
| **ML experiment tracking** | **Croissant** for datasets, PROV-O for runs | Do not adopt the dormant ML ontologies. See [`prior-art-map.md`](prior-art-map.md). |

**Keep EXPO's grammar as a checklist even when you swap the vocabulary.** Goal, design, model, planned actions, hypothesis mode, results, typed errors, admin metadata — that list is the durable contribution, and every substitute above under-covers at least one of its cells.

## Adapting the extension sheets

`governance-role-extensions.md` and `lifecycle-and-staged-protocol-extensions.md` are patterns generalised from apparatus with strong separation of powers and staged autonomy. Map them to what you have:

| If your system | Then |
|---|---|
| Has no authority separation (one team, one pipeline) | Drop the governance sheet to a single "who did what" row using PROV-O `hadRole`. Do not model prohibitions that do not exist. |
| Has human approval steps | The archetypes still apply — the judge is a person. Authorisation records matter *more*, not less. |
| Has no persistent entities (each run is independent) | Drop the lifecycle sheet entirely. |
| Has no partial intervention (things are on or off) | Drop the influence coefficient; keep transitions-as-events. |
| Never relaxes its protocol | Drop scaffolds. But check first — "we always ran it this way" often means an undeclared scaffold nobody noticed. |
| Is a wet lab | Scaffolds map to pilot studies and positive controls; regimes map to instrument settings. The pattern holds; the examples change. |

**Deleting an irrelevant sheet is correct.** A pack that models structure you do not have teaches people to invent it.

## Adapting the failure catalogue

The router's catalogue is the critic's checklist. Add to it from your own incidents — that is where the best entries come from.

**Adding an entry:** name the defect, describe the failure it produces, name the sheet that closes it, and give the critic a mechanical check. An entry without a check is a slogan.

**Removing an entry:** only if the defect is structurally impossible in your context. "We're careful about that" is not structural.

Candidates you may need that this pack does not carry: domain-specific units confusion; consent and access-control leakage into shared graphs; multi-site identity collisions; instrument calibration drift unrecorded; jurisdictional retention conflicts with retain-everything.

## Scaling down

The pack in one page, for a small project:

1. Write 5–10 competency questions. Interview one person who will be asked to justify a result.
2. Check whether the current records answer them. **If yes, stop and write that down.**
3. For the ones they cannot answer, list the fields you need.
4. Map each field to a **verified** EXPO or PROV-O term, or mint it in your namespace. Record the verification.
5. Where nothing fits, record the gap in a **gap register** rather than forcing a mapping — one line per gap, with why nothing fit. Tier 1 is not complete without it.
6. Write a JSON-LD context binding your existing fields to those terms.
7. Write the queries. Add a positive control each.
8. Add the sync check to CI.

**That is a complete Tier 1 formalisation.** No OWL, no triple store, no reasoner. It is genuinely enough for the large majority of projects, and it degrades gracefully if abandoned.

## Scaling up

If you are building something publishable or externally federated:

- Add the OWL module and SHACL shapes ([`validation-and-conformance.md`](validation-and-conformance.md)) — but only against [Gate 2's four conditions](formalisation-triage.md).
- Add RO-Crate packaging for anything that leaves the building.
- Add STATO if statistical method needs typing; EXACT2 if protocols need step-level semantics.
- Publish the module with a resolvable IRI, a version IRI, a changelog, and a maintainer. **If you cannot commit to that, do not publish it** — an unmaintained ontology at a dead IRI is exactly what this pack spends a whole sheet warning about, and you would be adding to the problem.

## Adapting the agents and commands

- **`formalisation-critic`** — extend its checklist as you extend the failure catalogue. Point it at your own `verified-terms.json`; that file is meant to grow.
- **`experiment-formalisation-architect`** — if you have swapped the experiment model, tell it so in the dispatch. It reads the sheets; keep them accurate and it follows.
- **Commands** — `/formalise-experiment` halts at a review gate per stage by design. Do not remove the gates to make it faster; the gates are what prevent a one-shot formalisation full of confident, unverified terms.

## Keeping the pack honest over time

Two maintenance obligations, both cheap and both routinely skipped:

1. **Re-verify the term inventory** when you next cite from it. Record the date. An inventory with a stale verification date is a liability wearing the costume of an asset.
2. **Re-check the prior-art map** before relying on any Dormant or Unverified row — the procedure is at the end of that sheet. Ontology projects go quiet without announcing it, and a status column that says "Live" because someone checked in 2026 is worse than no status column at all.

**If you fork this pack, fork the verification obligation with it.** The pack's central claim is that it checked. That claim expires.
