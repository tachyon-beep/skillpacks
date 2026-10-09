---
name: fact-checking
description: "Use when the user asks to verify factual claims, quantitative statements, or attributed citations in a paper or document against external evidence."
---

# Source-Based Fact Checking

Verify claims against evidence rather than model recall. Do not initiate a full
claim audit speculatively when the user requested a narrower edit or review.

## Scope and capabilities

Read the supplied files in order and preserve context. Use available search,
source retrieval, document-reading and output tools; exact tool names and
subagents are not prerequisites. If evidence cannot be retrieved, mark affected
claims unresolved and report coverage limits. Do not replace retrieval with
confident memory. Select an output location within the authorized workspace;
preserve existing reports rather than overwrite them without authorization.

## 1. Build the claim ledger

Extract externally verifiable claims, including qualified, probabilistic, or
hedged claims when their underlying assertion can be tested. Preserve quantifiers,
units, timeframe, population, conditions, and attribution. Separate compound
assertions when they require different evidence. Opinions, novel stipulative
terms, and internal methodology are not automatically external facts, but may
contain factual premises worth extracting.

Use one JSON array consistently, with records shaped as:

```json
[{"id": 1, "text": "Verbatim claim", "category": "scientific", "section": "2.1", "context": "Surrounding context", "status": "pending"}]
```

Useful categories are quantitative, citation, scientific, historical and
definitional. Preserve stable IDs; report omissions or sampling if exhaustive
extraction is infeasible. Do not claim every statement was covered without a
coverage check against the source.

## 2. Gather support and counterevidence

For each in-scope claim, retrieve relevant sources and read the supporting span
in context. Prefer original papers, official datasets/standards and direct
records over summaries. Check version/date, population and conditions, corrections
or retractions, source independence, and whether the source actually entails the
claim. A quotation's existence does not make its content true.

Record URLs/identifiers actually accessed, concise source excerpts where permitted,
locations, access/version dates when relevant, and how each source relates to the
claim. Seek plausible counterevidence and alternative interpretations. Absence of
contradiction is not positive support; inability to disprove is not verification.

## 3. Choose review effort by uncertainty and consequence

A single well-grounded pass is sufficient for straightforward bounded claims.
Use an independent critical pass for consequential, ambiguous, contested, or
weakly supported claims, or when the user requests dual verification. Keep that
pass blind to the first verdict until evidence gathering completes when feasible.
Batch size and concurrency follow context, tool limits and budget; there is no
fixed batch-of-eight requirement. Two agents citing one source are not two
independent evidence streams. Reconcile substance, not votes.

## 4. Decide and reconcile

| Status | Required basis |
|---|---|
| `verified` | Reliable accessible evidence supports the claim with its material qualifications. |
| `refuted` | Reliable evidence contradicts the claim as written; explain the actual correction. |
| `disputed` | Credible evidence or interpretations conflict and cannot yet be resolved. |
| `uncertain` | Insufficient evidence, inaccessible sources, unresolved scope, or failed coverage. |

Track `not_contradicted` as a search observation when useful, never as a synonym
for verified. A second-pass failure leaves that pass unperformed; preserve any
first-pass support but do not label the claim dual-verified. If review was required
for acceptance and is incomplete, final status remains uncertain. Disagreement
requires inspecting source context, not mechanically averaging confidence.

## 5. Deliver an auditable result

Write `fact-check-results.json` and `fact-check-exceptions.md` if files are requested
or useful for the authorized task. The results file remains a top-level JSON array
of stable claim records, extended with verdict/reasoning, evidence, and optional
independent-review results. Put run metadata (input paths, scope/coverage, timestamp
and counts) in the exception report or a separate manifest when needed; do not
wrap the claim array in an object. Validate the JSON and check IDs/counts.
Do not place a synthetic example or invented URL in an evidence record.

The exception report lists refuted, disputed and uncertain claims with locations,
reason, sources and a proposed correction or next verification step. Summarize
verified coverage without reproducing every successful check. Report retrieval
failures and unreviewed claims separately from negative findings.

Fact verification does not by itself assess methodology, causal identification,
plagiarism, grammar, citation style, or the paper's overall validity. Address
those only when requested and with appropriate evidence.
