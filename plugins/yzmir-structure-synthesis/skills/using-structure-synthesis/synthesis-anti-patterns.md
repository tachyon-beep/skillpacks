---
name: synthesis-anti-patterns
description: Use when auditing an existing structure-synthesis pipeline end to end - the eight recurring ways generation, verification, and utility judgement re-merge, each with the sheet that names the discipline and the fix. Use for a pre-ship review, a design retrospective, or when something about a generator/verifier split feels off but you can't yet name why.
---

# Synthesis Anti-Patterns

## When to Use

- Auditing an existing or proposed generator + verifier + evaluator pipeline before it ships
- Something about the design feels like it's re-merging generation, verification, and judgement, but the specific violation isn't obvious yet
- Writing a review checklist for structure-synthesis code
- Onboarding someone to why this pack draws such a sharp line between "imagines," "permits," and "judges"

This is the catalogue sheet. Each entry cross-references the sheet that covers the fix in depth — read this one to locate the problem, then the cited sheet to fix it properly.

## The Eight Patterns

### 1. Generator grades its own examination

The generator filters, ranks, or drops its own candidates using an internally-computed utility signal — predicted score, confidence, an auxiliary head's guess — before the pool ever reaches a downstream evaluator.

**Why it's dangerous**: the candidates most likely to be filtered are exactly the ones the generator's own (imperfect) model is most likely to be wrong about. Silent filtering removes them before anyone with better information gets a chance to disagree.

**Evidence to look for**: a `generate()` or pool-returning function whose output size can be smaller than the count requested, for reasons other than a legality failure; any threshold comparison against a self-produced score gating what's returned.

→ Full writeup and worked RED/GREEN example: `learning-objectives-for-generators.md`

### 2. Verifier consumes reward or provenance in its decision

The structural verifier's accept/reject decision reads predicted or measured task utility, or which generator/lineage produced the candidate, as a live input.

**Why it's dangerous**: verification is supposed to be a rule-driven, checkable-by-proof legality gate. The moment it consumes a utility or provenance signal, "legal" and "looks promising" become the same decision, and the gate can no longer be argued about on structural grounds alone.

**Evidence to look for**: a verifier function signature with a `reward`, `predicted_utility`, `source`, or `generator_id` parameter that participates in the accept/reject branch (not just logged).

→ Full writeup: `structural-verification.md`

### 3. Canonicaliser corrupts identity — in either direction

The canonical form can be wrong two ways, and both live in the canonicaliser. It can **change semantics**: a "simplification" rule removes or rewrites structure without a proof that the rewrite preserves the computed function — most commonly, pruning by structural pattern (e.g., any degree-(1,1) node) rather than by a grammar-declared identity-function membership check. Or it can **fail to unify**: incidental raw-graph details (node IDs, emission order) leak into the normal form — most commonly via signature refinement that reads only predecessors, leaving nodes distinguished solely by their downstream role permanently tied and letting raw labels break the tie — so one structure gets several canonical identities.

**Why it's dangerous**: everything downstream — hashing, equivalence detection, diversity measurement, archive dedup — trusts the canonical form as a faithful, *unique* representative. A semantic change corrupts every consumer at once, invisibly. A false split is worse in one specific way: no downstream check can ever catch it, because the exact-isomorphism fallback only fires when hashes already agree.

**Evidence to look for**: any pruning or rewrite rule keyed on graph shape (degree, fan-in/out) rather than an explicit operator-identity check; refinement loops that iterate `predecessors()` with no matching `successors()` fold; no idempotence test (`canon(canon(x)) == canon(x)`); no test that unifies port-permuted or relabeled variants of symmetric structures.

→ Full writeup and worked RED/GREEN examples of both directions: `canonicalisation-and-normal-forms.md`

### 4. Diversity claimed in raw-syntax space

A diversity metric, duplicate-rate report, or coverage claim is computed on raw generated graphs — different node IDs, different serialization order — rather than on the canonical form.

**Why it's dangerous**: raw-syntax diversity is nearly always high regardless of how collapsed the generator actually is, because incidental relabeling looks like real variation. Trusting it hides mode collapse behind a dashboard that reads "100% unique."

**Evidence to look for**: a diversity or uniqueness metric computed before any canonicalisation step; a best-of-K pool reported as diverse with no canonical-hash duplicate count alongside it.

→ Full writeup and worked RED/GREEN example: `diversity-and-mode-collapse.md`

### 5. Hash unstable across serialization order or library version

A semantic hash is delegated to a library's default hash function (or to iteration order that isn't explicitly pinned), without an owned serialization layer and a version field.

**Why it's dangerous**: an unstable hash silently invalidates archived identities the moment a dependency upgrades or a serialization detail shifts — with no error, just quietly wrong equality comparisons from that point forward.

**Evidence to look for**: a persisted "semantic hash" that is a direct call to a general-purpose graph-hash library function; no `hash_version` field; no test asserting the hash is stable across an equivalent-but-differently-ordered input.

→ Full writeup and worked RED/GREEN example (including a live library warning demonstrating the exact failure): `equivalence-detection-and-semantic-hashing.md`

### 6. Grammar expanded without a reliability gate

New operators, topology freedom, or a new staged expressiveness level are added to the grammar without first measuring that the current level's generator/verifier/canonicaliser triple is healthy.

**Why it's dangerous**: verification cost and illegal-candidate risk grow combinatorially with grammar size, not linearly with each addition. Ungated expansion compounds until search and verification become unreliable, usually with no single change identifiable as the cause.

**Evidence to look for**: whitelist or ceiling changes merged as isolated PRs with no rejection-rate, duplicate-rate, or verification-latency check against the pre-expansion baseline.

→ Full writeup and worked demonstration of the growth curve: `search-space-evolution-and-explosion-control.md`

### 7. Mode collapse hidden behind syntactic variation

A specific case of pattern 4 worth naming separately because it survives casual review: a pool that canonicalises to one structure but arrives with enough surface variation (different parameter values, different but functionally-irrelevant sub-structure) that even a canonical-hash check might miss it if the canonicaliser itself under-normalizes.

**Why it's dangerous**: this is the pattern that specifically defeats "we already check canonical diversity" — the fix for pattern 4 assumes a correct canonicaliser; this pattern is what happens when the canonicaliser has its own gaps (see pattern 3).

**Evidence to look for**: canonical-diversity metric reports healthy numbers, but functional-diversity spot-checks (actual output comparison on probe inputs) reveal duplicates the canonical hash didn't catch — the gap between canonical and functional duplicate rate is the signal.

→ Full writeup: `diversity-and-mode-collapse.md` (functional diversity section), root-caused via `canonicalisation-and-normal-forms.md`

### 8. Validity left entirely post-hoc when cheap constrained decoding existed

A generator emits freely and relies entirely on a post-generation verifier to catch illegal candidates, in a domain where a cheap decode-time mask (whitelist membership, running-budget tracking) could have prevented most of them before generation compute was even spent.

**Why it's dangerous**: not incorrect — a correct post-hoc verifier is still a correct gate — but wasteful in a way that compounds: every illegal candidate the mask could have prevented instead consumes full generation cost, full verification cost, and shows up as a rejection-rate number that looks like a generator-quality problem when it's actually a missed decode-time opportunity.

**Evidence to look for**: a high structural-rejection rate concentrated in checks that are locally decidable from a prefix (whitelist membership, running totals) rather than globally decidable only at completion (full-graph reachability, final-node contract matching).

→ Full writeup and worked RED/GREEN example: `validity-by-construction-vs-post-hoc.md`

## How to Use This Catalogue in a Review

Run it as a checklist, in order — patterns 1–2 are the sharpest boundary violations and worth ruling out first: a pipeline that fails either one has a design-level problem that no amount of correctness on patterns 3–8 can compensate for:

1. Trace every path where a generated candidate's fate (returned, filtered, ranked) is decided. Does any of that logic read a self-produced or internally-predicted utility signal? → Pattern 1.
2. Read the verifier's function signature and decision branches. Does anything besides the candidate, its birth parameters, the grammar, and the interface contract reach the accept/reject decision? → Pattern 2.
3. Find every canonicalisation rule and the refinement loop. Does each rewrite cite a specific proof obligation (an operator-identity membership check, a proven-dead-code condition), or does any rely on structural resemblance alone? Does the refinement fold in both in-edges and out-edges? → Pattern 3, both directions.
4. Find every diversity or duplicate-rate metric in logs, dashboards, or papers. Is it computed before or after canonicalisation? → Pattern 4.
5. Find the hash function used for candidate identity. Is it a pinned, versioned, owned serialization, or a library default? → Pattern 5.
6. Review the last several grammar/whitelist changes. Was each one checked against a reliability-gate baseline, or merged on its own merits alone? → Pattern 6.
7. If canonical-diversity numbers look healthy, spot-check functional diversity on a probe set. Does the gap suggest the canonicaliser itself under-normalizes? → Pattern 7.
8. Look at the structural-rejection rate's breakdown by check type. Are locally-decidable failures (whitelist, budget) a large share of it? → Pattern 8.

A review that finds zero instances of any of these eight patterns in a nontrivial pipeline is itself worth treating with suspicion — either the pipeline genuinely has excellent discipline (possible, and worth documenting exactly how, as a template for the next one) or the review didn't look hard enough. Per `synthesis-integrity-reviewer`'s design (see the router `SKILL.md`), a zero-findings audit run is treated as a defect of the audit, not a clean bill of health, until the reviewer can positively demonstrate why each pattern doesn't apply.

## Cross-References

Every pattern above links to its owning sheet. For the full pack overview and routing table, return to the router: `SKILL.md`.
