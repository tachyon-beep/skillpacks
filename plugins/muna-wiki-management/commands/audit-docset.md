---
description: Run a full consistency, derivation, path, and link integrity audit on a document set
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[manifest_path]"
---

# Audit Document Set

You are running a comprehensive audit across all documents in the set. This invokes four audit types in sequence: consistency, derivation integrity, path completeness, and link integrity.

## Core Principle

**Audit against registries, not against intuition.** Without a terminology registry, claim registry, and persona registry, you have no objective standard. If registries don't exist, build minimum viable versions first (see `/onboard-docset`).

## Prerequisites

Before running a full audit, verify these artifacts exist:

- [ ] Document manifest (document IDs, tiers, derivation lineage)
- [ ] Terminology registry (key terms with canonical definitions and tier variants)
- [ ] Claim registry (factual claims with canonical sources and propagation lists)
- [ ] Persona registry (roles with depth tolerance and reading paths)

If any are missing, **ask the user** whether to bootstrap them now or proceed with a partial audit.

## Audit Sequence

### Phase 1: Consistency Audit

Check that all documents tell the same story using the same language.

**Step 1 — Terminology scan:**
For each term in the terminology registry, verify it is used correctly across all documents:
- Tier 2 documents use the canonical form
- Tier 1 documents use an approved tier variant
- Tier 0 documents use an approved tier variant
- No document uses a prohibited conflation

**Step 2 — Claim verification:**
For each claim in the claim registry:
- Read the canonical source section — verify the claim still matches
- Read each section in the propagation list — verify the claim matches the canonical form
- Flag any mismatches (stale claims, rounded numbers, paraphrased assertions)

**Step 3 — Link integrity:**
For every cross-document link in every document:
- Verify the target file exists
- Verify the target heading/anchor exists
- Flag broken links, redirected links, and links to deprecated sections

**Step 4 — Structural conventions:**
If structural conventions are defined, verify compliance (heading hierarchy, numbering, citation format, callout patterns).

### Phase 2: Derivation Integrity

Check that derivatives faithfully represent their roots.

**For each derivative section with a derivation recipe:**
1. Verify the recipe's `required_content` items all appear in the derivative
2. Run a deferral audit — count and classify all cross-references to root documents
3. Check claim fidelity — every claim matches the canonical form
4. Check for phantom content — assertions in the derivative not present in any root

**For derivative sections without recipes:**
Flag them as `recipe-missing` defects. A derivative section without a recipe cannot be audited for completeness.

### Phase 3: Path Completeness

Check that every persona can navigate the set successfully.

**For each persona in the registry:**
1. Verify entry point exists and is at or below their depth tier
2. Walk the reading path step by step — at each step, apply stepping stone validation:
   - Does the reader have enough context from previous steps?
   - Does this step give enough to proceed to the next?
   - If this step links to a higher-tier document, can the reader skip it and still complete their task?
3. Verify terminal state — can the persona do what they came to do at the end of the path?

**Coverage matrix check:**
Build or update the personas × topics matrix. Flag empty cells as coverage gaps.

### Phase 4: Compile Results

Collect all defects into a unified defect list using this format:

```
Defect:
  location: [document-id §section]
  type: [one of: terminology-drift, stale-claim, broken-link, lazy-deferral,
         orphan-section, convention-violation, coverage-gap, path-dead-end,
         phantom-content, self-sufficiency-failure, recipe-missing]
  severity: [error | warning]
  description: [what's wrong]
  suggested_fix: [how to fix it]
  related_registry: [which registry entry is violated, if applicable]
```

### Phase 5: Triage and Present

Prioritize defects:

**Priority 1 (fix immediately)** — defects that cause readers to act on wrong information:
- `stale-claim`, `self-sufficiency-failure`, `phantom-content`

**Priority 2 (fix before next publish)** — defects that degrade navigation or trust:
- `broken-link`, `lazy-deferral`, `path-dead-end`

**Priority 3 (fix during maintenance)** — defects that cause friction but don't mislead:
- `terminology-drift`, `coverage-gap`, `convention-violation`, `orphan-section`, `recipe-missing`

**Within each priority level:** Fix root-document defects before derivative defects. Fix higher-tier documents (Tier 0, Tier 1) before Tier 2.

Present the triaged defect list to the user with:
1. Total defect count by priority level
2. The P1 defects in full (these need immediate attention)
3. Summary counts for P2 and P3
4. Recommended fix order

**Ask the user** which defects to address now vs defer to a later maintenance cycle.
