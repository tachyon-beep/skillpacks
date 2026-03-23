---
description: Propagate a root document change through all affected derivatives with impact trace and quality gates
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[root_document_path] [changed_section]"
---

# Propagate Change

You are propagating a change from a root document through all affected derivatives. This chains four competencies: Evolution → Derivation → Consistency → Governance.

## Core Principle

**Every root change has a blast radius. Trace it before you act.**

A change to one section of a root document may affect multiple sections across multiple derivatives. The impact trace tells you exactly what needs updating. Without it, you either miss affected sections (stale content) or wastefully review everything (overkill).

## Mandatory Workflow

### Step 1: Identify the Change

Read the changed section of the root document. If the user provided a specific section, read it. If they said "I updated the paper," ask which section(s) changed.

Use `git diff` if available to see exactly what changed:

```bash
git diff HEAD~1 -- [root_document_path]
```

### Step 2: Classify the Change

Determine the change type:

- **Cosmetic** — typo fixes, formatting, rewording that doesn't change meaning → **Stop here. No propagation needed.**
- **Clarification** — same claim, better explanation → check derivatives for stale phrasing but meaning unchanged
- **Substantive** — a claim, recommendation, or structural element changed → mandatory propagation to all derivative sections in the claim's propagation list
- **Structural** — sections added, removed, or reorganized → reading paths, cross-document links, and manifest all need review

**Tell the user** your classification and confirm before proceeding.

### Step 3: Run Impact Trace

For substantive and structural changes:

1. Read the document manifest to identify which derivatives draw from the changed root
2. Read the derivation graph (if it exists) to find the specific derivative *sections* affected
3. Check the claim registry for any claims whose canonical source is the changed section — their propagation lists identify additional affected locations
4. Check the terminology registry if the change altered any key terms

Output: A **work list** — each entry names: derivative document, affected section, derivation mode, and what specifically needs updating.

Present the work list to the user before proceeding.

### Step 4: Re-Derive Affected Sections

For each entry in the work list, in priority order (Tier 0 first, then Tier 1, multi-root last):

1. Read the updated root section
2. Retrieve the derivation recipe for the affected derivative section (if no recipe exists, draft one first)
3. Re-derive the section following the recipe:
   - **Distillation**: compress the updated content, preserving new claims exactly
   - **Translation**: translate the updated content using the target persona's vocabulary
   - **Extraction**: re-evaluate which content is relevant for the target persona
4. Run the **self-sufficiency test**: hide the root document, read the derivative section as the target persona — can they do their job?
5. Run a **deferral audit**: flag every cross-reference to the root. Classify each as acceptable (optional depth) or lazy (substitutes for needed content). Fix any lazy deferrals.

### Step 5: Verify Consistency

After all derivative sections are updated:

1. Check that every claim in the claim registry that was affected by the root change now matches across all documents
2. Check that any changed terminology is consistent across all documents at tier-appropriate variants
3. Verify all cross-document links still resolve (the root change may have altered heading anchors)

If any inconsistencies remain, fix them before proceeding.

### Step 6: Update Registries

- Update the claim registry if any claim values changed
- Update the terminology registry if any terms changed
- Update the external dependencies file if the change was triggered by an external standard update
- Update the derivation graph if section structure changed

Commit registry updates alongside the derivative updates:

```bash
git add [derivative-files] [registry-files]
git commit -m "propagate: update derivatives for [brief change description]"
```

### Step 7: Surface Quality Gate Results

Present to the user:

1. **What changed**: the root change classification and description
2. **What was affected**: the work list (which derivatives, which sections)
3. **What was updated**: summary of each derivative update
4. **Deferral audit results**: total deferrals found, how many acceptable vs lazy, how many fixed
5. **Consistency check results**: any remaining issues
6. **Recommended spot-check**: which derivative section the user should manually review (pick the most complex — multi-root, synthesis mode, or highest deferral count)

**Do not self-approve.** The user reviews and confirms before the work is considered complete.
