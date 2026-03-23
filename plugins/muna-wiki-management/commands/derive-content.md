---
description: Write or update a derivative document section with enforced self-sufficiency and deferral audit
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[derivative_path] [section]"
---

# Derive Content

You are writing or updating a specific section of a derivative document. This is the command that most directly fights LLM laziness — every step enforces faithful synthesis over deferral.

## Core Principle

**The derivative exists so the reader does NOT have to open the root.** If you write "see the paper for details," you have transferred your work to the reader. That is a failure, not a cross-reference.

## Before You Start

Gather these inputs:
1. **The derivative document** — which file, which section
2. **The root document section** — where the source content lives
3. **The target persona** — who is this section written for? What is their depth tolerance, domain vocabulary, and task?
4. **The derivation recipe** — if one exists, retrieve it. If not, you will create one in Step 1.

If the user hasn't specified all of these, **ask**.

## Mandatory Workflow

### Step 1: Write or Retrieve the Derivation Recipe

The recipe declares what this section must contain. Format:

```
Source: [root document §section]
Mode: distillation | translation | extraction
Target persona: [persona-id — role, depth tolerance, domain vocabulary]
Self-sufficiency claim: [what the reader can DO after reading this section alone]
Acceptable deferrals: [what CAN link to the root for optional depth]
Required content: [what MUST appear here, not behind a link]
```

**If no recipe exists**, draft one now by:
1. Reading the root section
2. Identifying the target persona's needs (what they came to do)
3. Determining the derivation mode:
   - **Distillation** — same argument, fewer words (Tier 2 → Tier 0)
   - **Translation** — same meaning, different vocabulary (Tier 2 → Tier 1 in a different domain)
   - **Extraction** — curated subset (Tier 2 → Tier 1, selecting what's relevant for the persona)
   - A section may have a primary + secondary mode (e.g., extract then translate)
4. Listing required content (claims, findings, recommendations the persona needs)
5. Listing acceptable deferrals (evidence, methodology, full data tables)

**Present the recipe to the user** for approval before writing content.

### Step 2: Read the Root Section

Read the complete root section specified in the recipe. Take note of:
- Every factual claim (especially quantitative — counts, percentages, named lists)
- The argument structure (premise → evidence → conclusion)
- Key terminology and how it maps to the persona's vocabulary
- Any caveats, qualifiers, or limitations

### Step 3: Write the Derivative Section

Follow the recipe's mode:

**Distillation:** Preserve the argument structure. Compress by removing supporting evidence, methodology details, and extended examples — but keep every conclusion and every quantitative claim exactly. The reader should reach the same understanding, just faster.

**Translation:** Restate the content using the persona's vocabulary. Map technical terms to their domain equivalents. Restructure if the persona thinks about the problem differently than the root presents it. The meaning must be identical; the language changes.

**Extraction:** Select only the content relevant to the persona's task. Omit sections that don't affect their decisions or actions. For included content, preserve it faithfully — extraction curates, it doesn't rewrite.

**For all modes:** Every item in the recipe's `required_content` list MUST appear inline. Not behind a link. Not summarized as "several factors." Inline, with the actual content.

### Step 4: Run the Self-Sufficiency Test

1. **Hide the root.** Mentally remove all access to the root document.
2. **Read your section as the target persona.** Can they complete their stated task?
3. **Check claim grounding.** Are all claims supported with enough context that the reader trusts them?
4. **Check for comprehension gaps.** Are there concepts that require the root to understand?

**Pass criteria:** The persona can do their job using only this section. All quantitative claims appear verbatim. No required content is behind a link.

**Fail criteria:** Any step requires opening the root. Any deferral replaces content the reader needs. Claims are paraphrased in ways that lose precision.

If the test fails, go back to Step 3 and fix the failing sections before proceeding.

### Step 5: Run the Deferral Audit

Scan the section for every cross-reference to the root or any other document. For each:

| Classify as... | If... | Action |
|---|---|---|
| **Acceptable** | Points to optional additional depth the reader doesn't need | Keep it |
| **Lazy** | Substitutes for content the reader actually needs to act | Replace with inline content |
| **Borderline** | Points to evidence that supports but may be needed for trust | Bring key evidence inline, keep link for full dataset |

**Zero lazy deferrals is the target.** If you cannot eliminate a lazy deferral (the root content is too complex to synthesize at this depth), mark it explicitly as a known gap rather than hiding it behind a link.

### Step 6: Verify Claim Fidelity

For every factual claim in your section:
- Does it match the root's canonical form exactly? (13, not "about a dozen")
- Does it match the claim registry entry if one exists?
- Did you introduce any claims not present in the root? (phantom content)

### Step 7: Present Results

Show the user:
1. The derivation recipe (new or existing)
2. The written/updated section
3. Self-sufficiency test result (pass/fail, with details if fail)
4. Deferral audit results (total count, acceptable, lazy, any remaining gaps)
5. Claim fidelity check results
6. **Recommended spot-check section** — the most complex part of the derivation

**Do not self-approve.** The user reviews the output before it is finalized.
