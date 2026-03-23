---
name: fact-checking
description: Dual-verified research paper fact-checking with structured JSON output and exception reports
---

# Research Paper Fact-Checker

## Overview

This skill orchestrates a four-phase pipeline to fact-check research papers with maximum rigor. Every verifiable claim is extracted, then independently verified by two agents (a researcher and an adversarial verifier) using web search. Results are reconciled and output as structured JSON plus a human-readable exception report.

**This is a deliberately expensive, token-heavy skill.** Only invoke it when you need rigorous fact-checking, not speculatively.

## When to Use

- User explicitly asks to fact-check a research paper or document
- User invokes `/fact-check <paths>`
- User asks to verify claims in a document against external sources

## Prerequisites

This skill requires the following tools to be available:
- **Agent** — to spawn subagents for each phase
- **WebSearch** and **WebFetch** — for evidence gathering (subagents)
- **Read** — to read the paper files (extraction subagent)
- **Write** — to produce output files

If WebSearch or WebFetch are unavailable, this skill cannot function. Inform the user and stop.

## Inputs

The user provides one or more file paths to the paper:
- Single file: `/fact-check path/to/paper.md`
- Multiple sections: `/fact-check path/to/section1.md path/to/section2.md`

All paths are read in order and treated as one continuous document.

---

## Phase 1 — Claim Extraction

Spawn a single agent to extract all verifiable claims from the paper. This agent reads but does NOT verify.

### Extractor Agent Prompt

Use the Agent tool with these parameters:
- **description**: "Extract verifiable claims from paper"
- **subagent_type**: leave default (general-purpose)

Prompt the agent with:

---

You are a claim extractor. Read the following research paper files and extract every verifiable factual claim.

**Files to read:** [INSERT FILE PATHS]

For each claim, output a JSON object on its own line with these fields:
- `id`: sequential integer starting at 1
- `text`: the verbatim claim from the paper (quote exactly)
- `category`: one of `quantitative`, `citation`, `scientific`, `historical`, `definitional`
- `section`: the section heading or location where the claim appears
- `context`: 1-2 surrounding sentences for disambiguation

**Extraction rules:**
- Extract every statement that asserts something verifiable about the external world
- DO NOT extract: opinions, hedged speculation ("may", "might", "could"), definitions of the paper's own novel terms, or methodology descriptions that are internal to the paper
- For citation claims: extract both the attributed finding AND the citation details (author, year, title if given)
- For quantitative claims: extract the exact number with units, timeframe, and stated source
- Be exhaustive — miss nothing. It is better to over-extract than under-extract.

Output ONLY a JSON array of claim objects. No commentary, no markdown formatting, no explanation.

---

### Processing the extraction result

Parse the returned JSON array. Count the claims and report to the user:

> "Extracted N claims from the paper. Organizing into batches of 8 for dual verification..."

If the JSON is malformed, retry the extraction once. If it fails again, report the error to the user and stop.

---

## Phase 2 — Research (Agent A)

For each batch of 8 claims, spawn a researcher agent. Multiple batches can run in parallel.

### Researcher Agent Prompt Template

Use the Agent tool with:
- **description**: "Research-verify claims batch N"
- **subagent_type**: leave default (general-purpose)

Prompt the agent with:

---

You are a research fact-checker. For each claim below, search the web for evidence that supports or refutes it.

**CRITICAL RULES:**
1. You MUST use WebSearch and WebFetch to find evidence. Do NOT rely on your training knowledge.
2. For each claim, perform at least one web search. Read at least one source page.
3. If you cannot find evidence, say so — do NOT fabricate sources or URLs.

**Claims to verify:**

[INSERT JSON ARRAY OF CLAIMS FOR THIS BATCH]

For each claim, output a JSON object with:
- `id`: the claim's id (match the input)
- `verdict`: one of `verified`, `refuted`, `uncertain`
  - `verified`: found reliable source(s) confirming the claim
  - `refuted`: found reliable source(s) contradicting the claim
  - `uncertain`: could not find sufficient evidence either way
- `confidence`: one of `high`, `medium`, `low`
- `reasoning`: 1-3 sentences explaining how the evidence supports your verdict
- `evidence`: array of objects, each with:
  - `url`: the actual URL you visited
  - `quote`: a relevant excerpt from the page
  - `relevance`: why this source matters

Output ONLY a JSON array of result objects. No commentary.

---

## Phase 3 — Verification (Agent B)

For the same batches, spawn independent verifier agents. These run IN PARALLEL with the researchers — they must never see Agent A's results.

### Verifier Agent Prompt Template

Use the Agent tool with:
- **description**: "Adversarial-verify claims batch N"
- **subagent_type**: leave default (general-purpose)

Prompt the agent with:

---

You are an adversarial fact-checker. Your job is to attempt to DISPROVE each claim below. Search for counter-evidence, check if numbers are misquoted, verify that cited sources actually say what is attributed to them.

**CRITICAL RULES:**
1. You MUST use WebSearch and WebFetch to find evidence. Do NOT rely on your training knowledge.
2. Actively search for COUNTER-evidence. Try to find sources that contradict the claim.
3. If you cannot disprove a claim despite genuine effort, mark it `verified`.
4. If you cannot find any evidence at all, mark it `uncertain` — do NOT default to verified.
5. Do NOT fabricate sources or URLs.

**Claims to check:**

[INSERT JSON ARRAY OF CLAIMS FOR THIS BATCH]

For each claim, output a JSON object with:
- `id`: the claim's id (match the input)
- `verdict`: one of `verified`, `refuted`, `uncertain`
- `confidence`: one of `high`, `medium`, `low`
- `reasoning`: 1-3 sentences explaining your adversarial assessment
- `evidence`: array of objects, each with:
  - `url`: the actual URL you visited
  - `quote`: a relevant excerpt from the page
  - `relevance`: why this source matters

Output ONLY a JSON array of result objects. No commentary.

---

## Orchestrating Phases 2 and 3

### Batching

Split the claims array into batches of 8. For N claims, create ceil(N/8) batches. Batch size of 8 balances agent context limits against per-batch overhead.

### Parallel execution

Launch up to 2 batch-pairs per message. For each batch-pair, spawn the Researcher and Verifier agents **in the same message** (parallel Agent tool calls):

```
Message 1:
  Agent tool call 1: Researcher batch 1
  Agent tool call 2: Verifier batch 1
  Agent tool call 3: Researcher batch 2
  Agent tool call 4: Verifier batch 2
```

After these agents return, report progress, then launch the next set of batch-pairs.

### Collecting results

As each agent returns, parse its JSON output and store the results keyed by claim ID. If an agent fails or returns malformed JSON, retry that specific agent once. If it fails again, mark all claims in that batch as `uncertain` with reasoning "agent failure — verification could not be completed".

### Progress reporting

After each message's agents return, report progress before launching the next set:

> "Batches N-M of T complete — verified: X, refuted: Y, uncertain: Z so far..."

---

## Phase 4 — Reconciliation

After ALL batches complete, reconcile the results. This runs in the main conversation — no subagent needed.

### Reconciliation rules

For each claim, compare the researcher verdict and verifier verdict:

| Researcher | Verifier | Final Verdict |
|-----------|----------|---------------|
| verified | verified | `verified` |
| refuted | refuted | `refuted` |
| uncertain | uncertain | `uncertain` |
| uncertain (agent failure) | any definitive | `uncertain` |
| any definitive | uncertain (agent failure) | `uncertain` |
| Any other mismatch | Any other mismatch | `disputed` |

### Building the output

For each claim, construct the final record:

```json
{
  "id": 1,
  "text": "<verbatim claim>",
  "category": "<category>",
  "section": "<section>",
  "context": "<context>",
  "verdict": "<reconciled verdict>",
  "research": {
    "verdict": "<researcher verdict>",
    "confidence": "<researcher confidence>",
    "reasoning": "<researcher reasoning>",
    "evidence": [{"url": "...", "quote": "...", "relevance": "..."}]
  },
  "verification": {
    "verdict": "<verifier verdict>",
    "confidence": "<verifier confidence>",
    "reasoning": "<verifier reasoning>",
    "evidence": [{"url": "...", "quote": "...", "relevance": "..."}]
  }
}
```

---

## Output — JSON Results

Write `fact-check-results.json` to the same directory as the first input file. If the file already exists or the directory is not writable, ask the user for an alternative path before writing.

### Schema

```json
{
  "metadata": {
    "paper_files": ["<list of input file paths>"],
    "total_claims": 142,
    "verified": 118,
    "refuted": 12,
    "disputed": 8,
    "uncertain": 4,
    "timestamp": "<ISO 8601 timestamp>"
  },
  "claims": [
    {
      "id": 1,
      "text": "The transformer architecture was introduced in 2017",
      "category": "historical",
      "section": "Section 2.1 - Background",
      "context": "...surrounding text...",
      "verdict": "verified",
      "research": {
        "verdict": "verified",
        "confidence": "high",
        "reasoning": "Multiple sources confirm Vaswani et al. published 'Attention Is All You Need' in June 2017",
        "evidence": [
          {
            "url": "https://arxiv.org/abs/1706.03762",
            "quote": "Submitted on 12 Jun 2017",
            "relevance": "Primary source — the original paper"
          }
        ]
      },
      "verification": {
        "verdict": "verified",
        "confidence": "high",
        "reasoning": "Independently confirmed via multiple secondary sources",
        "evidence": [
          {
            "url": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
            "quote": "introduced in 2017 by a team at Google Brain",
            "relevance": "Secondary source confirming date and attribution"
          }
        ]
      }
    }
  ]
}
```

Use the Write tool to create this file. Pretty-print the JSON (2-space indent).

---

## Output — Exception Report

Write `fact-check-exceptions.md` to the same directory as the JSON results.

This report contains ONLY claims that are NOT `verified`. Verified claims are omitted entirely.

### Template

```
# Fact-Check Exception Report
**Paper**: <comma-separated input file paths>
**Date**: <YYYY-MM-DD>
**Summary**: <total> claims checked — <verified> verified, <refuted> refuted, <disputed> disputed, <uncertain> uncertain

---

## Refuted (<count>)

### Claim #<id> — <category> | <section>
> "<verbatim claim text>"

**Research**: <researcher verdict> (<researcher confidence>)
- <researcher reasoning> — [source](<url>)

**Verification**: <verifier verdict> (<verifier confidence>)
- <verifier reasoning> — [source](<url>)

---

## Disputed (<count>)

[Same format as Refuted, for each disputed claim]

---

## Uncertain (<count>)

[Same format as Refuted, for each uncertain claim]
```

Omit any section that has zero claims (e.g., if there are no disputed claims, omit the Disputed section entirely).

---

## Error Handling

- **Malformed extraction JSON**: Retry extraction once. If it fails again, report to user and stop.
- **Agent failure/timeout**: Retry the failed agent once. If it fails again, mark all claims in that batch as `uncertain` with reasoning "agent failure".
- **No search results for a claim**: Agent marks it `uncertain` with reasoning "no web sources found". Agent must NOT fall back to training knowledge.
- **Paper too large (200+ claims)**: Report the count to the user and proceed. No artificial cap.

## What This Skill Does NOT Do

- Plagiarism checking
- Style or grammar review
- Citation formatting
- Summarization or rewriting
- Methodology evaluation — it verifies facts, not research design
