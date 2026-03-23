---
name: reader-panel-review
description: Orchestrate a simulated reader panel review — spawn persona-reader agents as teammates, manage the chapter pump, enforce the Step A gate, and coordinate synthesis
---

# Reader Panel Review — Coordinator Guide

You are the coordinator for a simulated reader panel review. You spawn persona-reader agents as teammates, feed them chapters one at a time, enforce the Step A expectations gate, and manage the full lifecycle from config parsing through synthesis.

## Overview

A panel review simulates diverse readers encountering a document suite chapter by chapter. Each persona reads independently, writes mood journals capturing genuine reactions, and produces an overall verdict. You then spawn a synthesiser to aggregate findings into editorial recommendations.

The primary output is **editorial intelligence**: how the document lands with different audiences, what it communicates vs what it intends, where it loses readers, and what derivative documents the audience actually needs.

## Cost Warning

**This is an expensive skill.** Each persona spawns as a sonnet agent that reads process.md (~1000 lines), writes a suite orientation, then cycles through Step A → chapter read → Steps B+C for every chapter they choose to read. A 3-persona panel reading 4 chapters each generates ~15+ agent turns. A 13-persona panel on a 20-chapter document suite can run hundreds of turns. The opus synthesiser at the end adds further cost.

Before starting, confirm with the user:
- Number of personas (the full panel, or a reduced set?)
- The user understands this will take significant time and tokens

## Prerequisites

- A config file defining the document suite, personas, and optional scenario framing. See `config-template.md` for the format specification. See `config.md` for a fully worked 6-persona example.
- Document files accessible at the paths specified in the config.
- An output directory (you will create it).

Both `config-template.md` and `config.md` live in the same directory as `process.md`. The user will provide the path to their config file.

---

## Phase 0: Panel Design (optional)

If the user does not have a config file, offer to spawn the persona-designer agent to create one.

Use `subagent_type: "muna-panel-review:persona-designer"` with `mode: bypassPermissions`. The spawn prompt provides:
- Paths to the document files (or a directory for the designer to Glob)
- Optional: user context about intended audiences
- Optional: desired panel size
- Path to process.md (designer reads Phase 1)
- Path to config-template.md (output format)
- Output path for the config file

The designer reads all documents freely, analyses audiences and decision chains, and writes a complete config file including a "Panel gaps and deferred personas" section documenting audiences it identified but chose not to include. It is fire-and-forget.

When the designer returns, present the generated config to the user for review. The user may edit it before proceeding to Phase 1.

If the user already has a config file, skip Phase 0 and go directly to Phase 1.

---

## Phase 1: Config Parsing

When the user provides a config file, read it and extract the following.

### Required elements

1. **Document suite** — the table of documents with paths, descriptions, chapter counts, and source file patterns. Validate that every referenced file path exists on disk. If any are missing, report the missing paths and stop.

2. **Personas** — each persona block must contain all required fields:
   - Name/Role
   - Lens
   - Background
   - Reading behaviour
   - Key question
   - Blind spots
   - Voice sample
   - Directory name

   If any required field is missing from any persona, report it and stop.

3. **Panel configuration** — control persona designation, unreliable narrator (if any), persona priority order, collision test pairings (if any).

### Optional elements

4. **Scenario framing** — the blockquote framing statement and framing effects table. If present, every persona must also have a `Consultation voice` field. If absent, the `Consultation voice` field should not be present in any persona, the "Consultation response" section is removed from verdicts, and the "Consultation Response Forecast" section is removed from the synthesis.

5. **Misconception** — only on the persona designated as the unreliable narrator.

### Validation summary

After parsing, report to the user:
- Number of personas found
- Number of documents in the suite
- Whether scenario framing is defined
- Which persona is the control
- Which persona (if any) is the unreliable narrator
- Any warnings or missing data

Wait for user confirmation before proceeding.

---

## Phase 2: Working Directory Setup

Use Bash to set up the output directory. If the user specifies an output path, use it. Otherwise, default to a `panel-review/` directory inside the current project (next to the config file). **Always use a path inside the project directory** — avoid `/tmp/` or other system directories that may trigger permission prompts for agents.

### Step 1: Create directory structure

```bash
mkdir -p "{output-dir}"
```

Create one subdirectory per persona, using the `Directory name` field from each persona spec:

```bash
mkdir -p "{output-dir}/lead-eng"
mkdir -p "{output-dir}/junior-dev"
mkdir -p "{output-dir}/vp-eng"
# ... one per persona
```

**All directories must exist before any agents are spawned.** Agents have Write access only — they cannot create directories. If you skip this step, agents will fail trying to write their journal files.

### Step 2: Create the hashed chapter store

```bash
mkdir -p "{output-dir}/.chapters"
```

Copy all chapter files into `.chapters/` with hashed filenames. The hash prevents agents from browsing or guessing file paths — the coordinator controls access.

```bash
for f in /path/to/source/chapters/*.md; do
  hash=$(sha256sum "$f" | cut -c1-6)
  cp "$f" "{output-dir}/.chapters/${hash}.md"
done
```

Run this for every document in the suite. If multiple documents have chapter files in different directories, run the loop for each source directory. Handle hash collisions by extending to 8 characters if a 6-character hash is not unique.

### Step 3: Write the manifest

Write `{output-dir}/.chapters/manifest.json` mapping original paths to hashed filenames. Structure:

```json
{
  "documents": {
    "Document Title": {
      "chapters": {
        "01: Chapter Name": {
          "original": "/absolute/path/to/original.md",
          "hashed": "a7f3b2.md"
        },
        "02: Next Chapter": {
          "original": "/absolute/path/to/next.md",
          "hashed": "c9d4e1.md"
        }
      }
    },
    "Second Document Title": {
      "chapters": {
        "01: First Chapter": {
          "original": "/absolute/path/to/other.md",
          "hashed": "f2a8b0.md"
        }
      }
    }
  }
}
```

### Step 4: Build tables of contents

For each document in the suite, build a table of contents containing chapter titles and numbers only. No content, no file paths. These tables are sent to agents for navigation.

Example:

```
Document: Cloud Migration Architecture (Technical Specification)
  Chapter 00: Front Matter
  Chapter 01: Executive Summary
  Chapter 02: Current State Assessment
  Chapter 03: Target Architecture
  Chapter 04: Migration Strategy
  ...
  Chapter 13: References
```

---

## Phase 3: Team Creation and Agent Spawning

### Create the team

Use `TeamCreate` to create a team:
- Name: `panel-review-{timestamp}` or a user-specified name

### Spawn agents

Spawn one `persona-reader` agent per persona as a teammate. For each persona:

- `name` = the persona's directory slug (e.g., `"lead-eng"`, `"junior-dev"`, `"vp-eng"`)
- `subagent_type` = `muna-panel-review:persona-reader`
- `mode` = `bypassPermissions`

**Why bypassPermissions:** Persona-readers only have Read and Write tools (restricted by frontmatter). A single persona reading 5 chapters generates ~15+ tool calls (read process.md, read chapters, write journals, write verdict). A 13-persona panel produces hundreds of Read/Write calls. Without bypass, every one triggers a permission prompt, making the workflow unusable. The tool restriction is the safety mechanism — bypass the permission prompts.

### Spawn prompt template

Each agent's spawn prompt must include ALL of the following. Fill in the values from the parsed config.

```
You are about to begin a panel review reading session.

## Your Persona

{Paste the full persona spec inline from config — all fields: Name/Role,
Lens, Background, Reading behaviour, Key question, Blind spots, Voice sample,
Consultation voice (if scenario framing defined), Misconception (if unreliable narrator),
Directory name}

## Scenario Framing

{Paste the scenario framing blockquote from config, or:
"No scenario framing is defined for this review."}

## Methodology

Read the methodology for your reading session from:
{absolute path to process.md}

Read Phases 2 and 3. They cover:
- Journal format (full and light variants)
- Voice guidelines
- Reading path autonomy
- Voice re-anchoring
- Overall verdict format

## Documents Available

{Per-document table of contents — titles and chapter numbers only,
no content, no file paths}

Document 1: {Title}
  Chapter 00: {Chapter title}
  Chapter 01: {Chapter title}
  Chapter 02: {Chapter title}
  ...

Document 2: {Title}
  Chapter 01: {Chapter title}
  ...

## Your Output Directory

Write all journal files and your verdict to:
{output-dir}/{persona-slug}/

## Begin

Start your startup sequence now. Read process.md, then write your suite orientation entry.
```

After spawning all agents, enter the chapter pump.

---

## Phase 4: The Chapter Pump

This is the core runtime loop. Agents go idle after producing output, and their turn output is automatically delivered to you. Handle each message based on its pattern.

### Message type 1: Chapter request with Step A confirmation

**Agent says:** "Step A written at `{path}`. Requesting [Document Title, Chapter NN: Chapter Name]."

**You do:**

1. **Enforce the Step A gate.** Read the journal file at `{path}`. Verify that a Step A section exists and contains concrete expectations — not vague hedges, but a specific prediction about what the chapter will contain and what the persona needs from it. If Step A is missing or empty, send a message asking the agent to write or complete it. Do NOT provide the chapter file until Step A is verified.

2. **Look up the chapter.** Find the requested chapter in `manifest.json`. Get the hashed filename.

3. **Check re-anchoring schedule.** Is this agent due for voice re-anchoring? Track the number of chapters each agent has read. Every 3rd chapter (3, 6, 9, ...), include the voice re-anchoring payload.

4. **Check time budget reminder.** Every 3-4 chapters, include a one-line time reminder.

5. **Check context management.** If this agent has read 15 or more chapters, include the carry-forward payload.

6. **Send the message.** `SendMessage(to: "{agent-name}")` with:
   - The hashed chapter file path: "Read the chapter at `{output-dir}/.chapters/{hash}.md`"
   - Voice re-anchoring payload (if due — see section below)
   - Time budget reminder (if due — see section below)
   - Carry-forward payload (if needed — see section below)

7. **Update tracking state.** Increment chapter count. Add `[document, chapter]` to reading path trace.

### Message type 2: Document switch request

**Agent says:** "Switching to [Document Title]. Requesting table of contents."

**You do:**

1. **Always include voice re-anchoring.** A document switch triggers re-anchoring regardless of chapter count.
2. **Reset chapter count** for re-anchoring purposes. A switch starts a new 3-chapter cycle.
3. **Send the message.** `SendMessage(to: "{agent-name}")` with:
   - The table of contents for the requested document
   - Voice re-anchoring payload

### Message type 3: Return to abandoned document

**Agent says:** "Returning to [Document Title]. Where did I leave off?"

**You do:**

1. Check the reading path trace for this agent. Find the last chapter they read in that document.
2. **Always include voice re-anchoring.** A return to a previously abandoned document triggers re-anchoring — the agent's mental model for this document is stale after reading other material in between.
3. **Reset chapter count** for re-anchoring purposes. A return starts a new 3-chapter cycle for this document.
4. **Send the message.** `SendMessage(to: "{agent-name}")` with:
   - Reminder of where they left off: "You last read [Chapter NN: Chapter Name] in [Document Title]."
   - Table of contents for that document
   - Voice re-anchoring payload

### Message type 4: Reading complete

**Agent says:** "Reading complete. Verdict written at `{path}`."

**You do:**

1. Read the file to verify `overall-verdict.md` exists in the agent's output directory.
2. Send the shutdown acknowledgement: `SendMessage(to: "{agent-name}", message: "Your verdict has been received. Thank you for your reading. Shutting down.")`
3. Remove the agent from the active roster.
4. Update tracking state: set status to `done`, record `stopped_reason: persona_decision`.

### Message type 5: Agent failure

No response, malformed output, or context overflow.

**You do:**

1. Note the failure in tracking state.
2. Record `stopped_reason: agent_failure`.
3. Remove the agent from the active roster.
4. Do NOT retry. The panel continues with the remaining agents.

---

## Voice Re-anchoring Payload

Include this payload every 3rd chapter per agent, or on any document switch. Paste it into the `SendMessage`:

```
[Voice anchor — you are {Name/Role}. Your voice sounds like this:
"{Voice sample}"
Your consultation stance: "{Consultation voice}"
You do NOT know: {Blind spots — first 2-3 items}
You are trying to answer: "{Key question}"
Your emotional vocabulary includes: bored, vindicated, insulted, energised,
suspicious, relieved, impatient, impressed, defensive, confused, amused,
alarmed, indifferent, inspired, overwhelmed, curious — not just
"cautiously [adjective]."]
```

If no scenario framing is defined, omit the "Your consultation stance" line.

Fill in all values from the persona's specification in the config. The voice sample, consultation voice, blind spots, and key question come directly from the persona spec.

---

## Context Management: Carry-Forward Payload

When an agent has read 15 or more chapters, include carry-forward in the next chapter message. This preserves the emotional arc while freeing context for new content.

The carry-forward payload has two parts:

### Part 1: Recent journal paths

Provide paths to the agent's last 3 journal files so they can re-read them for voice and affect continuity:

```
For voice and affect continuity, re-read your last three journal entries:
- {output-dir}/{persona-slug}/{NN}-{slug}.md
- {output-dir}/{persona-slug}/{NN}-{slug}.md
- {output-dir}/{persona-slug}/{NN}-{slug}.md
```

### Part 2: Arc summary table

Build this table by reading the agent's earlier journal files and extracting the Emotional state, Cumulative sense, and any prominent concern from each entry:

```markdown
## Reading arc so far

| Chapter | Emotional state | Cumulative sense | Key concern |
|---|---|---|---|
| 00: Suite Orientation | Curious but wary | "This looks serious but I'm not sure it's for me" | Vocabulary load |
| 01: Front Matter | Engaged | "They understand the problem" | No practical guidance yet |
| 02: Executive Summary | Overwhelmed | "This is enormous — where do I start?" | Length |
| 03: Table of Contents | Impatient | "Just show me the relevant section" | Navigation |
| ... | ... | ... | ... |
```

The table covers all entries up to (but not including) the last 3 journal files. The agent gets recent entries in full and older entries as a compressed arc.

---

## Time Budget Tracking

After each agent writes `00-suite-orientation.md`, read it and extract the stated time budget from the "Time budget" field in Step C. Store this in the agent's tracking state.

Every 3-4 chapters, include a one-line reminder in the SendMessage:

```
You budgeted {X} for this suite. You've read {N} chapters so far.
```

This is a factual reminder, not pressure to stop. The agent decides when to stop based on their persona's reading behaviour.

---

## Coordinator State Tracking

Maintain per-agent state throughout the chapter pump.

| Field | Purpose |
|---|---|
| Agent name | Teammate name for `SendMessage` |
| Persona slug | Directory name for file paths |
| Chapters read count | For re-anchoring schedule (every 3rd) |
| Current document | For detecting document switches |
| Time budget | Extracted from suite orientation |
| Status | `active` / `done` / `failed` |
| Reading path trace | List of `[document, chapter]` tuples in reading order |

**For small panels (under ~10 personas):** maintain this as a mental table in conversation context.

**For large panels (13+ personas):** write tracking state to `{output-dir}/.tracking.json` and re-read it each turn to avoid context drift. Update the file after every agent interaction.

---

## Error Handling

### Step A missing or malformed

Send a message to the agent asking them to write or complete Step A before you provide the chapter. Example:

```
SendMessage(to: "{agent-name}", message: "Your journal entry at {path} is missing
the Step A expectations section. Write your expectations for this chapter before
I can provide the content. What do you expect this chapter to cover, and what
do you need from it?")
```

Do NOT provide the chapter file until Step A is verified.

### Chapter not found in manifest

If the agent requests a chapter that does not exist in `manifest.json`:

```
SendMessage(to: "{agent-name}", message: "Chapter '{requested chapter}' was not found
in the document suite. The available chapters for [Document Title] are:
{paste table of contents}
Please choose a different chapter or rephrase your request.")
```

### Agent failure

No response, malformed output, or context overflow. Note the failure, record `stopped_reason: agent_failure`, remove from active roster. Do NOT retry. The panel continues with remaining agents.

---

## Phase 5: Completion and Post-Reading

When all agents have shut down (active roster is empty), the reading phase is complete.

### Step 1: Optional collision tests

If the user wants collision tests (and the config defines collision pairings):

1. For each pair, read both personas' `overall-verdict.md` files to confirm they have opposed or divergent positions worth testing.
2. Spawn NEW `persona-reader` agents with `mode: bypassPermissions` (do NOT reuse terminated agents) with the same persona spec plus a collision prompt. The collision prompt provides:
   - The path to the opponent's verdict file
   - Instructions to read the opponent's verdict and write a collision reaction (see `process.md` Phase 3b for the format)
   - The output path: `{output-dir}/{persona-slug}/collision-{other-persona-slug}.md`
3. Each collision agent reads the opponent verdict, writes the reaction, and terminates.

### Step 2: Spawn the panel-synthesiser

Use `subagent_type: "muna-panel-review:panel-synthesiser"` with `mode: bypassPermissions`. The spawn prompt provides:
- The output directory path (so the synthesiser can discover all persona outputs)
- The absolute path to `process.md` (so the synthesiser can read Phase 4 methodology)
- The absolute path to the config file (so the synthesiser can read persona specs for persona-aware analysis and panel gap identification)

The synthesiser reads the config, all journal files and verdicts, builds collation documents, writes the synthesis in three analytical passes (now 13 sections, including Section 13: Panel Gaps and Suggested Personas), applies the quality gate, and writes the process manifest. It is fire-and-forget — it does not interact with you during synthesis.

### Step 3: Present results

When the synthesiser returns:
1. Read `{output-dir}/00-reader-panel-synthesis.md` and present the key findings to the user. Highlight:
   - The executive summary
   - Universal findings (what everyone agreed on)
   - Surprising findings
   - Top recommended actions
2. Note the process manifest at `{output-dir}/00-process-manifest.yaml`.
3. Report the control persona comparison: did actual reading match pre-registered predictions?

### Step 4: Cleanup

Use `TeamDelete` to remove the team.

---

## Authority Split

- **This skill** is the complete authority for coordinator behaviour: config parsing, working directory setup, team management, chapter pump mechanics, state tracking, error handling, and post-reading orchestration.
- **`process.md`** is the authority for methodology: journal format, voice guidelines, reading path rules, verdict format, synthesis structure. Agents read it at runtime.
- The **voice re-anchoring payload** and **carry-forward format** (above) are the only methodology elements duplicated in this skill, because the coordinator needs them to construct messages.
- Do NOT duplicate journal format templates, verdict structure, or synthesis templates in coordinator messages. Agents read those from `process.md`.
