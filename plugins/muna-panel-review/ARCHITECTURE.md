# muna-panel-review — Architecture

## Purpose

Simulate a diverse panel of readers — each representing a distinct position in a document's potential audience — reading a document chapter by chapter and recording their reactions as mood journals. The primary output is **editorial intelligence**: how the document lands with different audiences, what it communicates vs what it intends, where it loses readers, and what derivative documents the audience actually needs.

This is audience simulation for editorial improvement. Not expert technical review — that's a separate concern.

## Component Architecture

### Agents (3)

#### 1. `persona-reader`

**Role:** A configurable reader agent. Given a persona specification (inline in spawn prompt) and a reference to process.md, reads chapter by chapter writing mood journals, then produces an overall verdict.

**Model:** sonnet (cost-driven — a 13-persona panel generates hundreds of agent turns).

**Tools:** `Read`, `Write` only. No `Glob`, no `Bash`, no `SendMessage`.

**Prompt:** Thin — identity, operating rules, shutdown handling, journal memory rules, tool restriction reassurance. Methodology (journal format, voice guidelines, verdict format) is read from process.md at runtime.

**How it works:**
- Spawned as a teammate by the coordinator, with `name` set to the persona's directory slug (e.g., `"ses"`, `"junior-dev"`)
- At startup: reads process.md Phases 2-3, reads inline persona spec, writes suite orientation, requests first chapter
- Enforces the Step A gate: writes expectations BEFORE reading each chapter, then goes idle
- Coordinator verifies Step A exists, provides the chapter file path via `SendMessage`
- Agent reads chapter, writes Step B + C, saves journal entry, requests next chapter or declares done
- Exercises full navigation autonomy: skip, jump, switch documents, abandon and return, stop early
- Cannot discover chapter files — filenames are hashed; agent must request by document name and chapter title
- Produces overall verdict in the persona's professional format before declaring done
- Can be re-engaged for collision tests (new agent instance with same persona spec + collision prompt)

**No-read-ahead enforcement:** Hashed filenames + no Glob + no Bash = structurally impossible for agents to browse ahead. The coordinator is the only entity that knows the filename mapping.

#### 2. `persona-designer`

**Role:** Audience analyst that reads a document suite cold and produces a complete panel review config file.

**Model:** sonnet (cost-driven — structured analysis and output, not cross-persona reasoning).

**Tools:** `Read`, `Write`, `Glob`.

**Execution:** Fire-and-forget. Gets document paths, reads everything freely (no-read-ahead does not apply — this is an analyst, not a reader), produces a config file, returns.

**What it does:**
- Reads all documents in the suite to identify audiences, decision chains, institutional perspectives
- Designs personas following process.md Phase 1 principles
- Produces a complete config file with document suite, scenario framing, persona specs, panel configuration
- Documents panel gaps: audiences identified but not included, with reasoning
- Contamination constraint: can define who a persona *is* but must not predict how they would react to specific chapters (reading behaviour is a character trait, not a content-informed route)

#### 3. `panel-synthesiser`

**Role:** Reads all journal outputs and verdicts across personas, produces the cross-panel synthesis.

**Model:** opus (quality-driven — cross-panel pattern recognition, epistemic tier grading, convergent-reason testing).

**Tools:** `Read`, `Write`, `Glob`.

**Execution:** Fire-and-forget. Gets the output directory path, discovers all journal files via Glob, produces synthesis + process manifest, returns. No pump loop needed.

**What it does:**
- Reads the config file for persona-aware analysis (evaluating whether reactions are characteristic of each persona's declared lens and blind spots)
- Collates persona outputs into cross-cutting views (reading path map, per-chapter index, keyword/theme extraction)
- Three analytical passes: reading path analysis, per-chapter cross-persona comparison, thematic aggregation
- Grades findings by epistemic confidence (Tier 1: text-surface, Tier 2: affective, Tier 3: institutional)
- Tests for convergent reasons, not just convergent conclusions (detects model-prior echo)
- Generates derivative document specifications from audience gap analysis
- Identifies panel coverage gaps and suggests additional personas (Section 13 — identity fields only, no behavioural predictions due to contamination constraint)
- Produces process checksum manifest (YAML)

### Skills (1)

#### `reader-panel-review` (orchestration skill)

**Role:** Teaches the primary Claude instance (the **coordinator**) how to run a panel review using the team model.

**What it covers:**
- Optional Phase 0: spawn persona-designer to create a config file from documents (if user doesn't have one)
- Config parsing: read user's config file, extract document suite, scenario framing, persona specs
- Working directory setup: create output dirs, copy chapters to `.chapters/` with hashed filenames, write manifest.json
- Team creation: `TeamCreate`, spawn N persona-reader agents as teammates
- Chapter pump: async event-driven message handling — verify Step A gate, look up hashed filename, send chapter path via `SendMessage`, handle document switches and returns
- Voice re-anchoring: every 3rd chapter per agent (or on document switch), coordinator includes re-anchor payload
- Context management: carry-forward payloads for long sessions (15+ chapters)
- Time budget tracking: extract from suite orientation, remind agents periodically
- Individual dropout: agents finish independently, coordinator sends shutdown requests, shrinks active roster
- Completion: optional collision tests (new agent instances), spawn panel-synthesiser, present synthesis, `TeamDelete`
- Error handling: failed agents removed from roster, panel continues with remaining agents

**Authority split:** The skill owns coordinator behaviour (setup, pump, state tracking). process.md owns methodology (journal format, voice guidelines, synthesis structure). Agents read process.md; the coordinator does not.

## Workflow

```
User provides config file + document paths
(or: user provides document paths → persona-designer produces config)
  │
  ▼
Coordinator (primary instance, using reader-panel-review skill)
  │
  ├─ Parse config: extract personas, documents, scenario framing
  │
  ├─ Setup: create output dirs, hash chapter files → .chapters/
  │
  ├─ TeamCreate, spawn N persona-readers as teammates
  │    ├─ persona-reader ("ses")
  │    ├─ persona-reader ("ciso")
  │    ├─ persona-reader ("junior-dev")
  │    └─ ... (one per persona in config)
  │
  ├─ Chapter pump (async, event-driven, individual dropout)
  │    ├─ Agent goes idle with chapter request
  │    ├─ Coordinator verifies Step A, sends chapter path
  │    ├─ Agent reads, journals, requests next (or declares done)
  │    └─ Repeat until all agents shut down
  │
  ├─ Optional: collision tests (spawn new agent instances)
  │
  ├─ Spawn panel-synthesiser (fire-and-forget)
  │    └─ Returns: synthesis document + process manifest
  │
  └─ Present synthesis → TeamDelete
```

## Output Directory Structure

```
{output-dir}/
├── .chapters/                    # Hashed chapter files (coordinator's working copy)
│   ├── manifest.json             # Mapping: original filename → hashed filename
│   ├── a7f3b2.md
│   ├── c9d1e4.md
│   └── ...
├── ses/
│   ├── 00-suite-orientation.md
│   ├── 01-executive-summary.md
│   ├── ...
│   ├── overall-verdict.md
│   └── collision-vendor-lead.md  (if collision tests run)
├── ciso/
│   └── ...
├── junior-dev/
│   └── ...
├── 00-reader-panel-synthesis.md
└── 00-process-manifest.yaml
```

## File Inventory

```
plugins/muna-panel-review/
├── .claude-plugin/
│   └── plugin.json
├── process.md                    # Full methodology (agents read at runtime)
├── config.md                     # Worked 13-persona example
├── config-template.md            # Config format reference
├── ARCHITECTURE.md               # This file
├── skills/
│   └── reader-panel-review/
│       └── SKILL.md              # Orchestration skill
└── agents/
    ├── persona-reader.md          # Configurable reader agent
    ├── persona-designer.md        # Cold-start panel designer
    └── panel-synthesiser.md       # Synthesis agent
```
