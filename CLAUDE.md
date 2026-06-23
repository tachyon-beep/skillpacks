# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **Skillpacks Marketplace** - a modular collection of 45 professional skillpacks (56 router/skill files plus 430+ reference sheets, 110+ commands, 90+ agents) for Claude Code across AI/ML, Python & Rust engineering, web backend, DevOps, SDLC and program/product management, solution architecture, game development, security, documentation, and UX design.

**Status**: Marketplace v3.22.0 - Production ready, CC BY-SA 4.0 licensed, publicly available

## Architecture

### Marketplace Structure

```plaintext
skillpacks/
├── .claude-plugin/
│   └── marketplace.json          # Marketplace catalog defining all 45 plugins
├── plugins/                       # 45 independent plugin directories
│   ├── [plugin-name]/
│   │   ├── .claude-plugin/
│   │   │   └── plugin.json       # Plugin metadata (name, version, description)
│   │   └── skills/
│   │       └── [skill-name]/
│   │           └── SKILL.md      # Skill implementation
└── docs/
    └── future-*.md               # Future planning docs
```

### Plugin Categories

Packs are grouped by faction prefix. Most modern packs follow a **router + reference-sheets** shape (a `using-X` router `SKILL.md` plus numbered sheets, slash commands, and SME agents), so the count that matters per pack is sheets/commands/agents rather than "skills". See `FACTIONS.md` for the thematic per-pack catalog and `.claude-plugin/marketplace.json` for the canonical list.

1. **Engineering & process (Axiom faction)** - 20 plugins
   - Python/Rust: `axiom-python-engineering`, `axiom-rust-engineering`, `axiom-rust-workspaces`, `axiom-pyo3-interop`
   - Backend & data: `axiom-web-backend`, `axiom-embedded-database`, `axiom-mcp-engineering`
   - Architecture & analysis: `axiom-system-architect`, `axiom-system-archaeologist`, `axiom-solution-architect`, `axiom-procedural-architecture`
   - Process & delivery: `axiom-engineering-foundations`, `axiom-planning`, `axiom-sdlc-engineering`, `axiom-program-management`, `axiom-product-management`, `axiom-devops-engineering`
   - Specialized discipline: `axiom-static-analysis-engineering`, `axiom-determinism-and-replay`, `axiom-audit-pipelines`

2. **AI/ML (Yzmir faction)** - 11 plugins
   - `yzmir-ai-engineering-expert` (router), `yzmir-pytorch-engineering`, `yzmir-training-optimization`, `yzmir-deep-rl`, `yzmir-neural-architectures`, `yzmir-llm-specialist`, `yzmir-ml-production`, `yzmir-simulation-foundations`, `yzmir-dynamic-architectures`, `yzmir-morphogenetic-rl`, `yzmir-systems-thinking`

3. **Game Development (Bravos faction)** - 2 plugins
   - `bravos-simulation-tactics`, `bravos-systems-as-experience`

4. **Design & creative (Lyra faction)** - 4 plugins
   - `lyra-ux-designer`, `lyra-site-designer`, `lyra-tui-designer`, `lyra-creative-writing`

5. **Documentation (Muna faction)** - 4 plugins
   - `muna-technical-writer`, `muna-document-designer`, `muna-wiki-management`, `muna-panel-review`

6. **Security & quality (Ordis faction)** - 2 plugins
   - `ordis-security-architect`, `ordis-quality-engineering`

7. **Meta (cross-faction)** - 2 plugins
   - `meta-sme-protocol`, `meta-skillpack-maintenance`

### Slash Commands (Router Skills)

**IMPORTANT**: All router skills (`using-X` skills) are available as slash commands in `.claude/commands/` due to skill context limits.

To use a router skill, invoke it as a slash command:

```
/ai-engineering      # Routes to AI/ML skills
/system-archaeologist # Routes to architecture analysis
/solution-architect  # Routes to forward solution design
/deep-rl            # Routes to RL algorithms
/python-engineering  # Routes to Python skills
```

**Why slash commands?** Router skills exceeded the context budget for skill discovery. Slash commands provide:
- No context limits
- Explicit user invocation
- Faster loading
- Better control flow

See [.claude/SLASH_COMMANDS.md](.claude/SLASH_COMMANDS.md) for the complete list of router commands.

### Skill File Format

Each `SKILL.md` follows this structure:

- **Front matter**: YAML with `name` and `description`
- **Content**: Expert-level guidance with examples, patterns, and anti-patterns
- Skills range from 200-2000 lines of production-ready content

### Router Patterns

Several plugins use "using-X" router skills that direct users to appropriate specialized skills. **These are now available as slash commands** (see Slash Commands section above):

- `axiom-system-archaeologist/using-system-archaeologist/SKILL.md` → `/system-archaeologist` - Routes to architecture analysis specialists
- `yzmir-ai-engineering-expert/using-ai-engineering/SKILL.md` → `/ai-engineering` - Routes to all AI/ML packs
- `yzmir-deep-rl/using-deep-rl/SKILL.md` → `/deep-rl` - Routes to 12 RL algorithm skills
- Similar routers exist for most plugins (see `.claude/SLASH_COMMANDS.md` for complete list)

## Installation & Testing

### Add to Claude Code

```bash
# Add marketplace
/plugin marketplace add tachyon-beep/skillpacks

# Browse available
/plugin

# Install specific pack
/plugin install yzmir-deep-rl
```

### Development Testing

When testing changes to skills:

1. Make edits to `plugins/[plugin-name]/skills/[skill-name]/SKILL.md`
2. If plugin is installed, changes take effect immediately
3. Test by asking Claude to use the skill

### Local Development Installation

```bash
# From skillpacks directory
/plugin marketplace add .
```

## Version Management

### Plugin Versioning

Each plugin has independent versioning in `.claude-plugin/plugin.json`:

```json
{
  "name": "yzmir-deep-rl",
  "version": "1.0.0",
  "description": "...",
  "category": "ai-ml"
}
```

### Marketplace Versioning

The marketplace catalog (`.claude-plugin/marketplace.json`) coordinates all 45 plugins:

- Lists all plugins with their source paths
- Maintains marketplace metadata (version, homepage)
- Uses `"pluginRoot": "./plugins"` to locate plugin directories

## Working With This Repository

### Adding a New Skill

1. Navigate to appropriate plugin: `plugins/[plugin-name]/skills/`
2. Create directory: `mkdir new-skill-name`
3. Create skill file: `new-skill-name/SKILL.md`
4. Add YAML front matter:

   ```yaml
   ---
   name: new-skill-name
   description: Brief description for skill discovery
   ---
   ```

5. Write skill content following existing skill patterns
6. Test with Claude Code

### Creating a New Plugin

1. Create directory: `plugins/new-plugin-name/`
2. Create metadata: `plugins/new-plugin-name/.claude-plugin/plugin.json`
3. Add skills directory: `plugins/new-plugin-name/skills/`
4. Register in marketplace: Add entry to `.claude-plugin/marketplace.json`
5. Update README.md with new plugin

### Testing Strategy

This repository uses a unique testing methodology:

- **RED-GREEN-REFACTOR pattern** applied to skills (process documentation)
- Test scenarios verify skills guide Claude correctly
- Historical test artifacts removed in v1.0.0 cleanup
- New tests should validate skill effectiveness in real usage

## Key Design Principles

### Modularity

- Each plugin is independently installable
- No cross-plugin dependencies in core functionality
- Router skills guide users to appropriate packs

### Faction Organization

- **Yzmir** (AI/ML): Mathematical, systematic, optimization-focused
- **Bravos** (Game Dev): Emergent systems, simulation, player-driven
- **Lyra** (UX): User-centered, accessible, experience-driven
- **Ordis** (Security): Compliance, threat modeling, defense
- **Muna** (Documentation): Clarity, structure, maintainability

### Skill Quality Standards

- Production-ready: Real implementations, not tutorials
- Expert-level: Assumes competence, teaches mastery
- Pattern-focused: Reusable approaches, not one-off solutions
- Anti-pattern aware: Explicitly warns against common mistakes

## File Manifest

**Total**: 700+ production files

- 56 router/skill files (SKILL.md) + 430+ reference sheets
- 110+ slash commands, 90+ SME agents
- 45 plugin metadata files (plugin.json)
- 1 marketplace catalog (marketplace.json)
- Core documentation (README, LICENSE, CLAUDE.md, FACTIONS.md, CONTRIBUTING.md, LICENSE_ADDENDUM.md)
- TDD artifacts (test scenarios, baseline results, methodology documentation)

## Git Workflow

### Branches

- `main` - Production-ready code only
- Feature branches for new skills/plugins

### Commits

Use conventional commits:

- `feat:` - New skills or plugins
- `fix:` - Skill corrections or improvements
- `docs:` - Documentation updates
- `chore:` - Repository maintenance

### Worktrees

Repository uses git worktrees (`.worktrees/` is gitignored):

```bash
git worktree add .worktrees/feature-name -b feature-name
```

## Common Operations

### Update a Skill

```bash
# Edit skill
vim plugins/yzmir-deep-rl/skills/policy-gradient-methods/SKILL.md

# Test with Claude Code (if plugin installed)
# Changes are live immediately
```

### Check Plugin Status

```bash
# List all plugins
ls plugins/

# Count skills in a plugin
find plugins/yzmir-deep-rl/skills -name "SKILL.md" | wc -l

# View plugin metadata
cat plugins/yzmir-deep-rl/.claude-plugin/plugin.json
```

### Validate Marketplace Structure

```bash
# Verify all plugins are registered
cat .claude-plugin/marketplace.json | grep '"name":'

# Check for missing plugin.json files
for dir in plugins/*/; do
  [ -f "$dir/.claude-plugin/plugin.json" ] || echo "Missing: $dir"
done

# Count total skills
find plugins -name "SKILL.md" | wc -l  # Should be 56
```

## Important Notes

### Repository History

- v1.2.0 (2025-11-12): Added axiom-system-archaeologist plugin (5 skills for architecture analysis)
- v1.1.0 (2025-11-10): Added axiom-python-engineering plugin (10 skills)
- v1.0.0 (2025-10-31): Public release with 513 internal files removed
- All removed files preserved in git history
- Original source structure backed up in commits

### Skill Validation

- Skills have been systematically validated through RED-GREEN-REFACTOR testing
- Historical validation artifacts removed for clean public release
- Each skill represents tested, production-ready guidance

### License

CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International) - See LICENSE for details

**Important**: Faction names (Axiom, Bravos, Lyra, Muna, Ordis, Yzmir) from Altered TCG are NOT covered by this license - see LICENSE_ADDENDUM.md for details

### Contributing

See source/CONTRIBUTING.md for contribution guidelines

## Quick Reference

```bash
# Install marketplace
/plugin marketplace add tachyon-beep/skillpacks

# Count skills
find plugins -name "SKILL.md" | wc -l

# List all plugins
ls plugins/

# View marketplace catalog
cat .claude-plugin/marketplace.json

# Check plugin versions
grep -r "version" plugins/*/.claude-plugin/plugin.json
```

<!-- filigree:instructions:v3.0.1:65e6fb25 -->
<!-- filigree:last-writer:filigree install -->
## Filigree Issue Tracker

`filigree` tracks tasks for this project. Data lives in `.filigree/`. Prefer
the MCP tools (`mcp__filigree__*`) when available; fall back to the `filigree`
CLI otherwise.

### Workflow

```bash
# At session start
filigree session-context                            # ready / in-progress / critical path

# Pick up the next startable issue (atomic claim + transition into its working status)
filigree start-next-work --assignee <name>
# ...or claim a specific issue
filigree start-work <id> --assignee <name>

# Do the work, commit, then
filigree close <id>
```

Use the atomic claim+transition verbs — `work_start` / `work_start_next`
(MCP) or `start-work` / `start-next-work` (CLI). Do **not** chain
`work_claim` (MCP) or `filigree claim` (CLI) with a subsequent status
update — the two-step form races against other agents; the combined verb is
atomic.

**Ready ≠ startable.** The working status is type-specific (tasks →
`in_progress`, features → `building`). Bugs start at `triage`, which has no
single-hop transition into work (`triage → confirmed → fixing`), so a triage
bug is *ready* but not directly *startable*: `work_start` on one returns
`INVALID_TRANSITION` naming the next status, and `work_start_next` skips it.
`work_ready` items carry a `startable` flag (plus a `next_action` hint when
false). Pass `advance=true` (MCP) / `--advance` (CLI) to walk the soft
transitions to the nearest working status automatically.

### Observations: when (and when not) to use them

`observation_create` is a fire-and-forget scratchpad for *incidental* defects — things
you notice *outside the scope of your current task* (a code smell in a
neighbouring file, a stale TODO, a missing test for an edge case you happened
to spot). Notes expire after 14 days unless promoted. Include `file_path` and
`line` when relevant. At session end, skim `observation_list` and either
`observation_dismiss` or `observation_promote` for what has accumulated.

**You fix bugs in your currently defined scope. You do NOT use observations
to finish work prematurely.** If a defect, gap, or follow-up belongs to your
current task, you own it — handle it as part of that task: fix it now, expand
the task's scope, file a proper issue with a dependency, or surface it to the
user. Filing it as an observation and closing the task is *not* completing
the task; it is shipping known-broken work and hiding the debt in a 14-day
expiring scratchpad. The test is "would I have noticed this even if I weren't
working on this task?" If no, it's task scope, not an observation.

### Priority scale

- P0: Critical (drop everything)
- P1: High (do next)
- P2: Medium (default)
- P3: Low
- P4: Backlog

### Reaching for tools

MCP tool schemas describe each tool; `filigree --help` and `filigree <verb>
--help` are the authoritative CLI reference. You do not need to memorise
either catalogue. The verbs you will reach for most:

- **Find work:** `work_ready`, `work_blocked`, `issue_list`, `issue_search`
- **Claim work:** `work_start`, `work_start_next`
- **Update:** `comment_add`, `label_add`, `issue_update`, `issue_close`
- **Admin (irreversible):** `issue_delete` (MCP) / `delete-issue` (CLI) —
  hard-deletes a terminal issue and its rows; `admin_undo_last` cannot reverse it.
- **Scratchpad:** `observation_create`, `observation_list`, `observation_promote`, `observation_dismiss`
- **Cross-product entity bindings (ADR-029):** `entity_association_add`,
  `entity_association_remove`, `entity_association_list`,
  `entity_association_list_by_entity`. Used when a sibling tool (e.g.
  Loomweave) needs to bind a Filigree issue to a function, class, or
  module identifier it owns. The `entity_id` is an opaque external string
  from Filigree's perspective and may be a `loomweave:eid:...` SEI or a legacy
  locator; callers may also supply `entity_kind` explicitly. The consumer (the sibling tool's read
  path) does drift detection against the stored
  `content_hash_at_attach`. `entity_association_list_by_entity` is the
  reverse-lookup surface — given an opaque external entity ID, return every
  Filigree issue bound to it (project isolation is by DB file). Also
  reachable over HTTP as
  `GET/POST /api/issue/{issue_id}/entity-associations`,
  `DELETE /api/issue/{issue_id}/entity-associations?entity_id=…`,
  and `GET /api/entity-associations?entity_id=…`.
- **Health:** `stats_get`, `metrics_get`, `mcp_status_get`

Pass `--actor <name>` (CLI) so events attribute to your agent identity. It
works in either position — before the verb (`filigree --actor X update …`) or
after it (`filigree update … --actor X`); the post-verb value overrides the
group-level one.

### Error handling

Errors return `{error: str, code: ErrorCode, details?: dict}`. Switch on
`code`, not on message text. Codes: `VALIDATION`, `NOT_FOUND`, `CONFLICT`,
`INVALID_TRANSITION`, `PERMISSION`, `NOT_INITIALIZED`, `IO`,
`INVALID_API_URL`, `FILE_REGISTRY_DISPLACED`, `REGISTRY_UNAVAILABLE`,
`LOOMWEAVE_REGISTRY_VERSION_MISMATCH`, `LOOMWEAVE_OUT_OF_SYNC`,
`BRIEFING_BLOCKED`, `STOP_FAILED`, `SCHEMA_MISMATCH`, `INTERNAL`.

On `INVALID_TRANSITION`, call `workflow_transition_list` (MCP) or
`filigree transitions <id>` to see what the workflow allows from here.

Two failure modes deserve a specific response:

- **`SCHEMA_MISMATCH`** — the installed `filigree` is older than the project
  database. The error message contains upgrade guidance. Surface it to the
  user; do not retry.
- **`ForeignDatabaseError`** — filigree found a parent project's database
  but no local `.filigree.conf`. Run `filigree init` in the current
  directory. Do **not** `cd` upward to a different project unless that was
  the actual intent.
<!-- /filigree:instructions -->
