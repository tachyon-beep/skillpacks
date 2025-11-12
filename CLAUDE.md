# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **Skillpacks Marketplace** - a modular collection of 15 professional skillpacks providing 135 production-ready skills for Claude Code across AI/ML, Python engineering, game development, security, documentation, and UX design.

**Status**: v1.2.0 - Production ready, CC BY-SA 4.0 licensed, publicly available

## Architecture

### Marketplace Structure

```plaintext
skillpacks/
├── .claude-plugin/
│   └── marketplace.json          # Marketplace catalog defining all 15 plugins
├── plugins/                       # 15 independent plugin directories
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

1. **Python Engineering (Axiom faction)** - 2 plugins, 15 skills
   - `axiom-python-engineering` (10 skills)
   - `axiom-system-archaeologist` (5 skills - router + specialists)

2. **AI/ML (Yzmir faction)** - 8 plugins, 70 skills
   - `yzmir-ai-engineering-expert` (router)
   - `yzmir-pytorch-engineering`
   - `yzmir-training-optimization`
   - `yzmir-deep-rl`
   - `yzmir-neural-architectures`
   - `yzmir-llm-specialist`
   - `yzmir-ml-production`
   - `yzmir-simulation-foundations`

3. **Game Development (Bravos faction)** - 2 plugins, 20 skills
   - `bravos-simulation-tactics`
   - `bravos-systems-as-experience`

4. **UX Design (Lyra faction)** - 1 plugin, 11 skills
   - `lyra-ux-designer`

5. **Security (Ordis faction)** - 1 plugin, 9 skills
   - `ordis-security-architect`

6. **Documentation (Muna faction)** - 1 plugin, 9 skills
   - `muna-technical-writer`

### Slash Commands (Router Skills)

**IMPORTANT**: All router skills (`using-X` skills) are available as slash commands in `.claude/commands/` due to skill context limits.

To use a router skill, invoke it as a slash command:

```
/ai-engineering      # Routes to AI/ML skills
/system-archaeologist # Routes to architecture analysis
/deep-rl            # Routes to RL algorithms
/python-engineering  # Routes to Python skills
```

**Why slash commands?** Router skills exceeded the context budget for skill discovery. Slash commands provide:
- No context limits
- Explicit user invocation
- Faster loading
- Better control flow

See [.claude/SLASH_COMMANDS.md](.claude/SLASH_COMMANDS.md) for complete list of all 15 router commands.

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
- Similar routers exist for all 15 plugins (see `.claude/SLASH_COMMANDS.md` for complete list)

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

The marketplace catalog (`.claude-plugin/marketplace.json`) coordinates all 13 plugins:

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

**Total**: 201+ production files

- 135 skill files (SKILL.md)
- 15 plugin metadata files (plugin.json)
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
find plugins -name "SKILL.md" | wc -l  # Should be 135
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
