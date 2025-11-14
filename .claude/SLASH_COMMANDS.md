# Slash Commands Reference

This document maps the router skills to their slash command equivalents.

## Why Slash Commands?

Router skills were converted to slash commands due to context limit issues. Skills have a limited context budget, and comprehensive router skills exceeded this limit. Slash commands are user-invoked (explicit) rather than model-invoked (automatic), which works better for navigation/routing scenarios.

## Available Slash Commands

All router skills from the 18 plugins are now available as slash commands:

### Python Engineering (Axiom)
- **`/python-engineering`** - Routes to Python expertise (testing, packaging, async, performance, etc.)

### Architecture Analysis (Axiom)
- **`/system-archaeologist`** - Routes to codebase architecture analysis and documentation
- **`/system-architect`** - Routes to architectural assessment and technical debt analysis

### Web Backend Development (Axiom)
- **`/web-backend`** - Routes to web backend development (FastAPI, Django, Express, APIs, microservices)

### Game Development (Bravos)
- **`/simulation-tactics`** - Routes to simulation and game development tactics
- **`/systems-as-experience`** - Routes to game systems design and player experience

### UX Design (Lyra)
- **`/ux-designer`** - Routes to UX design, accessibility, and user research

### Documentation (Muna)
- **`/technical-writer`** - Routes to technical documentation and API docs

### Security (Ordis)
- **`/security-architect`** - Routes to security architecture and threat modeling

### AI/ML Engineering (Yzmir)
- **`/ai-engineering`** - Master router for all AI/ML engineering tasks
- **`/pytorch-engineering`** - Routes to PyTorch-specific skills
- **`/training-optimization`** - Routes to model training and optimization
- **`/deep-rl`** - Routes to deep reinforcement learning algorithms
- **`/llm-specialist`** - Routes to LLM fine-tuning and deployment
- **`/neural-architectures`** - Routes to neural architecture selection
- **`/ml-production`** - Routes to ML deployment and production
- **`/simulation-foundations`** - Routes to simulation fundamentals
- **`/systems-thinking`** - Routes to systems thinking methodology and modeling

## Usage

Simply type the slash command in Claude Code to load the router skill:

```
/ai-engineering
```

The router will then guide you to the appropriate specialized skill for your task.

## Mapping Table

| Plugin | Original Skill | Slash Command |
|--------|----------------|---------------|
| axiom-python-engineering | using-python-engineering | /python-engineering |
| axiom-system-archaeologist | using-system-archaeologist | /system-archaeologist |
| axiom-system-architect | using-system-architect | /system-architect |
| axiom-web-backend | using-web-backend | /web-backend |
| bravos-simulation-tactics | using-simulation-tactics | /simulation-tactics |
| bravos-systems-as-experience | using-systems-as-experience | /systems-as-experience |
| lyra-ux-designer | using-ux-designer | /ux-designer |
| muna-technical-writer | using-technical-writer | /technical-writer |
| ordis-security-architect | using-security-architect | /security-architect |
| yzmir-ai-engineering-expert | using-ai-engineering | /ai-engineering |
| yzmir-deep-rl | using-deep-rl | /deep-rl |
| yzmir-llm-specialist | using-llm-specialist | /llm-specialist |
| yzmir-ml-production | using-ml-production | /ml-production |
| yzmir-neural-architectures | using-neural-architectures | /neural-architectures |
| yzmir-pytorch-engineering | using-pytorch-engineering | /pytorch-engineering |
| yzmir-simulation-foundations | using-simulation-foundations | /simulation-foundations |
| yzmir-systems-thinking | using-systems-thinking | /systems-thinking |
| yzmir-training-optimization | using-training-optimization | /training-optimization |

## Implementation Details

- **Source**: Router skills from `plugins/*/skills/using-*/SKILL.md`
- **Target**: Slash commands in `.claude/commands/*.md`
- **Conversion**: YAML frontmatter removed, content preserved
- **Naming**: "using-" prefix removed from skill names

## Benefits

1. **No context limits** - Slash commands don't count against skill discovery context
2. **Explicit invocation** - User controls when routers are loaded
3. **Faster discovery** - No need for Claude to scan descriptions
4. **Cleaner workflow** - Router → specialized skill is now explicit

## Maintenance

To update a slash command:
1. Edit the source skill: `plugins/[plugin]/skills/using-[name]/SKILL.md`
2. Re-run conversion script: `./convert-routers-to-commands.sh`
3. Slash command automatically updates

Or edit the slash command directly in `.claude/commands/[name].md`
