# Maintainer instructions

This repository publishes 48 independent plugin packs with 54 `SKILL.md` entrypoints, 530 optional reference sheets, 141 plugin commands and 113 agents. Maintenance fixtures and operational `.agents`/`.claude/skills` are excluded from these counts. Marketplace version: 4.0.0.

## Source and packaging

- `.claude-plugin/marketplace.json` is the install catalog; each listed source must match `plugins/<name>/.claude-plugin/plugin.json`.
- `plugins/<pack>/skills/<name>/SKILL.md` owns activation and the small task contract. Sibling references are optional, selected for a concrete uncertainty.
- Plugin `commands/` and `agents/` must work when that pack is installed alone. Inline the concise evidence rules they need; do not depend on repository-only docs or an uninstalled protocol plugin.
- `.claude/commands/` contains thin repository shortcuts. Do not duplicate runtime workflows there.
- README and FACTIONS list actual installable packs. `reviews/` contains dated historical evidence, not current certification.

## Editing and verification

Read [CONTRIBUTING.md](CONTRIBUTING.md), the applicable pack and affected callers before editing. Preserve unrelated dirty work. Honor current user authorization and project policy; do not add approval gates inferred from examples. No fixed reviewer count, minimum duration, finding quota or compulsory router invocation applies.

Retain specialist invariants and useful output fields. Remove duplicate tutorials and unsupported universal rules. Scope references to the actual runtime, authority and artifact contract. A clean review is valid with coverage and gaps; unknown is not verified.

Run `python3 scripts/check_marketplace_integrity.py`, `python3 scripts/check_skillpack_contracts.py` and `git diff --check`. Use focused executable checks where a recipe changed. For model behavior claims, compare outcomes and cost against a baseline; adherence to the skill alone is not proof of utility. Report checks not run and limitations.

## Version and Git policy

Use a feature branch for changes, conventional commit messages and explicit staging that preserves unrelated files. Bump each affected plugin according to compatibility; update marketplace metadata with catalog-wide changes. Major version 4.0.0 removes four packs and three SDLC leaves; see [the consolidation record](docs/relevance-refresh.md).

<!-- filigree:instructions:v3.1.0:c1c023c3 -->
<!-- filigree:last-writer:filigree install -->
## Filigree Issue Tracker

`filigree` tracks this project's work. Use it to find, claim, update and close
issues: `filigree session-context` at session start, then
`filigree start-next-work --assignee <name>`.

Full reference: the **filigree-workflow** skill (patterns, priorities,
observations, error codes), `filigree --help`, and the `mcp__filigree__*` tool
schemas. Prefer the MCP tools when available; fall back to the CLI.

Two rules `--help` will not tell you:

1. Claim atomically: `work_start` / `work_start_next` (MCP) or `start-work` /
   `start-next-work` (CLI). Never chain a claim with a separate status update;
   that two-step form races other agents.
2. On `SCHEMA_MISMATCH` the installed filigree is older than the project
   database. Surface it to the user; do not retry.
<!-- /filigree:instructions -->
