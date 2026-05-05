# Refresh: meta-skillpack-maintenance

**Verdict:** MEDIUM / M effort. Do this **second** (after meta-sme-protocol) — this pack governs maintenance of all 32 plugins, so its frontmatter examples set the convention downstream refreshes will copy.

## Context

- Pack path: `/home/john/skillpacks/plugins/meta-skillpack-maintenance/`
- Full review: `/tmp/skillpack-refresh-review/meta-skillpack-maintenance.md`
- Purpose: workflow + reference sheets for maintaining other Claude Code plugins (skills, commands, agents, hooks).

## Why refresh

Architecturally sound (v2.0.1, recent refactor, good pressure-resistance language) but the frontmatter examples have drifted from the repo's own evolved conventions — and inaccuracies in a meta-pack propagate to all 32 plugins it governs.

Specific drifts:
- Agent example frontmatter includes a `tools:` key that real repo agents don't use. Verify with `grep -l "^tools:" /home/john/skillpacks/plugins/*/agents/*.md`.
- Command example frontmatter uses unquoted YAML arrays for `allowed-tools`, but every actual command in the repo uses quoted JSON-style arrays. Verify with `head -10 /home/john/skillpacks/plugins/*/commands/*.md`.
- Pack does not reference the SME Agent Protocol (`meta-sme-protocol`) that real repo SME agents now follow.
- Misses the `.claude/commands/` slash-command-as-router pattern documented in the repo's `CLAUDE.md` (router skills exposed as slash commands).

## Scope — DO

- Bring agent frontmatter examples into line with what real agents in the repo actually use (verify by sampling existing agents).
- Bring command frontmatter examples into line with the quoted-array convention used in real commands.
- Add a section pointing maintainers to `meta-sme-protocol` for SME-style agents.
- Document the slash-command-router pattern (skills exposed via `.claude/commands/`) and when to use it.
- Keep the pressure-resistance language and the workflow skeleton.

## Scope — DO NOT

- Do not refactor the workflow skeleton — it works.
- Do not introduce new processes the maintainer must follow.

## Acceptance criteria

1. Every frontmatter example in the pack compiles to the same shape as a randomly sampled real-repo agent / command.
2. Pack explicitly references `meta-sme-protocol` when discussing SME agents.
3. Pack documents the slash-command-router pattern with one concrete example.
4. `plugin.json` version bumped (minor).
5. Workflow skeleton unchanged.

## Process

1. Read `/tmp/skillpack-refresh-review/meta-skillpack-maintenance.md` for full evidence.
2. Read every SKILL.md in this pack.
3. Sample real agents and commands across plugins:
   - `head -15 /home/john/skillpacks/plugins/axiom-python-engineering/agents/*.md`
   - `head -15 /home/john/skillpacks/plugins/lyra-ux-designer/agents/*.md`
   - `head -10 /home/john/skillpacks/plugins/*/commands/*.md | head -100`
4. Diff actual conventions against the pack's examples, list the deltas.
5. Edit. Re-sample to verify alignment.
6. Bump `plugin.json` version.

## Constraints

- Touches only this pack (no edits to other plugins).
- No fabrication — every frontmatter key/format must be observed in real repo files first.
- If `meta-sme-protocol` was refreshed first (recommended order), reference its current section names.
