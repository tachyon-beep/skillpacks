# Refresh: meta-sme-protocol

**Verdict:** MEDIUM / S effort. Do this **first** in the refresh campaign — 50+ agents across the marketplace reference this protocol, so corrections here propagate.

## Context

- Pack path: `/home/john/skillpacks/plugins/meta-sme-protocol/`
- Full review: `/tmp/skillpack-refresh-review/meta-sme-protocol.md`
- Defines the SME (Subject Matter Expert) Agent Protocol used by other packs' specialist agents — fact-finding contract, output structure, confidence/risk grading.

## Why refresh

Identified issues:
- Dated tool-name examples (e.g. `firecrawl`, raw "LSP" references) that no longer reflect current Claude Code tooling.
- Missing modern-agent context: subagent dispatch via the Agent tool, MCP integration, optional JSON summary blocks for machine-readable verdicts.
- The four-section output contract itself (Findings / Confidence / Risks / Recommendations) is sound and **load-bearing** — 50+ agents conform to it.

## Scope — DO

- Update tool-name examples to current Claude Code tools (Read, Grep, Bash, Agent, WebFetch, WebSearch).
- Add a short note on subagent dispatch as one fact-finding option.
- Add an optional machine-readable summary block convention (so callers can parse verdicts).
- Mention MCP-server-backed fact-finding as legitimate (Filigree, sentry-style sources).

## Scope — DO NOT

- **Do not change the four-section output contract.** The section names, order, and required fields are referenced verbatim by ~50 agents elsewhere in the marketplace. Verify with: `grep -r "SME Agent Protocol" /home/john/skillpacks/plugins/`.
- Do not rename or restructure the protocol skill.
- Do not introduce new mandatory sections — additions must be optional.

## Acceptance criteria

1. All examples reference tools that actually exist in current Claude Code.
2. Output contract section names unchanged from current version (diff the section headings).
3. New optional sections clearly marked OPTIONAL.
4. Grep across plugins for `SME Agent Protocol` returns no broken references.
5. `plugin.json` version bumped (patch).

## Process

1. Read `/tmp/skillpack-refresh-review/meta-sme-protocol.md` for full evidence.
2. Read every SKILL.md in this pack (small pack).
3. `grep -r "SME Agent Protocol\|sme-agent-protocol" /home/john/skillpacks/plugins/ | head -50` — confirm what downstream agents depend on.
4. Plan minimal additive edits. Run by user before applying if structural questions arise.
5. Edit. Verify section headings unchanged via `git diff`.
6. Bump `plugin.json` version.

## Constraints

- Read-only on other plugins; this PR touches only `plugins/meta-sme-protocol/` + `plugin.json`.
- No fabrication of tool names or features — verify every tool/MCP-server name exists.
