# Refresh: yzmir-ai-engineering-expert

**Verdict:** HIGH / S effort. Small but high-leverage — this is the top-level Yzmir router.

## Context

- Pack path: `/home/john/skillpacks/plugins/yzmir-ai-engineering-expert/`
- Full review: `/tmp/skillpack-refresh-review/yzmir-ai-engineering-expert.md`
- Purpose: top-level router that directs AI/ML questions to the appropriate Yzmir specialist pack.

## Why refresh

The router lists 6 downstream packs but **9 sibling `yzmir-*` packs exist** in the marketplace. Missing:
- `yzmir-dynamic-architectures`
- `yzmir-simulation-foundations`
- `yzmir-systems-thinking`

Verify with `ls /home/john/skillpacks/plugins/ | grep yzmir`.

Routing signals are also dated:
- No reasoning-model keywords (o-series, extended thinking, R1, Gemini thinking)
- No agentic-systems keywords (tool-use loops, multi-agent, MCP)
- No multimodal-by-default routing
- No diffusion model routing
- No RAG keywords

Pressure-resistance scaffolding is solid but oversized relative to the actual routing logic.

## Scope — DO

1. Add the 3 missing sibling packs to the router catalog with one-line "use when" descriptions.
2. Add 2026-era routing signals to the decision table:
   - reasoning models → `yzmir-llm-specialist`
   - agentic / tool-use → `yzmir-llm-specialist` + cross-ref to `axiom-engineering-foundations`
   - multimodal → `yzmir-llm-specialist` and/or `yzmir-neural-architectures`
   - diffusion → `yzmir-neural-architectures`
   - RAG → `yzmir-llm-specialist`
3. Trim pressure-resistance prose if disproportionate to routing content (judgment call — keep if it's working).

## Scope — DO NOT

- Do not change the router's decision-tree shape if it's working.
- Do not add content that belongs in downstream specialist packs.
- Do not reference packs that don't exist.

## Acceptance criteria

1. `ls /home/john/skillpacks/plugins/ | grep yzmir | wc -l` matches the count in the router catalog.
2. Every `yzmir-*` pack mentioned in the router exists; every existing `yzmir-*` pack is mentioned (or explicitly excluded with reason).
3. Router includes routing signals for: reasoning, agentic, multimodal, diffusion, RAG.
4. `plugin.json` version bumped (minor).

## Process

1. Read `/tmp/skillpack-refresh-review/yzmir-ai-engineering-expert.md`.
2. Read this pack's router skill end-to-end.
3. List all `yzmir-*` packs and their `description` from each `plugin.json`.
4. Edit catalog + decision table.
5. Bump version.

## Constraints

- Read-only on downstream packs.
- No fabrication of pack names — verify `plugin.json` for each.
- Router stays a router; resist the urge to teach AI/ML here.
