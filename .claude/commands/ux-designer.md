---
description: Use when starting any UX/UI design task — visual design, IA, interaction, accessibility (WCAG 2.2), user research, mobile/web/desktop/game/AI surfaces, plus first-principles audience-derived needs analysis. Routes to 11 specialist sheets, 3 commands, 3 agents (ux-critic, accessibility-auditor, ux-theorist).
---

# UX Designer Routing

**Accessibility constraints make design better for everyone, not just users with disabilities. Different UX tasks need different competencies — load only what the surface and the brief actually require. For static documentation/marketing sites use `/site-designer`; for prompt-engineering and model selection use `/llm-specialist`.**

Use the `using-ux-designer` skill from the `lyra-ux-designer` plugin to route to the right specialist sheet. Content authority lives in `plugins/lyra-ux-designer/skills/using-ux-designer/SKILL.md` — this wrapper is a thin pointer.

## When to Use

- Starting any UX/UI design task across mobile / web / desktop / game / AI surfaces
- Critiquing or reviewing an existing interface
- Designing a new interface or feature with accessibility built in
- Auditing for WCAG 2.2 AA compliance
- Relitigating inherited UX premises that prior reviews have ossified

**Don't use** for: backend logic, database design, prompt engineering (`/llm-specialist`), static documentation/marketing sites (`/site-designer`), security threat modelling (`/security-architect`).

## Sheets

### Foundations
- **ux-fundamentals** — core UX principles, mental models, design-thinking framing
- **visual-design-foundations** — colour, typography, hierarchy, layout, contrast (Visual Hierarchy Analysis Framework)
- **information-architecture** — navigation systems, content organisation, findability, faceted browsing
- **interaction-design-patterns** — affordances, feedback, states, motion, errors; enumerates the WCAG 2.2 new SCs
- **accessibility-and-inclusive-design** — Universal Access Model (6 dimensions), WCAG 2.2 AA, inclusive design — referenced by every other sheet
- **user-research-and-validation** — interviews, usability testing, mental models, journey mapping (5-phase User Understanding Model)

### Platform Extensions
- **mobile-design-patterns** — iOS 17+ HIG, Material 3, touch targets, gestures, thumb zones, platform conventions
- **web-application-design** — SaaS / dashboards, data tables, command palette, responsive, progressive enhancement (not for marketing sites)
- **desktop-software-design** — multi-window, keyboard-first, workspace customisation, power-user paths (Windows / macOS / Linux / Electron)
- **game-ui-design** — HUD, diegetic UI, controller / gamepad, immersion-vs-visibility, performance

### AI / Conversational Surfaces
- **ai-experience-patterns** — AI-UX Trust Stack (Legibility, Grounding, Steering, Refusal & Recovery, Reversibility, Calibration) for chat assistants, agent loops, RAG UIs, streaming output, citation hover-cards, preview-then-confirm for tool actions; AI-specific WCAG 2.2 traps (aria-live token buffering, prefers-reduced-motion on streaming cursor, SC 3.3.8)

## Commands

- `/lyra-ux-designer:design-review` — multi-competency critique across visual, IA, interaction, accessibility; add `ai-experience-patterns` if surface is a chat / agent / AI interface
- `/lyra-ux-designer:create-interface` — design a new component with platform-aware patterns and accessibility built-in; add `ai-experience-patterns` if the component is an AI surface
- `/lyra-ux-designer:accessibility-audit` — full WCAG 2.2 AA audit using the 6-dimension Universal Access Model, including the six new 2.2 success criteria (2.4.11, 2.5.7, 2.5.8, 3.2.6, 3.3.7, 3.3.8)

## Agents

- `ux-critic` — multi-competency design review against best practice (visual + IA + interaction + accessibility); the default critic
- `accessibility-auditor` — WCAG 2.2 AA compliance specialist applying the Universal Access Model; produces remediation-prioritised findings
- `ux-theorist` — first-principles needs derivation; invoke BEFORE design review when prior reviews keep waving inherited decisions through (premise relitigation)

All three follow the SME Agent Protocol with Confidence / Risk / Information Gaps / Caveats sections.

## Routing by Surface

- AI / chatbot / agent / RAG / streaming → `ai-experience-patterns` + `interaction-design-patterns` + `accessibility-and-inclusive-design` (+ platform extension)
- Mobile feature design → `visual-design-foundations` + `interaction-design-patterns` + `accessibility-and-inclusive-design` + `mobile-design-patterns`
- Web dashboard redesign → `information-architecture` + `visual-design-foundations` + `web-application-design` + `accessibility-and-inclusive-design`
- Game HUD → `visual-design-foundations` + `game-ui-design` + `accessibility-and-inclusive-design`
- "Explain X" / teaching → `ux-fundamentals`

## Cross-references

- Static documentation / marketing site IA and styling → `/site-designer`
- Prompt engineering, model selection, RAG retrieval quality → `/llm-specialist`
- Documentation UX, microcopy, UI text → `/technical-writer`
- Authentication UX threats, secure-feedback design → `/security-architect`
