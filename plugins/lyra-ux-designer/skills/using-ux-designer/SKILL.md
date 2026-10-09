---
name: using-ux-designer
description: "Use when designing or reviewing application interfaces, audience needs, interaction flows, accessibility, or AI surfaces across web, mobile, desktop, and games."
---

# Application UX

Ground design decisions in the product's purpose, actual audience, tasks, and
observed constraints. Existing chrome and familiar patterns must earn their
place; an attractive mockup is not usability evidence.

## Choose the useful pass

Infer the requested task from context. For a bounded edit, solve that interaction
and its relevant states. For a broader review or redesign:

1. Read product intent, current UI, constraints, and available user evidence.
2. State consequential assumptions. Separate observed user needs from proposed
   personas or preferences; ask only for information that changes the decision.
3. Trace each proposed surface to a user goal and decision. Revisit inherited
   premises when they no longer serve that goal; do not invent a quota of premises.
4. Specify interaction states: initial/empty, loading, success, failure, partial
   progress, permission denial, and recovery where applicable.
5. Check input modes, focus order, readable hierarchy, labels, contrast, zoom,
   reduced motion, and assistive-technology behavior relevant to the surface.
6. Choose the smallest useful validation: browser interaction, keyboard test,
   screen-reader test, observed task trial, or another discriminating experiment.
   Record what was actually tested and what remains a design hypothesis.

## Retrieve only a consequential reference

References are beside this file. Basic design questions need no compulsory
briefing. Read one or more sections only when they govern a material uncertainty.

| Need | Reference |
|---|---|
| Accessibility criteria and validation | [accessibility-and-inclusive-design.md](accessibility-and-inclusive-design.md) |
| User evidence and test design | [user-research-and-validation.md](user-research-and-validation.md) |
| Streaming, sources, steering, agent actions | [ai-experience-patterns.md](ai-experience-patterns.md) |
| Mobile, desktop, web, game-specific constraints | [mobile-design-patterns.md](mobile-design-patterns.md), [desktop-software-design.md](desktop-software-design.md), [web-application-design.md](web-application-design.md), [game-ui-design.md](game-ui-design.md) |
| General rubric | [ux-fundamentals.md](ux-fundamentals.md), [visual-design-foundations.md](visual-design-foundations.md), [information-architecture.md](information-architecture.md), [interaction-design-patterns.md](interaction-design-patterns.md) |

Use `lyra-tui-designer` for terminal substrate/lifecycle and
`lyra-site-designer` for static documentation-site implementation.

## AI and agent surfaces

Show task state, sources, stop/steering controls, consequential action previews,
results, and recovery. Distinguish retrieval evidence from model inference and
avoid confidence decorations unsupported by reliability measurements. Approval
friction should follow impact, reversibility, existing authorization, and the
active platform's controls; ordinary authorized edits need no repeated modal.

## Output and optional roles

Return a decision, design artifact, or prioritized findings with the triggering
state, evidence, consequence, and proposed remedy. Separate prototype/build,
automated checks, accessibility evaluation, user validation, and live acceptance.
Do not label a visual review a full accessibility audit or infer real audience
reactions from simulated personas.

`ux-critic`, `accessibility-auditor`, and `ux-theorist` are optional focused roles.
Use the theorist for unsupported audience/premise assumptions, the auditor for
criterion-based accessibility, and the critic for interaction review. Delegate
only when independent coverage improves the decision; no panel is required.
