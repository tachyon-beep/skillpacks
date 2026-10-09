# Reader Review Architecture

The pack supports two modes: a focused audience-lens review and an optional staged
cold-reading experiment. Both produce editorial hypotheses, not human audience
evidence. The staged mode preserves pre-exposure expectations and source-linked
observations; it is useful only when information order/navigation matters.

## Components and authority

- `skills/reader-panel-review/SKILL.md`: scope, runtime capabilities, coordination,
  persistent state, recovery and delivery contract.
- `process.md`: experimental contracts, journal/verdict/synthesis formats and
  evidence limits. Roles read only the relevant sections.
- `config-template.md`: minimal run configuration with optional richer metadata.
- `config.md`: illustrative larger panel; not a required size or cost baseline.
- Optional `persona-designer`, `persona-reader`, `panel-synthesiser` roles.
- Commands `panel-designer`, `panel-config`, `panel-review` expose these tasks.

## Staged mode

The coordinator validates sources and maintains each reader's supplied content,
position, expectations, observations, requests and failures. A reader saves A
(expectations); the coordinator verifies A and supplies only the requested chapter;
the reader saves B (observations) and C (next decision). Read-ahead/context
contamination is reported and may require a fresh probe.

Use actual runtime delegation/content restrictions when available. Hashed copies
can organize a chapter store but do not enforce file access. Tool declarations
may narrow capabilities on a supported host; they are not universal permission
grants. Never instruct a host to bypass permissions. If isolated contexts are
unavailable, use a labeled single-context lens review instead of pretending it
was a cold read.

## Synthesis and validation

Aggregate directly unless an independent synthesis role is useful. Check claims
against original passages. Distinguish inspectable text properties, simulated
audience interpretations/affect and institutional/commercial hypotheses. Repeated
persona outputs are correlated observations from this run, not independent survey
responses. Control matches and collision exercises check internal consistency,
not validity against real people.

Deliver actual coverage, prioritized source-grounded hypotheses, material
alternatives and the cheapest relevant human/mechanical validation. Keep model
judgment, stakeholder acceptance and publication separate. Optional journals and
manifest support reproducibility; no fixed persona count, chapter interval,
model family or thirteen-section synthesis is required.
