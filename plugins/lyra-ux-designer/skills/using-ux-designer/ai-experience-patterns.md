# AI and Agent Interface Contracts

Design for legibility, grounding, steering, recovery, reversible action and honest
calibration. Treat these as interaction contracts to verify, not decoration or
universal claims about every audience. Backend model/evaluation quality and
security controls need their own evidence.

## State and steering

Expose the current useful task state: waiting, retrieving, using a tool, streaming,
completed, stopped, partially completed or failed. Do not fake determinate progress
without a basis. Provide reachable stop/cancel and explain which work or effects
remain after cancellation. Preserve useful partial output and user edits.

Let users refine or correct the task without discarding work: edit inputs, adjust
constraints, accept/reject relevant hunks, retry a failed step or continue from
known state. Do not auto-regenerate on incidental edits or treat closing the UI
as proof that server-side actions stopped.

For long work, surface consequential budget limits and recovery at exhaustion.
Use real runtime signals; avoid invented ETAs, token counts or hidden failure.

## Grounding

Attach source references to the claims they support and make the actual source
span accessible. Distinguish quotes, paraphrases and inference; quotations can
also be misleading without context. Cite sources actually retrieved and checked,
not plausible generated URLs. Label when retrieval found no useful evidence and
an answer relies on general knowledge. A source list alone does not prove entailment.

Source panels should support verification without overwhelming the task. Retrieval
scores are not truth probabilities. Do not infer calibrated correctness from a
model's self-reported confidence, raw token probability or stylistic certainty.

## Actions and authorization

Approval policy follows impact, reversibility, actual user authorization and host
controls. Show concrete previews when they let users evaluate consequential
choices. Existing authorization can cover routine reversible edits or reads;
do not require another confirmation for every tool call.

For actions outside the authorized scope or genuinely consequential irreversible
choices, obtain the required approval through the platform. Keep action enforcement
outside model prose. Distinguish proposed, authorized, attempted, completed and
failed actions. Never imply a preview or a click guarantees downstream execution.

Expose undo/recovery when supported and be honest about its limits. Keep an
action/result record useful to the user. Do not prescribe type-to-confirm or a
second modal as a universal rule; choose friction based on the real consequence.

## Failure and uncertainty

Report the relevant cause and actionable retry/edit/skip/recovery without leaking
secrets or internal policy detail. Distinguish empty retrieval, inaccessible source,
tool failure, refusal and uncertain answer. Do not silently transform failures into
successful-looking summaries or bury useful content under disclaimer boilerplate.

Confidence cues require measured reliability in the relevant conditions. If that
evidence is absent, communicate concrete uncertainty and missing evidence rather
than display an invented percentage. A generic caveat on every answer is not
calibration. Verify important claims against sources or task-specific checks.

## Accessibility

Use semantic controls, labels, keyboard/focus behavior and relevant WCAG/platform
criteria. Streaming announcements should preserve comprehension without reading
every token repeatedly; test buffering/live-region behavior with actual assistive
technology. Preserve a usable stop control, reading order and reduced motion.
Treat cognitive load and authentication alternatives as concrete design/test
questions, not assumptions that dense output is acceptable because it is fluent.

See [accessibility-and-inclusive-design.md](accessibility-and-inclusive-design.md)
and [user-research-and-validation.md](user-research-and-validation.md).

## Validation and output

Exercise relevant streaming, stop, retry, edited-input, partial acceptance,
empty-source, tool-failure and authorized-action paths. Inspect actual behavior and
logs/results; a mockup cannot verify execution or recovery. Report tested paths,
remaining uncertainty and any need for human task validation. Synthetic personas
can suggest hypotheses but cannot establish real trust, comprehension or acceptance.
