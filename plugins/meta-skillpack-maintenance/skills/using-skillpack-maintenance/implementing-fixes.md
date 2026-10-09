# Implement Pack Changes

Work within the user's authorized scope and preserve unrelated changes. A prior
approved review or explicit implementation request supplies authority for routine
reversible edits; do not invent another approval phase.

## Change coherently

1. Inspect current status, canonical sources, callers, links and output consumers.
2. Keep the useful capability while removing obsolete duplicate instructions.
   Prefer a concise skill plus optional references or executable checks.
3. Update commands/agents that would reintroduce a retired question, orchestration
   ritual, permission assumption or output contract. Change actual parser/schema
   consumers together when required.
4. Keep test fixtures intentionally defective; do not repair them as product code.
5. Regenerate wrappers using the repository mechanism if one exists. Update release
   metadata under the current policy, including major changes for removed public
   components/contracts. Inspect the final diff rather than assume regeneration
   preserved unrelated content.

## Validation

Run repository catalog/integrity checks and targeted syntax/link/schema checks.
Exercise material workflow changes with representative task inputs and inspect
resulting artifacts. Check tools and supported runtime semantics rather than
invent tool availability. Record any model comparisons or runtime tests not run;
static editing alone cannot establish incremental model uplift.

New skills need a concrete trigger, useful output and appropriate validation;
no external authoring plugin or mandatory red/green ritual is a dependency.
Installed authoring tools may help where their workflow serves the task.

## Delivery custody

Report changed capability, reasons, checks/results, remaining limits and exact
release state. Commit, push, marketplace publication and user acceptance are
separate actions. Follow the user's instructions and repository release policy
for them; do not include unrelated dirty work in a commit.
