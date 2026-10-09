# Analyze Domain and Utility

## Inventory before recommendation

Read the task, intended consumers, repository instructions, current changes,
plugin metadata and canonical skill. Distinguish published skills, optional
references, commands, agents, hooks, generated wrappers, and fixtures. Record
entry points and actual dependencies rather than infer them from file names.

For this marketplace, inspect `plugins/<pack>/`, `.claude-plugin/marketplace.json`,
root wrappers, relevant scripts and existing review evidence. Do not count the
intentionally flawed fixture plugin as shipped capability.

## Map decisions, not an encyclopedia

For each component, record:

| Field | Question |
|---|---|
| Trigger | What user problem should activate it? |
| Incremental value | What does it add beyond a capable model and available tools? |
| Local knowledge | Which project/tool contracts cannot be inferred safely? |
| Output | What reviewable artifact or action results? |
| Verification | What would establish the artifact works? |
| Cost | What retrieval, delegation, latency or friction does it impose? |
| Overlap | Which existing owner already covers this? |

A missing textbook topic is not automatically a coverage gap. Add material only
when a real task or observed failure needs it. Stable principles can still be
misapplied; evolving APIs/standards require current primary-source verification.

## Disposition

Keep concrete workflows, environment adapters, persistent state contracts and
validated failure probes. Refocus broad routers around decisive conditions.
Merge duplicated ownership and move useful facts to optional references. Retire
rituals or components with no supported task value after updating consumers.
Separate direct content evidence from inferred usefulness and measured outcomes.

Ask about intent only when a consequential uncertainty remains. Report inventory,
recommendations, consumer impact and verification needed within the user's scope.
