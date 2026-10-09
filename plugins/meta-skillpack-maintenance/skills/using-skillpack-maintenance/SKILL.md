---
name: using-skillpack-maintenance
description: "Use when reviewing, simplifying, extending, or validating this marketplace’s plugins, skills, commands, agents, references, hooks, or packaging."
---

# Skillpack Maintenance

Maintain useful task outcomes and actual repository contracts. A skill is not
valuable merely because an agent follows it or its files parse.

## Establish scope and custody

Read the task, repository instructions, current diff, relevant manifests and
existing checks. Preserve unrelated work. Infer clear authorized scope; ask only
when a consequential decision is unresolved. Treat a review as read-only unless
changes are requested. Existing authorization persists through routine phases.

Inventory relevant published components separately from test fixtures, generated
wrappers, and historical reviews. The deliberately flawed plugin under
`.test-fixtures/flawed-plugin/` is a negative test corpus, not a published pack.

## Evaluate incremental value

Identify the concrete failure the component prevents, the project/tool knowledge
it supplies, and the artifact or check it enables. Compare against the model's
baseline and available native tools. Prefer a short evidence/output contract,
selective reference retrieval, or executable validation over generic lectures,
mandatory ceremonies, model personas, and duplicate framework introductions.

For substantial behavior changes use representative scenarios with the same
model/tools under no skill, current skill, and concise candidate. Score task
correctness, verification, user friction, cost/latency, and regressions. Separate
model behavior from tool failure. If no comparative run is practical, report that
limit; do not claim measured uplift. See [testing-skill-quality.md](testing-skill-quality.md).

## Implement within the approved task

1. Trace callers, wrappers, agents, references, packaging, and any parsed output
   schemas before deleting or changing a public contract.
2. Rewrite the canonical source, remove obsolete duplicate instructions, and
   update consumers together. Reference facts and examples remain on demand.
3. Keep descriptions discoverable by task/symptom. Use actual platform metadata
   conventions; tool declarations are restrictions in some runtimes.
4. Validate syntax, local links, catalog/manifest consistency and changed workflows.
   Run repository checks appropriate to the change, including
   `python scripts/check_marketplace_integrity.py` when working in this repository.
5. Update versions, changelogs, and generated wrappers as required by the current
   repository release policy; do not hand-edit generated outputs blindly.

No external authoring plugin is mandatory. An installed authoring/evaluation
skill can help, but its absence does not block a tested change. Do not require
uniform SME headings or a particular model family unless a real consumer needs
them. Honor the active platform's available tools and permission boundaries.

## References and report

- [analyzing-pack-domain.md](analyzing-pack-domain.md): scope, inventory, utility.
- [reviewing-pack-structure.md](reviewing-pack-structure.md): packaging and consumers.
- [testing-skill-quality.md](testing-skill-quality.md): outcome-based comparisons.
- [implementing-fixes.md](implementing-fixes.md): edits, validation, release custody.

Load only the relevant section. Report material changes, exact checks and
results, unresolved coverage, and release state. Distinguish source edits,
local validation, commit, publication, and actual user acceptance.
