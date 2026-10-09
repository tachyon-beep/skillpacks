# Contributing

Start with the user-visible problem, existing source contract and affected consumers. Keep activation specific: a strong model should load a skill for useful constraints or verification, not because a broad keyword appeared. Answer routine questions directly.

## Content and packaging

A plugin needs matching catalog/metadata names and an existing source directory. A skill needs YAML frontmatter with `name` and `description`, a concise task/evidence contract and valid local links. Keep deep recipes in optional references; do not reproduce entire workflows in commands and wrappers.

Preserve non-obvious failure checks, project authority, persistent state and meaningful artifact fields. Size deliverables to the request and risk. No fixed fan-out, elapsed-time quota, mandatory disagreement or hostility to a clean review belongs in a default workflow. Examples are not universal policy, external standards or authorization.

Each plugin is independently installable. Agent evidence rules belong inline; cross-pack guidance is optional and should identify the relevant concern. Repository shortcuts link to plugin source. Exclude operational skills and intentional fixtures from published counts.

## Validation

```bash
python3 scripts/check_marketplace_integrity.py
python3 scripts/check_skillpack_contracts.py
git diff --check
```

The frontmatter checker requires PyYAML in the Python environment.

For content-only changes, use direct source/link/schema checks and an independent reader where useful. Test modified executable recipes against the relevant runtime when feasible; state omissions. Do not add unit tests that merely mirror prose.

For a claimed skill benefit, compare task outcomes, defects, evidence quality and context/tool cost with an appropriate baseline on representative cases. Skill obedience, an invented persona's agreement and a passed local syntax check do not establish real user value. Keep empirical claims bounded by the observations actually made.

## Delivery

Preserve unrelated work, use a feature branch and stage only the intended change. Use conventional commits. Version affected plugins; use a major bump for removed entrypoints or materially incompatible default contracts. Update the catalog, README, shortcuts and migration documentation together. Distinguish implementation, local checks, CI, deployment and user acceptance. Do not publish or push unless authorized.

See [the evidence contract](docs/sme-agent-protocol.md) and [October 2026 consolidation](docs/relevance-refresh.md) for the current rationale. Historical artifacts remain in Git history and clearly labeled review files.
