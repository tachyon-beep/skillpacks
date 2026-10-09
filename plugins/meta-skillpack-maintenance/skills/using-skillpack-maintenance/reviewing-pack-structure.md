# Review Pack Structure

## Inspect contracts

- Skill frontmatter has a valid name and a task/symptom description.
- Commands have arguments and tool declarations appropriate to their runtime.
- Agent prompts have bounded roles, available capabilities and useful outputs.
- Relative references, pack names, command names and declared dependencies resolve.
- Hooks match supported events and have executable, scoped scripts.
- Catalog metadata, plugin metadata and generated wrappers agree with canonical sources.
- Fixtures and historical reports are not mistaken for active product instructions.

In this repository, run `python scripts/check_marketplace_integrity.py` for
catalog/pack/command cross-references. It intentionally does not validate every
relative Markdown link; check changed real links separately. Illustrative links
inside examples need not name actual repository files.

## Runtime boundaries

Metadata varies across hosts. Tool declarations may restrict rather than grant
capabilities; inspect the host's actual semantics. Do not copy tool/model names or
permission bypass settings into another runtime as universal requirements.
Use available search/read/edit/validation capabilities directly. Optional specialist
roles should not become mandatory delegation for routine work.

## Findings

Classify actual effects: broken activation or packaging; conflicting instructions;
missing consequential workflow; duplicate ownership; stale API/standard details;
or low-value prose. Do not assign severity from an arbitrary percentage of domain
topics missing. Evidence should identify file/location, failing path or consumer,
consequence and repair.

Preserve useful schemas when a real consumer parses them. General reviewers need
concise evidence, uncertainty and remaining checks, not uniform ceremonial headings.
Before deleting a protocol or component, trace references and parsed contracts.

For duplicate components, keep one canonical owner and migrate unique content;
update callers, descriptions, wrappers, versions and validation together.
