# GitHub and Azure DevOps Governance Recipes

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Preserved operational contracts from the former SDLC platform-integration leaf. Adapt to the installed service/version and repository policy using official documentation.

## Requirements and work records

Keep a stable requirement/work-item ID and links to design, pull request/change, test run and acceptance decision. Verify link semantics and permissions in the configured tracker; do not create a second backlog. GitHub issues/projects and Azure Boards are possible adapters, not mandatory choices.

## Configuration and review gates

Inspect branch protections/rulesets or Azure branch policies, required checks, bypass identities and the actual merge path. Verify required checks run on the intended commit and cannot be skipped by a parallel path. Review scope/independence follows selected policy; no duration/comment quota.

## Build/release and audit trail

Capture source revision, dependency/build inputs, immutable artifact digest, test/attestation references, promotion/deployment identity and observed health. Retain records according to actual policy. A pipeline YAML or Git history alone does not establish deployed state or complete audit history.

## Credentials and environments

Use scoped short-lived credentials where supported; separate environment state/config and verify secret handling. Inspect environment gates and deployment permissions. Do not echo secrets into logs or documentation.

## Measurements

Extract events with timestamp/timezone, identity, population/window and exclusions. Validate against real records; distinguish opened/merged/built/deployed/accepted states. Preserve query/calculation for reproduction.

## Platform migration

Inventory records, IDs, links, permissions, automation, secrets references and retention. Trial export/import with reconciliation; define mapping, cutover, ownership and rollback limits. Validate consumer workflows before retiring the old platform.

Official starting points: [GitHub Actions](https://docs.github.com/en/actions), [GitHub rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets), [Azure DevOps](https://learn.microsoft.com/en-us/azure/devops/). Verify current API/UI and policy semantics before configuring enforcement.
