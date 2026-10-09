# Mod boundaries and compatibility

Use when third-party content or code can change a persistent game/runtime. Reuse target-engine/platform integration rather than copying a generic workshop implementation.

## Contract

- Define supported extension points, data/code permissions, dependency/load order and API/version policy. Separate trusted code mods from sandboxed content.
- Validate schemas/identifiers and distinguish missing, incompatible and malicious content. Path containment, resource budgets and process/interpreter isolation depend on the actual runtime.
- Bind save/replay/content identity to required mod versions/configuration. Define missing-mod behavior, migration and rollback without silently corrupting state.
- Resolve conflicts deterministically with diagnosable precedence; dependency cycles and duplicate IDs need explicit failure behavior.
- Check hot reload/unload ownership, callbacks/resources, cancellation and partial initialization cleanup. Removing code does not automatically remove persistent effects.
- Preserve user control over enabling/removing mods and explain consequential compatibility choices. Policy/licensing/distribution decisions require the actual owner's terms.
- Test load-order changes, incompatible versions, malformed/oversized content, interrupted updates and save round trips with representative combinations.
- Measure runtime overhead and protect secrets/sensitive filesystem/network access at enforced boundaries; documentation alone is not a sandbox.

## Deliverable

Extension/save-compatibility contract or repair with tested failure cases, content/version identity and unresolved platform limits. Verify current engine/workshop APIs in upstream docs. See [sandbox affordances](sandbox-design-patterns.md) and security architecture when the trust model needs review.
