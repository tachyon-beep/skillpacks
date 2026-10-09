# Debugging and Refactoring Evidence

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

## Debugging

- Trace the actual failing entry point and state transition. Record observed behavior, impact, environment and a minimal reproduction or specific missing evidence.
- Separate hypotheses from observations. Choose a discriminating check: what result would falsify this explanation, and what alternative would remain?
- Use available execution, logs, debugger or instrumentation directly within authorization; ask for user-run evidence only when access is unavailable.
- Change one relevant cause at a time where possible. Verify the failing behavior and nearby failure/compatibility paths; a green unrelated suite is weak evidence.
- If investigation stalls, revisit the baseline, reproduce assumptions and counterevidence. Do not keep accumulating patches around an unfalsified story.

## Refactoring

- Name the behavior to preserve and the reason for change. Inspect callers, state ownership, public interfaces and operational assumptions before moving code.
- Characterize consequential existing behavior when tests are missing; do not freeze a known bug as intended behavior without a decision.
- Choose a bounded integration seam. For live compatibility requirements, expand/migrate/contract with named consumers and removal criteria; where hard cuts are authorized, remove obsolete paths rather than invent shims.
- Keep unrelated work intact. Run checks that exercise affected contracts and report any unverified deployment/data effects.

## Review comment

Location and evidence → observable consequence → suggested correction/check. Distinguish defect, tradeoff and uncertainty. A clean review is valid when supported by the inspected scope; tone and finding counts do not establish rigor.
