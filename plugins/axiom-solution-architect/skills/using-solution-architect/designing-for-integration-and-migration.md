# Integration and Migration Contracts

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Name provider/consumer, interface/schema version, identity/authority, deadlines, rate/size limits, compatibility and observable errors. State retry/idempotency and ordering assumptions; trace external side effects and partial success.

For migration, inventory old/new producers, readers and stored data. Define mapping/validation, coexistence or hard-cut policy, cutover trigger, ownership and rollback limits. A rollback of binaries may not reverse schema/data changes.

Test duplicate/out-of-order/missing messages, interrupted cutover and mismatched versions where they apply. Rehearse recovery with a realistic fixture or environment when risk warrants it. Link acceptance to actual consumer evidence, not merely provider completion.

Output: contract table, migration sequence, validation/reconciliation checks, owner and evidence required before cutover. Honor authorized hard cuts; do not add compatibility layers by default.
