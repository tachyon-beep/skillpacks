# API Contract Evidence

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Exercise the real route/schema/auth/storage boundary appropriate to the change: success; invalid/missing/denied; conflicting version; duplicate/retry; persistence/partial failure; bounded results.

Use independent expected behavior and realistic fixtures. Mocks can isolate a dependency but cannot prove its integration contract. Check tenant/object permissions and error envelope, not merely response status.

Run focused tests and relevant integration/acceptance checks; report environment, revision, result and omissions. Add performance measurements only where the claim depends on them. No universal coverage/test-count ratio applies.
