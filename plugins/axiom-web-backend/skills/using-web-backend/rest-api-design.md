# REST Compatibility

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Specify method/resource, request/response/error schema, authentication and per-object/action authorization. Define invalid/missing/denied/conflict behavior and safe retry semantics.

Bound collection results with stable ordering/cursors and explicit truncation. State idempotency/concurrency/preconditions for mutations. Compatibility includes behavior, meaning and error semantics, not just field syntax; test existing consumers before removing or changing fields.

Use the project's OpenAPI/versioning conventions. Check current HTTP semantics in [RFC 9110](https://www.rfc-editor.org/rfc/rfc9110). A naming convention alone does not prove compatibility.
