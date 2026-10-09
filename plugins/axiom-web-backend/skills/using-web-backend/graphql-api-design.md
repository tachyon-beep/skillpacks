# GraphQL Production Checks

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Authorize per resolver/object/action, including nested and batch paths. Validate query depth/complexity and result-size limits against expensive fixtures.

Check N+1 behavior with measured query counts; request-scoped loaders must not leak results across identities/tenants. Define mutation atomicity, errors and retry semantics.

Treat schema evolution and nullability as consumer contracts. Test variable/input coercion, partial errors, pagination and subscription lifecycle. Use [GraphQL specification](https://spec.graphql.org/) and the installed implementation's docs for exact semantics.
