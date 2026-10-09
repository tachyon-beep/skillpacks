# Express Version-aware Recipe

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Identify supported Node/Express/TypeScript versions from manifests, lockfile and deployment images. Do not adopt a hardcoded tutorial image tag as a support policy.

Trace middleware ordering, identity/authorization, input validation, error propagation and response lifecycle. Verify version-specific async exception handling in the installed Express major version.

Check timeouts, body/result limits, connection shutdown and background task ownership. Test denied/invalid/error/retry paths. Use [Express docs](https://expressjs.com/) and [Node docs](https://nodejs.org/docs/latest/api/) matching the selected runtime; preserve existing structure unless a real change needs it.
