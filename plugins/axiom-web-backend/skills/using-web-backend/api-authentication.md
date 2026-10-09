# Authentication and Authorization

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Trace identity proof → principal → delegated scopes → object/action permission at the enforcement point. Test tenant/object separation and denied access, not only valid tokens.

Document token/session lifecycle: issuance, expiry, revocation, refresh, audience/issuer, key rotation, storage and replay protection as applicable. Authenticate WebSocket/background/admin paths too. Keep credentials out of logs, errors and model-visible payloads.

Cookie/browser flows need their actual CSRF/same-site/CORS model; CORS is not authorization. Validate the configured identity provider/SDK against current primary documentation. Use dedicated threat review for unresolved trust boundaries.
