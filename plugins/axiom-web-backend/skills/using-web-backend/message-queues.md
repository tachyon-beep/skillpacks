# Message Delivery and Recovery

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Declare delivery/ordering scope, identity/deduplication, retry budget, poison-message disposition and observability. A broker ACK is not proof a business effect happened.

Commit processed-message identity and effect atomically where possible. Coordinate DB+publish with an outbox/inbox or another explicit atomic bridge; test crash after effect/before ACK and duplicate/out-of-order delivery.

Bound buffers/in-flight work and define backpressure/shedding. Track logical operation versus attempt, and acceptance/status returned to callers. Verify chosen broker/client semantics in its official supported-version docs; do not label delivery exactly-once without scope/proof.
