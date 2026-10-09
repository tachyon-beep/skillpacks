# Database Transaction Boundaries

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Trace each operation's reads/writes and transaction owner. Check atomicity, isolation, uniqueness/CAS constraints and what happens on cancellation, timeout or retry.

External calls cannot usually share a local DB transaction; use an explicit consistency/recovery design (such as an outbox) where needed. Record committed effects separately from response delivery. Make idempotency durable when retries can follow a lost response.

Check connection/pool lifecycle, migrations, query/index shape and actual driver semantics for the supported version. For SQLite/DuckDB operational discipline use the embedded-database pack. ORM syntax does not establish transaction correctness.
