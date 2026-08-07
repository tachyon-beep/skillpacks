# Response Shapes and Error Codes

Load this reference when parsing `--json` output or MCP responses, or when
deciding how to react to a failure.

## Response Envelopes (2.0)

- **Batch ops** → `{succeeded: [...], failed: [{id, error, code}, ...], newly_unblocked?: [...]}`.
  `failed` is always present (empty list if none); `newly_unblocked` is
  present only when non-empty (omitted when the op unblocked nothing). Pass `--detail=full` (CLI) or
  `response_detail="full"` (MCP) to get full records back.
- **List ops** → `{items: [...], has_more: bool, next_offset?: int}`.
  `next_offset` only appears when there is a next page.
- **Errors** → `{error: str, code: ErrorCode, details?: dict}`.

The issue ID is always `issue_id` in 2.0 — in MCP inputs, response payloads,
and CLI JSON. Status is always `status`; "state" was retired as a
user-facing word.

## ErrorCode — the complete set

Switch on `code`, never on message text. The full enum:

`VALIDATION`, `NOT_FOUND`, `CONFLICT`, `INVALID_TRANSITION`, `PERMISSION`,
`NOT_INITIALIZED`, `IO`, `INVALID_API_URL`, `FILE_REGISTRY_DISPLACED`,
`REGISTRY_UNAVAILABLE`, `LOOMWEAVE_REGISTRY_VERSION_MISMATCH`,
`LOOMWEAVE_OUT_OF_SYNC`, `BRIEFING_BLOCKED`, `STOP_FAILED`,
`SCHEMA_MISMATCH`, `INTERNAL`.

Branch on `code` for retry policy: `CONFLICT` → CLI exit 4, retryable
(another agent owns the claim — retry against a different issue); everything
at exit 1 needs operator intervention.

## Failure modes that deserve a specific response

- **`INVALID_TRANSITION`** — the workflow does not allow that status hop from
  here. Call `workflow_transition_list` (MCP) or `filigree transitions <id>`
  to see what *is* allowed, then walk it (or pass `--advance` / `advance=true`
  to walk the soft transitions automatically).
- **`SCHEMA_MISMATCH`** — the installed `filigree` is older than the project
  database. The error message contains upgrade guidance. Surface it to the
  user; do not retry.
- **`CONFLICT`** — someone else holds the claim, or the record changed under
  you. Safe to retry against different work; never force-overwrite.
- **`ForeignDatabaseError`** — filigree found a parent project's database but
  no local `.filigree.conf`. Run `filigree init` in the current directory. Do
  **not** `cd` upward to a different project unless that was the actual intent.
