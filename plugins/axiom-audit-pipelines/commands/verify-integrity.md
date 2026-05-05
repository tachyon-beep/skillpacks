---
description: Verify the integrity of an existing audit trail. Replays the chain against signatures and fingerprints, confirms anchor provenance, locates gaps and redactions, and reports findings as a verification statement. Dispatches the integrity-auditor agent for the actual verification.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "AskUserQuestion"]
argument-hint: "[trail_path_or_export_envelope]"
---

# Verify Integrity Command

You are verifying the integrity of an audit trail or an exported subset. The output is a *verification statement* — what was processed, what was anchored, what gaps and redactions were found, and whether the integrity claim holds.

This command does NOT design or scaffold. For design, use the `using-audit-pipelines` skill. For scaffolding, use `/scaffold-audit-trail`. For "what counts as a decision," use `/design-decision-log`.

## Invocation Path

`/verify-integrity` dispatches the `integrity-auditor` agent with the inputs and emits a verification statement to disk. The agent does the verification work; this command resolves inputs, frames the scope, and consolidates output.

## Preconditions

The command takes a single argument: a path to a trail directory, an export envelope file, or a chain segment.

### Resolve the argument

```bash
INPUT="${ARGUMENTS}"

if [ -z "${INPUT}" ]; then
  # AskUserQuestion:
  # "What are we verifying? Provide one of:
  #    - a path to a trail directory (entries + chain anchor),
  #    - a path to an export envelope file,
  #    - a path to a chain segment (entries plus anchoring metadata)."
  :
fi

if [ -d "${INPUT}" ]; then
  echo "Verifying trail directory: ${INPUT}"
  TYPE="trail"
elif [ -f "${INPUT}" ]; then
  # Inspect to distinguish export envelope vs single segment file
  echo "Verifying file: ${INPUT}"
  TYPE="file"
else
  echo "ERROR: ${INPUT} not found."
  exit 1
fi
```

### Identify the corresponding spec

The verification rules come from the trail's `99-audit-pipeline-specification.md`. Locate it:

```bash
# Adjacent to the trail
ls "${INPUT}/../audit-pipeline/" 2>/dev/null
ls audit-pipeline/ 2>/dev/null

# Or asked from the user
```

If no spec is available, the verification proceeds in *spec-inferred* mode — the auditor infers chain construction, hash function, signature scheme, and canonicalisation from the trail itself, and reports inference confidence in the verification statement. Unspecced verification is weaker than specced verification; flag this in the output.

## Workflow

### Step 1 — Frame the scope

Determine:

- **What is being verified.** Full trail, contiguous segment from M to N, export envelope, single batch.
- **What anchor is being used.** Self-anchored (chain head as published by producer), externally-anchored (notary-cosigned, public-ledger-committed), or unanchored (no external evidence; verification verifies internal integrity only).
- **What public keys / KMS endpoints are needed.** Resolve `key_id`s in the trail to public keys; for HMAC trails, identify the verifier's key holder.

If externally-anchored, fetch the anchor from its publication channel via an *independent* path — not through the producer. If only the producer's anchor is available, state in the output that anchor provenance is producer-attested.

### Step 2 — Dispatch the integrity-auditor agent

Use the Task tool with subagent_type: "integrity-auditor". Provide:

- Path to the trail / segment / envelope.
- The `99-` spec (or "spec-inferred mode").
- The chain anchor and its provenance.
- Public keys / KMS endpoints.
- The verification scope.

The agent performs:

- Walk the chain from anchor forward (linked-hash) or verify inclusion proofs (Merkle).
- Re-hash each entry's canonical bytes; compare to stored `entry_hash`.
- Verify signatures against `key_id`-resolved public keys.
- Detect gap-markers and redaction events; record them in the report.
- For Merkle: verify consistency proofs across batches.
- Report time-trust posture (notary anchors used, time bounds achieved).

### Step 3 — Consolidate verification statement

Write the verification statement to `${INPUT}/verification-statement-YYYY-MM-DD.md`:

```markdown
# Verification Statement

- **Verified**: <trail / segment / envelope path>
- **Spec**: <99-spec reference, or "spec-inferred">
- **Verifier**: <this command + agent + library versions>
- **Anchor**: <hash, provenance source, signature, time>
- **Scope**: entries from <first> to <last>, count = N
- **Result**: PASS | FAIL | PARTIAL

## Findings

### Chain integrity
- Entries verified: N / N
- Hash mismatches: 0
- Chain-link breaks: 0

### Signatures
- Entries with valid signatures: M / N
- Failed signatures: list with entry_ids and key_ids

### Gaps and redactions
- Gap markers: list with entry_ids, scope, evidence references
- Redactions: list with entry_ids, redaction-event references

### Time
- Notary anchors used: list with timestamps
- Time bounds: <range>

## Caveats
- <e.g., "spec-inferred mode; signature scheme assumed Ed25519 from key_id format">
- <e.g., "anchor obtained only from producer; external corroboration absent">

## Result statement
<plain-language summary suitable for the consumer of the verification>
```

### Step 4 — Sign the verification statement

The verification statement is itself signed by the verifier's signing key (separate from the producer's). The signature binds *who verified, what, and when* — an auditor's verification statement is meaningful evidence only if the auditor's identity is established.

For internal verifications without an external auditor, the verifier identity is the running operator + the verifier tool's code-version. For external verifications (regulator, third-party auditor), the verifier signs with their organisational key.

### Step 5 — Output and handoffs

The verification statement is the deliverable. Suggest:

- For PASS: file the statement; communicate to the requesting party (auditor, regulator, internal compliance).
- For FAIL: invoke incident response; the failure is itself an audit event — the trail's owners must record it as `audit.integrity-failure` in the audit pipeline (see `using-audit-pipelines/SKILL.md` for the schema).
- For PARTIAL: document the partial scope; gaps and redactions need explanation; producer's records corroborated against external evidence.

## Failure-Mode Handling

| Failure | Action |
|---------|--------|
| Hash mismatch on a single entry | Report entry_id; do NOT modify; this is a tamper-or-corruption finding |
| Signature failure with valid hash | Possible key compromise; check `04-` revocation records |
| Multiple consecutive hash failures | Possible storage corruption; suggest backup restore (which is itself an audit event) |
| Gap marker without supporting evidence | The gap is sealed but cause is unverified; report as PARTIAL |
| Anchor unavailable | Verification is internal-only; explicitly stated in caveats |
| Spec-inferred mode finds inconsistency | The trail does not match its inferred form; possibly multiple chain segments concatenated, or genuine corruption |

## Output Location

Verification statements are written to the trail directory (or alongside the export envelope), with a date-stamped filename. They are not modified after writing; subsequent verifications produce new statements.

## Downstream Handoffs

- A FAIL or PARTIAL result triggers incident response per `axiom-sdlc-engineering` runbooks.
- The verification statement becomes input to compliance reporting, regulator submissions, or court evidence as applicable.
- For routine verifications, the cadence (per `10-performance-budget.md`) determines the next run.

## Scope Boundaries

Covered: chain integrity verification, signature verification, gap and redaction reporting, time-trust assessment, statement output.

Not covered: investigation of *why* an integrity failure occurred (incident response); remediation of broken chains (chain-breaking-event procedures live in `03-`); legal-evidence chain-of-custody beyond the verification statement itself (legal/compliance owns this).

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Verifying using producer-supplied anchor with no independent path | Fetch anchor from external source; flag as caveat if not available |
| Spec-inferred mode treated as full verification | Lower confidence; explicit caveat in statement |
| Verification statement unsigned | Statement is unauthenticated; sign with verifier's key |
| FAIL silently retried until PASS | The FAIL is the finding; do not paper over with retry |
| Verifying without resolving redactions | Redacted entries are valid; misclassifying as failure is wrong |
| Missing anchor provenance check | Producer-attested-only anchor weakens the verification |
